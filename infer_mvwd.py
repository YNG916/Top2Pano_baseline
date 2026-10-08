"""Shared occupancy, independent perspective diffusion queries; no target reads."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mvwd_runtime import REPO, load_config, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--robot-assets")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--robots", type=int, nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    if args.robot_assets:
        config["data"]["robot_assets_root"] = str(Path(args.robot_assets).expanduser().resolve())
    dataset = make_dataset(config, args.split, include_targets=False, scenes=args.scene,
                           max_episodes=args.max_episodes, frames=args.frames, robots=args.robots)
    dataset[0]  # Check SAM and robot states before model allocation.
    if dataset.robot_assets is not None:
        config["robot_asset_provenance"] = dataset.robot_assets.provenance()
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data._utils.collate import default_collate
    from adapters.camera_geometry import PerspectiveRenderer
    from adapters.mvwd_raw import sha256_file
    from mvwd_model import load_trained
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model = load_trained(config, args.checkpoint).to(args.device).eval()
    renderer = PerspectiveRenderer(**config["renderer"])
    settings = config["inference"]
    metadata = {"method": "Top2Pano-Persp", "dataset": dataset.provenance(), "config": config,
                "checkpoint_sha256": sha256_file(args.checkpoint), "ground_truth_loaded": False,
                "robot_conditioning": dataset.robot_assets is not None,
                "seed_policy": "sha256(base_seed, release_id, episode_id, robot_id, frame_index)"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    key, voxel = None, None
    with torch.no_grad(), (output / "predictions.jsonl").open("w") as manifest:
        for index in range(len(dataset)):
            sample = dataset[index]
            batch = default_collate([sample])
            batch = {name: value.to(args.device) if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
            if sample["bev_key"] != key:
                # Stable scene geometry independent of query ordering/seed.
                geometry_identity = f"{settings['seed']}:{dataset.raw.metadata['release_id']}:{sample['bev_key']}"
                torch.manual_seed(int(hashlib.sha256(geometry_identity.encode()).hexdigest()[:8], 16))
                voxel = model.predict_occupancy(batch["bev_rgb"])
                key = sample["bev_key"]
            identity = f"{settings['seed']}:{dataset.raw.metadata['release_id']}:{sample['episode_id']}:{sample['robot_id']}:{sample['frame_index']}"
            seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            coarse = renderer(voxel, batch)
            image = model.refine(coarse["rgb"], coarse["depth_condition"],
                                 settings["ddim_steps"], settings["guidance_scale"])
            directory = output / sample["episode_id"] / f"robot_{sample['robot_id']:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"frame_{sample['frame_index']:03d}.png"
            pixels = ((image[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(destination)
            saved = {name: coarse[name][0].cpu().numpy() for name in
                     ("rgb", "depth_z", "robot_id", "robot_weight", "robot_depth_z") if name in coarse}
            np.savez_compressed(destination.with_suffix(".coarse.npz"), **saved)
            manifest.write(json.dumps({"episode_id": sample["episode_id"], "robot_id": sample["robot_id"],
                                       "frame_index": sample["frame_index"], "seed": seed,
                                       "path": str(destination.relative_to(output))}) + "\n")
            if index % 10 == 0:
                print(f"Generated {index + 1}/{len(dataset)} queries", flush=True)


if __name__ == "__main__":
    main()
