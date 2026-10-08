"""Project all known robots at real poses; no scene GT or simulator needed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from mvwd_runtime import REPO, load_config, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--robot-assets")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--robots", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 10, 20])
    parser.add_argument("--output", default=str(REPO / "artifacts/mvwd/robot_inspection"))
    parser.add_argument("--with-targets", action="store_true", help="Save RGB GT for visual comparison only; not a rendering input")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    if args.robot_assets:
        config["data"]["robot_assets_root"] = str(Path(args.robot_assets).expanduser().resolve())
    if not config["data"].get("robot_rendering"):
        parser.error("Enable robot_rendering in the config")
    dataset = make_dataset(config, args.split, scenes=args.scene, max_episodes=args.max_episodes,
                           robots=args.robots, frames=args.frames, include_targets=False)
    import torch
    from torch.utils.data._utils.collate import default_collate
    from adapters.camera_geometry import pinhole_rays
    from adapters.robot_assets import robot_surfaces
    # This CPU projection diagnostic does not allocate either diffusion model.
    torch.set_num_threads(4)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(len(dataset)):
        sample = dataset[index]
        batch = default_collate([sample])
        height, width = map(int, sample["target_hw"])
        origin, directions, z = pinhole_rays(batch["K_rgb"], batch["T_wc_rgb"], height, width)
        surfaces = robot_surfaces(batch, origin, directions)
        rgb = surfaces["rgb"][0].numpy().reshape(height, width, 3)
        ids = surfaces["id"][0].numpy().reshape(height, width)
        depth = (surfaces["distance"] * z)[0].numpy().reshape(height, width)
        depth[~np.isfinite(depth)] = 0
        directory = output / sample["episode_id"] / ("robot_%02d" % sample["robot_id"])
        directory.mkdir(parents=True, exist_ok=True)
        prefix = "frame_%03d" % sample["frame_index"]
        Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(directory / (prefix + ".robot_rgb.png"))
        palette = np.asarray([[242, 79, 14], [10, 87, 235], [13, 158, 64]], dtype=np.uint8)
        colored = np.zeros((height, width, 3), np.uint8)
        colored[ids >= 0] = palette[ids[ids >= 0]]
        Image.fromarray(colored).save(directory / (prefix + ".robot_id.png"))
        np.savez_compressed(directory / (prefix + ".robots.npz"), rgb=rgb, robot_id=ids, depth_z=depth)
        if args.with_targets:
            record, robot, frame = dataset.query(index)
            gt = dataset.raw.views(record, robot, ("rgb",), dataset.cache_root)["rgb"][frame]
            Image.fromarray(gt).resize((width, height), Image.Resampling.BOX).save(directory / (prefix + ".gt_rgb.png"))
        row = {"episode_id": sample["episode_id"], "robot_id": sample["robot_id"], "frame_index": sample["frame_index"],
               "mesh_pixels_by_identity": {str(r): int((ids == r).sum()) for r in sample["robot_ids"]},
               "robot_T_wb": sample["robot_T_wb"].tolist(), "robot_camera_heights": sample["robot_camera_heights"].tolist()}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("robot_id", "frame_index", "mesh_pixels_by_identity")}), flush=True)
    report = {"dataset": dataset.provenance(), "queries": rows, "ground_truth_loaded": args.with_targets,
              "ground_truth_used_for_rendering": False,
              "scene_occlusion_included": False,
              "purpose": "robot-only projection diagnostic; environment occlusion is handled by PerspectiveRenderer"}
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved robot projection diagnostics to " + str(output))


if __name__ == "__main__":
    main()
