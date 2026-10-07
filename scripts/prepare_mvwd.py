"""Generate SAM-from-RGB conditions and/or decompressed frame caches."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from adapters.mvwd_raw import read_json, sha256_file
from mvwd_runtime import REPO, load_config, make_dataset


def atomic_npy(path, array):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def colorize_masks(masks, shape, seed):
    # SAM is class-free: never read taxonomy/GT semantic/height/occupancy.
    rng = np.random.default_rng(seed)
    canvas = np.zeros(shape, dtype=np.uint8)
    for mask in sorted(masks, key=lambda mask: mask["area"], reverse=True):
        canvas[mask["segmentation"]] = rng.integers(32, 256, size=3, dtype=np.uint8)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--mode", choices=("sam", "views", "all"), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--scene", action="append")
    parser.add_argument("--max-episodes", type=int, help="Debug limit per split")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-model", choices=("vit_b", "vit_l", "vit_h"), default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--points-per-side", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    generator, recipe, seen = None, None, set()
    if args.mode in ("sam", "all"):
        if not args.sam_checkpoint or not Path(args.sam_checkpoint).is_file():
            parser.error("--sam-checkpoint must point to a downloaded SAM checkpoint")
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        recipe = {"method": "sam", "checkpoint_sha256": sha256_file(args.sam_checkpoint),
                  "model_type": args.sam_model, "points_per_side": args.points_per_side,
                  "color_seed": config["training"]["seed"], "bev_size": config["data"]["bev_size"],
                  "preprocessing": "letterbox-v1", "colorization": "area-descending-random-rgb-v1"}
        model = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(args.device)
        generator = SamAutomaticMaskGenerator(model, points_per_side=args.points_per_side)
    for split in args.splits:
        dataset = make_dataset(config, split, include_targets=False, scenes=args.scene, max_episodes=args.max_episodes)
        release = dataset.raw.metadata["release_id"]
        if generator is not None:
            segmentation = Path(config["data"]["segmentation_root"]) / release
            segmentation.mkdir(parents=True, exist_ok=True)
            metadata_path = segmentation / "sam_metadata.json"
            if metadata_path.exists() and read_json(metadata_path) != recipe:
                raise ValueError("SAM recipe changed; use a separate segmentation_root for this experiment")
            metadata_path.write_text(json.dumps(recipe, indent=2) + "\n")
        for index, record in enumerate(dataset.raw.records):
            if generator is not None:
                _, rgb, valid, _, _, key = dataset.processed_bev(record)
                if key not in seen:
                    seen.add(key)
                    receipt_path = segmentation / f"{key}.json"
                    image_path = segmentation / f"{key}.npy"
                    if not receipt_path.exists() or not image_path.exists():
                        masks = generator.generate(rgb)
                        colored = colorize_masks(masks, rgb.shape, recipe["color_seed"])
                        colored[~valid] = 0
                        atomic_npy(image_path, colored)
                        receipt = {"bev_key": key, "release_id": release,
                                   "configuration_path": record["configuration_path"],
                                   "recipe": recipe, "source_rgb_sha256": __import__("hashlib").sha256(rgb.tobytes()).hexdigest()}
                        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if args.mode in ("views", "all"):
                for robot in dataset.robots:
                    path = dataset.raw.episode_path(record) / "robot_views/before" / f"robot_{robot:02d}.npz"
                    directory = Path(config["data"]["cache_root"]) / "views" / release / record["id"] / f"robot_{robot:02d}"
                    directory.mkdir(parents=True, exist_ok=True)
                    stat = path.stat()
                    receipt = {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                               "episode_path": record["episode_path"], "release_id": release,
                               "modalities": ["rgb", "depth_linear"]}
                    receipt_path = directory / "metadata.json"
                    complete = all((directory / f"{name}.npy").exists() for name in receipt["modalities"])
                    if receipt_path.exists() and read_json(receipt_path) == receipt and complete:
                        continue
                    with np.load(path, allow_pickle=False) as packet:
                        for name in receipt["modalities"]:
                            atomic_npy(directory / f"{name}.npy", packet[name])
                    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if index % 25 == 0:
                print(f"{split}: prepared {index + 1}/{len(dataset.raw.records)} episodes", flush=True)


if __name__ == "__main__":
    main()
