"""Check split isolation and save BEV/calibration/target diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from adapters.mvwd_raw import MVWDRaw
from mvwd_runtime import REPO, load_config, make_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO / "configs/mvwd_level1.yaml"))
    parser.add_argument("--root")
    parser.add_argument("--robot-assets")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--index", type=int, default=0, help="Episode index within selected split")
    parser.add_argument("--load-query", action="store_true", help="Also validate cached SAM conditions and training fields")
    parser.add_argument("--output", default=str(REPO / "artifacts/mvwd/inspection"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.root:
        config["data"]["root"] = str(Path(args.root).expanduser().resolve())
    if args.robot_assets:
        config["data"]["robot_assets_root"] = str(Path(args.robot_assets).expanduser().resolve())
    records = {split: MVWDRaw(config["data"]["root"], split) for split in ("train", "val", "test")}
    report = {"splits": {split: {"episodes": len(raw.records), "scenes": len({r['scene'] for r in raw.records})}
                         for split, raw in records.items()}, "overlaps": {}}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        intersection = {field: sorted({r[field] for r in records[a].records} & {r[field] for r in records[b].records})
                        for field in ("id", "scene", "configuration_id", "configuration_path")}
        if any(intersection.values()):
            raise ValueError(f"Split overlap: {a}/{b}: {intersection}")
        report["overlaps"][f"{a}/{b}"] = {field: len(values) for field, values in intersection.items()}
    dataset = make_dataset(config, args.split, scenes=args.scene)
    record = dataset.raw.records[args.index]
    bev, rgb, valid, w2p, p2w, key = dataset.processed_bev(record)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    positions = []
    for robot in dataset.robots:
        points = []
        for frame in range(record["frames"]):
            camera = dataset.raw.camera(record, robot, frame)
            T = np.asarray(camera.get("modality_camera_to_world", {}).get("rgb", camera["camera_to_world"]))
            point = w2p @ T[:, 3]
            back = p2w @ point
            if not np.allclose(back, T[:, 3], atol=1e-6):
                raise ValueError("BEV round-trip calibration failed")
            points.append(tuple(point[:2]))
            if frame == 0:
                tip = w2p @ np.r_[T[:3, 3] + T[:3, 2], 1]
                color = ((255, 0, 0), (0, 180, 0), (0, 80, 255))[robot % 3]
                draw.line([tuple(point[:2]), tuple(tip[:2])], fill=color, width=3)
                draw.text(tuple(point[:2]), f"robot_{robot:02d}", fill=color)
                views = dataset.raw.views(record, robot, ("rgb",), dataset.cache_root)
                Image.fromarray(views["rgb"][0]).save(output / f"robot_{robot:02d}_frame_000.png")
        draw.line(points, fill=((255, 0, 0), (0, 180, 0), (0, 80, 255))[robot % 3], width=2)
        positions.append({"robot_id": robot, "first_bev_pixel": points[0], "last_bev_pixel": points[-1]})
    image.save(output / "bev_trajectories.png")
    Image.fromarray((valid * 255).astype(np.uint8)).save(output / "bev_valid.png")
    report["sample"] = {"episode_id": record["id"], "scene": record["scene"], "bev_key": key,
                        "raw_bev_shape": list(bev["rgb"].shape), "world_to_bev": w2p.tolist(),
                        "bev_to_world": p2w.tolist(), "floor_z": float(bev["floor_z"]), "robots": positions}
    if args.load_query:
        index = sum(len(dataset._frames[i]) * len(dataset.robots) for i in range(args.index))
        sample = dataset[index]
        report["query_fields"] = {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                  for name, value in sample.items() if isinstance(value, np.ndarray)}
        Image.fromarray((sample["segmentation"] * 255).astype(np.uint8)).save(output / "sam_segmentation.png")
    (output / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "splits": report["splits"],
                      "episode_id": record["id"], "round_trip": "passed"}, indent=2))


if __name__ == "__main__":
    main()
