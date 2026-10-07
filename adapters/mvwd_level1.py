"""Single-view Level 1 queries and calibrated image preprocessing."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from .mvwd_raw import MVWDRaw, cache_key, read_json


def letterbox_bev(rgb, world_to_pixel, size):
    """Pad first, resize uniformly; BEV calibration uses integer pixel centers."""
    height, width = rgb.shape[:2]
    side = max(height, width)
    left, top = (side - width) // 2, (side - height) // 2
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    canvas[top:top + height, left:left + width] = rgb
    valid = np.zeros((side, side), dtype=np.uint8)
    valid[top:top + height, left:left + width] = 255
    image = np.asarray(Image.fromarray(canvas).resize((size, size), Image.Resampling.BOX)).copy()
    valid = np.asarray(Image.fromarray(valid).resize((size, size), Image.Resampling.NEAREST)) > 0
    scale = size / side
    transform = np.eye(4, dtype=np.float64)
    transform[0, 0] = transform[1, 1] = scale
    transform[0, 3] = scale * (left + 0.5) - 0.5
    transform[1, 3] = scale * (top + 0.5) - 0.5
    calibrated = transform @ world_to_pixel
    return image, valid, calibrated, np.linalg.inv(calibrated)


def scale_intrinsics(K, old_hw, new_hw):
    # MVWD perspective intrinsics use corner-origin coordinates and u/v + .5.
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    return np.diag([new_w / old_w, new_h / old_h, 1.0]) @ K


class MVWDLevel1:
    def __init__(self, root, split="train", *, bev_size=512, target_size=(256, 448),
                 robots=(0, 1, 2), frame_stride=1, frames=None, scenes=None,
                 max_episodes=None, segmentation_root=None, cache_root=None,
                 include_targets=True, include_depth=True, wall_prior="rgb_black"):
        self.raw = MVWDRaw(root, split, scenes, max_episodes)
        self.bev_size = int(bev_size)
        self.target_size = tuple(map(int, target_size))
        if self.bev_size % 64 or any(s <= 0 or s % 64 for s in self.target_size):
            raise ValueError("BEV and target dimensions must be positive multiples of 64")
        self.robots = tuple(int(r) for r in robots)
        if not self.robots or len(set(self.robots)) != len(self.robots):
            raise ValueError("Choose unique robots")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if wall_prior not in ("rgb_black", "none"):
            raise ValueError("wall_prior must be rgb_black or none")
        self.wall_prior = wall_prior
        self.segmentation_root = Path(segmentation_root).expanduser().resolve() if segmentation_root else None
        self.cache_root = Path(cache_root).expanduser().resolve() if cache_root else None
        self.include_targets = include_targets
        self.include_depth = include_depth and include_targets
        self._processed_cache = OrderedDict()
        self._segmentation_cache = OrderedDict()
        self._frames, self._ends = [], []
        total = 0
        for record in self.raw.records:
            selected = tuple(range(0, record["frames"], frame_stride)) if frames is None else tuple(map(int, frames))
            if not selected or len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= record["frames"]:
                raise ValueError(f"Invalid frame selection for {record['id']}")
            if min(self.robots) < 0 or max(self.robots) >= record["robots"]:
                raise ValueError(f"Invalid robot selection for {record['id']}")
            self._frames.append(selected)
            total += len(selected) * len(self.robots)
            self._ends.append(total)

    def __len__(self):
        return self._ends[-1]

    def query(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect_right(self._ends, index)
        local = index - (self._ends[episode - 1] if episode else 0)
        robot_slot, frame_slot = divmod(local, len(self._frames[episode]))
        return self.raw.records[episode], self.robots[robot_slot], self._frames[episode][frame_slot]

    def processed_bev(self, record):
        bev = self.raw.bev(record)
        key = cache_key(record, bev["floor_index"], self.bev_size)
        if key in self._processed_cache:
            self._processed_cache.move_to_end(key)
            return self._processed_cache[key]
        rgb, valid, w2p, p2w = letterbox_bev(bev["rgb"], bev["world_to_pixel"], self.bev_size)
        return self.raw._remember(self._processed_cache, key, (bev, rgb, valid, w2p, p2w, key), 2)

    def __getitem__(self, index):
        record, robot, frame = self.query(index)
        bev, rgb, valid, w2p, p2w, key = self.processed_bev(record)
        if self.segmentation_root is None:
            raise ValueError("SAM segmentation cache is required; run scripts/prepare_mvwd.py --mode sam")
        directory = self.segmentation_root / self.raw.metadata["release_id"]
        if key not in self._segmentation_cache:
            receipt = read_json(directory / f"{key}.json")
            if receipt["bev_key"] != key or receipt["release_id"] != self.raw.metadata["release_id"]:
                raise ValueError("Segmentation cache does not match the BEV/release")
            segmentation = np.load(directory / f"{key}.npy", allow_pickle=False)
            if segmentation.shape != rgb.shape or segmentation.dtype != np.uint8:
                raise ValueError("Expected an RGB uint8 SAM segmentation at processed BEV resolution")
            self.raw._remember(self._segmentation_cache, key, segmentation, 2)
        segmentation = self._segmentation_cache[key]
        camera = self.raw.camera(record, robot, frame)
        poses = camera.get("modality_camera_to_world", {})
        T = np.asarray(poses.get("rgb", camera["camera_to_world"]), dtype=np.float64)
        K = scale_intrinsics(np.asarray(camera["pixel_intrinsics"], dtype=np.float64),
                             (camera["height"], camera["width"]), self.target_size)
        walls = valid & np.all(rgb <= 5, axis=-1) if self.wall_prior == "rgb_black" else np.zeros_like(valid)
        source = rgb.copy()
        source[walls] = 255  # Preserve the original RGB-derived wall heuristic.
        sample = {
            "episode_id": record["id"], "scene": record["scene"], "bev_key": key,
            "robot_id": robot, "frame_index": frame, "time_seconds": frame / record["fps"],
            "bev_rgb": source.astype(np.float32) / 255.0,
            "segmentation": segmentation.astype(np.float32) / 255.0,
            "bev_valid": valid, "wall_mask": walls,
            "world_to_bev": w2p.astype(np.float32), "bev_to_world": p2w.astype(np.float32),
            "floor_z": np.float32(bev["floor_z"]), "K_rgb": K.astype(np.float32),
            "T_wc_rgb": T.astype(np.float32), "target_hw": np.asarray(self.target_size, dtype=np.int64),
            "near_m": np.float32(camera["near_m"]), "far_m": np.float32(camera["far_m"]),
        }
        if self.include_targets:
            modalities = ("rgb", "depth_linear") if self.include_depth else ("rgb",)
            arrays = self.raw.views(record, robot, modalities, self.cache_root)
            image = np.asarray(Image.fromarray(arrays["rgb"][frame]).resize(
                self.target_size[::-1], Image.Resampling.BOX)).copy()
            sample["target_rgb"] = image.astype(np.float32) / 127.5 - 1.0
            if self.include_depth:
                depth = np.asarray(arrays["depth_linear"][frame], dtype=np.float32).copy()
                sample["target_depth_z"] = depth
                sample["depth_valid"] = np.isfinite(depth) & (depth > 0) & (depth <= camera["far_m"])
                sample["K_depth"] = np.asarray(camera["geometry_pixel_intrinsics"], dtype=np.float32)
                sample["T_wc_depth"] = np.asarray(poses.get("depth_linear", camera["camera_to_world"]), dtype=np.float32)
        return sample

    def provenance(self):
        return {**self.raw.provenance(), "query_count": len(self), "robots": self.robots,
                "frame_selections": [list(frames) for frames in sorted(set(self._frames))],
                "episode_ids": [record["id"] for record in self.raw.records], "bev_size": self.bev_size,
                "target_size": self.target_size, "wall_prior": self.wall_prior,
                "include_depth": self.include_depth}
