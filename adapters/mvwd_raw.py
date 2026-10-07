"""Read the pinned MVWD payload format directly, without importing mvwd.py."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def within(root, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError(f"Expected root-relative payload path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Payload escapes dataset root: {relative}")
    return path


def cache_key(record, floor, size):
    # configuration_id alone does not identify the source-specific payload.
    identity = [record["configuration_path"], int(floor), int(size), "letterbox-v1"]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class MVWDRaw:
    def __init__(self, root, split="train", scenes=None, max_episodes=None):
        self.root = Path(root).expanduser().resolve()
        self.metadata = read_json(self.root / "dataset.json")
        if split not in ("train", "val", "test"):
            raise ValueError("Choose a pinned train/val/test split")
        manifest = within(self.root, self.metadata["splits"][split]["path"])
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(records) != self.metadata["splits"][split]["count"]:
            raise ValueError("Split count differs from pinned dataset.json")
        expected = self.metadata.get("checksums", {}).get(str(manifest.relative_to(self.root)))
        if expected and sha256_file(manifest) != expected:
            raise ValueError("Split checksum differs from pinned dataset.json")
        if any(r["split"] != split for r in records):
            raise ValueError("Manifest contains records from another split")
        if scenes:
            names = {scenes} if isinstance(scenes, str) else set(scenes)
            records = [r for r in records if r["scene"] in names]
        if max_episodes is not None:
            if max_episodes <= 0:
                raise ValueError("max_episodes must be positive")
            records = records[:max_episodes]
        if not records:
            raise ValueError("Empty episode selection")
        self.records = records
        self.split = split
        self.manifest_hash = sha256_file(manifest)
        self._bev_cache = OrderedDict()
        self._camera_cache = OrderedDict()
        self._view_cache = OrderedDict()
        self._floor_cache = {}

    @staticmethod
    def _remember(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
        return value

    def episode_path(self, record):
        return within(self.root, record["episode_path"])

    def floor_index(self, record):
        if record["id"] not in self._floor_cache:
            metrics = read_json(self.episode_path(record) / "generation_metrics.json")
            self._floor_cache[record["id"]] = int(metrics["trajectory"]["floor_index"])
        return self._floor_cache[record["id"]]

    def bev(self, record):
        floor = self.floor_index(record)
        key = (record["configuration_path"], floor)
        if key in self._bev_cache:
            self._bev_cache.move_to_end(key)
            return self._bev_cache[key]
        path = within(self.root, record["configuration_path"]) / "bev/environment_base.npz"
        prefix = f"floor_{floor:02d}/"
        with np.load(path, allow_pickle=False) as packet:
            data = {"rgb": packet[prefix + "rgb"], "floor_index": floor}
            for name in ("world_bounds", "pixel_to_world", "world_to_pixel", "meters_per_pixel", "floor_z"):
                data[name] = packet[prefix + "calibration_" + name]
        if data["rgb"].ndim != 3 or data["rgb"].shape[-1] != 3 or data["rgb"].dtype != np.uint8:
            raise ValueError(f"Unexpected BEV RGB format: {path}")
        return self._remember(self._bev_cache, key, data, 2)

    def camera(self, record, robot, frame):
        key = record["id"]
        if key not in self._camera_cache:
            observations = read_json(self.episode_path(record) / "observations_before.json")
            cameras = {(o["robot_id"], int(o["physical_time_index"])): o["camera"] for o in observations}
            self._remember(self._camera_cache, key, cameras, 8)
        return self._camera_cache[key][(f"robot_{robot:02d}", frame)]

    def views(self, record, robot, modalities=("rgb", "depth_linear"), cache_root=None):
        path = self.episode_path(record) / "robot_views/before" / f"robot_{robot:02d}.npz"
        if cache_root is not None:
            directory = Path(cache_root) / "views" / self.metadata["release_id"] / record["id"] / f"robot_{robot:02d}"
            receipt_path = directory / "metadata.json"
            if receipt_path.exists():
                receipt = read_json(receipt_path)
                stat = path.stat()
                if receipt["source_size"] != stat.st_size or receipt["source_mtime_ns"] != stat.st_mtime_ns:
                    raise ValueError(f"Stale view cache: {directory}")
                return {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in modalities}
        key = (str(path), tuple(modalities))
        if key in self._view_cache:
            return self._view_cache[key]
        with np.load(path, allow_pickle=False) as packet:
            arrays = {name: packet[name] for name in modalities}
        return self._remember(self._view_cache, key, arrays, 1)

    def provenance(self):
        return {"release_id": self.metadata["release_id"], "split": self.split,
                "split_sha256": self.manifest_hash, "episode_count": len(self.records)}
