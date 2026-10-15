"""Read saved Top2Pano manifests and official MVWD targets, without dataset SDKs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

from adapters.mvwd_raw import MVWDRaw, within, sha256_file
from adapters.mvwd_level1 import stratified_episode_frames
from .geometry import register_depth, camera


def load_prediction_sets(directories, protocol, preview=False):
    by_seed = defaultdict(dict)
    inputs = []
    raw = None
    identity = None
    records = None
    size = None
    for directory in directories:
        directory = Path(directory).expanduser().resolve()
        meta = json.loads((directory / "metadata.json").read_text())
        rows = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if (
            not rows
            or len(rows) != meta["dataset"]["query_count"]
            or meta.get("ground_truth_loaded") is not False
        ):
            raise ValueError(
                "Incomplete manifest or GT-free generation not recorded: "
                + str(directory)
            )
        seed = int(meta["config"]["inference"]["seed"])
        inference = {
            k: v for k, v in meta["config"]["inference"].items() if k != "seed"
        }
        signature = (
            meta["dataset"]["release_id"],
            meta["dataset"]["split_sha256"],
            meta["dataset"]["split"],
            meta["checkpoint_sha256"],
            json.dumps(inference, sort_keys=True),
            json.dumps(meta["config"].get("renderer", {}), sort_keys=True),
            meta["dataset"].get("robot_assets", {}).get("manifest_sha256"),
        )
        if identity is None:
            identity = signature
            raw = MVWDRaw(meta["config"]["data"]["root"], meta["dataset"]["split"])
            records = {r["id"]: r for r in raw.records}
            if (
                raw.metadata["release_id"] != signature[0]
                or raw.manifest_hash != signature[1]
            ):
                raise ValueError(
                    "Saved dataset release/split checksum differs from current GT"
                )
        elif identity != signature:
            raise ValueError(
                "Input folders mix checkpoints, generation settings, assets or splits"
            )
        for row in rows:
            e, r, t = row["episode_id"], row["robot_id"], row["frame_index"]
            if e not in records or not isinstance(r, int) or not isinstance(t, int):
                raise ValueError("Unknown episode or noninteger query identity")
            record = records[e]
            if not 0 <= r < record["robots"] or not 0 <= t < record["frames"]:
                raise ValueError("Query outside episode range")
            key = (e, r, t)
            if key in by_seed[seed]:
                raise ValueError("Duplicate seed/query across input manifests")
            path = within(directory, row["path"])
            with Image.open(path) as im:
                if im.mode != "RGB":
                    raise ValueError("Expected RGB prediction PNG")
                shape = (im.height, im.width)
            if size is None:
                size = shape
            if shape != size or shape != tuple(protocol["target_hw"]):
                raise ValueError(
                    "Prediction dimensions differ from frozen metric protocol"
                )
            material = "%s:%s:%s:%s:%s" % (seed, signature[0], e, r, t)
            expected = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)
            if row["seed"] != expected:
                raise ValueError(
                    "Query random seed differs from recorded base seed policy"
                )
            by_seed[seed][key] = {"path": path, "scene": record["scene"]}
        inputs.append(
            {
                "directory": str(directory),
                "base_seed": seed,
                "manifest_sha256": sha256_file(directory / "predictions.jsonl"),
                "metadata_sha256": sha256_file(directory / "metadata.json"),
                "checkpoint_sha256": meta["checkpoint_sha256"],
            }
        )
    seeds = sorted(by_seed)
    if not seeds:
        raise ValueError("No prediction sets")
    reference = set(by_seed[seeds[0]])
    if any(set(by_seed[s]) != reference for s in seeds):
        raise ValueError("Every seed must cover the exact same queries")
    expected = {
        (r["id"], robot, frame)
        for r in raw.records
        for robot in protocol["robots"]
        for frame in stratified_episode_frames(
            r["frames"],
            protocol["frames_per_episode"],
            r["id"],
            protocol["frame_sample_seed"],
        )
    }
    coverage = {
        "split": raw.split,
        "episode_count": len({k[0] for k in reference}),
        "official_split_episode_count": len(raw.records),
        "queries_per_seed": len(reference),
        "official_protocol_queries_per_seed": len(expected),
        "missing_protocol_queries": len(expected - reference),
        "extra_protocol_queries": len(reference - expected),
        "observed_seeds": seeds,
        "expected_seeds": protocol["seeds"],
        "full_test_protocol": raw.split == "test"
        and reference == expected
        and set(seeds) == set(protocol["seeds"]),
    }
    if not preview and not coverage["full_test_protocol"]:
        raise ValueError(
            "Main benchmark requires all test episodes, frozen frames/views and seeds. "
            + "Use --preview for explicitly diagnostic subsets. Coverage: "
            + json.dumps(coverage)
        )
    return by_seed, raw, records, inputs, coverage


def load_gt_episode(raw, record, units, target_hw, cache_root=None, geometry=False):
    """GT RGB is BOX-resized exactly as the training adapter; geometry is separately registered."""
    h, w = target_hw
    images = {}
    cameras = {}
    for robot in sorted({r for r, t in units}):
        modalities = ("rgb", "depth_linear") if geometry else ("rgb",)
        arrays = raw.views(record, robot, modalities, cache_root)
        for r, t in sorted(units):
            if r != robot:
                continue
            image = (
                np.asarray(
                    Image.fromarray(arrays["rgb"][t]).resize(
                        (w, h), Image.Resampling.BOX
                    ),
                    dtype=np.float32,
                )
                / 255.0
            )
            images[r, t] = image.transpose(2, 0, 1).copy()
            if not geometry:
                continue
            cam = raw.camera(record, r, t)
            poses = cam.get("modality_camera_to_world", {})
            T_rgb = poses.get("rgb", cam["camera_to_world"])
            T_depth = poses.get("depth_linear", cam["camera_to_world"])
            K = np.asarray(cam["pixel_intrinsics"], dtype=np.float64).copy()
            K[0] *= w / cam["width"]
            K[1] *= h / cam["height"]
            depth = register_depth(
                np.asarray(arrays["depth_linear"][t], dtype=np.float64),
                cam["geometry_pixel_intrinsics"],
                T_depth,
                K,
                T_rgb,
                (h, w),
                cam["near_m"],
                cam["far_m"],
            )
            cameras[r, t] = camera(K, T_rgb, depth, cam["near_m"], cam["far_m"])
    return images, cameras


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            raise ValueError("NaN cannot be serialized as a metric")
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    return value


def weight_provenance(weights_dir, include_met3r=False):
    """Record downloaded weight bytes, not just mutable upstream model names."""
    import os

    files = set()
    root = Path(weights_dir).resolve()
    if root.exists():
        files.update(
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt", ".safetensors")
        )
    if include_met3r:
        hf = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                str(
                    Path(
                        os.environ.get(
                            "HF_HOME", str(Path.home() / ".cache/huggingface")
                        )
                    )
                    / "hub"
                ),
            )
        )
        for directory in hf.glob("models--naver--MASt3R*"):
            files.update(
                p.resolve()
                for p in (directory / "snapshots").rglob("*")
                if p.is_file() and p.suffix in (".pt", ".pth", ".bin", ".safetensors")
            )
    return [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(files)
    ]
