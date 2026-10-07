import json
from pathlib import Path, PurePath

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from adapters.mvwd_level1 import MVWDLevel1, letterbox_bev, scale_intrinsics
from adapters.mvwd_raw import MVWDRaw, cache_key, sha256_file, within
from mvwd_runtime import align_depth_to_rgb, original_batch


@pytest.fixture
def payload(tmp_path):
    root = tmp_path / "dataset"
    (root / "splits").mkdir(parents=True)
    record = {"id": "ep_train", "scene": "room_train", "split": "train",
              "configuration_id": "cfg_train", "configuration_path": "data/cfg_train",
              "episode_path": "data/ep_train", "frames": 4, "fps": 10, "robots": 1}
    cfg, ep = root / record["configuration_path"], root / record["episode_path"]
    (cfg / "bev").mkdir(parents=True)
    (ep / "robot_views/before").mkdir(parents=True)
    rgb = np.full((64, 32, 3), 100, dtype=np.uint8)
    matrix = np.array([[10, 0, 0, 15.5], [0, -10, 0, 31.5], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.savez(cfg / "bev/environment_base.npz", **{
        "floor_00/rgb": rgb, "floor_00/calibration_world_to_pixel": matrix,
        "floor_00/calibration_pixel_to_world": np.linalg.inv(matrix),
        "floor_00/calibration_world_bounds": [-1.6, -3.2, 1.6, 3.2],
        "floor_00/calibration_meters_per_pixel": .1, "floor_00/calibration_floor_z": 0.0})
    (ep / "generation_metrics.json").write_text(json.dumps({"trajectory": {"floor_index": 0}}))
    camera = {"camera_to_world": np.eye(4).tolist(),
              "pixel_intrinsics": [[100, 0, 128], [0, 100, 64], [0, 0, 1]],
              "geometry_pixel_intrinsics": [[50, 0, 64], [0, 50, 32], [0, 0, 1]],
              "height": 128, "width": 256, "near_m": .1, "far_m": 15.0}
    actual = np.eye(4); actual[:3, 3] = [1, 2, 1]
    camera["modality_camera_to_world"] = {"rgb": actual.tolist(), "depth_linear": actual.tolist()}
    observations = [{"robot_id": "robot_00", "physical_time_index": t, "camera": camera} for t in range(4)]
    (ep / "observations_before.json").write_text(json.dumps(observations))
    depth = np.broadcast_to(np.linspace(1, 4, 128, dtype=np.float32), (4, 64, 128)).copy()
    np.savez(ep / "robot_views/before/robot_00.npz", rgb=np.full((4, 128, 256, 3), 127, np.uint8), depth_linear=depth)
    splits, checksums = {}, {}
    for split in ("train", "val", "test"):
        r = {**record, "id": f"ep_{split}", "scene": f"room_{split}", "split": split}
        path = root / f"splits/{split}.jsonl"
        path.write_text(json.dumps(r) + "\n")
        splits[split] = {"path": f"splits/{split}.jsonl", "count": 1}
        checksums[f"splits/{split}.jsonl"] = sha256_file(path)
    (root / "dataset.json").write_text(json.dumps({"release_id": "test_release", "splits": splits, "checksums": checksums}))
    segmentation = tmp_path / "sam"
    directory = segmentation / "test_release"
    directory.mkdir(parents=True)
    key = cache_key(record, 0, 64)
    np.save(directory / f"{key}.npy", np.full((64, 64, 3), 80, np.uint8))
    (directory / f"{key}.json").write_text(json.dumps({"bev_key": key, "release_id": "test_release"}))
    return root, segmentation, record


def test_letterbox_is_metric_and_preserves_pixel_centers():
    rgb = np.zeros((90, 30, 3), np.uint8)
    world_to_pixel = np.eye(4)
    resized, valid, w2p, p2w = letterbox_bev(rgb, world_to_pixel, 64)
    assert resized.shape == (64, 64, 3)
    # Padding+resize maps the same world point to its new pixel center.
    point = np.array([4, 12, 1.2, 1])
    expected = (point[:2] + np.array([30.5, .5])) * (64 / 90) - .5
    np.testing.assert_allclose((w2p @ point)[:2], expected)
    np.testing.assert_allclose(p2w @ w2p @ point, point)
    assert valid[:, 0].sum() == 0
    assert w2p[0, 0] == w2p[1, 1]


def test_adapter_uses_measured_poses_and_keeps_query_identity(payload):
    root, segmentation, record = payload
    ds = MVWDLevel1(root, bev_size=64, target_size=(64, 128), robots=(0,),
                   frames=[1, 3], segmentation_root=segmentation)
    assert len(ds) == 2
    sample = ds[1]
    assert sample["episode_id"] == record["id"] and sample["frame_index"] == 3
    assert sample["time_seconds"] == .3
    np.testing.assert_allclose(sample["T_wc_rgb"][:3, 3], [1, 2, 1])
    np.testing.assert_allclose(sample["K_rgb"], sample["K_depth"])
    assert sample["target_rgb"].shape == (64, 128, 3)
    assert sample["target_depth_z"].shape == (64, 128)
    assert not sample["wall_mask"][~sample["bev_valid"]].any()
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_inference_does_not_open_ground_truth(payload):
    root, segmentation, record = payload
    (root / record["episode_path"] / "robot_views/before/robot_00.npz").unlink()
    ds = MVWDLevel1(root, bev_size=64, target_size=(64, 128), robots=(0,),
                   segmentation_root=segmentation, include_targets=False)
    sample = ds[0]
    assert "target_rgb" not in sample and "target_depth_z" not in sample


def test_depth_registration_and_original_batch_layout(payload):
    root, segmentation, _ = payload
    ds = MVWDLevel1(root, bev_size=64, target_size=(64, 128), robots=(0,), segmentation_root=segmentation)
    batch = default_collate([ds[0]])
    depth, valid = align_depth_to_rgb(batch)
    assert valid.all()
    torch.testing.assert_close(depth, batch["target_depth_z"])
    legacy = original_batch(batch, {"prompts": {"occupancy": "source", "refinement": "target"}, "renderer": {}})
    assert legacy["jpg"].shape == (1, 64, 64, 3)
    assert legacy["hint"].shape == (1, 64, 64, 6)
    assert legacy["ground_truth"].shape == (1, 64, 128, 3)
    assert legacy["ground_truth_depth"].shape == (1, 64, 128, 3)
    assert legacy["ground_truth_depth"].min() == 0 and legacy["ground_truth_depth"].max() == 1


def test_split_checksum_and_root_path_validation(payload):
    root, _, _ = payload
    raw = MVWDRaw(root)
    bad = {**raw.records[0], "episode_path": "../outside"}
    with pytest.raises(ValueError, match="escapes"):
        raw.episode_path(bad)
    with (root / "splits/train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        MVWDRaw(root)


def test_configuration_payload_identity_is_not_collapsed():
    a = {"configuration_path": "data/source_a", "configuration_id": "same"}
    b = {"configuration_path": "data/source_b", "configuration_id": "same"}
    assert cache_key(a, 0, 512) != cache_key(b, 0, 512)


def test_root_path_validation_without_is_relative_to(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "dataset-outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with monkeypatch.context() as legacy:
        legacy.delattr(PurePath, "is_relative_to", raising=False)
        assert within(root, "data/episode") == root / "data/episode"
        with pytest.raises(ValueError, match="root-relative"):
            within(root, outside)
        for path in ("../dataset-outside", "escape/episode"):
            with pytest.raises(ValueError, match="escapes"):
                within(root, path)


def test_resize_keeps_fov_with_corner_origin_intrinsics():
    K = np.array([[640, 0, 448], [0, 640, 256], [0, 0, 1]])
    small = scale_intrinsics(K, (512, 896), (256, 448))
    np.testing.assert_allclose(np.linalg.inv(K) @ [101, 201, 1],
                               np.linalg.inv(small) @ [50.5, 100.5, 1])
