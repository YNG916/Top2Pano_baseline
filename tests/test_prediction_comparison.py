"""Match saved queries to GT and preserve the GT-free inference artifacts."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import mvwd_prediction_comparison as comparison
from scripts import run_mvwd_experiment as runner


class RawTargets:
    calls = []

    def __init__(self, root, split):
        assert root == "/dataset" and split == "val"
        self.root = Path("/dataset")
        self.metadata = {"release_id": "release"}
        self.manifest_hash = "split-hash"
        self.records = [{"id": "episode", "scene": "office", "robots": 3, "frames": 3, "fps": 10, "configuration_path": "config"}]

    def views(self, record, robot, modalities, cache_root):
        self.calls.append((record["id"], robot, modalities))
        frames = np.zeros((3, 4, 8, 3), dtype=np.uint8)
        frames[2] = [20, 40, 60]
        return {"rgb": frames}

    def episode_path(self, record):
        return self.root / record["id"]

    def bev(self, record):
        return {"rgb": np.full((4, 8, 3), [7, 8, 9], dtype=np.uint8), "floor_index": 0,
                "world_bounds": np.zeros(4), "world_to_pixel": np.eye(4), "pixel_to_world": np.eye(4),
                "meters_per_pixel": np.asarray(.1), "floor_z": np.asarray(0.)}

    def world_bev(self, record):
        packet = self.bev(record)
        packet["rgb"] = np.zeros((3, 4, 8, 3), dtype=np.uint8)
        packet["rgb"][2] = [90, 100, 110]
        return packet

    def provenance(self):
        return {"release_id": "release", "split": "val"}


def saved_prediction(tmp_path):
    directory = tmp_path / "experiment/predictions/infer"
    directory.mkdir(parents=True)
    metadata = {"config": {"data": {"root": "/dataset"}},
                "dataset": {"split": "val", "release_id": "release", "split_sha256": "split-hash", "query_count": 1},
                "ground_truth_loaded": False, "checkpoint_sha256": "checkpoint-hash"}
    (directory / "metadata.json").write_text(json.dumps(metadata))
    row = {"episode_id": "episode", "robot_id": 1, "frame_index": 2, "seed": 42, "path": "frame.png"}
    (directory / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    Image.new("RGB", (4, 2), (100, 120, 140)).save(directory / "frame.png")
    np.savez(directory / "frame.coarse.npz", rgb=np.ones((2, 4, 3), dtype=np.float32) * .5)
    return directory


def test_export_matches_query_and_preserves_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "MVWDRaw", RawTargets)
    RawTargets.calls = []
    predictions = saved_prediction(tmp_path)
    before = {p.name: p.read_bytes() for p in predictions.iterdir()}
    report = comparison.export_comparisons(predictions)
    assert RawTargets.calls == [("episode", 1, ("rgb",))]
    output = Path(report["report_directory"])
    assert output == predictions
    gt = np.asarray(Image.open(output / report["queries"][0]["gt_path"]))
    assert gt.shape == (2, 4, 3) and np.all(gt == [20, 40, 60])
    panel = np.asarray(Image.open(output / report["queries"][0]["comparisons_path"]))
    assert np.all(panel[54:, 8:] == [100, 120, 140])
    assert report["ground_truth_loaded_for_generation"] is False
    assert report["ground_truth_loaded_for_comparison"] is True
    assert {name: (predictions / name).read_bytes() for name in before} == before
    assert (predictions / "frame.gt.png").exists()
    assert (predictions / "frame.comparison.png").exists()
    assert not (tmp_path / "experiment/reports").exists()
    assert np.all(np.asarray(Image.open(predictions / "episode/gt_env_bev.png")) == [7, 8, 9])
    assert np.all(np.asarray(Image.open(predictions / "frame.world_bev.png")) == [90, 100, 110])
    assert Image.open(predictions / "frame.world_bev.png").size == (8, 4)
    bev_report = json.loads((predictions / "bev_comparison.json").read_text())
    assert bev_report["queries"][0]["frame_index"] == 2
    assert bev_report["ground_truth_loaded_for_generation"] is False
    with pytest.raises(FileExistsError):
        comparison.export_comparisons(predictions)


def test_incomplete_or_wrong_release_rejected_before_gt_read(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "MVWDRaw", RawTargets)
    RawTargets.calls = []
    predictions = saved_prediction(tmp_path)
    path = predictions / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["dataset"]["query_count"] = 2
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete"):
        comparison.export_comparisons(predictions)
    metadata["dataset"]["query_count"] = 1
    metadata["dataset"]["release_id"] = "different"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="releases differ"):
        comparison.export_comparisons(predictions)
    assert RawTargets.calls == []
    assert not (predictions / "comparison.json").exists()


def test_compare_routes_to_saved_predictions(tmp_path):
    directory = tmp_path / "experiment"
    directory.mkdir()
    (directory / "config.yaml").write_text("training: {output_dir: /unused}\n")
    saved = saved_prediction(tmp_path)
    _, _, output, command = runner.plan_step(directory, "gt", "compare", ["--predictions", str(saved)])
    assert output == saved
    assert command[command.index("--output") + 1] == str(output)


def test_bev_only_backfill_keeps_existing_view_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "MVWDRaw", RawTargets)
    predictions = saved_prediction(tmp_path)
    (predictions / "comparison.json").write_text("existing view comparison")
    report = comparison.export_bevs(predictions)
    assert report["query_count"] == 1
    assert (predictions / "comparison.json").read_text() == "existing view comparison"
    assert not (predictions / ".bev_comparison.lock").exists()
    with pytest.raises(FileExistsError):
        comparison.export_bevs(predictions)


def test_full_comparison_reuses_matching_bev_backfill(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "MVWDRaw", RawTargets)
    predictions = saved_prediction(tmp_path)
    comparison.export_bevs(predictions)
    before = (predictions / "bev_comparison.json").read_bytes()
    comparison.export_comparisons(predictions)
    assert (predictions / "bev_comparison.json").read_bytes() == before
    assert (predictions / "GT_COMPARISON.md").exists()
