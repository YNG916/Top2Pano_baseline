"""Metric definitions, scientific aggregation and GT-only pair selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from mvwd_metrics import PSNRMetric, EpisodeMetricAggregator, evaluate_level1
from mvwd_metrics.core import FeatureMoments
from mvwd_metrics.geometry import camera, register_depth, visible_surface_overlap
from mvwd_metrics.io import load_prediction_sets, json_safe
from scripts import run_mvwd_experiment as runner


def test_psnr_known_rgb_error_and_perfect_image():
    metric = PSNRMetric()
    target = np.zeros((2, 3, 8, 8), np.float32)
    pred = target.copy()
    pred[0] = 0.1
    scores = metric(pred, target)
    assert scores[0] == pytest.approx(20.0, abs=1e-6)
    assert np.isposinf(scores[1])
    with pytest.raises(ValueError):
        metric(pred * 30, target)
    with pytest.raises(ValueError):
        metric(pred.astype(np.uint8), target)
    assert json_safe({"perfect": scores[1]}) == {"perfect": "Infinity"}


def test_episode_and_seed_macro_mean_not_pooled_frame_mean():
    agg = EpisodeMetricAggregator([42, 43], bootstrap_samples=1000, bootstrap_seed=12)
    for seed, small, large in ((42, 0.0, 8.0), (43, 2.0, 10.0)):
        agg.add("lpips", "short", seed, "one", small)
        for frame in range(9):
            agg.add("lpips", "long", seed, str(frame), large)
    result = agg.summarize("lpips")
    assert result["mean"] == 5.0  # (mean(0,2) + mean(8,10))/2, not 8.2 pooled.
    assert result["episode_count"] == 2 and result["ci95"] == [1.0, 9.0]
    assert result == agg.summarize("lpips")
    with pytest.raises(ValueError, match="Duplicate"):
        agg.add("lpips", "short", 42, "one", 0)


def test_seed_support_and_undefined_overlap_cannot_silently_bias_mean():
    agg = EpisodeMetricAggregator([1, 2])
    agg.add("psnr", "a", 1, "view0", 20)
    with pytest.raises(ValueError, match="same episodes"):
        agg.summarize("psnr")
    agg.add("psnr", "a", 2, "view1", 20)
    with pytest.raises(ValueError, match="same metric units"):
        agg.summarize("psnr")
    empty = agg.summarize("met3r")
    assert empty["mean"] is None and empty["episode_count"] == 0
    with pytest.raises(ValueError):
        agg.add("lpips", "a", 1, "nan", float("nan"))


def test_streaming_fid_moments_equal_full_numpy_statistics():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(51, 7)) + 1e6
    moments = FeatureMoments()
    for batch in np.array_split(features, 8):
        moments.update(batch)
    mean, cov = moments.statistics()
    np.testing.assert_allclose(mean, features.mean(0), rtol=0, atol=1e-9)
    np.testing.assert_allclose(cov, np.cov(features, rowvar=False), atol=1e-9)
    assert moments.count == 51
    with pytest.raises(ValueError):
        FeatureMoments().statistics()


def plane(depth=2.0, shift=0.0):
    K = np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1.0]])
    T = np.eye(4)
    T[0, 3] = shift
    return camera(K, T, np.full((8, 12), depth))


def test_overlap_identity_partial_overlap_occlusion_and_behind_camera():
    a = plane()
    identity = visible_surface_overlap(a, a, boundary=0)
    assert identity["ratio"] == 1.0
    assert visible_surface_overlap(a, plane(shift=0.4), boundary=0)[
        "ratio"
    ] == pytest.approx(10 / 12)
    assert visible_surface_overlap(a, plane(depth=1.0), boundary=0)["ratio"] == 0.0
    reverse = plane()
    reverse["T_wc"][:3, :3] = np.diag([-1, 1, -1])
    assert visible_surface_overlap(a, reverse, boundary=0)["ratio"] == 0.0
    invalid = plane()
    invalid["depth_z"][:] = 0
    assert visible_surface_overlap(invalid, a)["forward"]["valid_pixels"] == 0


def test_gt_depth_registration_respects_modal_camera_pose_and_pixel_centers():
    a = plane()
    out = register_depth(
        a["depth_z"], a["K"], a["T_wc"], a["K"], a["T_wc"], (8, 12), 0.1, 15.0
    )
    np.testing.assert_array_equal(out, a["depth_z"])
    translated = plane(shift=0.4)
    out = register_depth(
        a["depth_z"],
        a["K"],
        a["T_wc"],
        translated["K"],
        translated["T_wc"],
        (8, 12),
        0.1,
        15.0,
    )
    np.testing.assert_array_equal(out[:, :10], 2)
    np.testing.assert_array_equal(out[:, 10:], 0)


class FakeMEt3R:
    def __init__(self):
        self.count = 0

    def __call__(self, a, b):
        self.count += len(a)
        return np.full(len(a), 0.25)

    def provenance(self):
        return {"implementation": "test double"}


def test_level1_api_pairs_same_time_only_and_records_no_overlap_episodes():
    pred = np.full((2, 2, 3, 2, 3, 8, 8), 0.2, dtype=np.float32)
    gt = np.zeros((2, 3, 2, 3, 8, 8), dtype=np.float32)
    overlap = {
        (e, t, i, j): {
            "source": "ground_truth_depth_and_camera",
            "ratio": 0.3 if (e, t, i, j) == (0, 1, 0, 2) else 0.0,
        }
        for e in range(2)
        for t in range(2)
        for i in range(3)
        for j in range(i + 1, 3)
    }
    backend = FakeMEt3R()
    result = evaluate_level1(
        pred,
        gt,
        overlap_metadata=overlap,
        seeds=[42, 43],
        metrics=["psnr", "met3r"],
        backends={"met3r": backend},
    )
    assert backend.count == 2  # One eligible synchronized pair x two seeds.
    assert result["metrics"]["met3r"]["mean"] == 0.25
    assert result["metrics"]["met3r"]["episode_count"] == 1
    assert result["metrics"]["met3r"]["ci95"] is None
    assert len(result["pair_eligibility"]) == 12
    assert result["metrics"]["psnr"]["episode_count"] == 2
    overlap[0, 1, 0, 2]["source"] = "generated_depth"
    with pytest.raises(ValueError, match="GT geometry"):
        evaluate_level1(
            pred,
            gt,
            overlap_metadata=overlap,
            seeds=[42, 43],
            metrics=["met3r"],
            backends={"met3r": backend},
        )


def saved_sets(tmp_path, monkeypatch, split="test"):
    import mvwd_metrics.io as module

    class Raw:
        def __init__(self, root, chosen):
            self.metadata = {"release_id": "release"}
            self.manifest_hash = "split-hash"
            self.split = chosen
            self.records = [
                {"id": "episode", "scene": "house", "robots": 1, "frames": 1}
            ]

    monkeypatch.setattr(module, "MVWDRaw", Raw)
    protocol = {
        "target_hw": [8, 8],
        "robots": [0],
        "frames_per_episode": 1,
        "frame_sample_seed": 42,
        "seeds": [42, 43],
    }
    dirs = []
    for seed in protocol["seeds"]:
        d = tmp_path / str(seed)
        d.mkdir()
        dirs.append(d)
        metadata = {
            "ground_truth_loaded": False,
            "checkpoint_sha256": "checkpoint",
            "dataset": {
                "query_count": 1,
                "split": split,
                "release_id": "release",
                "split_sha256": "split-hash",
            },
            "config": {
                "data": {"root": "/dataset"},
                "inference": {"seed": seed, "ddim_steps": 50},
            },
        }
        row = {
            "episode_id": "episode",
            "robot_id": 0,
            "frame_index": 0,
            "path": "prediction.png",
            "seed": int(
                hashlib.sha256(("%s:release:episode:0:0" % seed).encode()).hexdigest()[
                    :8
                ],
                16,
            ),
        }
        (d / "metadata.json").write_text(json.dumps(metadata))
        (d / "predictions.jsonl").write_text(json.dumps(row) + "\n")
        Image.new("RGB", (8, 8)).save(d / "prediction.png")
    return dirs, protocol


def test_formal_coverage_requires_full_test_and_all_fixed_seeds(tmp_path, monkeypatch):
    dirs, protocol = saved_sets(tmp_path, monkeypatch)
    sets, raw, records, inputs, coverage = load_prediction_sets(dirs, protocol)
    assert coverage["full_test_protocol"]
    with pytest.raises(ValueError, match="Main benchmark"):
        load_prediction_sets(dirs[:1], protocol)
    assert not load_prediction_sets(dirs[:1], protocol, preview=True)[-1][
        "full_test_protocol"
    ]
    metadata = json.loads((dirs[0] / "metadata.json").read_text())
    metadata["dataset"]["split"] = "train"
    (dirs[0] / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Main benchmark"):
        load_prediction_sets(dirs[:1], protocol)


def test_duplicate_images_and_mixed_checkpoints_are_rejected(tmp_path, monkeypatch):
    dirs, protocol = saved_sets(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Duplicate"):
        load_prediction_sets([dirs[0], dirs[0]], protocol, True)
    path = dirs[1] / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["checkpoint_sha256"] = "other"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mix checkpoints"):
        load_prediction_sets(dirs, protocol)


def test_metrics_runner_routes_outputs_under_evaluations(tmp_path):
    (tmp_path / "config.yaml").write_text("training: {output_dir: /unused}\n")
    _, snapshot, output, command = runner.plan_step(
        tmp_path, "metrics", "metrics_level1", ["--predictions", "/saved", "--preview"]
    )
    assert output == tmp_path / "evaluations/metrics"
    assert command[command.index("--output") + 1] == str(output)
    assert command[-1] == "--preview"


class FakeFID:
    def features(self, images):
        return np.asarray(images).mean((2, 3)).astype(np.float64)

    def distance(self, first, second):
        return float(np.square(first.mean - second.mean).sum())

    def provenance(self):
        return {"implementation": "test double"}


def test_fid_pools_whole_population_per_seed_and_counts_gt_once():
    from mvwd_metrics import Level1Evaluator

    evaluator = Level1Evaluator(
        seeds=[42, 43], metrics=["fid"], backends={"fid": FakeFID()}
    )
    for episode, n in [("short", 1), ("long", 3)]:
        targets = np.zeros((n, 3, 8, 8), np.float32)
        for seed, error in [(42, 0.2), (43, 0.4)]:
            evaluator.update_views(
                episode, seed, list(range(n)), targets + error, targets
            )
    result = evaluator.finish()["metrics"]["fid"]
    assert result["gt_images"] == 4 and result["images_per_seed"] == 4
    assert evaluator.gt_moments.count == 4
    assert result["mean"] == pytest.approx((3 * 0.2**2 + 3 * 0.4**2) / 2)
    assert result["ci95"] is None


def test_invalid_met3r_reconstruction_propagates_undefined_without_zero_or_pair_reselection():
    from mvwd_metrics.core import UndefinedMetricError
    from mvwd_metrics import Level1Evaluator

    class InvalidEstimator:
        def __call__(self, a, b):
            raise UndefinedMetricError("official estimator has no overlap")

        def provenance(self):
            return {"implementation": "test double"}

    evaluator = Level1Evaluator(
        seeds=[42], metrics=["psnr", "met3r"], backends={"met3r": InvalidEstimator()}
    )
    a = np.full((2, 3, 8, 8), 0.2, np.float32)
    evaluator.update_views("episode", 42, ["a", "b"], a, np.zeros_like(a))
    row = evaluator.update_pair("episode", 42, "pair", a[:1], a[1:])
    assert row["met3r"] is None and row["status"] == "undefined_reconstruction"
    result = evaluator.finish()["metrics"]
    assert np.isfinite(result["psnr"]["mean"])
    assert result["met3r"]["mean"] is None
    assert result["met3r"]["failure_fraction"] == 1.0
    assert result["met3r"]["evaluated_pair_count"] == 1


def test_cli_streams_query_pair_and_gt_reference_separately(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import yaml
    import scripts.evaluate_mvwd_level1 as cli
    from mvwd_metrics import Level1Evaluator

    protocol = {
        "level": 1,
        "metrics": ["psnr", "met3r"],
        "target_hw": [8, 8],
        "bootstrap_samples": 50,
        "bootstrap_seed": 42,
        "weights_dir": str(tmp_path / "weights"),
        "overlap": {"threshold": 0.1},
    }
    pp = tmp_path / "protocol.yaml"
    pp.write_text(yaml.safe_dump(protocol))
    units = [(0, 0), (1, 0), (2, 0)]
    sets = {42: {}}
    for r, t in units:
        p = tmp_path / ("robot%d.png" % r)
        Image.new("RGB", (8, 8), (30, 60, 90)).save(p)
        sets[42]["episode", r, t] = {"path": p}
    raw = SimpleNamespace(provenance=lambda: {"release_id": "release"})
    monkeypatch.setattr(cli, "load_config", lambda path: {"data": {"cache_root": None}})
    monkeypatch.setattr(
        cli,
        "load_prediction_sets",
        lambda *a: (
            sets,
            raw,
            {"episode": {"scene": "house"}},
            [],
            {"full_test_protocol": True},
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_gt_episode",
        lambda *a, **kw: (
            {u: np.zeros((3, 8, 8), np.float32) for u in units},
            {u: {} for u in units},
        ),
    )
    monkeypatch.setattr(
        cli,
        "visible_surface_overlap",
        lambda *a, **kw: {"ratio": 1.0, "source": "ground_truth_depth_and_camera"},
    )
    monkeypatch.setattr(
        cli,
        "Level1Evaluator",
        lambda *a, **kw: Level1Evaluator(*a, **kw, backends={"met3r": FakeMEt3R()}),
    )
    monkeypatch.setattr(cli, "weight_provenance", lambda *a, **kw: [])
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--config",
            "unused",
            "--protocol",
            str(pp),
            "--output",
            str(output),
            "--predictions",
            "unused",
            "--preview",
            "--device",
            "cpu",
            "--met3r-gt-reference",
        ],
    )
    cli.main()
    result = json.loads((output / "metrics.json").read_text())
    assert result["metrics"]["met3r"]["mean"] == 0.25
    assert result["benchmark_ready"] is False
    assert result["all_requested_metrics_defined"] is True
    pairs = [
        json.loads(x) for x in (output / "per_pair.jsonl").read_text().splitlines()
    ]
    queries = [
        json.loads(x) for x in (output / "per_query.jsonl").read_text().splitlines()
    ]
    assert len(pairs) == 3 and all(x["gt_reference_met3r"] == 0.25 for x in pairs)
    assert len(queries) == 3 and all("gt_reference_met3r" not in x for x in queries)


def test_inference_seed_override_is_recorded_and_changes_the_same_query(
    tmp_path, monkeypatch
):
    """K=3 support must change samples and preserve verifiable seed metadata."""
    from types import SimpleNamespace
    import copy
    import torch
    import infer_mvwd
    import mvwd_model
    import adapters.camera_geometry as geometry

    class TinyDataset:
        robot_assets = None
        raw = SimpleNamespace(metadata={"release_id": "release"})

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "episode_id": "episode",
                "bev_key": "bev",
                "robot_id": 0,
                "frame_index": 0,
                "bev_rgb": np.zeros((8, 8, 3), np.float32),
            }

        def provenance(self):
            return {"query_count": 1, "release_id": "release", "split": "test"}

    class TinyModel(torch.nn.Module):
        def predict_occupancy(self, bev):
            return torch.zeros(1)

        def refine(self, rgb, depth, steps, guidance):
            return torch.rand((1, 3, 8, 8)) * 2 - 1

    class TinyRenderer:
        def __init__(self, **kwargs):
            pass

        def __call__(self, voxel, batch):
            return {
                "rgb": torch.zeros((1, 8, 8, 3)),
                "depth_condition": torch.zeros((1, 8, 8)),
                "depth_z": torch.zeros((1, 8, 8)),
            }

    config = {
        "training": {"seed": 42},
        "inference": {"seed": 42, "ddim_steps": 50, "guidance_scale": 9},
        "data": {"cache_root": None},
        "renderer": {},
    }
    monkeypatch.setattr(infer_mvwd, "load_config", lambda path: copy.deepcopy(config))
    monkeypatch.setattr(infer_mvwd, "make_dataset", lambda *a, **kw: TinyDataset())
    monkeypatch.setattr(mvwd_model, "load_trained", lambda *a: TinyModel())
    monkeypatch.setattr(geometry, "PerspectiveRenderer", TinyRenderer)
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_text("test-only tiny checkpoint")
    images = []
    for seed in (42, 43):
        output = tmp_path / ("seed%d" % seed)
        monkeypatch.setattr(
            "sys.argv",
            [
                "infer",
                "--config",
                "unused",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--seed",
                str(seed),
                "--no-gt-comparison",
            ],
        )
        infer_mvwd.main()
        metadata = json.loads((output / "metadata.json").read_text())
        query = json.loads((output / "predictions.jsonl").read_text().strip())
        assert metadata["config"]["inference"]["seed"] == seed
        expected = int(
            hashlib.sha256(("%s:release:episode:0:0" % seed).encode()).hexdigest()[:8],
            16,
        )
        assert query["seed"] == expected
        images.append(np.asarray(Image.open(output / query["path"])))
    assert not np.array_equal(*images)
