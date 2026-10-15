"""Analytic GT correspondence and feature tests, independent of model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mvwd_metrics import MVGTWarpDINOMetric, evaluate_level1
from mvwd_metrics.geometry import camera, gt_surface_correspondences, correspondence_receipt


def plane(shift=0.0, depth=2.0):
    K = np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1.]])
    T = np.eye(4)
    T[0, 3] = shift
    return camera(K, T, np.full((8, 12), depth))


class PixelFeatures:
    def __call__(self, images):
        return torch.from_numpy(images.copy())

    def provenance(self):
        return {"implementation": "test-only pixel features, not DINO"}


def test_safe_bilinear_correspondences_identity_and_half_pixel_shift():
    a = plane()
    same = gt_surface_correspondences(a, a, boundary=0)
    assert same["ratio"] == 1
    np.testing.assert_array_equal(same["forward"]["target_uv"],
                                  same["forward"]["source_yx"][:, ::-1] + .5)
    moved = gt_surface_correspondences(a, plane(shift=.1), boundary=0)
    assert moved["ratio"] == pytest.approx(11 / 12)
    np.testing.assert_allclose(moved["forward"]["target_uv"][:, 0],
                               moved["forward"]["source_yx"][:, 1])
    assert "target_uv" not in correspondence_receipt(moved)["forward"]


def test_bilinear_visibility_rejects_occlusion_invalid_and_boundary_footprints():
    a, b = plane(), plane(shift=.1)
    b["depth_z"][:, 6:] = 1.0
    matches = gt_surface_correspondences(a, b, boundary=0)
    # Samples whose bilinear footprint touches the nearer surface or its edge are rejected.
    assert np.all(matches["forward"]["target_uv"][:, 0] <= 4.5)
    assert gt_surface_correspondences(a, plane(depth=1), boundary=0)["ratio"] == 0
    invalid = plane()
    invalid["depth_z"][:] = 0
    assert gt_surface_correspondences(invalid, a)["ratio"] == 0
    with pytest.raises(ValueError):
        gt_surface_correspondences(a, a, depth_atol_m=-1)


def test_analytic_cosine_symmetric_directions_and_swap_invariance():
    metric = MVGTWarpDINOMetric(extractor=PixelFeatures())
    matches = gt_surface_correspondences(plane(), plane(), boundary=0)
    a = torch.zeros(3, 8, 12)
    a[0] = 1
    b = torch.zeros_like(a)
    b[1] = 1
    score = metric.score_features(a, b, matches)
    assert score["score"] == pytest.approx(1.)
    assert metric.score_features(a, a, matches)["score"] == pytest.approx(0., abs=1e-6)
    assert metric.score_features(a, -a, matches)["score"] == pytest.approx(2.)
    reverse = {**matches, "forward": matches["backward"], "backward": matches["forward"]}
    assert metric.score_features(b, a, reverse)["score"] == pytest.approx(score["score"])


def test_fractional_warp_align_corners_and_normalization_after_interpolation():
    from mvwd_metrics.gtwarp import _directed_feature_error

    source = torch.zeros(2, 1, 2)
    source[:, 0, 0] = torch.tensor([1., 1.])
    source[:, 0, 1] = torch.tensor([1., 1.])
    target = torch.tensor([[[2., 0.]], [[0., 1.]]])
    match = {"source_yx": np.array([[0, 0]]), "target_uv": np.array([[1., .5]]),
             "visible_pixels": 1}
    # Interpolate [2,0] and [0,1] first -> [1,.5], THEN normalize.
    expected = 1 - 1.5 / np.sqrt(2 * 1.25)
    assert _directed_feature_error(source, target, match) == pytest.approx(expected, abs=1e-6)


def test_pair_averages_directions_equally_instead_of_pooling_pixel_counts():
    metric = MVGTWarpDINOMetric(extractor=PixelFeatures())
    a = torch.tensor([[[1., 1., 1.]], [[0., 0., 0.]]])
    b = torch.tensor([[[1., 1., 0.]], [[0., 0., 1.]]])
    matches = {
        "source": "ground_truth_depth_and_camera",
        "forward": {"source_yx": np.array([[0, 0], [0, 1]]),
                    "target_uv": np.array([[.5, .5], [1.5, .5]]), "visible_pixels": 2},
        "backward": {"source_yx": np.array([[0, 2]]),
                     "target_uv": np.array([[2.5, .5]]), "visible_pixels": 1},
    }
    result = metric.score_features(a, b, matches)
    assert result["forward"] == pytest.approx(0)
    assert result["backward"] == pytest.approx(1)
    assert result["score"] == pytest.approx(.5)  # not pooled 1/3


def test_prediction_errors_cannot_shrink_mask_or_be_silently_filled():
    metric = MVGTWarpDINOMetric(extractor=PixelFeatures())
    matches = gt_surface_correspondences(plane(), plane(), boundary=0)
    a = torch.ones(3, 8, 12)
    b = a.clone()
    b[:, 3, 3] = float("nan")
    with pytest.raises(ValueError, match="Invalid feature"):
        metric.score_features(a, b, matches)
    with pytest.raises(ValueError, match="Invalid feature"):
        metric.score_features(a, torch.zeros_like(a), matches)
    empty = gt_surface_correspondences(plane(), plane(depth=1), boundary=0)
    with pytest.raises(ValueError, match="empty masks"):
        metric.score_features(a, a, empty)
    with pytest.raises(ValueError, match="GT-only"):
        metric.score_features(a, a, {**matches, "source": "generated_depth"})


def test_api_reuses_features_same_time_only_and_excludes_no_overlap_episode():
    rng = np.random.default_rng(4)
    gt = rng.uniform(.1, 1, (2, 2, 2, 3, 8, 12)).astype(np.float32)
    pred = np.stack((gt, gt.copy()))
    cameras = {(e, r, t): plane(depth=1 if e == 1 and r == 1 else 2)
               for e in range(2) for r in range(2) for t in range(2)}
    result = evaluate_level1(pred, gt, camera_params=cameras, seeds=[42, 43],
                            metrics=["mv_gtwarp_dino"], overlap_options={"boundary": 0},
                            backends={"mv_gtwarp_dino": MVGTWarpDINOMetric(extractor=PixelFeatures())})
    metric = result["metrics"]["mv_gtwarp_dino"]
    assert metric["episode_count"] == 1
    assert metric["episodes"][0]["units_per_seed"] == 2  # two synchronized times, no temporal pairs
    assert metric["mean"] > 0
    assert len(result["pair_eligibility"]) == 4
    with pytest.raises(ValueError, match="actual GT cameras"):
        evaluate_level1(pred, gt, overlap_metadata={}, seeds=[42, 43], metrics=["mv_gtwarp_dino"])


def test_wrong_spatial_correspondence_increases_error_for_textured_surface():
    metric = MVGTWarpDINOMetric(extractor=PixelFeatures())
    a = np.random.default_rng(2).uniform(.01, 1, (1, 3, 8, 12)).astype(np.float32)
    matches = gt_surface_correspondences(plane(), plane(), boundary=0)
    correct = metric(a, a, matches)["score"]
    shifted = metric(a, np.roll(a, 3, axis=-1), matches)["score"]
    assert correct < 1e-6 and shifted > .05


def test_cli_gtwarp_references_and_pair_receipts_are_separate_from_scores(tmp_path, monkeypatch):
    import yaml
    import scripts.evaluate_mvwd_level1 as cli
    from mvwd_metrics import Level1Evaluator

    protocol = yaml.safe_load((Path(__file__).resolve().parents[1] /
                              "configs/mvwd_metrics_level1_gtwarp.yaml").read_text())
    protocol.update(metrics=["mv_gtwarp_dino"], target_hw=[8, 12],
                    weights_dir=str(tmp_path / "weights"), bootstrap_samples=20)
    protocol["overlap"]["boundary"] = 0
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    units = [(0, 0), (1, 0)]
    sets = {seed: {} for seed in [42, 43]}
    rgb = np.random.default_rng(2).integers(20, 240, (8, 12, 3), dtype=np.uint8)
    for seed in sets:
        for robot, time in units:
            dest = tmp_path / ("s%d_r%d.png" % (seed, robot))
            Image.fromarray(rgb if robot == 0 else np.roll(rgb, 3, axis=1)).save(dest)
            sets[seed]["episode", robot, time] = {"path": dest}
    monkeypatch.setattr(cli, "load_config", lambda _: {"data": {}})
    monkeypatch.setattr(cli, "load_prediction_sets", lambda *a: (
        sets, SimpleNamespace(provenance=lambda: {}), {"episode": {"scene": "house"}}, [],
        {"full_test_protocol": False}))
    monkeypatch.setattr(cli, "load_gt_episode", lambda *a, **kw: (
        {u: rgb.transpose(2, 0, 1).astype(np.float32) / 255 for u in units}, {u: plane() for u in units}))
    monkeypatch.setattr(cli, "Level1Evaluator", lambda *a, **kw: Level1Evaluator(*a, **kw,
        backends={"mv_gtwarp_dino": MVGTWarpDINOMetric(extractor=PixelFeatures())}))
    monkeypatch.setattr(cli, "weight_provenance", lambda *a, **kw: [])
    output = tmp_path / "result"
    monkeypatch.setattr("sys.argv", ["evaluate", "--config", "unused", "--output", str(output),
                        "--protocol", str(path), "--predictions", "unused", "--preview",
                        "--device", "cpu", "--gtwarp-gt-reference"])
    cli.main()
    result = json.loads((output / "metrics.json").read_text())
    assert result["metrics"]["mv_gtwarp_dino"]["mean"] > .01
    rows = [json.loads(x) for x in (output / "per_pair.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["gt_reference_mv_gtwarp_dino"] < 1e-6 for r in rows)
    assert all(r["gtwarp_details"]["forward_pixels"] == 96 for r in rows)
    assert result["pair_selection"]["per_metric"]["mv_gtwarp_dino"]["eligible_pairs"] == 1
    assert not result["benchmark_ready"]
