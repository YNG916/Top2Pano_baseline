"""Shared streaming and [B,N,T,3,H,W] APIs for Level-1 image evaluation."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np

from .core import (
    PSNRMetric,
    LPIPSMetric,
    FIDMetric,
    MEt3RMetric,
    FeatureMoments,
    EpisodeMetricAggregator,
    images01,
    UndefinedMetricError,
)
from .geometry import visible_surface_overlap, gt_surface_correspondences, correspondence_receipt
from .gtwarp import MVGTWarpDINOMetric, METRIC_NAME


class Level1Evaluator:
    def __init__(
        self,
        seeds=(42, 43, 44),
        metrics=("psnr", "lpips", "fid", METRIC_NAME),
        device="cuda",
        weights_dir="models/metrics",
        bootstrap_samples=2000,
        bootstrap_seed=20261015,
        backends=None,
    ):
        self.metrics = tuple(metrics)
        if (
            not self.metrics
            or len(set(metrics)) != len(metrics)
            or set(metrics) - {"psnr", "lpips", "fid", "met3r", METRIC_NAME}
        ):
            raise ValueError("Choose distinct Level1 metrics: psnr lpips fid met3r mv_gtwarp_dino")
        self.seeds, self.device = tuple(seeds), device
        self.aggregate = EpisodeMetricAggregator(
            seeds, bootstrap_samples, bootstrap_seed
        )
        self.backends = dict(backends or {})
        self.weights_dir = str(Path(weights_dir).resolve())
        self.gt_moments = FeatureMoments()
        self.pred_moments = {seed: FeatureMoments() for seed in seeds}
        self.gt_units = set()
        self.view_units = set()
        self.pair_units = set()
        self.gtwarp_units = set()
        self.failed_pairs = []

    def backend(self, metric):
        if metric not in self.backends:
            if metric != "psnr":
                import torch

                torch.hub.set_dir(str(Path(self.weights_dir) / "torch_hub"))
            if metric == "psnr":
                self.backends[metric] = PSNRMetric()
            elif metric == "lpips":
                self.backends[metric] = LPIPSMetric(self.device)
            elif metric == "fid":
                self.backends[metric] = FIDMetric(self.device, self.weights_dir)
            elif metric == "met3r":
                self.backends[metric] = MEt3RMetric(self.device)
            elif metric == METRIC_NAME:
                self.backends[metric] = MVGTWarpDINOMetric(self.device, self.weights_dir)
        return self.backends[metric]

    def update_views(self, episode, seed, units, predictions, targets):
        a, b = images01(predictions), images01(targets)
        if a.shape != b.shape or len(units) != len(a) or seed not in self.seeds:
            raise ValueError("View batch/seed mismatch")
        keys = [(str(episode), seed, unit) for unit in units]
        if len(set(keys)) != len(keys) or any(k in self.view_units for k in keys):
            raise ValueError("Duplicate evaluated query")
        self.view_units.update(keys)
        records = [
            {"episode_id": str(episode), "seed": seed, "unit": unit} for unit in units
        ]
        for metric in ("psnr", "lpips"):
            if metric in self.metrics:
                values = self.backend(metric)(a, b)
                if len(values) != len(a):
                    raise ValueError("Metric returned wrong batch length")
                for row, value, unit in zip(records, values, units):
                    self.aggregate.add(metric, episode, seed, unit, value)
                    row[metric] = float(value)
        if "fid" in self.metrics:
            metric = self.backend("fid")
            self.pred_moments[seed].update(metric.features(a))
            fresh = [
                i for i, u in enumerate(units) if (str(episode), u) not in self.gt_units
            ]
            if fresh:
                self.gt_moments.update(metric.features(b[fresh]))
                self.gt_units.update((str(episode), units[i]) for i in fresh)
        return records

    def update_pair(self, episode, seed, unit, first, second):
        key = (str(episode), seed, unit)
        if key in self.pair_units:
            raise ValueError("Duplicate evaluated camera pair")
        self.pair_units.add(key)
        try:
            value = float(self.backend("met3r")(images01(first), images01(second))[0])
        except UndefinedMetricError as exc:
            row = {
                "episode_id": str(episode),
                "seed": seed,
                "unit": unit,
                "met3r": None,
                "status": "undefined_reconstruction",
                "reason": str(exc),
            }
            self.failed_pairs.append(row)
            return row
        self.aggregate.add("met3r", episode, seed, unit, value)
        return {
            "episode_id": str(episode),
            "seed": seed,
            "unit": unit,
            "met3r": value,
            "status": "ok",
        }

    def finish(self):
        # Include views even if only FID/MEt3R were selected, and check seed support.
        supports = {
            seed: {(e, u) for e, s, u in self.view_units if s == seed}
            for seed in self.seeds
        }
        if not supports[self.seeds[0]] or any(
            supports[s] != supports[self.seeds[0]] for s in self.seeds
        ):
            raise ValueError(
                "All seeds must evaluate exactly the same nonempty query set"
            )
        report = {}
        for name in self.metrics:
            if name == "fid":
                continue
            if name == "met3r" and self.failed_pairs:
                # Do not redefine the GT pair set based on generated reconstruction success.
                report[name] = {
                    "mean": None,
                    "ci95": None,
                    "status": "undefined_pairs",
                    "failed_pair_count": len(self.failed_pairs),
                    "evaluated_pair_count": len(self.pair_units),
                    "failure_fraction": len(self.failed_pairs) / len(self.pair_units),
                    "failures": list(self.failed_pairs),
                    "note": "Main MEt3R is undefined if any GT-eligible pair fails. No zero fill or success-only mean.",
                }
            else:
                report[name] = self.aggregate.summarize(name)
        if "fid" in self.metrics:
            count = self.gt_moments.count
            if any(x.count != count for x in self.pred_moments.values()):
                raise ValueError("FID GT/prediction population sizes differ")
            scores = {
                str(seed): self.backend("fid").distance(x, self.gt_moments)
                for seed, x in self.pred_moments.items()
            }
            report["fid"] = {
                "mean": float(np.mean(list(scores.values()))),
                "seed_scores": scores,
                "images_per_seed": count,
                "gt_images": count,
                "ci95": None,
                "aggregation": "whole-population FID per seed, mean across seeds",
                "small_population": count < 2048,
            }
        return {
            "metrics": report,
            "seeds": list(self.seeds),
            "aggregation": "unit mean within episode/seed; seed mean within episode; equal episode mean",
            "backends": {
                name: backend.provenance() for name, backend in self.backends.items()
            },
        }

    def update_gtwarp_pair(self, episode, seed, unit, matches, *, features):
        key = (str(episode), seed, unit)
        if seed not in self.seeds or key in self.gtwarp_units:
            raise ValueError("Invalid seed or duplicate evaluated GTWarp pair")
        details = self.backend(METRIC_NAME).score_features(*features, matches)
        self.aggregate.add(METRIC_NAME, episode, seed, unit, details["score"])
        self.gtwarp_units.add(key)
        return {
            "episode_id": str(episode), "seed": seed, "unit": unit,
            METRIC_NAME: float(details["score"]),
            "gtwarp_details": details, "gtwarp_status": "ok",
        }


def evaluate_level1(
    pred_images,
    gt_images,
    camera_params=None,
    overlap_metadata=None,
    *,
    episode_ids=None,
    seeds=(42,),
    overlap_threshold=0.1,
    overlap_options=None,
    **kwargs,
):
    """pred: [K,B,N,T,3,H,W] or [B,N,T,3,H,W]; GT: [B,N,T,3,H,W].

    Camera params: mapping (episode_index, view_index, time_index) -> GT geometry camera.
    Alternatively overlap_metadata maps (b,t,i,j) -> a ground_truth_depth_and_camera receipt.
    Time indices define independent synchronized multi-view conditions, never temporal pairs.
    """
    pred, gt = np.asarray(pred_images), np.asarray(gt_images)
    if pred.ndim == 6:
        pred = pred[None]
    if (
        pred.ndim != 7
        or gt.ndim != 6
        or pred.shape[1:] != gt.shape
        or len(seeds) != len(pred)
    ):
        raise ValueError(
            "Expected matching K,B,N,T,3,H,W predictions and B,N,T,3,H,W GT"
        )
    if not 0 <= overlap_threshold < 1:
        raise ValueError("Invalid overlap threshold")
    k, b, n, t, c, h, w = pred.shape
    episodes = (
        list(episode_ids) if episode_ids is not None else [str(i) for i in range(b)]
    )
    if len(episodes) != b or len(set(episodes)) != b:
        raise ValueError("Episode IDs must be unique")
    evaluator = Level1Evaluator(seeds=seeds, **kwargs)
    pairs = []
    units = [(i, time) for i in range(n) for time in range(t)]
    for e in range(b):
        for trial, seed in enumerate(seeds):
            evaluator.update_views(
                episodes[e],
                seed,
                units,
                pred[trial, e].reshape(n * t, c, h, w),
                gt[e].reshape(n * t, c, h, w),
            )
        if not {"met3r", METRIC_NAME}.intersection(evaluator.metrics):
            continue
        if overlap_metadata is None and camera_params is None:
            raise ValueError(
                "Cross-view metrics require GT overlap eligibility; no generated-depth pair selection"
            )
        if METRIC_NAME in evaluator.metrics and camera_params is None:
            raise ValueError("GTWarp requires actual GT cameras/depth, not just overlap ratios")
        for time in range(t):
            feature_cache = {}
            for i, j in combinations(range(n), 2):
                matches = None
                if METRIC_NAME in evaluator.metrics:
                    matches = gt_surface_correspondences(camera_params[e, i, time],
                                                         camera_params[e, j, time],
                                                         **(overlap_options or {}))
                if overlap_metadata is None and matches is not None and "met3r" not in evaluator.metrics:
                    receipt = correspondence_receipt(matches)
                elif overlap_metadata is None:
                    receipt = visible_surface_overlap(
                        camera_params[e, i, time],
                        camera_params[e, j, time],
                        **(overlap_options or {}),
                    )
                else:
                    receipt = overlap_metadata[e, time, i, j]
                if (
                    not np.isfinite(receipt.get("ratio", np.nan))
                    or not 0 <= receipt["ratio"] <= 1
                ):
                    raise ValueError("Invalid GT overlap ratio")
                if receipt.get("source") != "ground_truth_depth_and_camera":
                    raise ValueError(
                        "Pair eligibility must be defined from GT geometry"
                    )
                if "met3r" in evaluator.metrics and receipt["ratio"] > overlap_threshold:
                    for trial, seed in enumerate(seeds):
                        evaluator.update_pair(
                            episodes[e],
                            seed,
                            (time, i, j),
                            pred[trial, e, i, time][None],
                            pred[trial, e, j, time][None],
                        )
                if matches is not None and matches["ratio"] > overlap_threshold:
                    for trial, seed in enumerate(seeds):
                        for view in (i, j):
                            if (trial, view) not in feature_cache:
                                feature_cache[trial, view] = evaluator.backend(METRIC_NAME).features(
                                    pred[trial, e, view, time][None])[0]
                        evaluator.update_gtwarp_pair(
                            episodes[e], seed, (time, i, j), matches,
                            features=(feature_cache[trial, i], feature_cache[trial, j]))
                pairs.append(
                    {
                        "episode_id": episodes[e],
                        "time_index": time,
                        "views": [i, j],
                        **receipt,
                        "eligible": receipt["ratio"] > overlap_threshold,
                        **({"gtwarp": correspondence_receipt(matches),
                            "gtwarp_eligible": matches["ratio"] > overlap_threshold}
                           if matches is not None else {}),
                    }
                )
    result = evaluator.finish()
    result["pair_eligibility"] = pairs
    return result
