"""Evaluate saved Level1 predictions using the frozen episode/seed protocol."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import yaml

from mvwd_runtime import REPO, load_config
from mvwd_metrics import Level1Evaluator
from mvwd_metrics.core import UndefinedMetricError, EpisodeMetricAggregator
from mvwd_metrics.geometry import (visible_surface_overlap, gt_surface_correspondences, correspondence_receipt)
from mvwd_metrics.gtwarp import METRIC_NAME
from mvwd_metrics.io import (
    load_prediction_sets,
    load_gt_episode,
    json_safe,
    weight_provenance,
)
from adapters.mvwd_raw import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--protocol", default=str(REPO / "configs/mvwd_metrics_level1_gtwarp.yaml")
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        type=Path,
        help="Same checkpoint/protocol; may include disjoint scene shards for each seed",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Explicit subset/single-seed diagnostic, never a formal benchmark row",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=("psnr", "lpips", "fid", "met3r", METRIC_NAME),
        help="Metric subset for smoke checks; formal benchmark requires the protocol metric set",
    )
    parser.add_argument(
        "--met3r-gt-reference",
        action="store_true",
        help="Diagnostic only: also run official MEt3R on the same GT camera pairs",
    )
    parser.add_argument("--gtwarp-gt-reference", action="store_true",
                        help="Diagnostic only: GT feature warp error on the same fixed correspondences")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    config = load_config(args.config)
    protocol_path = Path(args.protocol).resolve()
    protocol_bytes = protocol_path.read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol.get("level") != 1:
        parser.error("This evaluator implements Level1 only")
    if (
        protocol.get(
            "met3r_undefined_policy", "null_entire_metric_and_report_failure_fraction"
        )
        != "null_entire_metric_and_report_failure_fraction"
    ):
        parser.error("Unsupported MEt3R failure protocol")
    metrics = args.metrics or protocol["metrics"]
    if not args.preview and set(metrics) != set(protocol["metrics"]):
        parser.error("Formal benchmark requires every metric in the selected protocol")
    if args.met3r_gt_reference and "met3r" not in metrics:
        parser.error("--met3r-gt-reference requires met3r in the selected metrics")
    if args.gtwarp_gt_reference and METRIC_NAME not in metrics:
        parser.error("--gtwarp-gt-reference requires mv_gtwarp_dino in the selected metrics")
    if METRIC_NAME in metrics:
        expected = {
            "implementation": "mvwd_gtwarp_dino_v1", "feature_backbone": "dino_vits16",
            "upsampler": "featup", "use_norm": True, "input_resolution": "native",
            "visibility": "valid_bilinear_footprint", "distance": "cosine",
            "normalization": "after_bilinear_sampling",
            "pair_aggregation": "mean_of_two_directions", "surfaces": "all_gt_visible_same_time",
        }
        if protocol.get("gtwarp") != expected:
            parser.error("Unsupported or missing frozen GTWarp feature/visibility protocol")
    pair_metrics = set(metrics).intersection({"met3r", METRIC_NAME})
    overlap = dict(protocol["overlap"])
    threshold = overlap.pop("threshold")
    if not 0 <= threshold < 1:
        parser.error("overlap threshold must be in [0,1)")
    sets, raw, records, inputs, coverage = load_prediction_sets(
        args.predictions, protocol, args.preview
    )
    seeds = sorted(sets)
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.yaml").write_bytes(protocol_bytes)
    evaluator = Level1Evaluator(
        seeds,
        metrics,
        args.device,
        str((REPO / protocol["weights_dir"]).resolve()),
        protocol["bootstrap_samples"],
        protocol["bootstrap_seed"],
    )
    reference_aggregate = EpisodeMetricAggregator([0], protocol["bootstrap_samples"], protocol["bootstrap_seed"])
    runtime = {}
    if set(metrics) - {"psnr"}:
        import torch

        torch.manual_seed(protocol["bootstrap_seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
        if args.device.startswith("cuda"):
            runtime["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
    episodes = sorted({key[0] for key in sets[seeds[0]]})
    overlap_count = eligible_count = 0
    episodes_with_pairs = set()
    pair_coverage = {name: {"eligible_pairs": 0, "episodes": set()} for name in pair_metrics}
    with (directory / "per_query.jsonl").open("w") as qlog, (
        directory / "per_pair.jsonl"
    ).open("w") as plog, (directory / "overlap.jsonl").open("w") as olog:

        def write(stream, row):
            stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
            stream.flush()

        for position, episode in enumerate(episodes, 1):
            units = sorted((r, t) for e, r, t in sets[seeds[0]] if e == episode)
            targets, cameras = load_gt_episode(
                raw,
                records[episode],
                units,
                protocol["target_hw"],
                config["data"].get("cache_root"),
                geometry=bool(pair_metrics),
            )
            plans = {}
            for frame in sorted({t for r, t in units}) if pair_metrics else []:
                frame_plans = []
                for first, second in combinations(sorted(r for r, t in units if t == frame), 2):
                    key = (frame, first, second)
                    matches = (gt_surface_correspondences(cameras[first, frame], cameras[second, frame],
                               **overlap) if METRIC_NAME in metrics else None)
                    receipts = {}
                    if matches is not None:
                        receipts[METRIC_NAME] = correspondence_receipt(matches)
                    if "met3r" in metrics:
                        receipts["met3r"] = visible_surface_overlap(
                            cameras[first, frame], cameras[second, frame], **overlap)
                    eligible = {name: receipt["ratio"] > threshold for name, receipt in receipts.items()}
                    primary = receipts[METRIC_NAME if matches is not None else "met3r"]
                    overlap_count += 1
                    eligible_count += int(any(eligible.values()))
                    if any(eligible.values()):
                        episodes_with_pairs.add(episode)
                    for name, accepted in eligible.items():
                        if accepted:
                            pair_coverage[name]["eligible_pairs"] += 1
                            pair_coverage[name]["episodes"].add(episode)
                    write(olog, {"episode_id": episode, "scene": records[episode]["scene"],
                                 "frame_index": frame, "robot_ids": [first, second],
                                 "eligible": any(eligible.values()), **primary,
                                 "per_metric": {name: {**receipt, "eligible": eligible[name]}
                                                for name, receipt in receipts.items()}})
                    frame_plans.append({"key": key, "matches": matches, "receipts": receipts,
                                        "eligible": eligible})
                plans[frame] = frame_plans
            gt_reference = {}
            for seed in seeds:
                predictions = {}
                for unit in units:
                    p = sets[seed][(episode, *unit)]["path"]
                    with Image.open(p) as im:
                        predictions[unit] = (
                            np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
                        )
                for start in range(0, len(units), args.batch_size):
                    batch = units[start : start + args.batch_size]
                    rows = evaluator.update_views(
                        episode,
                        seed,
                        batch,
                        np.stack([predictions[u] for u in batch]),
                        np.stack([targets[u] for u in batch]),
                    )
                    for row, (robot, frame) in zip(rows, batch):
                        row.update(
                            scene=records[episode]["scene"],
                            robot_id=robot,
                            frame_index=frame,
                        )
                        write(qlog, row)
                for frame, frame_plans in plans.items():
                    gtwarp_views = sorted({r for plan in frame_plans
                                          if plan["eligible"].get(METRIC_NAME)
                                          for r in plan["key"][1:]})
                    features, gt_features = {}, {}
                    if gtwarp_views:
                        backend = evaluator.backend(METRIC_NAME)
                        dense = backend.features(np.stack([predictions[r, frame] for r in gtwarp_views]))
                        features = dict(zip(gtwarp_views, dense))
                        del dense
                        if args.gtwarp_gt_reference and seed == seeds[0]:
                            dense = backend.features(np.stack([targets[r, frame] for r in gtwarp_views]))
                            gt_features = dict(zip(gtwarp_views, dense))
                            del dense
                    for plan in frame_plans:
                        if not any(plan["eligible"].values()):
                            continue
                        key = plan["key"]
                        _, first, second = key
                        row = {"episode_id": episode, "seed": seed, "unit": key}
                        if plan["eligible"].get("met3r"):
                            row.update(evaluator.update_pair(episode, seed, key,
                                       predictions[first, frame][None], predictions[second, frame][None]))
                            if args.met3r_gt_reference:
                                try:
                                    reference = evaluator.backend("met3r")(
                                        targets[first, frame][None], targets[second, frame][None])
                                    row["gt_reference_met3r"] = float(reference[0])
                                except UndefinedMetricError as exc:
                                    row["gt_reference_met3r"] = None
                                    row["gt_reference_failure"] = str(exc)
                        if plan["eligible"].get(METRIC_NAME):
                            row.update(evaluator.update_gtwarp_pair(
                                episode, seed, key, plan["matches"],
                                features=(features[first], features[second])))
                            if args.gtwarp_gt_reference:
                                if seed == seeds[0]:
                                    gt_reference[key] = backend.score_features(
                                        gt_features[first], gt_features[second], plan["matches"])
                                if seed == seeds[0]:
                                    reference_aggregate.add(METRIC_NAME, episode, 0, key, gt_reference[key]["score"])
                                row["gt_reference_mv_gtwarp_dino"] = gt_reference[key]["score"]
                                row["gt_reference_gtwarp_details"] = gt_reference[key]
                        row.update(scene=records[episode]["scene"], frame_index=frame,
                                   robot_ids=[first, second], gt_overlap_ratio=(
                                   plan["receipts"][METRIC_NAME if METRIC_NAME in metrics else "met3r"]["ratio"]))
                        write(plog, row)
                    # Drop dense maps before processing the next synchronized time.
                    features.clear()
                    gt_features.clear()
            print(
                "Evaluated %d/%d episodes (%s)"
                % (position, len(episodes), records[episode]["scene"]),
                flush=True,
            )
    result = evaluator.finish()
    if args.gtwarp_gt_reference:
        result["gtwarp_gt_reference"] = reference_aggregate.summarize(METRIC_NAME)
        result["gtwarp_gt_reference"]["role"] = "diagnostic GT control only; no subtraction or normalization of main score"
    result["all_requested_metrics_defined"] = all(
        value.get("mean") is not None for value in result["metrics"].values()
    )
    result["benchmark_ready"] = (
        not args.preview
        and coverage["full_test_protocol"]
        and result["all_requested_metrics_defined"]
    )
    notes = []
    if evaluator.failed_pairs:
        notes.append(
            "MEt3R is undefined: some GT-eligible pairs have invalid official reconstructions. Other metrics remain valid; no pair reselection or zero fill."
        )
    if args.preview:
        notes.append("Diagnostic preview: not a full-test K=3 benchmark result")
    if len(episodes) < 2:
        notes.append("One episode: no episode-bootstrap 95% CI")
    if pair_metrics and len(episodes_with_pairs) < len(episodes):
        notes.append(
            "Episodes without GT-eligible camera pairs have undefined cross-view metrics; excluded counts are explicit, never zero-filled"
        )
    if result.get("metrics", {}).get("fid", {}).get("small_population"):
        notes.append(
            "FID has fewer images than its 2048 feature dimensions; diagnostic estimate only"
        )
    result["downloaded_weights"] = (
        weight_provenance(
            evaluator.weights_dir, include_met3r="met3r" in evaluator.backends
        )
        if set(metrics) - {"psnr"}
        else []
    )
    result["implementation_sha256"] = {
        str(path.relative_to(REPO)): sha256_file(path)
        for path in [
            Path(__file__).resolve(),
            *sorted((REPO / "mvwd_metrics").glob("*.py")),
        ]
    }
    result.update(
        schema_version=2 if METRIC_NAME in metrics else 1,
        level=1,
        mode="preview" if args.preview else "benchmark",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        coverage=coverage,
        inputs=inputs,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        pair_selection={
            "candidate_pairs": overlap_count,
            "eligible_pairs": eligible_count,
            "episodes_with_eligible_pairs": len(episodes_with_pairs),
            "geometry_source": "ground truth only, shared by all methods/seeds",
            "episodes_without_eligible_pairs": (
                len(episodes) - len(episodes_with_pairs) if pair_metrics else None
            ),
            "per_metric": {name: {"eligible_pairs": counts["eligible_pairs"],
                                      "episodes_with_eligible_pairs": len(counts["episodes"]),
                                      "episodes_without_eligible_pairs": len(episodes) - len(counts["episodes"])}
                           for name, counts in pair_coverage.items()},
            "threshold_comparison": "minimum bidirectional GT overlap > threshold",
        },
        dataset=raw.provenance(),
        runtime=runtime,
        notes=notes,
        gt_reference_diagnostic=args.met3r_gt_reference,
        gtwarp_gt_reference_diagnostic=args.gtwarp_gt_reference,
        gt_resize="PIL BOX to the frozen target RGB resolution; no prediction resize",
        gt_use="evaluation only; RGB for paired/FID targets, depth/cameras for pair eligibility and GTWarp correspondences",
    )
    # Persist population statistics so FID can be recomputed without extracting features again.
    if "fid" in metrics:
        mean, cov = evaluator.gt_moments.statistics()
        np.savez_compressed(
            directory / "fid_gt_statistics.npz",
            mu=mean,
            sigma=cov,
            count=evaluator.gt_moments.count,
        )
        for seed, moments in evaluator.pred_moments.items():
            mean, cov = moments.statistics()
            np.savez_compressed(
                directory / ("fid_seed_%s_statistics.npz" % seed),
                mu=mean,
                sigma=cov,
                count=moments.count,
            )
    (directory / "metrics.json").write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    print(
        json.dumps(
            json_safe(
                {
                    "mode": result["mode"],
                    "coverage": coverage,
                    "metrics": {
                        m: {k: v for k, v in r.items() if k != "episodes"}
                        for m, r in result["metrics"].items()
                    },
                    "notes": notes,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
