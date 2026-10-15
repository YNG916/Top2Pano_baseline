"""Small GT/negative/degenerate controls for the MVWD GTWarp metric (diagnostic only)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageFilter
import yaml

from mvwd_runtime import REPO, load_config
from mvwd_metrics.gtwarp import MVGTWarpDINOMetric
from mvwd_metrics.geometry import gt_surface_correspondences, correspondence_receipt
from mvwd_metrics.io import load_prediction_sets, load_gt_episode, json_safe, weight_provenance
from adapters.mvwd_raw import sha256_file


def blurred(image):
    rgb = np.rint(image.transpose(1, 2, 0) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(8)),
                      dtype=np.float32).transpose(2, 0, 1).copy() / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--protocol", default=str(REPO / "configs/mvwd_metrics_level1_gtwarp.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-pairs", type=int, default=12)
    args = parser.parse_args()
    if args.max_pairs < 1:
        parser.error("max-pairs must be positive")
    protocol_bytes = Path(args.protocol).read_bytes()
    protocol = yaml.safe_load(protocol_bytes)
    config = load_config(args.config)
    sets, raw, records, inputs, coverage = load_prediction_sets(args.predictions, protocol, preview=True)
    seed = min(sets)
    episodes = sorted({e for e, r, t in sets[seed]})
    options = dict(protocol["overlap"])
    threshold = options.pop("threshold")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.yaml").write_bytes(protocol_bytes)
    import torch

    torch.manual_seed(protocol["bootstrap_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metric = MVGTWarpDINOMetric(args.device, REPO / protocol["weights_dir"])
    donor = None
    donor_episode = episodes[-1]
    if len(episodes) > 1:
        donor_unit = min((r, t) for e, r, t in sets[seed] if e == donor_episode)
        donor_gt, _ = load_gt_episode(raw, records[donor_episode], [donor_unit],
                                      protocol["target_hw"], config["data"].get("cache_root"))
        donor = metric.features(donor_gt[donor_unit][None])[0]
    values = defaultdict(list)
    rows = []
    with (output / "controls.jsonl").open("w") as stream:
        for episode in episodes:
            units = sorted((r, t) for e, r, t in sets[seed] if e == episode)
            gt, cameras = load_gt_episode(raw, records[episode], units,
                protocol["target_hw"], config["data"].get("cache_root"), geometry=True)
            for frame in sorted({t for r, t in units}):
                robots = sorted(r for r, t in units if t == frame)
                plans = [(i, j, gt_surface_correspondences(cameras[i, frame], cameras[j, frame], **options))
                         for i, j in combinations(robots, 2)]
                plans = [(i, j, m) for i, j, m in plans if m["ratio"] > threshold]
                if not plans:
                    continue
                images = np.stack([gt[r, frame] for r in robots])
                reference = dict(zip(robots, metric.features(images)))
                predicted = []
                for r in robots:
                    with Image.open(sets[seed][episode, r, frame]["path"]) as image:
                        predicted.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)
                predicted = dict(zip(robots, metric.features(np.stack(predicted))))
                # Shift the image, not the frozen GT coordinates/masks.
                shifted = dict(zip(robots, metric.features(np.roll(images, 32, axis=-1))))
                smooth = dict(zip(robots, metric.features(np.stack([blurred(im) for im in images]))))
                gray = metric.features(np.full_like(images[:1], .5))[0]
                for i, j, matches in plans:
                    controls = {
                        "gt": (reference[i], reference[j]),
                        "prediction": (predicted[i], predicted[j]),
                        "target_shifted_32px": (reference[i], shifted[j]),
                        "duplicate_first_view": (reference[i], reference[i]),
                        "blurred_gt_pair": (smooth[i], smooth[j]),
                        "constant_gray_pair": (gray, gray),
                    }
                    if donor is not None and episode != donor_episode:
                        controls["wrong_episode_target"] = (reference[i], donor)
                    scores = {name: metric.score_features(a, b, matches)["score"]
                              for name, (a, b) in controls.items()}
                    row = {"episode_id": episode, "scene": records[episode]["scene"],
                           "frame_index": frame, "robot_ids": [i, j], "seed": seed,
                           "correspondences": correspondence_receipt(matches), "scores": scores}
                    for name, value in scores.items():
                        values[name].append(value)
                    rows.append(row)
                    stream.write(json.dumps(json_safe(row), allow_nan=False) + "\n")
                    stream.flush()
                    print("Validated %d pairs (%s)" % (len(rows), records[episode]["scene"]), flush=True)
                    if len(rows) >= args.max_pairs:
                        break
                if len(rows) >= args.max_pairs:
                    break
            if len(rows) >= args.max_pairs:
                break
    if not rows:
        raise ValueError("No GT-eligible pairs for validation")
    result = {
        "mode": "diagnostic_controls_only", "pair_count": len(rows),
        "control_means": {k: float(np.mean(v)) for k, v in values.items()},
        "control_pair_counts": {k: len(v) for k, v in values.items()},
        "mean_definition": "diagnostic pair means, not formal episode-macro benchmark scores",
        "limitations": ["No forced score ordering; GT is not a strict mathematical lower bound",
                        "Constant/blurred/repeated views may score well on consistency alone; retain fidelity metrics",
                        "Wrong-episode control requires two input episodes, with one held as donor"],
        "backends": metric.provenance(), "inputs": inputs, "coverage": coverage,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "implementation_sha256": {str(p.relative_to(REPO)): sha256_file(p) for p in
            [Path(__file__).resolve(), *sorted((REPO / "mvwd_metrics").glob("*.py"))]},
        "downloaded_weights": weight_provenance(REPO / protocol["weights_dir"]),
    }
    (output / "validation.json").write_text(json.dumps(json_safe(result), indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("pair_count", "control_means", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
