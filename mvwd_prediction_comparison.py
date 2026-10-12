"""Export matched GT/coarse/final panels from finished predictions, without a model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from adapters.mvwd_raw import MVWDRaw, sha256_file, within


def default_report_directory(predictions):
    return Path(predictions).resolve()


def comparison_panel(images, title):
    width, height = images[0].size
    panel = Image.new("RGB", (width * 3, height + 54), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((8, 5), title, fill="black")
    for index, (label, image) in enumerate(zip(("GT", "Coarse", "Prediction"), images)):
        if image.size != (width, height):
            raise ValueError("Comparison image sizes differ")
        draw.text((index * width + 8, 30), label, fill="black")
        panel.paste(image, (index * width, 54))
    return panel


def prediction_context(predictions):
    metadata = json.loads((predictions / "metadata.json").read_text())
    rows = [json.loads(line) for line in (predictions / "predictions.jsonl").read_text().splitlines() if line.strip()]
    keys = [(r["episode_id"], r["robot_id"], r["frame_index"]) for r in rows]
    if not rows or len(set(keys)) != len(rows):
        raise ValueError("Expected a nonempty prediction manifest with unique queries")
    if len(rows) != metadata["dataset"]["query_count"]:
        raise ValueError("Prediction manifest is incomplete; finish inference before exporting GT")
    if metadata.get("ground_truth_loaded") is not False:
        raise ValueError("Prediction metadata must record GT-free generation")
    raw = MVWDRaw(metadata["config"]["data"]["root"], metadata["dataset"]["split"])
    if raw.metadata["release_id"] != metadata["dataset"]["release_id"]:
        raise ValueError("Prediction and GT dataset releases differ")
    if raw.manifest_hash != metadata["dataset"]["split_sha256"]:
        raise ValueError("Prediction and GT split manifests differ")
    records = {r["id"]: r for r in raw.records}
    # Validate the complete query set and prediction files before creating output.
    for row in rows:
        record = records[row["episode_id"]]
        if not 0 <= row["robot_id"] < record["robots"] or not 0 <= row["frame_index"] < record["frames"]:
            raise ValueError("Prediction query is outside its episode")
        path = within(predictions, row["path"])
        if not path.is_file() or not path.with_suffix(".coarse.npz").is_file():
            raise FileNotFoundError("Prediction or coarse file is missing")
    return metadata, rows, raw, records


def export_comparisons(predictions, output=None, cache_root=None):
    """Read targets only from the pinned queries saved by completed inference.

    Does not modify prediction PNGs, manifests or metadata, and computes no metrics.
    Default output is alongside each prediction frame; derived files are never overwritten.
    """
    predictions = Path(predictions).resolve()
    output = Path(output).resolve() if output is not None else default_report_directory(predictions)
    inline = output == predictions
    if (output.exists() and not inline) or (output / "comparison.json").exists() or (output / "GT_COMPARISON.md").exists():
        raise FileExistsError("GT comparison already exists; original predictions are unchanged")
    metadata, rows, raw, records = prediction_context(predictions)
    for row in rows:
        for suffix in (".gt.png", ".coarse.png", ".comparison.png"):
            destination = within(output, str(Path(row["path"]).with_suffix(suffix)))
            if destination.exists():
                raise FileExistsError("Comparison file already exists: %s" % destination)
    existing_bev = None
    if (output / "bev_comparison.json").exists():
        existing_bev = json.loads((output / "bev_comparison.json").read_text())
        expected = {(row["episode_id"], row["robot_id"], row["frame_index"]) for row in rows}
        saved = {(row["episode_id"], row["robot_id"], row["frame_index"]) for row in existing_bev["queries"]}
        if (expected != saved or existing_bev["query_count"] != len(rows)
                or existing_bev["prediction_manifest_sha256"] != sha256_file(predictions / "predictions.jsonl")
                or existing_bev["dataset"]["release_id"] != raw.metadata["release_id"]
                or existing_bev.get("ground_truth_loaded_for_generation") is not False):
            raise ValueError("Existing BEV comparison does not match the saved predictions")
        paths = [v["world_bev_path"] for v in existing_bev["queries"]]
        for episode in existing_bev["episodes"].values():
            paths.extend((episode["gt_env_bev_path"], episode["calibration_path"]))
        if any(not within(output, path).is_file() for path in paths):
            raise FileNotFoundError("Existing BEV comparison is incomplete")
    output.mkdir(parents=True, exist_ok=inline)
    lock = output / ".gt_comparison.lock"
    lock.touch(exist_ok=False)
    try:
        exported, overviews, page_panels = [], [], []
        group, page = None, 1

        def flush_page():
            if not page_panels:
                return
            episode, robot = group
            sheet = Image.new("RGB", (max(p.width for p in page_panels), sum(p.height for p in page_panels)), "white")
            y = 0
            for panel in page_panels:
                sheet.paste(panel, (0, y))
                y += panel.height
            relative = Path(episode) / ("robot_%02d" % robot) / ("comparison_overview_%02d.png" % page)
            (output / relative).parent.mkdir(parents=True, exist_ok=True)
            if (output / relative).exists():
                raise FileExistsError("Comparison overview already exists: %s" % (output / relative))
            sheet.save(output / relative)
            overviews.append({"episode_id": episode, "scene": records[episode]["scene"],
                              "robot_id": robot, "page": page, "path": str(relative)})
            page_panels.clear()

        for row in sorted(rows, key=lambda r: (r["episode_id"], r["robot_id"], r["frame_index"])):
            episode, robot, frame = row["episode_id"], row["robot_id"], row["frame_index"]
            next_group = (episode, robot)
            if group != next_group:
                flush_page()
                group, page = next_group, 1
            record = records[episode]
            prediction_path = within(predictions, row["path"])
            with Image.open(prediction_path) as image:
                prediction = image.convert("RGB")
            with np.load(prediction_path.with_suffix(".coarse.npz"), allow_pickle=False) as packet:
                coarse = packet["rgb"]
            if coarse.shape != (prediction.height, prediction.width, 3) or not np.isfinite(coarse).all():
                raise ValueError("Invalid coarse RGB shape or values")
            coarse = Image.fromarray((coarse.clip(0, 1) * 255).astype(np.uint8))
            rgb = raw.views(record, robot, ("rgb",), cache_root)["rgb"][frame]
            gt = Image.fromarray(rgb).resize(prediction.size, Image.Resampling.BOX)
            relative = Path(row["path"])
            title = "%s | robot %d | frame %d" % (record["scene"], robot, frame)
            panel = comparison_panel((gt, coarse, prediction), title)
            entry = {"episode_id": episode, "scene": record["scene"], "robot_id": robot,
                     "frame_index": frame, "prediction_path": row["path"], "seed": row["seed"]}
            for category, image in (("gt", gt), ("coarse", coarse), ("comparisons", panel)):
                suffix = {"gt": ".gt.png", "coarse": ".coarse.png", "comparisons": ".comparison.png"}[category]
                destination = within(output, str(relative.with_suffix(suffix)))
                destination.parent.mkdir(parents=True, exist_ok=True)
                image.save(destination)
                entry[category + "_path"] = str(destination.relative_to(output))
            exported.append(entry)
            page_panels.append(panel)
            if len(page_panels) == 6:
                flush_page()
                page += 1
        flush_page()
        if existing_bev is None:
            export_bevs(predictions, output)
        report = {"prediction_directory": str(predictions), "report_directory": str(output),
                  "query_count": len(exported), "dataset": raw.provenance(),
                  "checkpoint_sha256": metadata["checkpoint_sha256"],
                  "prediction_manifest_sha256": sha256_file(predictions / "predictions.jsonl"),
                  "prediction_metadata_sha256": sha256_file(predictions / "metadata.json"),
                  "ground_truth_loaded_for_generation": False,
                  "ground_truth_loaded_for_comparison": True,
                  "gt_resize": "PIL BOX to prediction resolution, matching the training RGB adapter",
                  "columns": ["GT", "Coarse", "Prediction"], "queries": exported, "overviews": overviews,
                  "bev_manifest": "bev_comparison.json"}
        (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# GT / coarse / prediction", "", "GT is read after generation for visual comparison only.",
                 "GT uses the same episode, observer and physical frame, resized with BOX to prediction resolution.",
                 "Original predictions are unchanged. This report does not compute benchmark metrics.", "",
                 "- `frame_*.gt.png`: matched target RGB beside the original prediction",
                 "- `frame_*.coarse.png`: displayed coarse RGB",
                 "- `frame_*.comparison.png`: GT / coarse / prediction panel",
                 "- `comparison_overview_*.png`: per-observer panels, at most six frames per page", "",
                 "- [Recorded environment/world BEV](BEV_COMPARISON.md)", "",
                 "Predictions: `%s`" % predictions, "", "Queries: %d" % len(exported), ""]
        for overview in overviews:
            lines.append("- [%s / robot %d / page %d](%s)" % (overview["scene"], overview["robot_id"], overview["page"], overview["path"]))
        (output / "GT_COMPARISON.md").write_text("\n".join(lines) + "\n")
        print("GT comparison: %s (%d queries)" % (output, len(exported)), flush=True)
        return report
    finally:
        lock.unlink()


def export_bevs(predictions, output=None):
    """Export raw recorded BEV RGB, never an overlay or a generation condition."""
    predictions = Path(predictions).resolve()
    output = Path(output).resolve() if output is not None else predictions
    if (output / "bev_comparison.json").exists() or (output / "BEV_COMPARISON.md").exists():
        raise FileExistsError("Recorded BEV comparison already exists")
    metadata, rows, raw, records = prediction_context(predictions)
    episodes = sorted({row["episode_id"] for row in rows})
    destinations = []
    for episode in episodes:
        destinations.extend([within(output, str(Path(episode) / "gt_env_bev.png")),
                             within(output, str(Path(episode) / "bev_calibration.json"))])
    for row in rows:
        destinations.append(within(output, str(Path(row["path"]).with_suffix(".world_bev.png"))))
    for path in destinations:
        if path.exists():
            raise FileExistsError("Recorded BEV output already exists: %s" % path)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".bev_comparison.lock"
    lock.touch(exist_ok=False)
    try:
        episode_exports, queries = {}, []
        fields = ("world_bounds", "pixel_to_world", "world_to_pixel", "meters_per_pixel", "floor_z")
        for row in sorted(rows, key=lambda v: (v["episode_id"], v["robot_id"], v["frame_index"])):
            episode, frame = row["episode_id"], row["frame_index"]
            record = records[episode]
            world = raw.world_bev(record)
            if episode not in episode_exports:
                env = raw.bev(record)
                relative = Path(episode) / "gt_env_bev.png"
                destination = within(output, str(relative))
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(env["rgb"]).save(destination)
                env_source = str(raw.root / record["configuration_path"] / "bev/environment_base.npz")
                world_source = str(raw.episode_path(record) / "bev/world_before.npz")
                calibration = {"phase": "before", "floor_index": env["floor_index"],
                               "environment": {name: np.asarray(env[name]).tolist() for name in fields},
                               "world": {name: np.asarray(world[name]).tolist() for name in fields},
                               "environment_rgb_shape": list(env["rgb"].shape),
                               "world_rgb_shape": list(world["rgb"].shape),
                               "environment_source": env_source, "world_source": world_source,
                               "environment_rgb_key": "floor_%02d/rgb" % env["floor_index"]}
                calibration_path = Path(episode) / "bev_calibration.json"
                (output / calibration_path).write_text(json.dumps(calibration, indent=2) + "\n")
                episode_exports[episode] = {"scene": record["scene"], "gt_env_bev_path": str(relative),
                                            "calibration_path": str(calibration_path),
                                            "environment_source": env_source, "world_source": world_source}
            relative = Path(row["path"]).with_suffix(".world_bev.png")
            destination = within(output, str(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(world["rgb"][frame]).save(destination)
            queries.append({"episode_id": episode, "scene": record["scene"], "robot_id": row["robot_id"],
                            "frame_index": frame, "time_seconds": frame / record["fps"],
                            "prediction_path": row["path"], "world_bev_path": str(relative),
                            "gt_env_bev_path": episode_exports[episode]["gt_env_bev_path"]})
        report = {"query_count": len(queries), "phase": "before", "dataset": raw.provenance(),
                  "prediction_manifest_sha256": sha256_file(predictions / "predictions.jsonl"),
                  "ground_truth_loaded_for_generation": False, "ground_truth_loaded_for_comparison": True,
                  "image_policy": "Recorded native-resolution RGB; no resize, wall whitening or robot overlays",
                  "same_frame_policy": "Same physical-frame world BEV for all observers, copied beside each query",
                  "episodes": episode_exports, "queries": queries}
        (output / "bev_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# Recorded environment and world BEV", "",
                 "These are recorded before-phase GT images, exported after generation for inspection only.",
                 "Environment BEV excludes robots; world BEV includes robots at the selected physical frame.",
                 "Images keep native resolution and colors. Calibration is saved per episode.",
                 "Observers at the same physical frame share the same world BEV.", ""]
        for episode, info in episode_exports.items():
            lines.append("- [%s environment BEV](%s)" % (info["scene"], info["gt_env_bev_path"]))
        lines.append("")
        for query in queries:
            lines.append("- [%s / robot %d / frame %d world BEV](%s)" %
                         (query["scene"], query["robot_id"], query["frame_index"], query["world_bev_path"]))
        (output / "BEV_COMPARISON.md").write_text("\n".join(lines) + "\n")
        print("Recorded BEV comparison: %s (%d queries)" % (output, len(queries)), flush=True)
        return report
    finally:
        lock.unlink()
