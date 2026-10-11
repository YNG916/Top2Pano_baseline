"""Compare saved predictions against GT after inference, without loading a model."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from adapters.mvwd_raw import MVWDRaw, sha256_file
from mvwd_runtime import load_config
from scripts.audit_upstream_weights import Archive, compare_prefix


def load_predictions(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines() if line]
    keyed = {(r["episode_id"], r["robot_id"], r["frame_index"]): r for r in rows}
    if len(keyed) != len(rows):
        raise ValueError("Duplicate prediction queries")
    return metadata, keyed


def psnr(a, b):
    mse = float(np.square(a.astype(np.float64) - b.astype(np.float64)).mean())
    return -10 * math.log10(max(mse, 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prediction-a", required=True)
    parser.add_argument("--prediction-b", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    directories = [Path(v).resolve() for v in (args.prediction_a, args.prediction_b)]
    (ma, a), (mb, b) = [load_predictions(d) for d in directories]
    if not a or a.keys() != b.keys():
        raise ValueError("Compare exactly the same nonempty query set")
    for metadata in (ma, mb):
        if metadata["config"]["data"]["root"] != config["data"]["root"]:
            raise ValueError("Prediction and review dataset roots differ")
    if ma["dataset"]["split"] != mb["dataset"]["split"]:
        raise ValueError("Prediction splits differ")
    raw = MVWDRaw(config["data"]["root"], ma["dataset"]["split"])
    if any(m["dataset"]["release_id"] != raw.metadata["release_id"] for m in (ma, mb)):
        raise ValueError("Dataset release differs")
    records = {r["id"]: r for r in raw.records}
    output = Path(args.output).resolve()
    if output.parent.name != "evaluations":
        raise ValueError("Use an evaluations/step output directory")
    report_dir = output.parent.parent / "reports" / output.name
    if output.exists() or report_dir.exists():
        raise FileExistsError("Choose a new review step name")
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = []
    panels = {}
    for key in sorted(a):
        episode, robot, frame = key
        if a[key]["seed"] != b[key]["seed"]:
            raise ValueError("Query seeds differ")
        arrays = []
        packets = []
        for directory, manifest in zip(directories, (a, b)):
            path = directory / manifest[key]["path"]
            arrays.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255)
            with np.load(path.with_suffix(".coarse.npz"), allow_pickle=False) as z:
                packets.append({k: z[k] for k in z.files})
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("Prediction resolutions differ")
        h, w = arrays[0].shape[:2]
        target = raw.views(records[episode], robot, ("rgb",), config["data"].get("cache_root"))["rgb"][frame]
        target = np.asarray(Image.fromarray(target).resize((w,h), Image.Resampling.BOX), dtype=np.float32) / 255
        coarses = [p["rgb"].clip(0,1) for p in packets]
        if any(c.shape != target.shape or not np.isfinite(c).all() for c in coarses):
            raise ValueError("Invalid coarse RGB")
        counts = [int(((p["robot_id"] >= 0) & (p["robot_id"] != robot) & (p["robot_weight"] > .1)).sum()) for p in packets]
        metrics.append({"episode_id":episode, "scene":records[episode]["scene"], "robot_id":robot,
                        "frame_index":frame,"seed":a[key]["seed"],
                        "coarse_a_psnr_db":psnr(coarses[0], target), "coarse_b_psnr_db":psnr(coarses[1], target),
                        "final_a_psnr_db":psnr(arrays[0], target), "final_b_psnr_db":psnr(arrays[1], target),
                        "coarse_pair_max_abs_delta":float(np.max(np.abs(coarses[0]-coarses[1]))),
                        "final_pair_mean_abs_delta":float(np.abs(arrays[0]-arrays[1]).mean()),
                        "predicted_other_robot_pixels_a":counts[0], "predicted_other_robot_pixels_b":counts[1]})
        panels.setdefault((episode,robot), []).append((frame,[target,coarses[0],arrays[0],coarses[1],arrays[1]]))
    archives = [Archive(p) for p in (args.checkpoint_a, args.checkpoint_b)]
    initial = Archive(config["model"]["init_checkpoint"])
    audit = []
    for path, arc in zip((args.checkpoint_a,args.checkpoint_b),archives):
        audit.append({"checkpoint":str(Path(path).resolve()),"global_step":arc.metadata.get("global_step"),
                      "density_vs_init":compare_prefix(arc,initial,"density_model."),
                      "render_vs_init":compare_prefix(arc,initial,"render_model.")})
    labels = ["GT (evaluation only)", "Coarse A", "Final A (%s steps)" % audit[0]["global_step"],
              "Coarse B", "Final B (%s steps)" % audit[1]["global_step"]]
    for (episode,robot), samples in panels.items():
        fig, axes = plt.subplots(len(samples),5,figsize=(17,3.2*len(samples)),squeeze=False)
        for row,(frame,images) in enumerate(samples):
            for col,img in enumerate(images):
                axes[row,col].imshow(img);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if row == 0:axes[row,col].set_title(labels[col],fontsize=11)
                if col == 0:axes[row,col].set_ylabel("robot %d / frame %d" % (robot,frame))
        fig.suptitle("%s | %s | identical queries and seeds" % (records[episode]["scene"],episode[:16]),fontsize=12)
        fig.tight_layout()
        fig.savefig(report_dir / ("%s_robot_%02d.png" % (episode,robot)),dpi=120)
        plt.close(fig)
    names = ("coarse_a_psnr_db","coarse_b_psnr_db","final_a_psnr_db","final_b_psnr_db")
    summary = {"query_count":len(metrics),"scene_count":len({m["scene"] for m in metrics}),
               "episode_count":len({m["episode_id"] for m in metrics}),
               "mean_per_query_psnr_db":{k:float(np.mean([m[k] for m in metrics])) for k in names},
               "max_coarse_pair_delta":max(m["coarse_pair_max_abs_delta"] for m in metrics),
               "queries_with_predicted_other_robots":sum(m["predicted_other_robot_pixels_a"]>0 for m in metrics)}
    report = {"summary":summary,"queries":metrics,"checkpoints":audit,"dataset":raw.provenance(),
              "prediction_manifests":[{"path":str(d),"manifest_sha256":sha256_file(d/"predictions.jsonl"),
                                      "metadata_sha256":sha256_file(d/"metadata.json")} for d in directories],
              "figures":str(report_dir),"ground_truth_loaded_for_evaluation_only":True,
              "limitations":["One selected preview subset; not full validation or test performance.",
                             "PSNR does not establish robot correctness or temporal/multiview consistency.",
                             "Coarse robot pixels are predicted visibility, not GT segmentation.",
                             "Checkpoint comparisons use full-storage CRC32 screening, not cryptographic tensor equality."]}
    (output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    (report_dir/"README.md").write_text("# Saved prediction comparison\n\nColumns: GT, coarse A, final A, coarse B, final B.\n\n"+json.dumps(summary,indent=2)+"\n\nGT is read only after inference, for this evaluation. See ../../evaluations/"+output.name+"/metrics.json for provenance and limitations.\n")
    print(json.dumps(summary,indent=2),flush=True)
    print("Comparison figures: " + str(report_dir),flush=True)


if __name__ == "__main__":
    main()
