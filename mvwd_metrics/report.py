"""Readable companion to saved metric results; no model inference or GT loading."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path


LABELS = {
    "psnr": ("PSNR", "↑"),
    "lpips": ("LPIPS", "↓"),
    "fid": ("FID", "↓"),
    "mv_gtwarp_dino": ("MV-GTWarp-DINO", "↓"),
    "met3r": ("MEt3R", "↓"),
}


def cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def number(value):
    if value is None:
        return "未定义"
    value = float(value)
    if math.isinf(value):
        return "+∞" if value > 0 else "−∞"
    if math.isnan(value):
        raise ValueError("Cannot report NaN as a metric")
    return "%.4f" % value


def mean(values):
    return math.fsum(float(v) for v in values) / len(values)


def completed_time(value):
    if not value:
        return "未记录"
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo
    except ImportError:  # Historical training env is Python3.8; metrics env is 3.10.
        return date.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return date.astimezone(ZoneInfo("Europe/Riga")).strftime("%Y-%m-%d %H:%M:%S") + "（Europe/Riga）"


def scene_mapping(directory):
    path = Path(directory) / "per_query.jsonl"
    scenes = {}
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                episode, scene = row.get("episode_id"), row.get("scene")
                if episode is None or scene is None:
                    continue
                if str(episode) in scenes and scenes[str(episode)] != scene:
                    raise ValueError("Conflicting scene labels in saved evaluation")
                scenes[str(episode)] = scene
    return scenes


def render_metrics_markdown(result, scenes=None):
    """Render stored means/CI, with derived seed/scene episode-macro summaries."""
    metrics = result["metrics"]
    coverage, protocol = result.get("coverage", {}), result.get("protocol", {})
    pairs = result.get("pair_selection", {})
    seeds = result.get("seeds", coverage.get("observed_seeds", []))
    title = "完整评估" if result.get("mode") == "benchmark" else "预览 / 子集诊断"
    lines = ["# Level 1 指标记录", "", "**%s**" % title, ""]
    if result.get("benchmark_ready") is True:
        lines += ["已通过本评估器的完整覆盖与指标有效性检查。该标记不代表生成质量达标或模型完全收敛。", ""]
    else:
        lines += ["**本记录不满足完整有效 benchmark 的条件；请查看覆盖范围、未定义指标和备注。**", ""]
    query_count = coverage.get("queries_per_seed")
    lines += [
        "- 评估完成：%s。" % completed_time(result.get("completed_utc")),
        "- 数据 split：%s；episodes：%s / %s。" % (
            cell(coverage.get("split", "未记录")), coverage.get("episode_count", "—"),
            coverage.get("official_split_episode_count", "—")),
        "- 生成 seeds：%s；每 seed %s 张；总计 %s 张。" % (
            cell(", ".join(map(str, seeds))), query_count if query_count is not None else "—",
            query_count * len(seeds) if query_count is not None else "—"),
        "- 分辨率 H×W：%s；每 episode 固定抽帧数：%s；robot IDs：%s。" % (
            "×".join(map(str, protocol.get("target_hw", []))) or "—",
            protocol.get("frames_per_episode", "—"), cell(protocol.get("robots", "—"))),
        "- 缺失 / 额外 protocol queries：%s / %s。" % (
            coverage.get("missing_protocol_queries", "—"), coverage.get("extra_protocol_queries", "—")),
        "",
    ]
    if result.get("mode") != "benchmark":
        lines += ["以上固定抽帧数是协议目标；本次 preview 的实际帧数以保存的 query 为准。", ""]
    lines += ["## 总体指标", "", "| 指标 | 方向 | 均值 | 95% episode bootstrap CI | 样本量 | 状态 |",
              "| --- | :---: | ---: | --- | --- | --- |"]
    for name, metric in metrics.items():
        label, direction = LABELS.get(name, (name, "—"))
        interval = metric.get("ci95")
        ci = "[%s, %s]" % tuple(map(number, interval)) if interval is not None else "—"
        support = ("%s 张/seed" % metric.get("images_per_seed", "—") if name == "fid"
                   else "%s episodes" % metric.get("episode_count", "—"))
        status = metric.get("status", "ok" if metric.get("mean") is not None else "undefined")
        lines.append("| %s | %s | %s | %s | %s | %s |" % (
            cell(label), direction, number(metric.get("mean")), ci, support, cell(status)))
        if metric.get("failed_pair_count"):
            lines.append("\nMEt3R 重建失败：%s / %s 对；失败比例 %s。\n" % (
                metric["failed_pair_count"], metric.get("evaluated_pair_count", "—"),
                number(metric.get("failure_fraction"))))
    # Keep any failure paragraph outside the table.
    failures = [line for line in lines if line.startswith("\nMEt3R")]
    lines = [line for line in lines if not line.startswith("\nMEt3R")]
    lines += ["", *failures,
              "CI 按 episode bootstrap；仅一个 episode 时不估计该 CI。FID 按每 seed 全图像集合计算，再对 seeds 取均值，不计算 episode CI。", ""]
    pair_metrics = [name for name in ("mv_gtwarp_dino", "met3r") if name in metrics]
    if pair_metrics:
        lines += ["## 跨视角覆盖", ""]
        for name in pair_metrics:
            counts = pairs.get("per_metric", {}).get(name, pairs)
            lines += ["- %s：每 seed %s 个 GT 合格视角对；有合格对的 episodes 为 %s；无合格对的 episodes 为 %s。" % (
                LABELS[name][0], counts.get("eligible_pairs", "—"),
                counts.get("episodes_with_eligible_pairs", "—"), counts.get("episodes_without_eligible_pairs", "—"))]
        lines += ["", "无 GT 合格重叠的 episode 不填零；这与 MEt3R 对合格视角对发生重建失败是两种情况。", ""]
    if seeds:
        lines += ["## 各 seed", "", "| Seed | " + " | ".join(LABELS.get(n, (n, ""))[0] for n in metrics) + " |",
                  "| --- | " + " | ".join("---:" for _ in metrics) + " |"]
        for seed in seeds:
            values = []
            for name, metric in metrics.items():
                value = metric.get("seed_scores", {}).get(str(seed))
                # An undefined aggregate must not become a successful-subset seed score.
                if value is None and metric.get("mean") is not None:
                    episode_values = [e["seed_means"][str(seed)] for e in metric.get("episodes", [])
                                      if str(seed) in e.get("seed_means", {})]
                    value = mean(episode_values) if episode_values else None
                values.append(number(value))
            lines.append("| %s | %s |" % (cell(seed), " | ".join(values)))
        lines.append("")
    scenes = scenes or {}
    paired = [n for n, m in metrics.items() if m.get("episodes")]
    if scenes and paired:
        grouped = defaultdict(lambda: defaultdict(list))
        unmapped = set()
        for name in paired:
            for episode in metrics[name]["episodes"]:
                eid = str(episode["episode_id"])
                if eid not in scenes:
                    unmapped.add(eid)
                    continue
                grouped[scenes[eid]][name].append(episode["mean"])
        counts = Counter(scenes.values())
        lines += ["## 各场景", "", "表中 n 为该项指标参与平均的 episode 数；场景值按 episode 等权汇总。FID 保持全局计算。", "",
                  "| 场景 | Episodes | " + " | ".join(LABELS.get(n, (n, ""))[0] + " (n)" for n in paired) + " |",
                  "| --- | ---: | " + " | ".join("---:" for _ in paired) + " |"]
        for scene in sorted(counts):
            values = ["%s (%d)" % (number(mean(grouped[scene][n])), len(grouped[scene][n]))
                      if grouped[scene][n] else "未定义 (0)" for n in paired]
            lines.append("| %s | %d | %s |" % (cell(scene), counts[scene], " | ".join(values)))
        lines += [""]
        if unmapped:
            lines += ["场景明细缺少 %d 个 episode 的场景标签，上表不是完整场景汇总。" % len(unmapped), ""]
    reference = result.get("gtwarp_gt_reference")
    if reference is not None:
        lines += ["GT-Warp 的 GT 自身对照：%s，仅用于诊断，不从生成指标中扣除。" % number(reference.get("mean")), ""]
    lines += ["## 解释与备注", ""]
    notes = list(result.get("notes", []))
    if "fid" in metrics:
        notes.append("FID 必须在相同样本量、数据和预处理下比较，不与少量 preview FID 直接比较。")
    if "mv_gtwarp_dino" in metrics:
        notes.append("MV-GTWarp-DINO 是本基准定义的 GT 几何约束特征误差，不是官方 MEt3R；平滑/常量图可能低分，必须结合图像质量指标。")
    lines += ["- " + cell(note) for note in notes] or ["- 无额外备注。"]
    lines += ["", "## 来源", "", "- 原始完整数值：[metrics.json](metrics.json)。",
              "- 协议副本：[protocol.yaml](protocol.yaml)。",
              "- 逐图记录：[per_query.jsonl](per_query.jsonl)。",
              "- 逐视角对记录：[per_pair.jsonl](per_pair.jsonl)。"]
    for item in result.get("inputs", []):
        lines.append("- 输入 seed %s：%s；checkpoint SHA256：%s。" % (
            cell(item.get("base_seed", "—")), cell(item.get("directory", "—")),
            cell(item.get("checkpoint_sha256", "—"))))
    lines += ["", "本文件由保存的评估结果自动整理；原始 JSON、预测与指标定义不变。", ""]
    return "\n".join(lines)


def write_metrics_markdown(directory, result=None):
    directory = Path(directory)
    if result is None:
        result = json.loads((directory / "metrics.json").read_text())
    text = render_metrics_markdown(result, scene_mapping(directory))
    destination = directory / "METRICS.md"
    with destination.open("x") as stream:
        stream.write(text)
    return destination
