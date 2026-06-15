import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from feature.history.historyClass import History
from full_week_score_test import (
    ScoreStats,
    build_graph_with_labels,
    commit_window,
    predicted_label_from_score,
)
from test_no_label_leakage_wednesday import (
    approximate_auc,
    final_label_metrics,
    metrics_at_threshold,
    score_window,
    timing_summary,
)


THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default="Tuesday")
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    parser.add_argument("--packet-cap", type=int, required=True)
    return parser.parse_args()


def add_prediction_counts(counter, true_label, predicted_label):
    counter[f"true_{true_label}"] += 1
    counter[f"predicted_{predicted_label}"] += 1
    counter[f"{true_label}_as_{predicted_label}"] += 1


def calculate_f1(metrics):
    precision = metrics["attack_precision"]
    recall = metrics["attack_recall"]
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def log_progress(path, message):
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def cutoff_reason(path):
    reason = path.stem.rsplit("_", 1)[-1]
    return reason if reason in {"time", "cap", "eof"} else "unknown"


def allowed_processing_seconds(path, graph, window_seconds=10.0):
    reason = cutoff_reason(path)
    if reason == "time":
        return window_seconds
    if not graph.packets:
        return 0.0
    return max(
        float(graph.packets[-1].timestamp)
        - float(graph.packets[0].timestamp),
        0.0,
    )


def main():
    args = parse_args()
    files = sorted(args.window_dir.glob(f"{args.day}-ip_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no {args.day} windows found in {args.window_dir}"
        )

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.write_text("", encoding="utf-8")

    history = History(life_windows=30, detail_windows=5)
    predicted_counts = Counter()
    score_stats = {
        "normal": ScoreStats(),
        "attack": ScoreStats(),
    }
    reason_counts = Counter()
    timings = defaultdict(list)
    traffic = Counter()
    cutoff_counts = Counter()
    deadline_seconds = []
    realtime_violations = []
    started_at = time.perf_counter()

    for index, path in enumerate(files, start=1):
        window_started_at = time.perf_counter()
        stage_started_at = time.perf_counter()
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        build_seconds = time.perf_counter() - stage_started_at

        scores, observations, score_timings = score_window(
            graph,
            history,
            return_timing=True,
        )
        threshold_labels = {
            edge_key: predicted_label_from_score(score)
            for edge_key, score in scores.items()
        }

        stage_started_at = time.perf_counter()
        suspicious_result, final_labels = commit_window(
            history,
            graph,
            threshold_labels,
            suspicious_observations=observations,
        )
        commit_seconds = time.perf_counter() - stage_started_at

        stage_started_at = time.perf_counter()
        for edge_key, score in scores.items():
            true_label = true_edge_labels.get(edge_key, "normal")
            predicted_label = final_labels[edge_key]
            add_prediction_counts(
                predicted_counts,
                true_label,
                predicted_label,
            )
            score_stats[true_label].add(score)
        for decision in suspicious_result.decisions.values():
            reason_counts[decision.reason] += 1
        evaluation_seconds = time.perf_counter() - stage_started_at

        window_seconds = time.perf_counter() - window_started_at
        reason = cutoff_reason(path)
        allowed_seconds = allowed_processing_seconds(path, graph)
        cutoff_counts[reason] += 1
        deadline_seconds.append(allowed_seconds)
        if window_seconds > allowed_seconds:
            realtime_violations.append(
                {
                    "file": path.name,
                    "cutoff_reason": reason,
                    "packet_count": len(graph.packets),
                    "edge_count": len(graph.edges),
                    "allowed_seconds": allowed_seconds,
                    "processing_seconds": window_seconds,
                    "overrun_seconds": window_seconds - allowed_seconds,
                    "processing_to_allowed_ratio": (
                        window_seconds / allowed_seconds
                        if allowed_seconds > 0.0
                        else None
                    ),
                }
            )
        window_timings = {
            "build_graph": build_seconds,
            **score_timings,
            "commit_history": commit_seconds,
            "evaluation_only": evaluation_seconds,
            "pipeline_without_evaluation": (
                build_seconds
                + score_timings["score_total"]
                + commit_seconds
            ),
            "window_total": window_seconds,
        }
        for name, value in window_timings.items():
            timings[name].append(value)

        traffic["window_count"] += 1
        traffic["packet_count"] += len(graph.packets)
        traffic["edge_count"] += len(graph.edges)
        traffic["max_packets_per_window"] = max(
            traffic["max_packets_per_window"],
            len(graph.packets),
        )
        traffic["max_edges_per_window"] = max(
            traffic["max_edges_per_window"],
            len(graph.edges),
        )

        if index % 100 == 0 or index == len(files):
            log_progress(
                args.progress_path,
                f"progress index={index}/{len(files)} "
                f"file={path.name} packets={len(graph.packets)} "
                f"edges={len(graph.edges)} "
                f"window_seconds={window_seconds:.3f} "
                f"elapsed_seconds={time.perf_counter() - started_at:.1f}",
            )

    elapsed_seconds = time.perf_counter() - started_at
    label_metrics = final_label_metrics(predicted_counts)
    label_metrics["f1"] = calculate_f1(label_metrics)
    result = {
        "completed": True,
        "test_day": args.day,
        "packet_cap": args.packet_cap,
        "window_directory": str(args.window_dir),
        "window_count": len(files),
        "elapsed_seconds": elapsed_seconds,
        "predicted_counts": dict(predicted_counts),
        "final_label_metrics": label_metrics,
        "approximate_auc": approximate_auc(
            score_stats["normal"],
            score_stats["attack"],
        ),
        "score_distribution": {
            label: stats.as_dict()
            for label, stats in score_stats.items()
        },
        "threshold_metrics": {
            f"{threshold:.2f}": metrics_at_threshold(
                score_stats,
                threshold,
            )
            for threshold in THRESHOLDS
        },
        "timing": {
            name: timing_summary(values)
            for name, values in timings.items()
        },
        "traffic": {
            **dict(traffic),
            "mean_packets_per_window": (
                traffic["packet_count"] / traffic["window_count"]
            ),
            "mean_edges_per_window": (
                traffic["edge_count"] / traffic["window_count"]
            ),
        },
        "realtime_deadline": {
            "rule": (
                "time截停窗口允许10秒；cap/eof截停窗口允许时间为"
                "最后进入流量时间戳减第一条进入流量时间戳。"
            ),
            "cutoff_reason_counts": dict(cutoff_counts),
            "allowed_seconds": timing_summary(deadline_seconds),
            "violation_count": len(realtime_violations),
            "violation_rate": (
                len(realtime_violations) / len(files)
            ),
            "top_overruns": sorted(
                realtime_violations,
                key=lambda item: item["overrun_seconds"],
                reverse=True,
            )[:20],
        },
        "suspicious_decision_reasons": dict(reason_counts),
        "parameters": {
            "theta_suspicious": (
                history.suspicious_edge_history.theta_suspicious
            ),
            "theta_attack": history.suspicious_edge_history.theta_attack,
            "ttl_windows": history.suspicious_edge_history.ttl_windows,
            "attack_chain_threshold": (
                history.suspicious_edge_history.attack_chain_threshold
            ),
        },
        "evaluation_note": (
            f"测试从空 History 开始，使用 {args.day} 原始数据、"
            "当前算法和10秒边界；实验变量是单窗口最大流量数。"
        ),
    }
    args.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_progress(
        args.progress_path,
        f"done windows={len(files)} elapsed_seconds={elapsed_seconds:.1f}",
    )


if __name__ == "__main__":
    main()
