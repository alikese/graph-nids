import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.decision_engine import predicted_label_from_observation
from feature.history.historyClass import History
from feature.score_profile import load_active_score_profile, reset_active_score_profile
from feature.scoring_pipeline import commit_window, score_window
from full_week_score_test import ScoreStats, build_graph_with_labels
from test_full_week_integrated import (
    add_lifecycle_pending,
    flush_lifecycle_pending,
    lifecycle_key,
    pending_counts,
    resolved_label_metrics,
)
from test_no_label_leakage_wednesday import (
    approximate_auc,
    final_label_metrics,
    metrics_at_threshold,
    timing_summary,
)
from train.train_score_weights import load_trained_weights, reset_all_weights


THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-dir",
        type=Path,
        default=ROOT / "windows divide nb15",
    )
    parser.add_argument("--day", default="Monday")
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=ROOT / "train" / "nb15_trained_score_weights.json",
    )
    parser.add_argument("--score-profile-path", type=Path)
    parser.add_argument(
        "--result-path",
        type=Path,
        default=ROOT / "train" / "nb15_lifecycle_backfill_result.json",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=ROOT / "train" / "nb15_lifecycle_backfill_progress.log",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--attack-threshold", type=float)
    parser.add_argument("--suspicious-threshold", type=float)
    parser.add_argument("--strict-suspicious-promotion", action="store_true")
    parser.add_argument("--score-only-direct-attack", action="store_true")
    return parser.parse_args()


def add_prediction_counts(counter, true_label, predicted_label):
    counter[f"true_{true_label}"] += 1
    counter[f"predicted_{predicted_label}"] += 1
    counter[f"{true_label}_as_{predicted_label}"] += 1


def calculate_f1(metrics):
    precision = metrics["attack_precision"]
    recall = metrics["attack_recall"]
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def log_progress(path, message):
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def ordered_files(window_dir, day):
    return sorted(window_dir.glob(f"{day}-ip_*.csv"), key=lambda path: path.name)


def select_contiguous_files(files, start_index, max_windows):
    start_index = max(int(start_index), 0)
    if start_index >= len(files):
        return []
    if not max_windows or max_windows <= 0:
        return files[start_index:]
    return files[start_index : start_index + int(max_windows)]


def threshold_label(score, attack_threshold, suspicious_threshold):
    if score >= attack_threshold:
        return "attack"
    if score >= suspicious_threshold:
        return "suspicious"
    return "normal"


def score_result(score_stats):
    return {
        "approximate_auc": approximate_auc(
            score_stats["normal"],
            score_stats["attack"],
        ),
        "score_distribution": {
            label: stats.as_dict()
            for label, stats in score_stats.items()
        },
        "threshold_metrics": {
            f"{threshold:.2f}": metrics_at_threshold(score_stats, threshold)
            for threshold in THRESHOLDS
        },
    }


def main():
    args = parse_args()
    files = select_contiguous_files(
        ordered_files(args.window_dir, args.day),
        args.window_start,
        args.max_windows,
    )
    if not files:
        raise FileNotFoundError(
            f"no {args.day} windows found in {args.window_dir}"
        )

    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.write_text("", encoding="utf-8")

    reset_all_weights()
    reset_active_score_profile()
    loaded_weights = load_trained_weights(args.weights_path)
    loaded_score_profile = None
    if args.score_profile_path is not None:
        loaded_score_profile = load_active_score_profile(args.score_profile_path)

    history = History(life_windows=30, detail_windows=5)
    if args.attack_threshold is not None:
        history.suspicious_edge_history.theta_attack = float(args.attack_threshold)
    if args.suspicious_threshold is not None:
        history.suspicious_edge_history.theta_suspicious = float(
            args.suspicious_threshold
        )
    if args.attack_threshold is not None and args.suspicious_threshold is not None:
        history.suspicious_edge_history.promotion_evidence_threshold = (
            float(args.attack_threshold) + float(args.suspicious_threshold)
        ) / 2.0
    history.suspicious_edge_history.strict_suspicious_promotion = bool(
        args.strict_suspicious_promotion
    )
    history.suspicious_edge_history.score_only_attack_threshold = bool(
        args.score_only_direct_attack
    )
    predicted_counts = Counter()
    lifecycle_final_counts = Counter()
    lifecycle_pending = {}
    transition_counts = Counter()
    score_stats = {"normal": ScoreStats(), "attack": ScoreStats()}
    timings = {}
    traffic = Counter()
    started_at = time.perf_counter()

    log_progress(
        args.progress_path,
        (
            f"start day={args.day} windows={len(files)} "
            f"window_dir={args.window_dir} weights={args.weights_path}"
        ),
    )

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
        if args.attack_threshold is not None and args.suspicious_threshold is not None:
            threshold_labels = {
                edge_key: threshold_label(
                    score,
                    args.attack_threshold,
                    args.suspicious_threshold,
                )
                for edge_key, score in scores.items()
            }
        else:
            threshold_labels = {
                edge_key: predicted_label_from_observation(observations[edge_key])
                for edge_key in scores
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
            add_prediction_counts(predicted_counts, true_label, predicted_label)
            if predicted_label == "suspicious":
                add_lifecycle_pending(
                    lifecycle_pending,
                    lifecycle_key(history, edge_key),
                    args.day,
                    true_label,
                )
            else:
                add_prediction_counts(
                    lifecycle_final_counts,
                    true_label,
                    predicted_label,
                )
            score_stats[true_label].add(score)

        flushed_release_count = 0
        for edge_key in suspicious_result.released_normal_edges:
            flushed_release_count += flush_lifecycle_pending(
                lifecycle_pending,
                lifecycle_key(history, edge_key),
                "normal",
                lifecycle_final_counts,
                {args.day: Counter()},
            )
        flushed_attack_count = 0
        for edge_key in suspicious_result.promoted_attack_edges:
            flushed_attack_count += flush_lifecycle_pending(
                lifecycle_pending,
                lifecycle_key(history, edge_key),
                "attack",
                lifecycle_final_counts,
                {args.day: Counter()},
            )
        if flushed_release_count:
            transition_counts["lifecycle_backfilled_normal"] += flushed_release_count
        if flushed_attack_count:
            transition_counts["lifecycle_backfilled_attack"] += flushed_attack_count

        evaluation_seconds = time.perf_counter() - stage_started_at
        window_seconds = time.perf_counter() - window_started_at
        for name, value in {
            "build_graph": build_seconds,
            **score_timings,
            "commit_history": commit_seconds,
            "evaluation_only": evaluation_seconds,
            "window_total": window_seconds,
        }.items():
            timings.setdefault(name, []).append(value)

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

        if index % args.progress_every == 0 or index == len(files):
            log_progress(
                args.progress_path,
                (
                    f"progress={index}/{len(files)} file={path.name} "
                    f"packets={len(graph.packets)} edges={len(graph.edges)} "
                    f"window_seconds={window_seconds:.3f} "
                    f"elapsed_seconds={time.perf_counter() - started_at:.1f}"
                ),
            )

    elapsed_seconds = time.perf_counter() - started_at
    final_metrics = final_label_metrics(predicted_counts)
    final_metrics["f1"] = calculate_f1(final_metrics)
    lifecycle_metrics = resolved_label_metrics(lifecycle_final_counts)
    result = {
        "completed": True,
        "day": args.day,
        "window_dir": str(args.window_dir),
        "window_count": len(files),
        "window_start": args.window_start,
        "max_windows": args.max_windows,
        "elapsed_seconds": elapsed_seconds,
        "weights_path": str(args.weights_path),
        "attack_threshold_override": args.attack_threshold,
        "suspicious_threshold_override": args.suspicious_threshold,
        "strict_suspicious_promotion": bool(args.strict_suspicious_promotion),
        "score_only_direct_attack": bool(args.score_only_direct_attack),
        "score_profile_path": (
            str(args.score_profile_path) if args.score_profile_path else None
        ),
        "loaded_weights": loaded_weights,
        "loaded_score_profile_kind": (
            loaded_score_profile.get("kind") if loaded_score_profile else None
        ),
        "predicted_counts": dict(predicted_counts),
        "final_label_metrics_without_backfill": final_metrics,
        "lifecycle_final": {
            "predicted_counts": dict(lifecycle_final_counts),
            "final_label_metrics": lifecycle_metrics,
            "pending_counts": dict(pending_counts(lifecycle_pending)),
        },
        "suspicious_transitions": dict(transition_counts),
        **score_result(score_stats),
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
        "note": (
            "without_backfill counts the current window label. lifecycle_final "
            "caches suspicious observations and backfills them when later "
            "released as normal or promoted as attack; unresolved observations "
            "remain pending and are excluded from lifecycle F1."
        ),
    }
    args.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_progress(
        args.progress_path,
        f"done result={args.result_path} elapsed_seconds={elapsed_seconds:.1f}",
    )


if __name__ == "__main__":
    main()
