import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.decision_engine import predicted_label_from_observation
from feature.history.historyClass import History
from feature.score_profile import (
    PROTOCOL_GROUPS,
    load_active_score_profile,
    protocol_group_from_edge_key,
    reset_active_score_profile,
)
from feature.scoring_pipeline import commit_window, score_window
from full_week_score_test import BIN_COUNT, ScoreStats, build_graph_with_labels
from test_no_label_leakage_wednesday import approximate_auc, metrics_at_threshold
from train.train_score_weights import load_trained_weights, reset_all_weights


DEFAULT_THRESHOLDS = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
SUB_SCORE_NAMES = (
    "local_anomaly_score",
    "current_behavior_anomaly_score",
    "finite_history_offset_anomaly_score",
    "approximate_novelty_anomaly_score",
    "behavior_role_anomaly_score",
    "auth_bruteforce_score",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--days", nargs="+", required=True)
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--score-profile-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def ordered_files(window_dir, days):
    day_order = {day: index for index, day in enumerate(days)}
    files = [
        path
        for path in window_dir.glob("*.csv")
        if path.name.split("-", 1)[0] in day_order
    ]
    return sorted(
        files,
        key=lambda path: (
            day_order[path.name.split("-", 1)[0]],
            path.name,
        ),
    )


def limit_files_evenly(files, limit):
    if not limit or limit <= 0 or len(files) <= limit:
        return files
    if limit == 1:
        return [files[0]]
    indexes = {
        round(index * (len(files) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [files[index] for index in sorted(indexes)]


def label_key(label):
    return (
        "attack"
        if str(label).strip() not in {"", "0", "BENIGN", "Normal", "normal"}
        else "normal"
    )


def log(path, message):
    print(message, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def stats_dict(stats):
    return stats.as_dict() if stats.count else {"n": 0}


def separation_row(name, normal_stats, attack_stats):
    normal = stats_dict(normal_stats)
    attack = stats_dict(attack_stats)
    normal_mean = normal.get("mean")
    attack_mean = attack.get("mean")
    delta = (
        attack_mean - normal_mean
        if normal_mean is not None and attack_mean is not None
        else None
    )
    return {
        "name": name,
        "normal": normal,
        "attack": attack,
        "delta_mean_attack_minus_normal": delta,
        "normal_p95": normal.get("p95"),
        "attack_p50": attack.get("p50"),
        "attack_p75": attack.get("p75"),
    }


def best_thresholds(score_stats):
    if score_stats["normal"].count <= 0 or score_stats["attack"].count <= 0:
        return {}
    best_f1 = None
    best_f1_under_5pct_fpr = None
    best_recall_under_5pct_fpr = None
    for index in range(BIN_COUNT + 1):
        threshold = index / BIN_COUNT
        metrics = metrics_at_threshold(score_stats, threshold)
        row = {"threshold": threshold, **metrics}
        if best_f1 is None or row["f1"] > best_f1["f1"]:
            best_f1 = row
        if row["normal_false_positive_rate"] <= 0.05:
            if (
                best_f1_under_5pct_fpr is None
                or row["f1"] > best_f1_under_5pct_fpr["f1"]
            ):
                best_f1_under_5pct_fpr = row
            if (
                best_recall_under_5pct_fpr is None
                or row["attack_recall"] > best_recall_under_5pct_fpr["attack_recall"]
            ):
                best_recall_under_5pct_fpr = row
    return {
        "diagnostic_best_f1": best_f1,
        "diagnostic_best_f1_with_fpr_le_0_05": best_f1_under_5pct_fpr,
        "diagnostic_best_recall_with_fpr_le_0_05": best_recall_under_5pct_fpr,
        "note": "Diagnostic only. Do not write these thresholds back to a model without a separate calibration/validation split.",
    }


def group_report(group_stats, top_k):
    score_stats = group_stats["scores"]
    sub_scores = []
    for name in sorted(group_stats["sub_scores"]):
        stats = group_stats["sub_scores"][name]
        sub_scores.append(separation_row(name, stats["normal"], stats["attack"]))
    features = []
    for name in sorted(group_stats["features"]):
        stats = group_stats["features"][name]
        features.append(separation_row(name, stats["normal"], stats["attack"]))
    features_with_delta = [
        row for row in features if row["delta_mean_attack_minus_normal"] is not None
    ]
    return {
        "counts": dict(group_stats["counts"]),
        "auc": approximate_auc(score_stats["normal"], score_stats["attack"]),
        "score_distribution": {
            "normal": stats_dict(score_stats["normal"]),
            "attack": stats_dict(score_stats["attack"]),
        },
        "threshold_metrics": {
            f"{threshold:.2f}": metrics_at_threshold(score_stats, threshold)
            for threshold in DEFAULT_THRESHOLDS
        },
        "diagnostic_best_thresholds": best_thresholds(score_stats),
        "sub_score_separation": sub_scores,
        "feature_separation": features,
        "top_attack_higher_features": sorted(
            features_with_delta,
            key=lambda row: row["delta_mean_attack_minus_normal"],
            reverse=True,
        )[:top_k],
        "top_normal_higher_features": sorted(
            features_with_delta,
            key=lambda row: row["delta_mean_attack_minus_normal"],
        )[:top_k],
    }


def new_group_stats():
    return {
        "counts": Counter(),
        "scores": {"normal": ScoreStats(), "attack": ScoreStats()},
        "sub_scores": defaultdict(lambda: {"normal": ScoreStats(), "attack": ScoreStats()}),
        "features": defaultdict(lambda: {"normal": ScoreStats(), "attack": ScoreStats()}),
    }


def main():
    args = parse_args()
    files = limit_files_evenly(ordered_files(args.window_dir, args.days), args.max_windows)
    if not files:
        raise FileNotFoundError(f"no windows found in {args.window_dir} for {args.days}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_path:
        args.progress_path.parent.mkdir(parents=True, exist_ok=True)
        args.progress_path.write_text("", encoding="utf-8")

    reset_all_weights()
    reset_active_score_profile()
    loaded_weights = load_trained_weights(args.weights_path)
    loaded_score_profile = None
    if args.score_profile_path is not None:
        loaded_score_profile = load_active_score_profile(args.score_profile_path)

    history = History(life_windows=30, detail_windows=5)
    groups = {group: new_group_stats() for group in ("ALL", *PROTOCOL_GROUPS)}
    traffic = Counter()
    started_at = time.perf_counter()
    log(
        args.progress_path,
        f"start windows={len(files)} days={','.join(args.days)} window_dir={args.window_dir}",
    )

    for index, path in enumerate(files, start=1):
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        scores, observations = score_window(graph, history)
        predicted = {
            edge_key: predicted_label_from_observation(observations[edge_key])
            for edge_key in scores
        }
        commit_window(
            history,
            graph,
            predicted,
            suspicious_observations=observations,
        )

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

        for edge_key, score in scores.items():
            label = label_key(true_edge_labels.get(edge_key, "normal"))
            group = protocol_group_from_edge_key(edge_key)
            observation = observations[edge_key]
            sub_scores = observation.get("sub_scores", {}) or {}
            feature_vector = observation.get("feature_vector", {}) or {}
            for target_group in ("ALL", group):
                group_stats = groups[target_group]
                group_stats["counts"][label] += 1
                group_stats["scores"][label].add(score)
                for name in SUB_SCORE_NAMES:
                    group_stats["sub_scores"][name][label].add(
                        sub_scores.get(name, 0.0)
                    )
                for name, value in feature_vector.items():
                    if isinstance(value, bool):
                        value = 1.0 if value else 0.0
                    if isinstance(value, (int, float)):
                        group_stats["features"][name][label].add(value)

        if index % args.progress_every == 0 or index == len(files):
            log(
                args.progress_path,
                (
                    f"progress={index}/{len(files)} file={path.name} "
                    f"elapsed_seconds={time.perf_counter() - started_at:.1f}"
                ),
            )

    result = {
        "completed": True,
        "method": "Score and feature separation diagnostic only; no training or profile update is performed.",
        "anti_overfit_note": (
            "Use this report to identify failure modes. Any threshold or feature "
            "change suggested by this report must be validated on a separate split "
            "or another dataset before being treated as a general method."
        ),
        "window_dir": str(args.window_dir),
        "days": args.days,
        "window_count": len(files),
        "weights_path": str(args.weights_path),
        "score_profile_path": str(args.score_profile_path) if args.score_profile_path else None,
        "loaded_weights_keys": sorted(loaded_weights.keys()) if isinstance(loaded_weights, dict) else [],
        "loaded_score_profile_kind": (
            loaded_score_profile.get("kind") if loaded_score_profile else None
        ),
        "elapsed_seconds": time.perf_counter() - started_at,
        "traffic": dict(traffic),
        "groups": {
            group: group_report(groups[group], args.top_k)
            for group in ("ALL", *PROTOCOL_GROUPS)
        },
    }
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(args.progress_path, f"done result={args.output_path}")


if __name__ == "__main__":
    main()
