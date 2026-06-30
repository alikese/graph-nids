import argparse
import json
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
from full_week_score_test import ScoreStats, build_graph_with_labels
from test_no_label_leakage_wednesday import approximate_auc, final_label_metrics
from train.train_score_weights import load_trained_weights, reset_all_weights


TOP_LEVEL_FEATURES = (
    "current_behavior_anomaly_score",
    "finite_history_offset_anomaly_score",
    "approximate_novelty_anomaly_score",
    "behavior_role_anomaly_score",
)
INTERNAL_GROUPS = {
    "current_behavior_anomaly_score": (
        "small_packet_ratio",
        "zero_payload_ratio",
        "syn_without_ack_ratio",
        "handshake_failure_score",
        "rst_ratio",
        "burstiness_score",
        "flags_entropy_score",
    ),
    "finite_history_offset_anomaly_score": (
        "edge_packet_count_active_offset",
        "edge_byte_count_active_offset",
        "edge_small_packet_ratio_active_offset",
        "edge_handshake_failure_active_offset",
        "edge_protocol_flags_active_drift_distance",
        "edge_time_behavior_active_drift_distance",
    ),
    "approximate_novelty_anomaly_score": (
        "recent_new_edge_mark",
        "edge_low_activity_score",
        "approximate_rare_edge_score",
        "source_destination_diversity_burst_score",
        "source_port_diversity_burst_score",
    ),
    "behavior_role_anomaly_score": (
        "source_out_degree_quantile",
        "destination_in_degree_quantile",
        "normalized_byte_ratio",
        "source_role_drift",
        "destination_role_drift",
        "previous_attack_dst_to_src_mark",
    ),
}


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = 1.0
        self.maximum = 0.0

    def add(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        value = min(max(value, 0.0), 1.0)
        self.count += 1
        self.total += value
        self.total_square += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_dict(self):
        if not self.count:
            return {"n": 0}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "n": self.count,
            "mean": mean,
            "variance": variance,
            "min": self.minimum,
            "max": self.maximum,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-dir", type=Path, required=True)
    parser.add_argument("--days", nargs="+", required=True)
    parser.add_argument("--weights-path", type=Path)
    parser.add_argument("--score-profile-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    return parser.parse_args()


def ordered_files(window_dir, days):
    day_order = {day: index for index, day in enumerate(days)}
    files = [
        path
        for path in window_dir.glob("*.csv")
        if path.name.split("-", 1)[0] in day_order
    ]
    return sorted(files, key=lambda path: (day_order[path.name.split("-", 1)[0]], path.name))


def label_key(label):
    return "attack" if str(label).strip() not in {"", "0", "BENIGN", "Normal", "normal"} else "normal"


def add_prediction_counts(counter, true_label, predicted_label):
    counter[f"true_{true_label}"] += 1
    counter[f"predicted_{predicted_label}"] += 1
    counter[f"{true_label}_as_{predicted_label}"] += 1


def score_stats_dict(stats_by_label):
    result = {}
    for label, stats in stats_by_label.items():
        result[label] = stats.as_dict()
    normal = stats_by_label.get("normal")
    attack = stats_by_label.get("attack")
    result["approximate_auc"] = approximate_auc(normal, attack) if normal and attack else None
    return result


def feature_stats_dict(stats_by_label):
    output = {}
    for label, stats_by_name in stats_by_label.items():
        output[label] = {
            name: stats.as_dict()
            for name, stats in stats_by_name.items()
        }
    return output


def feature_auc_details(stats_by_label, feature_names):
    details = {}
    for name in feature_names:
        normal = stats_by_label["normal"][name]
        attack = stats_by_label["attack"][name]
        auc = approximate_auc(normal, attack) if normal.count and attack.count else None
        normal_dict = normal.as_dict()
        attack_dict = attack.as_dict()
        details[name] = {
            "normal_mean": normal_dict.get("mean", 0.0),
            "attack_mean": attack_dict.get("mean", 0.0),
            "mean_gap": attack_dict.get("mean", 0.0) - normal_dict.get("mean", 0.0),
            "approximate_auc": auc,
            "discrimination_signal": max((auc or 0.5) - 0.5, 0.0),
        }
    return details


def log(path, message):
    print(message, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def main():
    args = parse_args()
    files = ordered_files(args.window_dir, args.days)
    if not files:
        raise FileNotFoundError(f"no files for days={args.days} in {args.window_dir}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.progress_path:
        args.progress_path.parent.mkdir(parents=True, exist_ok=True)
        args.progress_path.write_text("", encoding="utf-8")

    reset_all_weights()
    reset_active_score_profile()
    if args.weights_path:
        load_trained_weights(args.weights_path)
    if args.score_profile_path:
        load_active_score_profile(args.score_profile_path)

    groups = ("ALL", *PROTOCOL_GROUPS)
    score_stats = {
        group: {label: ScoreStats() for label in ("normal", "attack", "all")}
        for group in groups
    }
    top_feature_stats = {
        group: {
            label: {name: ScoreStats() for name in TOP_LEVEL_FEATURES}
            for label in ("normal", "attack", "all")
        }
        for group in groups
    }
    internal_feature_stats = {
        group: {
            component: {
                label: {name: ScoreStats() for name in feature_names}
                for label in ("normal", "attack", "all")
            }
            for component, feature_names in INTERNAL_GROUPS.items()
        }
        for group in groups
    }
    predicted_counts = {group: Counter() for group in groups}
    protocol_counts = {group: Counter() for group in groups}

    history = History(life_windows=30, detail_windows=5)
    started_at = time.perf_counter()
    log(args.progress_path, f"start files={len(files)} days={','.join(args.days)}")

    for index, path in enumerate(files, start=1):
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        scores, observations = score_window(graph, history)
        predicted = {
            edge_key: predicted_label_from_observation(observations[edge_key])
            for edge_key in scores
        }
        _, final_labels = commit_window(
            history,
            graph,
            predicted,
            suspicious_observations=observations,
        )

        for edge_key, score in scores.items():
            group = protocol_group_from_edge_key(edge_key)
            true_label = true_edge_labels.get(edge_key, "normal")
            labels = (true_label, "all")
            for target_group in (group, "ALL"):
                protocol_counts[target_group]["edge_count"] += 1
                protocol_counts[target_group][f"true_{true_label}"] += 1
                add_prediction_counts(
                    predicted_counts[target_group],
                    true_label,
                    final_labels[edge_key],
                )
                for label in labels:
                    score_stats[target_group][label].add(score)

            observation = observations[edge_key]
            sub_scores = observation["sub_scores"]
            feature_vector = observation["feature_vector"]
            for target_group in (group, "ALL"):
                for label in labels:
                    for name in TOP_LEVEL_FEATURES:
                        top_feature_stats[target_group][label][name].add(
                            sub_scores.get(name, 0.0)
                        )
                    for component, feature_names in INTERNAL_GROUPS.items():
                        for name in feature_names:
                            internal_feature_stats[target_group][component][label][name].add(
                                feature_vector.get(name, 0.0)
                            )

        if index % args.progress_every == 0 or index == len(files):
            log(
                args.progress_path,
                f"progress={index}/{len(files)} file={path.name} elapsed={time.perf_counter() - started_at:.1f}",
            )

    protocol_profiles = {}
    for group in PROTOCOL_GROUPS:
        normal_scores = score_stats[group]["normal"]
        all_scores = score_stats[group]["all"]
        quantile_source = normal_scores if normal_scores.count else all_scores
        attack_threshold = max(0.70, quantile_source.quantile(1.0 - args.target_fpr) or 0.70)
        suspicious_threshold = max(0.55, quantile_source.quantile(0.75) or 0.55)
        protocol_profiles[group] = {
            "traffic": dict(protocol_counts[group]),
            "score_stats": score_stats_dict(score_stats[group]),
            "top_level_feature_details": feature_auc_details(
                top_feature_stats[group],
                TOP_LEVEL_FEATURES,
            ),
            "top_level_feature_stats": feature_stats_dict(top_feature_stats[group]),
            "internal_feature_details": {
                component: feature_auc_details(
                    internal_feature_stats[group][component],
                    feature_names,
                )
                for component, feature_names in INTERNAL_GROUPS.items()
            },
            "internal_feature_stats": {
                component: feature_stats_dict(internal_feature_stats[group][component])
                for component in INTERNAL_GROUPS
            },
            "decision_recommendation": {
                "target_fpr": args.target_fpr,
                "attack_threshold": attack_threshold,
                "suspicious_threshold": suspicious_threshold,
                "normal_threshold": 0.30,
                "min_strong_signals": 2,
            },
            "predicted_counts": dict(predicted_counts[group]),
            "final_label_metrics": final_label_metrics(predicted_counts[group]),
        }

    result = {
        "schema_version": 1,
        "kind": "environment_profile",
        "window_dir": str(args.window_dir),
        "days": args.days,
        "weights_path": str(args.weights_path) if args.weights_path else None,
        "score_profile_path": str(args.score_profile_path) if args.score_profile_path else None,
        "window_count": len(files),
        "elapsed_seconds": time.perf_counter() - started_at,
        "overall": {
            "traffic": dict(protocol_counts["ALL"]),
            "score_stats": score_stats_dict(score_stats["ALL"]),
            "top_level_feature_details": feature_auc_details(
                top_feature_stats["ALL"],
                TOP_LEVEL_FEATURES,
            ),
            "internal_feature_details": {
                component: feature_auc_details(
                    internal_feature_stats["ALL"][component],
                    feature_names,
                )
                for component, feature_names in INTERNAL_GROUPS.items()
            },
            "predicted_counts": dict(predicted_counts["ALL"]),
            "final_label_metrics": final_label_metrics(predicted_counts["ALL"]),
        },
        "protocol_profiles": protocol_profiles,
    }
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(args.progress_path, f"done result={args.output_path} elapsed={result['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
