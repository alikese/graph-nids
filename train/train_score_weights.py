import argparse
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.edge.edgeClass import (
    DEFAULT_CURRENT_BEHAVIOR_WEIGHTS,
    reset_current_behavior_weights,
    set_current_behavior_weights,
)
from feature.history.historyClass import (
    DEFAULT_BEHAVIOR_ROLE_WEIGHTS,
    History,
    reset_behavior_role_weights,
    set_behavior_role_weights,
)
from feature.history.history_feature.active_edge_features import (
    EdgeActiveHistoryFeature,
)
from feature.decision_engine import predicted_label_from_observation
from feature.scoring_pipeline import commit_window, recent_new_edge, score_window
from feature.sum_score import (
    DEFAULT_LOCAL_ANOMALY_WEIGHTS,
    reset_local_anomaly_weights,
    set_local_anomaly_weights,
)
from full_week_score_test import (
    ScoreStats,
    build_graph_with_labels,
)
from test_no_label_leakage_wednesday import (
    approximate_auc,
    final_label_metrics,
    metrics_at_threshold,
    timing_summary,
)

TOP_LEVEL_COMPONENT_NAMES = tuple(DEFAULT_LOCAL_ANOMALY_WEIGHTS)
INTERNAL_DEFAULT_WEIGHTS = {
    "current_behavior_anomaly_score": dict(
        DEFAULT_CURRENT_BEHAVIOR_WEIGHTS
    ),
    "finite_history_offset_anomaly_score": dict(
        EdgeActiveHistoryFeature.DEFAULT_FINITE_HISTORY_OFFSET_WEIGHTS
    ),
    "approximate_novelty_anomaly_score": dict(
        recent_new_edge.DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS
    ),
    "behavior_role_anomaly_score": dict(
        DEFAULT_BEHAVIOR_ROLE_WEIGHTS
    ),
}
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
PHASES = (
    "level1_internal_training",
    "level2_total_score_training",
    "two_level_trained_evaluation",
)


class ComponentStats:
    def __init__(self, bin_count=1000):
        self.bin_count = int(bin_count)
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.bins = [0] * (self.bin_count + 1)

    def add(self, value):
        value = min(max(float(value), 0.0), 1.0)
        self.count += 1
        self.total += value
        self.total_square += value * value
        index = min(int(value * self.bin_count), self.bin_count)
        self.bins[index] += 1

    @property
    def mean(self):
        return self.total / self.count if self.count else 0.0

    @property
    def variance(self):
        if not self.count:
            return 0.0
        return max(self.total_square / self.count - self.mean * self.mean, 0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-dir",
        type=Path,
        default=ROOT / "windows divide",
    )
    parser.add_argument(
        "--days",
        nargs="+",
        default=["Tuesday", "Wednesday", "Thursday", "Friday"],
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=ROOT / "train" / "trained_score_weights.json",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=ROOT / "train" / "trained_score_evaluation.json",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=ROOT / "train" / "train_score_progress.log",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=ROOT / "train" / "train_score_checkpoint.pkl",
    )
    parser.add_argument(
        "--internal-weights-path",
        type=Path,
        default=None,
        help="加载已有 internal_weights 并跳过一级内部权重训练。",
    )
    parser.add_argument("--prior-strength", type=float, default=0.20)
    parser.add_argument("--top-level-f1-weight", type=float, default=0.50)
    parser.add_argument("--top-level-recall-weight", type=float, default=0.30)
    parser.add_argument("--top-level-precision-weight", type=float, default=0.20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="删除已有 checkpoint，从头重新训练。",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="加载已有权重，只执行 two_level_trained_evaluation 阶段。",
    )
    parser.add_argument(
        "--internal-only",
        action="store_true",
        help="只训练 internal_weights，top-level 权重保持默认值。",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="训练完成后保留阶段 checkpoint，便于继续调二级目标。",
    )
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


def component_auc(normal_stats, attack_stats):
    if normal_stats.count <= 0 or attack_stats.count <= 0:
        return 0.5
    normal_below = 0
    wins = 0.0
    for attack_count, normal_count in zip(
        attack_stats.bins,
        normal_stats.bins,
    ):
        wins += attack_count * (normal_below + 0.5 * normal_count)
        normal_below += normal_count
    return wins / (normal_stats.count * attack_stats.count)


def train_weight_group(component_stats, default_weights, prior_strength):
    if not 0.0 <= prior_strength <= 1.0:
        raise ValueError("prior-strength must be between 0 and 1")

    details = {}
    signals = {}
    for name in default_weights:
        normal_stats = component_stats["normal"][name]
        attack_stats = component_stats["attack"][name]
        auc = component_auc(normal_stats, attack_stats)
        pooled_variance = (
            normal_stats.variance + attack_stats.variance
        ) / 2.0
        standardized_gap = (
            (attack_stats.mean - normal_stats.mean)
            / math.sqrt(pooled_variance + 1e-12)
        )
        signal = max(auc - 0.5, 0.0)
        signals[name] = signal
        details[name] = {
            "normal_count": normal_stats.count,
            "attack_count": attack_stats.count,
            "normal_mean": normal_stats.mean,
            "attack_mean": attack_stats.mean,
            "mean_gap": attack_stats.mean - normal_stats.mean,
            "standardized_mean_gap": standardized_gap,
            "approximate_auc": auc,
            "discrimination_signal": signal,
        }

    signal_total = sum(signals.values())
    if signal_total <= 0.0:
        learned = dict(default_weights)
    else:
        learned = {
            name: signals[name] / signal_total
            for name in default_weights
        }
    weights = {
        name: (
            prior_strength * default_weights[name]
            + (1.0 - prior_strength) * learned[name]
        )
        for name in default_weights
    }
    weight_total = sum(weights.values())
    weights = {
        name: value / weight_total
        for name, value in weights.items()
    }
    return weights, learned, details


def threshold_counts(normal_stats, attack_stats, threshold):
    index = min(
        max(math.ceil(float(threshold) * normal_stats.bin_count), 0),
        normal_stats.bin_count,
    )
    normal_positive = sum(normal_stats.bins[index:])
    attack_positive = sum(attack_stats.bins[index:])
    return normal_positive, attack_positive


def binary_metrics_at_component_threshold(
    normal_stats,
    attack_stats,
    threshold,
):
    normal_positive, attack_positive = threshold_counts(
        normal_stats,
        attack_stats,
        threshold,
    )
    total_attack = attack_stats.count
    predicted_positive = normal_positive + attack_positive
    precision = (
        attack_positive / predicted_positive
        if predicted_positive > 0
        else 0.0
    )
    recall = attack_positive / total_attack if total_attack > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    false_positive_rate = (
        normal_positive / normal_stats.count
        if normal_stats.count > 0
        else 0.0
    )
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "normal_positive": normal_positive,
        "attack_positive": attack_positive,
    }


def train_top_level_metric_weight_group(
    component_stats,
    default_weights,
    prior_strength,
    f1_weight,
    recall_weight,
    precision_weight,
):
    if not 0.0 <= prior_strength <= 1.0:
        raise ValueError("prior-strength must be between 0 and 1")
    objective_weights = {
        "f1": max(float(f1_weight), 0.0),
        "recall": max(float(recall_weight), 0.0),
        "precision": max(float(precision_weight), 0.0),
    }
    objective_total = sum(objective_weights.values())
    if objective_total <= 0.0:
        raise ValueError(
            "at least one top-level objective weight must be positive"
        )
    objective_weights = {
        name: value / objective_total
        for name, value in objective_weights.items()
    }

    details = {}
    signals = {}
    for name in default_weights:
        normal_stats = component_stats["normal"][name]
        attack_stats = component_stats["attack"][name]
        auc = component_auc(normal_stats, attack_stats)
        threshold_metrics = [
            binary_metrics_at_component_threshold(
                normal_stats,
                attack_stats,
                threshold,
            )
            for threshold in THRESHOLDS
        ]
        for item in threshold_metrics:
            item["objective"] = (
                objective_weights["f1"] * item["f1"]
                + objective_weights["recall"] * item["recall"]
                + objective_weights["precision"] * item["precision"]
            )
        best = max(
            threshold_metrics,
            key=lambda item: (
                item["objective"],
                item["f1"],
                item["recall"],
                item["precision"],
            ),
        )
        signal = max(best["objective"], 0.0)
        signals[name] = signal
        details[name] = {
            "normal_count": normal_stats.count,
            "attack_count": attack_stats.count,
            "normal_mean": normal_stats.mean,
            "attack_mean": attack_stats.mean,
            "mean_gap": attack_stats.mean - normal_stats.mean,
            "approximate_auc": auc,
            "objective_weights": objective_weights,
            "best_threshold": best["threshold"],
            "best_threshold_precision": best["precision"],
            "best_threshold_recall": best["recall"],
            "best_threshold_f1": best["f1"],
            "best_threshold_false_positive_rate": best[
                "false_positive_rate"
            ],
            "metric_objective_signal": signal,
            "threshold_metrics": threshold_metrics,
        }

    signal_total = sum(signals.values())
    if signal_total <= 0.0:
        learned = dict(default_weights)
    else:
        learned = {
            name: signals[name] / signal_total
            for name in default_weights
        }
    weights = {
        name: (
            prior_strength * default_weights[name]
            + (1.0 - prior_strength) * learned[name]
        )
        for name in default_weights
    }
    weight_total = sum(weights.values())
    weights = {
        name: value / weight_total
        for name, value in weights.items()
    }
    return weights, learned, details


def reset_all_weights():
    reset_current_behavior_weights()
    EdgeActiveHistoryFeature.reset_finite_history_offset_weights()
    recent_new_edge.reset_approximate_novelty_weights()
    reset_behavior_role_weights()
    reset_local_anomaly_weights()


def apply_internal_weights(internal_weights):
    set_current_behavior_weights(
        internal_weights["current_behavior_anomaly_score"]
    )
    EdgeActiveHistoryFeature.set_finite_history_offset_weights(
        internal_weights["finite_history_offset_anomaly_score"]
    )
    recent_new_edge.set_approximate_novelty_weights(
        internal_weights["approximate_novelty_anomaly_score"]
    )
    set_behavior_role_weights(
        internal_weights["behavior_role_anomaly_score"]
    )


def load_trained_weights(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    apply_internal_weights(payload["internal_weights"])
    set_local_anomaly_weights(payload["top_level_weights"])
    return {
        "internal_weights": payload["internal_weights"],
        "top_level_weights": payload["top_level_weights"],
    }


def load_internal_weights_payload(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    internal_weights = payload["internal_weights"]
    apply_internal_weights(internal_weights)
    return {
        "internal_weights": internal_weights,
        "internal_raw_learned_weights": payload.get(
            "internal_raw_learned_weights",
            {},
        ),
        "internal_training_details": payload.get(
            "internal_training_details",
            {},
        ),
    }


def new_internal_feature_stats():
    return {
        group_name: {
            label: {
                feature_name: ComponentStats()
                for feature_name in default_weights
            }
            for label in ("normal", "attack")
        }
        for group_name, default_weights in INTERNAL_DEFAULT_WEIGHTS.items()
    }


def new_top_level_stats():
    return {
        label: {
            name: ComponentStats()
            for name in TOP_LEVEL_COMPONENT_NAMES
        }
        for label in ("normal", "attack")
    }


def empty_scope(days):
    return {
        "predicted_counts": Counter(),
        "daily_predicted_counts": {
            day: Counter()
            for day in days
        },
        "score_stats": {
            "overall": {
                "normal": ScoreStats(),
                "attack": ScoreStats(),
            },
            **{
                day: {
                    "normal": ScoreStats(),
                    "attack": ScoreStats(),
                }
                for day in days
            },
        },
        "timings": defaultdict(list),
        "daily_timings": {
            day: defaultdict(list)
            for day in days
        },
        "traffic": Counter(),
        "daily_traffic": {
            day: Counter()
            for day in days
        },
        "decision_reasons": Counter(),
    }


def update_traffic(counter, graph):
    counter["window_count"] += 1
    counter["packet_count"] += len(graph.packets)
    counter["edge_count"] += len(graph.edges)
    counter["max_packets_per_window"] = max(
        counter["max_packets_per_window"],
        len(graph.packets),
    )
    counter["max_edges_per_window"] = max(
        counter["max_edges_per_window"],
        len(graph.edges),
    )


def log_progress(path, message):
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def files_signature(files):
    return tuple(path.name for path in files)


def load_checkpoint(path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(path)


def phase_checkpoint_path(base_path, phase):
    return base_path.with_name(
        f"{base_path.stem}_{phase}{base_path.suffix}"
    )


def checkpoint_matches(payload, phase, files, days, window_dir):
    return (
        payload
        and payload.get("phase") == phase
        and payload.get("days") == list(days)
        and payload.get("window_directory") == str(window_dir)
        and payload.get("files_signature") == files_signature(files)
    )


def run_phase(
    phase,
    files,
    days,
    window_dir,
    progress_path,
    progress_every,
    checkpoint_path=None,
    checkpoint_every=20,
    collect_internal_features=False,
    collect_top_level_scores=False,
):
    if checkpoint_path is not None:
        checkpoint_path = phase_checkpoint_path(checkpoint_path, phase)
    checkpoint_payload = (
        load_checkpoint(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    if checkpoint_matches(checkpoint_payload, phase, files, days, window_dir):
        if checkpoint_payload.get("status") == "completed":
            log_progress(
                progress_path,
                f"{phase} resume=completed "
                f"elapsed_seconds={checkpoint_payload['elapsed_seconds']:.1f}",
            )
            return (
                checkpoint_payload["scope"],
                checkpoint_payload["internal_feature_stats"],
                checkpoint_payload["top_level_stats"],
                checkpoint_payload["elapsed_seconds"],
            )

        history = checkpoint_payload["history"]
        scope = checkpoint_payload["scope"]
        internal_feature_stats = checkpoint_payload["internal_feature_stats"]
        top_level_stats = checkpoint_payload["top_level_stats"]
        next_index = int(checkpoint_payload["next_index"])
        base_elapsed = float(checkpoint_payload.get("elapsed_seconds", 0.0))
        log_progress(
            progress_path,
            f"{phase} resume next_index={next_index + 1}/{len(files)} "
            f"elapsed_seconds={base_elapsed:.1f}",
        )
    else:
        history = History(life_windows=30, detail_windows=5)
        scope = empty_scope(days)
        internal_feature_stats = new_internal_feature_stats()
        top_level_stats = new_top_level_stats()
        next_index = 0
        base_elapsed = 0.0

    started_at = time.perf_counter()

    def current_elapsed():
        return base_elapsed + time.perf_counter() - started_at

    def checkpoint(status, next_index_value):
        if checkpoint_path is None:
            return
        save_checkpoint(
            checkpoint_path,
            {
                "phase": phase,
                "status": status,
                "days": list(days),
                "window_directory": str(window_dir),
                "files_signature": files_signature(files),
                "next_index": next_index_value,
                "elapsed_seconds": current_elapsed(),
                "history": history,
                "scope": scope,
                "internal_feature_stats": internal_feature_stats,
                "top_level_stats": top_level_stats,
            },
        )

    for zero_index in range(next_index, len(files)):
        index = zero_index + 1
        path = files[zero_index]
        window_started_at = time.perf_counter()
        day = path.name.split("-", 1)[0]

        stage_started_at = time.perf_counter()
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        build_seconds = time.perf_counter() - stage_started_at

        scores, observations, score_timings = score_window(
            graph,
            history,
            return_timing=True,
        )
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
            add_prediction_counts(
                scope["predicted_counts"],
                true_label,
                predicted_label,
            )
            add_prediction_counts(
                scope["daily_predicted_counts"][day],
                true_label,
                predicted_label,
            )
            scope["score_stats"]["overall"][true_label].add(score)
            scope["score_stats"][day][true_label].add(score)
            if collect_internal_features:
                feature_vector = observations[edge_key]["feature_vector"]
                for group_name, default_weights in (
                    INTERNAL_DEFAULT_WEIGHTS.items()
                ):
                    for feature_name in default_weights:
                        internal_feature_stats[group_name][true_label][
                            feature_name
                        ].add(feature_vector.get(feature_name, 0.0))
            if collect_top_level_scores:
                sub_scores = observations[edge_key]["sub_scores"]
                for name in TOP_LEVEL_COMPONENT_NAMES:
                    top_level_stats[true_label][name].add(
                        sub_scores.get(name, 0.0)
                    )
        for decision in suspicious_result.decisions.values():
            scope["decision_reasons"][decision.reason] += 1
        evaluation_seconds = time.perf_counter() - stage_started_at

        window_seconds = time.perf_counter() - window_started_at
        window_timings = {
            "build_graph": build_seconds,
            **score_timings,
            "commit_history": commit_seconds,
            "evaluation_only": evaluation_seconds,
            "window_total": window_seconds,
        }
        for name, value in window_timings.items():
            scope["timings"][name].append(value)
            scope["daily_timings"][day][name].append(value)
        update_traffic(scope["traffic"], graph)
        update_traffic(scope["daily_traffic"][day], graph)

        if (
            index % progress_every == 0
            or index == len(files)
        ):
            log_progress(
                progress_path,
                f"{phase} progress={index}/{len(files)} day={day} "
                f"file={path.name} packets={len(graph.packets)} "
                f"edges={len(graph.edges)} "
                f"window_seconds={window_seconds:.3f} "
                f"elapsed_seconds={current_elapsed():.1f}",
            )
        if (
            checkpoint_every > 0
            and (index % checkpoint_every == 0 or index == len(files))
        ):
            checkpoint("running", index)

    checkpoint("completed", len(files))
    return (
        scope,
        internal_feature_stats,
        top_level_stats,
        current_elapsed(),
    )


def traffic_result(counter):
    windows = counter["window_count"]
    return {
        **dict(counter),
        "mean_packets_per_window": (
            counter["packet_count"] / windows
            if windows
            else 0.0
        ),
        "mean_edges_per_window": (
            counter["edge_count"] / windows
            if windows
            else 0.0
        ),
    }


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
            f"{threshold:.2f}": metrics_at_threshold(
                score_stats,
                threshold,
            )
            for threshold in THRESHOLDS
        },
    }


def scope_result(scope, days, elapsed_seconds):
    metrics = final_label_metrics(scope["predicted_counts"])
    metrics["f1"] = calculate_f1(metrics)
    daily = {}
    for day in days:
        daily_metrics = final_label_metrics(
            scope["daily_predicted_counts"][day]
        )
        daily_metrics["f1"] = calculate_f1(daily_metrics)
        daily[day] = {
            "predicted_counts": dict(
                scope["daily_predicted_counts"][day]
            ),
            "final_label_metrics": daily_metrics,
            **score_result(scope["score_stats"][day]),
            "timing": {
                name: timing_summary(values)
                for name, values in scope["daily_timings"][day].items()
            },
            "traffic": traffic_result(scope["daily_traffic"][day]),
        }
    return {
        "elapsed_seconds": elapsed_seconds,
        "predicted_counts": dict(scope["predicted_counts"]),
        "final_label_metrics": metrics,
        **score_result(scope["score_stats"]["overall"]),
        "timing": {
            name: timing_summary(values)
            for name, values in scope["timings"].items()
        },
        "traffic": traffic_result(scope["traffic"]),
        "suspicious_decision_reasons": dict(
            scope["decision_reasons"]
        ),
        "daily": daily,
    }


def main():
    args = parse_args()
    files = ordered_files(args.window_dir, args.days)
    if not files:
        raise FileNotFoundError(
            f"no windows for {args.days} in {args.window_dir}"
        )

    args.weights_path.parent.mkdir(parents=True, exist_ok=True)
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reset_checkpoint:
        for phase in PHASES:
            path = phase_checkpoint_path(args.checkpoint_path, phase)
            if path.exists():
                path.unlink()
        if args.checkpoint_path.exists():
            args.checkpoint_path.unlink()
    if args.reset_checkpoint or not args.progress_path.exists():
        args.progress_path.write_text("", encoding="utf-8")

    reset_all_weights()
    if args.eval_only:
        loaded_weights = load_trained_weights(args.weights_path)
        log_progress(
            args.progress_path,
            f"eval_only_start days={','.join(args.days)} "
            f"windows={len(files)} weights={args.weights_path} "
            f"checkpoint={args.checkpoint_path}",
        )
        trained_scope, _, _, trained_elapsed = run_phase(
            phase="two_level_trained_evaluation",
            files=files,
            days=args.days,
            window_dir=args.window_dir,
            progress_path=args.progress_path,
            progress_every=args.progress_every,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
        )
        result = {
            "completed": True,
            "eval_only": True,
            "evaluation_days": args.days,
            "window_directory": str(args.window_dir),
            "window_count": len(files),
            "weights_path": str(args.weights_path),
            "checkpoint_path": str(args.checkpoint_path),
            "loaded_internal_weights": loaded_weights["internal_weights"],
            "loaded_top_level_weights": loaded_weights["top_level_weights"],
            "trained_in_sample_evaluation": scope_result(
                trained_scope,
                args.days,
                trained_elapsed,
            ),
            "evaluation_warning": (
                "本次仅加载已有权重并执行第三阶段评估；"
                "新增 auth_bruteforce_score 不改变已训练权重，"
                "只作为额外认证服务爆破证据并入最终分数。"
            ),
        }
        args.result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log_progress(
            args.progress_path,
            f"eval_only_done result={args.result_path} "
            f"elapsed_seconds={trained_elapsed:.1f}",
        )
        return

    log_progress(
        args.progress_path,
        f"training_start days={','.join(args.days)} "
        f"windows={len(files)} checkpoint={args.checkpoint_path}",
    )
    baseline_scope = None
    baseline_elapsed = 0.0
    internal_weights_source = None
    if args.internal_weights_path:
        loaded_internal = load_internal_weights_payload(
            args.internal_weights_path
        )
        internal_weights = loaded_internal["internal_weights"]
        internal_raw_weights = loaded_internal[
            "internal_raw_learned_weights"
        ]
        internal_details = loaded_internal["internal_training_details"]
        internal_weights_source = str(args.internal_weights_path)
        log_progress(
            args.progress_path,
            f"level1_skipped internal_weights_path="
            f"{args.internal_weights_path}",
        )
    else:
        (
            baseline_scope,
            internal_feature_stats,
            _,
            baseline_elapsed,
        ) = run_phase(
            phase="level1_internal_training",
            files=files,
            days=args.days,
            window_dir=args.window_dir,
            progress_path=args.progress_path,
            progress_every=args.progress_every,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            collect_internal_features=True,
        )
        internal_weights = {}
        internal_raw_weights = {}
        internal_details = {}
        for group_name, default_weights in INTERNAL_DEFAULT_WEIGHTS.items():
            (
                internal_weights[group_name],
                internal_raw_weights[group_name],
                internal_details[group_name],
            ) = train_weight_group(
                internal_feature_stats[group_name],
                default_weights,
                args.prior_strength,
            )
        apply_internal_weights(internal_weights)
        log_progress(
            args.progress_path,
            f"level1_complete internal_weights={internal_weights} "
            f"elapsed_seconds={baseline_elapsed:.1f}",
        )

    internal_scope = None
    internal_elapsed = 0.0
    if args.internal_only:
        top_level_weights = dict(DEFAULT_LOCAL_ANOMALY_WEIGHTS)
        top_level_raw_weights = dict(DEFAULT_LOCAL_ANOMALY_WEIGHTS)
        top_level_details = {
            "skipped": True,
            "reason": (
                "internal-only mode keeps default top-level local "
                "anomaly weights"
            ),
        }
        reset_local_anomaly_weights()
        log_progress(
            args.progress_path,
            f"level2_skipped internal_only=true "
            f"top_level_weights={top_level_weights}",
        )
    else:
        (
            internal_scope,
            _,
            top_level_stats,
            internal_elapsed,
        ) = run_phase(
            phase="level2_total_score_training",
            files=files,
            days=args.days,
            window_dir=args.window_dir,
            progress_path=args.progress_path,
            progress_every=args.progress_every,
            checkpoint_path=args.checkpoint_path,
            checkpoint_every=args.checkpoint_every,
            collect_top_level_scores=True,
        )
        (
            top_level_weights,
            top_level_raw_weights,
            top_level_details,
        ) = train_top_level_metric_weight_group(
            top_level_stats,
            DEFAULT_LOCAL_ANOMALY_WEIGHTS,
            args.prior_strength,
            args.top_level_f1_weight,
            args.top_level_recall_weight,
            args.top_level_precision_weight,
        )
        set_local_anomaly_weights(top_level_weights)
        log_progress(
            args.progress_path,
            f"level2_complete top_level_weights={top_level_weights} "
            f"elapsed_seconds={internal_elapsed:.1f}",
        )

    weights_payload = {
        "training_days": args.days,
        "window_directory": str(args.window_dir),
        "window_count": len(files),
        "training_mode": (
            "internal_only"
            if args.internal_only
            else "two_level_metric"
        ),
        "method": (
            "第一层对四类异常分数内部特征分别计算 attack-vs-normal "
            "近似AUC区分信号。"
            + (
                "本次启用 internal-only，第二层 top-level 权重保持默认值。"
                if args.internal_only
                else (
                    "第二层对重新计算后的四类异常分数，按候选阈值下的 "
                    "F1、Recall、Precision 综合目标训练。两层均将学习信号"
                    "归一化，并与内置权重做先验混合。"
                )
            )
        ),
        "prior_strength": args.prior_strength,
        "internal_weights_source": internal_weights_source,
        "top_level_objective_weights": {
            "f1": args.top_level_f1_weight,
            "recall": args.top_level_recall_weight,
            "precision": args.top_level_precision_weight,
        },
        "internal_default_weights": INTERNAL_DEFAULT_WEIGHTS,
        "internal_raw_learned_weights": internal_raw_weights,
        "internal_weights": internal_weights,
        "internal_training_details": internal_details,
        "top_level_default_weights": DEFAULT_LOCAL_ANOMALY_WEIGHTS,
        "top_level_raw_learned_weights": top_level_raw_weights,
        "top_level_weights": top_level_weights,
        "top_level_training_details": top_level_details,
    }
    args.weights_path.write_text(
        json.dumps(weights_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trained_scope, _, _, trained_elapsed = run_phase(
        phase="two_level_trained_evaluation",
        files=files,
        days=args.days,
        window_dir=args.window_dir,
        progress_path=args.progress_path,
        progress_every=args.progress_every,
        checkpoint_path=args.checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )
    result = {
        "completed": True,
        "training_days": args.days,
        "evaluation_days": args.days,
        "window_directory": str(args.window_dir),
        "window_count": len(files),
        "training_mode": (
            "internal_only"
            if args.internal_only
            else "two_level_metric"
        ),
        "weights_path": str(args.weights_path),
        "checkpoint_path": str(args.checkpoint_path),
        "internal_weights_source": internal_weights_source,
        "internal_default_weights": INTERNAL_DEFAULT_WEIGHTS,
        "trained_internal_weights": internal_weights,
        "internal_training_details": internal_details,
        "top_level_default_weights": DEFAULT_LOCAL_ANOMALY_WEIGHTS,
        "top_level_objective_weights": {
            "f1": args.top_level_f1_weight,
            "recall": args.top_level_recall_weight,
            "precision": args.top_level_precision_weight,
        },
        "trained_top_level_weights": top_level_weights,
        "top_level_training_details": top_level_details,
        "baseline": (
            scope_result(
                baseline_scope,
                args.days,
                baseline_elapsed,
            )
            if baseline_scope is not None
            else None
        ),
        "internal_weights_only": scope_result(
            trained_scope if args.internal_only else internal_scope,
            args.days,
            trained_elapsed if args.internal_only else internal_elapsed,
        ),
        "trained_in_sample_evaluation": scope_result(
            trained_scope,
            args.days,
            trained_elapsed,
        ),
        "evaluation_warning": (
            "Tuesday、Wednesday、Thursday 同时用于权重训练和评估，"
            "因此 trained_in_sample_evaluation 是训练集内指标，"
            "不能代表对未见数据的泛化能力。"
        ),
    }
    args.result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_progress(
        args.progress_path,
        f"done result={args.result_path} "
        f"total_elapsed_seconds="
        f"{baseline_elapsed + internal_elapsed + trained_elapsed:.1f}",
    )
    if not args.keep_checkpoint:
        for phase in PHASES:
            path = phase_checkpoint_path(args.checkpoint_path, phase)
            if path.exists():
                path.unlink()
    if args.checkpoint_path.exists():
        args.checkpoint_path.unlink()


if __name__ == "__main__":
    main()
