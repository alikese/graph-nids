import argparse
import itertools
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
from feature.scoring_pipeline import commit_window, score_window
from full_week_score_test import build_graph_with_labels
from test_no_label_leakage_wednesday import final_label_metrics


def parse_float_list(value):
    return [float(item) for item in str(value).split(",") if item.strip()]


def parse_int_list(value):
    return [int(item) for item in str(value).split(",") if item.strip()]


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
        default=["Wednesday"],
        help="Days used to select the best parameter set.",
    )
    parser.add_argument(
        "--eval-days",
        nargs="+",
        default=None,
        help="Optional holdout days evaluated only for the best parameter set.",
    )
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument(
        "--window-ranges",
        default="",
        help=(
            "Optional 1-based inclusive ranges per day, e.g. "
            "Tuesday:180-220,Wednesday:860-910. Used only for offline sampling."
        ),
    )
    parser.add_argument("--life-windows", type=int, default=30)
    parser.add_argument("--detail-windows", type=int, default=5)
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional trained_score_weights.json to load before searching.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=ROOT / "train" / "optimized_experiment_params.json",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=ROOT / "train" / "optimize_experiment_params.log",
    )
    parser.add_argument(
        "--objective",
        choices=["f1", "recall", "detected_or_pending", "balanced", "robust_groups"],
        default="robust_groups",
    )
    parser.add_argument("--max-fpr", type=float, default=0.05)
    parser.add_argument("--fpr-penalty", type=float, default=3.0)
    parser.add_argument("--normal-suspicious-penalty", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--progress-every-config", type=int, default=1)
    parser.add_argument(
        "--attack-thresholds",
        default="0.62,0.66,0.70,0.74",
    )
    parser.add_argument(
        "--suspicious-thresholds",
        default="0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--normal-thresholds", default="0.30")
    parser.add_argument("--min-strong-signals-values", default="1,2")
    parser.add_argument("--theta-suspicious-values", default="0.50,0.55,0.60")
    parser.add_argument("--theta-attack-values", default="0.62,0.66,0.70")
    parser.add_argument(
        "--high-confidence-score-thresholds",
        default="0.58,0.62,0.66",
    )
    parser.add_argument("--min-consecutive-windows-values", default="1,2")
    return parser.parse_args()


def parse_window_ranges(value):
    ranges = {}
    if not value:
        return ranges
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        day, bounds = item.split(":", 1)
        start_text, end_text = bounds.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        if start <= 0 or end < start:
            raise ValueError(f"invalid window range: {item}")
        ranges[day] = (start, end)
    return ranges


def ordered_files(window_dir, days, max_windows=0, window_ranges=None):
    day_order = {day: index for index, day in enumerate(days)}
    window_ranges = window_ranges or {}
    files = sorted(
        [
            path
            for path in window_dir.glob("*.csv")
            if path.name.split("-", 1)[0] in day_order
        ],
        key=lambda path: (
            day_order[path.name.split("-", 1)[0]],
            path.name,
        ),
    )
    if window_ranges:
        selected = []
        per_day_index = Counter()
        for path in files:
            day = path.name.split("-", 1)[0]
            per_day_index[day] += 1
            if day not in window_ranges:
                selected.append(path)
                continue
            start, end = window_ranges[day]
            if start <= per_day_index[day] <= end:
                selected.append(path)
        files = selected
    if max_windows and max_windows > 0:
        return files[:max_windows]
    return files


def add_prediction_counts(counter, true_label, predicted_label):
    counter[f"true_{true_label}"] += 1
    counter[f"predicted_{predicted_label}"] += 1
    counter[f"{true_label}_as_{predicted_label}"] += 1


def f1_from_metrics(metrics):
    precision = metrics["attack_precision"]
    recall = metrics["attack_recall"]
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def objective_value(metrics, args, group_metrics=None):
    f1 = f1_from_metrics(metrics)
    recall = metrics["attack_recall"]
    detected = metrics["attack_detected_or_pending_rate"]
    fpr = metrics["normal_attack_false_positive_rate"]
    suspicious = metrics["normal_suspicious_rate"]

    if args.objective == "robust_groups" and group_metrics:
        group_values = list(group_metrics.values())
        group_f1 = [item["f1"] for item in group_values]
        group_recall = [item["attack_recall"] for item in group_values]
        group_detected = [
            item["attack_detected_or_pending_rate"]
            for item in group_values
        ]
        group_precision = [item["attack_precision"] for item in group_values]
        group_fpr = [
            item["normal_attack_false_positive_rate"]
            for item in group_values
        ]
        group_suspicious = [
            item["normal_suspicious_rate"]
            for item in group_values
        ]
        value = (
            0.30 * min(group_f1)
            + 0.20 * min(group_recall)
            + 0.15 * min(group_detected)
            + 0.20 * (sum(group_f1) / len(group_f1))
            + 0.15 * (sum(group_precision) / len(group_precision))
        )
        value -= args.fpr_penalty * max(
            max(rate - args.max_fpr, 0.0)
            for rate in group_fpr
        )
        value -= args.normal_suspicious_penalty * (
            sum(group_suspicious) / len(group_suspicious)
        )
        return value

    if args.objective == "f1":
        value = f1
    elif args.objective == "recall":
        value = recall
    elif args.objective == "detected_or_pending":
        value = detected
    else:
        value = 0.45 * f1 + 0.35 * recall + 0.20 * detected

    value -= args.fpr_penalty * max(fpr - args.max_fpr, 0.0)
    value -= args.normal_suspicious_penalty * suspicious
    return value


def build_configs(args):
    for (
        attack_threshold,
        suspicious_threshold,
        normal_threshold,
        min_strong_signals,
        theta_suspicious,
        theta_attack,
        high_confidence_score_threshold,
        min_consecutive_windows,
    ) in itertools.product(
        parse_float_list(args.attack_thresholds),
        parse_float_list(args.suspicious_thresholds),
        parse_float_list(args.normal_thresholds),
        parse_int_list(args.min_strong_signals_values),
        parse_float_list(args.theta_suspicious_values),
        parse_float_list(args.theta_attack_values),
        parse_float_list(args.high_confidence_score_thresholds),
        parse_int_list(args.min_consecutive_windows_values),
    ):
        if not normal_threshold < suspicious_threshold < attack_threshold:
            continue
        if not theta_suspicious < theta_attack:
            continue
        yield {
            "decision": {
                "attack_threshold": attack_threshold,
                "suspicious_threshold": suspicious_threshold,
                "normal_threshold": normal_threshold,
                "min_strong_signals": min_strong_signals,
            },
            "suspicious_history": {
                "theta_suspicious": theta_suspicious,
                "theta_attack": theta_attack,
                "high_confidence_score_threshold": (
                    high_confidence_score_threshold
                ),
                "min_consecutive_windows": min_consecutive_windows,
            },
        }


def run_config(files, config, args):
    history = History(
        life_windows=args.life_windows,
        detail_windows=args.detail_windows,
        suspicious_edge_history_params=config["suspicious_history"],
    )
    predicted_counts = Counter()
    group_predicted_counts = defaultdict(Counter)
    traffic = Counter()
    started_at = time.perf_counter()

    for path in files:
        group_name = path.name.split("-", 1)[0]
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        scores, observations = score_window(graph, history)
        threshold_labels = {
            edge_key: predicted_label_from_observation(
                observations[edge_key],
                **config["decision"],
            )
            for edge_key in scores
        }
        _, final_labels = commit_window(
            history,
            graph,
            threshold_labels,
            suspicious_observations=observations,
        )
        for edge_key in scores:
            true_label = true_edge_labels.get(edge_key, "normal")
            predicted_label = final_labels[edge_key]
            add_prediction_counts(
                predicted_counts,
                true_label,
                predicted_label,
            )
            add_prediction_counts(
                group_predicted_counts[group_name],
                true_label,
                predicted_label,
            )
        traffic["window_count"] += 1
        traffic["packet_count"] += len(graph.packets)
        traffic["edge_count"] += len(graph.edges)

    metrics = final_label_metrics(predicted_counts)
    metrics["f1"] = f1_from_metrics(metrics)
    group_metrics = {}
    for group_name, counts in group_predicted_counts.items():
        group_metric = final_label_metrics(counts)
        group_metric["f1"] = f1_from_metrics(group_metric)
        group_metrics[group_name] = group_metric
    metrics["objective_value"] = objective_value(metrics, args, group_metrics)
    return {
        "config": config,
        "metrics": metrics,
        "group_metrics": group_metrics,
        "predicted_counts": dict(predicted_counts),
        "group_predicted_counts": {
            group_name: dict(counts)
            for group_name, counts in group_predicted_counts.items()
        },
        "traffic": dict(traffic),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def log_progress(path, message):
    print(message, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main():
    args = parse_args()
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.parent.mkdir(parents=True, exist_ok=True)
    args.progress_path.write_text("", encoding="utf-8")

    if args.weights_path is not None:
        from train.train_score_weights import load_trained_weights

        load_trained_weights(args.weights_path)

    window_ranges = parse_window_ranges(args.window_ranges)
    train_files = ordered_files(
        args.window_dir,
        args.days,
        args.max_windows,
        window_ranges,
    )
    if not train_files:
        raise FileNotFoundError(
            f"no training windows for {args.days} in {args.window_dir}"
        )

    configs = list(build_configs(args))
    if not configs:
        raise ValueError("empty parameter grid after validation")

    log_progress(
        args.progress_path,
        f"search_start configs={len(configs)} days={','.join(args.days)} "
        f"windows={len(train_files)} objective={args.objective}",
    )

    results = []
    best = None
    for index, config in enumerate(configs, start=1):
        result = run_config(train_files, config, args)
        results.append(result)
        if (
            best is None
            or result["metrics"]["objective_value"]
            > best["metrics"]["objective_value"]
        ):
            best = result
        if (
            args.progress_every_config > 0
            and (
                index % args.progress_every_config == 0
                or index == len(configs)
            )
        ):
            metrics = result["metrics"]
            log_progress(
                args.progress_path,
                f"config={index}/{len(configs)} "
                f"objective={metrics['objective_value']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"recall={metrics['attack_recall']:.4f} "
                f"fpr={metrics['normal_attack_false_positive_rate']:.4f} "
                f"best={best['metrics']['objective_value']:.4f}",
            )

    ranked = sorted(
        results,
        key=lambda item: item["metrics"]["objective_value"],
        reverse=True,
    )
    output = {
        "completed": True,
        "objective": args.objective,
        "max_fpr": args.max_fpr,
        "training_days": args.days,
        "window_ranges": {
            day: list(bounds)
            for day, bounds in window_ranges.items()
        },
        "training_window_count": len(train_files),
        "searched_config_count": len(configs),
        "best": ranked[0],
        "top_results": ranked[: args.top_k],
    }

    if args.eval_days:
        eval_files = ordered_files(
            args.window_dir,
            args.eval_days,
            args.max_windows,
            window_ranges,
        )
        if not eval_files:
            raise FileNotFoundError(
                f"no evaluation windows for {args.eval_days} in {args.window_dir}"
            )
        output["evaluation_days"] = args.eval_days
        output["evaluation_window_count"] = len(eval_files)
        output["best_holdout_evaluation"] = run_config(
            eval_files,
            ranked[0]["config"],
            args,
        )

    args.result_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_progress(
        args.progress_path,
        f"search_done best_objective="
        f"{ranked[0]['metrics']['objective_value']:.4f} "
        f"result={args.result_path}",
    )


if __name__ == "__main__":
    main()
