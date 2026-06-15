import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from feature.history.historyClass import History
from full_week_score_test import (
    DAYS,
    ScoreStats,
    build_graph_with_labels,
    commit_window,
    ordered_weekday_files,
    predicted_label_from_score,
)
from test_no_label_leakage_wednesday import (
    approximate_auc,
    final_label_metrics,
    metrics_at_threshold,
    score_window,
    timing_summary,
)


ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "full_week_integrated_result.json"
PROGRESS_PATH = ROOT / "full_week_integrated_progress.log"
CHECKPOINT_PATH = ROOT / "full_week_integrated_checkpoint.pkl"
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def new_state():
    return {
        "next_index": 0,
        "history": History(life_windows=30, detail_windows=5),
        "predicted_counts": Counter(),
        "daily_predicted_counts": {
            day: Counter() for day in DAYS
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
                for day in DAYS
            },
        },
        "suspicious_reason_counts": Counter(),
        "daily_suspicious_reason_counts": {
            day: Counter() for day in DAYS
        },
        "window_timings": defaultdict(list),
        "daily_window_timings": {
            day: defaultdict(list) for day in DAYS
        },
        "traffic": Counter(),
        "daily_traffic": {
            day: Counter() for day in DAYS
        },
        "started_elapsed_seconds": 0.0,
    }


def save_checkpoint(state):
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(CHECKPOINT_PATH)


def load_state():
    if not CHECKPOINT_PATH.exists():
        return new_state()
    with CHECKPOINT_PATH.open("rb") as handle:
        return pickle.load(handle)


def log_progress(message):
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def add_prediction_counts(counter, true_label, predicted_label):
    counter[f"true_{true_label}"] += 1
    counter[f"predicted_{predicted_label}"] += 1
    counter[f"{true_label}_as_{predicted_label}"] += 1


def score_scope_result(score_stats):
    return {
        "score_distribution": {
            label: stats.as_dict()
            for label, stats in score_stats.items()
        },
        "approximate_auc": approximate_auc(
            score_stats["normal"],
            score_stats["attack"],
        ),
        "threshold_metrics": {
            f"{threshold:.2f}": metrics_at_threshold(
                score_stats,
                threshold,
            )
            for threshold in THRESHOLDS
        },
    }


def traffic_result(traffic):
    window_count = traffic["window_count"]
    return {
        "window_count": window_count,
        "packet_count": traffic["packet_count"],
        "edge_count": traffic["edge_count"],
        "max_packets_per_window": traffic["max_packets_per_window"],
        "max_edges_per_window": traffic["max_edges_per_window"],
        "mean_packets_per_window": (
            traffic["packet_count"] / window_count
            if window_count
            else 0.0
        ),
        "mean_edges_per_window": (
            traffic["edge_count"] / window_count
            if window_count
            else 0.0
        ),
    }


def build_result(state, files, elapsed_seconds, completed):
    overall_scores = state["score_stats"]["overall"]
    return {
        "completed": completed,
        "processed_window_count": state["next_index"],
        "total_window_count": len(files),
        "elapsed_seconds": elapsed_seconds,
        "day_order": list(DAYS),
        "overall": {
            "predicted_counts": dict(state["predicted_counts"]),
            "final_label_metrics": final_label_metrics(
                state["predicted_counts"]
            ),
            **score_scope_result(overall_scores),
            "suspicious_decision_reasons": dict(
                state["suspicious_reason_counts"]
            ),
            "timing": {
                name: timing_summary(values)
                for name, values in state["window_timings"].items()
            },
            "traffic": traffic_result(state["traffic"]),
        },
        "daily": {
            day: {
                "predicted_counts": dict(
                    state["daily_predicted_counts"][day]
                ),
                "final_label_metrics": final_label_metrics(
                    state["daily_predicted_counts"][day]
                ),
                **score_scope_result(state["score_stats"][day]),
                "suspicious_decision_reasons": dict(
                    state["daily_suspicious_reason_counts"][day]
                ),
                "timing": {
                    name: timing_summary(values)
                    for name, values in state[
                        "daily_window_timings"
                    ][day].items()
                },
                "traffic": traffic_result(
                    state["daily_traffic"][day]
                ),
            }
            for day in DAYS
        },
        "parameters": {
            "theta_suspicious": (
                state["history"].suspicious_edge_history.theta_suspicious
            ),
            "theta_attack": (
                state["history"].suspicious_edge_history.theta_attack
            ),
            "attack_chain_threshold": (
                state[
                    "history"
                ].suspicious_edge_history.attack_chain_threshold
            ),
            "ttl_windows": (
                state["history"].suspicious_edge_history.ttl_windows
            ),
        },
        "label_usage": (
            "CSV 真实标签仅在当前窗口预测与历史提交完成后用于离线评估；"
            "模型历史只接收预测标签。"
        ),
        "evaluation_warning": (
            "当前参数曾使用 Wednesday 标签调优，因此 Wednesday 不是独立测试集；"
            "Monday、Tuesday、Thursday、Friday 可用于观察跨日泛化，"
            "但全周 overall 指标仍包含 Wednesday 调参偏差。"
        ),
    }


def write_result(state, files, elapsed_seconds, completed):
    result = build_result(state, files, elapsed_seconds, completed)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    files = ordered_weekday_files()
    state = load_state()
    if state["next_index"] == 0:
        PROGRESS_PATH.write_text("", encoding="utf-8")

    run_started_at = time.perf_counter()
    base_elapsed = float(state.get("started_elapsed_seconds", 0.0))
    log_progress(
        f"start next_index={state['next_index']} total={len(files)}"
    )

    for index in range(state["next_index"], len(files)):
        window_started_at = time.perf_counter()
        path = files[index]
        day = path.name.split("-", 1)[0]

        stage_started_at = time.perf_counter()
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        build_seconds = time.perf_counter() - stage_started_at

        scores, observations, score_timings = score_window(
            graph,
            state["history"],
            return_timing=True,
        )
        threshold_labels = {
            edge_key: predicted_label_from_score(score)
            for edge_key, score in scores.items()
        }

        stage_started_at = time.perf_counter()
        suspicious_result, final_labels = commit_window(
            state["history"],
            graph,
            threshold_labels,
            suspicious_observations=observations,
        )
        commit_seconds = time.perf_counter() - stage_started_at

        evaluation_started_at = time.perf_counter()
        for edge_key, score in scores.items():
            true_label = true_edge_labels.get(edge_key, "normal")
            predicted_label = final_labels[edge_key]
            add_prediction_counts(
                state["predicted_counts"],
                true_label,
                predicted_label,
            )
            add_prediction_counts(
                state["daily_predicted_counts"][day],
                true_label,
                predicted_label,
            )
            state["score_stats"]["overall"][true_label].add(score)
            state["score_stats"][day][true_label].add(score)

        for decision in suspicious_result.decisions.values():
            state["suspicious_reason_counts"][decision.reason] += 1
            state["daily_suspicious_reason_counts"][day][
                decision.reason
            ] += 1
        evaluation_seconds = time.perf_counter() - evaluation_started_at

        total_seconds = time.perf_counter() - window_started_at
        timings = {
            "build_graph": build_seconds,
            **score_timings,
            "commit_history": commit_seconds,
            "evaluation_only": evaluation_seconds,
            "pipeline_without_evaluation": (
                build_seconds
                + score_timings["score_total"]
                + commit_seconds
            ),
            "window_total": total_seconds,
        }
        for name, value in timings.items():
            state["window_timings"][name].append(value)
            state["daily_window_timings"][day][name].append(value)

        for traffic in (
            state["traffic"],
            state["daily_traffic"][day],
        ):
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

        state["next_index"] = index + 1
        elapsed_seconds = (
            base_elapsed + time.perf_counter() - run_started_at
        )
        current_day_finished = (
            index == len(files) - 1
            or files[index + 1].name.split("-", 1)[0] != day
        )
        if index % 100 == 0 or current_day_finished:
            log_progress(
                f"progress index={index + 1}/{len(files)} "
                f"day={day} file={path.name} "
                f"packets={len(graph.packets)} edges={len(graph.edges)} "
                f"window_seconds={total_seconds:.3f} "
                f"elapsed_seconds={elapsed_seconds:.1f}"
            )

        if current_day_finished:
            state["started_elapsed_seconds"] = elapsed_seconds
            save_checkpoint(state)
            write_result(
                state,
                files,
                elapsed_seconds,
                completed=False,
            )
            log_progress(
                f"checkpoint day={day} next_index={state['next_index']}"
            )
    elapsed_seconds = (
        base_elapsed + time.perf_counter() - run_started_at
    )
    state["started_elapsed_seconds"] = elapsed_seconds
    write_result(state, files, elapsed_seconds, completed=True)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    log_progress(
        f"done windows={len(files)} elapsed_seconds={elapsed_seconds:.1f}"
    )


if __name__ == "__main__":
    main()
