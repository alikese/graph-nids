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
        "window_plus_released_counts": Counter(),
        "lifecycle_final_counts": Counter(),
        "daily_predicted_counts": {
            day: Counter() for day in DAYS
        },
        "daily_window_plus_released_counts": {
            day: Counter() for day in DAYS
        },
        "daily_lifecycle_final_counts": {
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
        "suspicious_transition_counts": Counter(),
        "daily_suspicious_transition_counts": {
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
        "suspicious_eval_true_labels": {},
        "suspicious_lifecycle_pending": {},
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
        state = pickle.load(handle)
    state.setdefault("suspicious_transition_counts", Counter())
    state.setdefault("window_plus_released_counts", Counter())
    state.setdefault("lifecycle_final_counts", Counter())
    state.setdefault(
        "daily_suspicious_transition_counts",
        {day: Counter() for day in DAYS},
    )
    state.setdefault(
        "daily_window_plus_released_counts",
        {day: Counter() for day in DAYS},
    )
    state.setdefault(
        "daily_lifecycle_final_counts",
        {day: Counter() for day in DAYS},
    )
    state.setdefault("suspicious_eval_true_labels", {})
    state.setdefault("suspicious_lifecycle_pending", {})
    history = state.get("history")
    suspicious_history = getattr(history, "suspicious_edge_history", None)
    if suspicious_history is not None:
        suspicious_history.ttl_windows = None
    return state


def log_progress(message):
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def add_prediction_counts(counter, true_label, predicted_label):
    add_prediction_count(counter, true_label, predicted_label, 1)


def add_prediction_count(counter, true_label, predicted_label, count):
    counter[f"true_{true_label}"] += count
    counter[f"predicted_{predicted_label}"] += count
    counter[f"{true_label}_as_{predicted_label}"] += count


def resolved_label_metrics(predicted_counts):
    attack_total = predicted_counts["true_attack"]
    normal_total = predicted_counts["true_normal"]
    total = attack_total + normal_total
    predicted_attack = predicted_counts["predicted_attack"]
    predicted_normal = predicted_counts["predicted_normal"]
    true_attack = predicted_counts["attack_as_attack"]
    true_normal = predicted_counts["normal_as_normal"]
    false_attack = predicted_counts["normal_as_attack"]
    false_normal = predicted_counts["attack_as_normal"]
    precision = true_attack / predicted_attack if predicted_attack else 0.0
    recall = true_attack / attack_total if attack_total else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "attack_recall": recall,
        "attack_precision": precision,
        "f1": f1,
        "normal_attack_false_positive_rate": (
            false_attack / normal_total if normal_total else 0.0
        ),
        "attack_false_negative_rate": (
            false_normal / attack_total if attack_total else 0.0
        ),
        "accuracy": (
            (true_attack + true_normal) / total if total else 0.0
        ),
        "resolved_count": total,
        "predicted_attack": predicted_attack,
        "predicted_normal": predicted_normal,
    }


def lifecycle_key(history, edge_key):
    return history.suspicious_edge_history._continuity_key(edge_key)


def add_lifecycle_pending(pending, key, day, true_label):
    record = pending.setdefault(key, {"overall": Counter(), "daily": {}})
    record["overall"][true_label] += 1
    record["daily"].setdefault(day, Counter())[true_label] += 1


def flush_lifecycle_pending(
    pending,
    key,
    predicted_label,
    overall_counter,
    daily_counters,
):
    record = pending.pop(key, None)
    if record is None:
        return 0
    flushed = 0
    for true_label, count in record["overall"].items():
        add_prediction_count(overall_counter, true_label, predicted_label, count)
        flushed += count
    for day, label_counts in record["daily"].items():
        daily_counter = daily_counters[day]
        for true_label, count in label_counts.items():
            add_prediction_count(daily_counter, true_label, predicted_label, count)
    return flushed


def pending_counts(pending):
    counts = Counter()
    for record in pending.values():
        for true_label, count in record["overall"].items():
            counts[f"true_{true_label}"] += count
            counts[f"{true_label}_as_pending"] += count
            counts["pending_total"] += count
    return counts


def pending_counts_for_day(pending, day):
    counts = Counter()
    for record in pending.values():
        for true_label, count in record["daily"].get(day, {}).items():
            counts[f"true_{true_label}"] += count
            counts[f"{true_label}_as_pending"] += count
            counts["pending_total"] += count
    return counts


def add_suspicious_transition_counts(counter, suspicious_result, graph):
    graph_edges = set(graph.edges)
    decision_edges = set(suspicious_result.decisions)
    released_edges = set(suspicious_result.released_normal_edges)
    active_edges = set(suspicious_result.active_suspicious_edges)
    promoted_edges = set(suspicious_result.promoted_attack_edges)

    counter["decision_total"] += len(decision_edges)
    counter["decision_current_window"] += len(decision_edges & graph_edges)
    counter["decision_historical_only"] += len(decision_edges - graph_edges)
    counter["released_normal_total"] += len(released_edges)
    counter["released_normal_current_window"] += len(released_edges & graph_edges)
    counter["released_normal_historical_only"] += len(released_edges - graph_edges)
    counter["active_suspicious_total"] += len(active_edges)
    counter["active_suspicious_current_window"] += len(active_edges & graph_edges)
    counter["active_suspicious_historical_only"] += len(active_edges - graph_edges)
    counter["promoted_attack_total"] += len(promoted_edges)
    counter["promoted_attack_current_window"] += len(promoted_edges & graph_edges)
    counter["promoted_attack_historical_only"] += len(promoted_edges - graph_edges)


def add_historical_release_counts(
    counter,
    daily_counter,
    suspicious_result,
    graph,
    suspicious_eval_true_labels,
):
    graph_edges = set(graph.edges)
    historical_releases = (
        set(suspicious_result.released_normal_edges) - graph_edges
    )
    missing_label_count = 0
    for edge_key in historical_releases:
        true_label = suspicious_eval_true_labels.get(edge_key)
        if true_label is None:
            missing_label_count += 1
            continue
        add_prediction_counts(counter, true_label, "normal")
        add_prediction_counts(daily_counter, true_label, "normal")
    return missing_label_count


def update_suspicious_eval_labels(
    labels,
    suspicious_result,
    graph,
    true_edge_labels,
):
    graph_edges = set(graph.edges)
    for edge_key in set(suspicious_result.active_suspicious_edges) & graph_edges:
        labels[edge_key] = true_edge_labels.get(edge_key, "normal")
    live_suspicious_edges = set(suspicious_result.active_suspicious_edges)
    stale_keys = [edge_key for edge_key in labels if edge_key not in live_suspicious_edges]
    for edge_key in stale_keys:
        labels.pop(edge_key, None)


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
    lifecycle_pending_counts = pending_counts(
        state["suspicious_lifecycle_pending"]
    )
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
            "window_plus_released": {
                "predicted_counts": dict(
                    state["window_plus_released_counts"]
                ),
                "final_label_metrics": final_label_metrics(
                    state["window_plus_released_counts"]
                ),
            },
            "lifecycle_final": {
                "predicted_counts": dict(
                    state["lifecycle_final_counts"]
                ),
                "final_label_metrics": resolved_label_metrics(
                    state["lifecycle_final_counts"]
                ),
                "pending_counts": dict(lifecycle_pending_counts),
            },
            **score_scope_result(overall_scores),
            "suspicious_decision_reasons": dict(
                state["suspicious_reason_counts"]
            ),
            "suspicious_transitions": dict(
                state["suspicious_transition_counts"]
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
                "window_plus_released": {
                    "predicted_counts": dict(
                        state["daily_window_plus_released_counts"][day]
                    ),
                    "final_label_metrics": final_label_metrics(
                        state["daily_window_plus_released_counts"][day]
                    ),
                },
                "lifecycle_final": {
                    "predicted_counts": dict(
                        state["daily_lifecycle_final_counts"][day]
                    ),
                    "final_label_metrics": resolved_label_metrics(
                        state["daily_lifecycle_final_counts"][day]
                    ),
                    "pending_counts": dict(
                        pending_counts_for_day(
                            state["suspicious_lifecycle_pending"],
                            day,
                        )
                    ),
                },
                **score_scope_result(state["score_stats"][day]),
                "suspicious_decision_reasons": dict(
                    state["daily_suspicious_reason_counts"][day]
                ),
                "suspicious_transitions": dict(
                    state["daily_suspicious_transition_counts"][day]
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
            "evidence_decay": (
                state["history"].suspicious_edge_history.evidence_decay
            ),
            "release_threshold": (
                state["history"].suspicious_edge_history.release_threshold
            ),
            "release_policy": (
                "release when decayed evidence_score <= release_threshold; "
                "ttl_windows is disabled"
            ),
            "window_plus_released_metric_note": (
                "window_plus_released adds historical suspicious releases "
                "back into the evaluation matrix using evaluation-only cached "
                "true labels; labels are not used by the model state."
            ),
            "lifecycle_final_metric_note": (
                "lifecycle_final does not count suspicious as a final class. "
                "Suspicious observations are cached and later backfilled into "
                "normal or attack when their lifecycle releases or promotes; "
                "still-active observations are reported as pending and are "
                "excluded from F1."
            ),
            "final_active_suspicious_count": (
                len(state["history"].suspicious_edge_history.edges)
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
            add_prediction_counts(
                state["window_plus_released_counts"],
                true_label,
                predicted_label,
            )
            add_prediction_counts(
                state["daily_window_plus_released_counts"][day],
                true_label,
                predicted_label,
            )
            if predicted_label == "suspicious":
                add_lifecycle_pending(
                    state["suspicious_lifecycle_pending"],
                    lifecycle_key(state["history"], edge_key),
                    day,
                    true_label,
                )
            else:
                add_prediction_counts(
                    state["lifecycle_final_counts"],
                    true_label,
                    predicted_label,
                )
                add_prediction_counts(
                    state["daily_lifecycle_final_counts"][day],
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
        add_suspicious_transition_counts(
            state["suspicious_transition_counts"],
            suspicious_result,
            graph,
        )
        add_suspicious_transition_counts(
            state["daily_suspicious_transition_counts"][day],
            suspicious_result,
            graph,
        )
        missing_release_labels = add_historical_release_counts(
            state["window_plus_released_counts"],
            state["daily_window_plus_released_counts"][day],
            suspicious_result,
            graph,
            state["suspicious_eval_true_labels"],
        )
        if missing_release_labels:
            state["suspicious_transition_counts"][
                "released_normal_missing_eval_label"
            ] += missing_release_labels
            state["daily_suspicious_transition_counts"][day][
                "released_normal_missing_eval_label"
            ] += missing_release_labels
        flushed_release_count = 0
        for edge_key in suspicious_result.released_normal_edges:
            flushed_release_count += flush_lifecycle_pending(
                state["suspicious_lifecycle_pending"],
                lifecycle_key(state["history"], edge_key),
                "normal",
                state["lifecycle_final_counts"],
                state["daily_lifecycle_final_counts"],
            )
        flushed_attack_count = 0
        for edge_key in suspicious_result.promoted_attack_edges:
            flushed_attack_count += flush_lifecycle_pending(
                state["suspicious_lifecycle_pending"],
                lifecycle_key(state["history"], edge_key),
                "attack",
                state["lifecycle_final_counts"],
                state["daily_lifecycle_final_counts"],
            )
        if flushed_release_count:
            state["suspicious_transition_counts"][
                "lifecycle_backfilled_normal"
            ] += flushed_release_count
            state["daily_suspicious_transition_counts"][day][
                "lifecycle_backfilled_normal"
            ] += flushed_release_count
        if flushed_attack_count:
            state["suspicious_transition_counts"][
                "lifecycle_backfilled_attack"
            ] += flushed_attack_count
            state["daily_suspicious_transition_counts"][day][
                "lifecycle_backfilled_attack"
            ] += flushed_attack_count
        update_suspicious_eval_labels(
            state["suspicious_eval_true_labels"],
            suspicious_result,
            graph,
            true_edge_labels,
        )
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
