import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from feature.attack_similar.previous_attack_edge import build_recent_attack_edge_index
from feature.history.historyClass import History
from feature.history.history_feature.active_edge_features import EdgeActiveHistoryFeature
from feature.sum_score import attack_chain_evidence_score, local_anomaly_score
from full_week_score_test import (
    ATTACK_THRESHOLD,
    NORMAL_THRESHOLD,
    ScoreStats,
    build_graph_with_labels,
    commit_window,
    predicted_label_from_score,
    recent_new_edge,
)


ROOT = Path(__file__).resolve().parent
WINDOW_DIR = ROOT / "windows divide"
RESULT_PATH = ROOT / "wednesday_no_label_leakage_result.json"
PROGRESS_PATH = ROOT / "wednesday_no_label_leakage_progress.log"
TEST_START_INDEX = 396
TEST_END_INDEX = 523
WARMUP_WINDOWS = 30
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)


def ordered_wednesday_files():
    return sorted(WINDOW_DIR.glob("Wednesday-*.csv"), key=lambda path: path.name)


def approximate_auc(normal_stats, attack_stats):
    if normal_stats.count <= 0 or attack_stats.count <= 0:
        return None
    normal_below = 0
    wins = 0.0
    for attack_count, normal_count in zip(attack_stats.bins, normal_stats.bins):
        wins += attack_count * (normal_below + 0.5 * normal_count)
        normal_below += normal_count
    return wins / (normal_stats.count * attack_stats.count)


def metrics_at_threshold(scores, threshold):
    normal_total = scores["normal"].count
    attack_total = scores["attack"].count
    false_positive = sum(
        count
        for index, count in enumerate(scores["normal"].bins)
        if index / 1000 >= threshold
    )
    true_positive = sum(
        count
        for index, count in enumerate(scores["attack"].bins)
        if index / 1000 >= threshold
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / attack_total if attack_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "attack_recall": recall,
        "normal_false_positive_rate": (
            false_positive / normal_total if normal_total else 0.0
        ),
        "precision": precision,
        "f1": f1,
    }


def final_label_metrics(predicted_counts):
    attack_total = predicted_counts["true_attack"]
    normal_total = predicted_counts["true_normal"]
    predicted_attack = predicted_counts["predicted_attack"]
    true_attack = predicted_counts["attack_as_attack"]
    false_attack = predicted_counts["normal_as_attack"]
    pending_attack = predicted_counts["attack_as_suspicious"]

    return {
        "attack_recall": true_attack / attack_total if attack_total else 0.0,
        "attack_pending_suspicious_rate": (
            pending_attack / attack_total if attack_total else 0.0
        ),
        "attack_detected_or_pending_rate": (
            (true_attack + pending_attack) / attack_total
            if attack_total
            else 0.0
        ),
        "normal_attack_false_positive_rate": (
            false_attack / normal_total if normal_total else 0.0
        ),
        "normal_suspicious_rate": (
            predicted_counts["normal_as_suspicious"] / normal_total
            if normal_total
            else 0.0
        ),
        "attack_precision": (
            true_attack / predicted_attack if predicted_attack else 0.0
        ),
    }


def compact_stats(stats):
    if stats.count <= 0:
        return {"n": 0}
    return {
        "n": stats.count,
        "mean": stats.total / stats.count,
        "p50": stats.quantile(0.50),
        "p90": stats.quantile(0.90),
        "p99": stats.quantile(0.99),
    }


def timing_summary(values):
    if not values:
        return {"n": 0}
    ordered = sorted(float(value) for value in values)

    def percentile(q):
        index = min(
            max(int(round((len(ordered) - 1) * q)), 0),
            len(ordered) - 1,
        )
        return ordered[index]

    return {
        "n": len(ordered),
        "total_seconds": sum(ordered),
        "mean_seconds": sum(ordered) / len(ordered),
        "p50_seconds": percentile(0.50),
        "p90_seconds": percentile(0.90),
        "p95_seconds": percentile(0.95),
        "p99_seconds": percentile(0.99),
        "max_seconds": ordered[-1],
    }


def confusion_group(true_label, predicted_label):
    if true_label == "attack":
        return {
            "attack": "attack_true_positive",
            "suspicious": "attack_pending_suspicious",
            "unknown": "attack_unknown",
            "normal": "attack_false_negative",
        }[predicted_label]
    return {
        "attack": "normal_false_positive",
        "suspicious": "normal_pending_suspicious",
        "unknown": "normal_unknown",
        "normal": "normal_true_negative",
    }[predicted_label]


def score_window(graph, history, return_timing=False):
    timings = {}
    stage_start = time.perf_counter()
    finite_scores = EdgeActiveHistoryFeature.compute_finite_history_offset_anomaly_scores(
        graph.edges.values(),
        history,
        current_window_only=False,
        return_components=True,
    )
    timings["finite_history"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    novelty_scores = recent_new_edge.compute_approximate_novelty_anomaly_scores(
        graph,
        history,
        return_components=True,
    )
    timings["novelty"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    current_scores = {}
    current_components = {}
    for edge_key, edge_obj in graph.edges.items():
        components = {
            "small_packet_ratio": edge_obj.small_packet_ratio(),
            "zero_payload_ratio": edge_obj.zero_payload_ratio(),
            "syn_without_ack_ratio": edge_obj.syn_without_ack_ratio(),
            "handshake_failure_score": edge_obj.handshake_failure_score(),
            "rst_ratio": edge_obj.rst_ratio(),
            "burstiness_score": edge_obj.burstiness_score(),
            "flags_entropy_score": edge_obj.flags_entropy_score(),
        }
        components["current_behavior_anomaly_score"] = (
            edge_obj.current_behavior_anomaly_score()
        )
        current_components[edge_key] = components
        current_scores[edge_key] = components["current_behavior_anomaly_score"]
    timings["current_behavior"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    role_scores = history.compute_behavior_role_anomaly_scores(
        graph,
        current_window_only=False,
        return_components=True,
    )
    timings["behavior_role"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    attack_index = build_recent_attack_edge_index(history)
    suspicious_diffusion_index = (
        history.suspicious_edge_history.build_destination_to_source_diffusion_index()
    )
    scores = {}
    observations = {}
    for edge_key, edge_obj in graph.edges.items():
        finite_components = finite_scores.get(id(edge_obj), {})
        novelty_components = novelty_scores.get(edge_key, {})
        role_components = role_scores.get(edge_key, {})
        score_components = local_anomaly_score(
            current_scores.get(edge_key, 0.0),
            finite_components.get("finite_history_offset_anomaly_score", 0.0),
            novelty_components.get("approximate_novelty_anomaly_score", 0.0),
            role_components.get("behavior_role_anomaly_score", 0.0),
            return_components=True,
            edge_or_key=edge_key,
            history=history,
            previous_attack_index=attack_index,
        )
        score = score_components["local_anomaly_score"]
        scores[edge_key] = score

        suspicious_diffusion_score = suspicious_diffusion_index.get(
            str(edge_key[0]),
            0.0,
        )
        attack_similarity_score = max(
            score_components["previous_attack_similarity_score"],
            suspicious_diffusion_score,
        )
        structural_expansion_score = max(
            role_components.get("source_out_degree_quantile", 0.0),
            role_components.get("destination_in_degree_quantile", 0.0),
            novelty_components.get(
                "source_destination_diversity_burst_score", 0.0
            ),
            novelty_components.get("source_port_diversity_burst_score", 0.0),
        )
        attack_chain_score = attack_chain_evidence_score(
            previous_attack_dst_to_src_mark=role_components.get(
                "previous_attack_dst_to_src_mark", 0.0
            ),
            edge_low_activity_score=novelty_components.get(
                "edge_low_activity_score", 0.0
            ),
            approximate_rare_edge_score=novelty_components.get(
                "approximate_rare_edge_score", 0.0
            ),
            zero_payload_ratio=current_components[edge_key][
                "zero_payload_ratio"
            ],
            recent_new_edge_mark=novelty_components.get(
                "recent_new_edge_mark", 0.0
            ),
        )
        feature_vector = {
            **current_components[edge_key],
            **finite_components,
            **novelty_components,
            **role_components,
            "suspicious_diffusion_score": suspicious_diffusion_score,
            "attack_chain_score": attack_chain_score,
        }
        observations[edge_key] = {
            "score": score,
            "sub_scores": {
                "current_behavior_anomaly_score": score_components[
                    "current_behavior_anomaly_score"
                ],
                "finite_history_offset_anomaly_score": score_components[
                    "finite_history_offset_anomaly_score"
                ],
                "approximate_novelty_anomaly_score": score_components[
                    "approximate_novelty_anomaly_score"
                ],
                "behavior_role_anomaly_score": score_components[
                    "behavior_role_anomaly_score"
                ],
            },
            "feature_vector": feature_vector,
            "attack_similarity_score": attack_similarity_score,
            "structural_expansion_score": structural_expansion_score,
            "attack_chain_score": attack_chain_score,
        }

    timings["assemble_and_score"] = time.perf_counter() - stage_start
    timings["score_total"] = sum(timings.values())
    if return_timing:
        return scores, observations, timings
    return scores, observations


def main():
    files = ordered_wednesday_files()
    warmup_start = max(TEST_START_INDEX - WARMUP_WINDOWS, 0)
    history = History(life_windows=30, detail_windows=5)
    score_stats = {"normal": ScoreStats(), "attack": ScoreStats()}
    predicted_counts = Counter()
    suspicious_counts = Counter()
    suspicious_reason_counts = Counter()
    evaluation_suspicious_reason_counts = Counter()
    evaluation_reason_label_counts = Counter()
    confusion_feature_stats = defaultdict(lambda: defaultdict(ScoreStats))
    all_window_timings = defaultdict(list)
    warmup_window_timings = defaultdict(list)
    evaluation_window_timings = defaultdict(list)
    all_window_packet_counts = []
    all_window_edge_counts = []
    started_at = time.perf_counter()
    PROGRESS_PATH.write_text("", encoding="utf-8")

    for absolute_index in range(warmup_start, TEST_END_INDEX + 1):
        window_start = time.perf_counter()
        path = files[absolute_index]
        stage_start = time.perf_counter()
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        build_seconds = time.perf_counter() - stage_start
        scores, suspicious_observations, score_timings = score_window(
            graph,
            history,
            return_timing=True,
        )
        score_threshold_labels = {
            edge_key: predicted_label_from_score(score)
            for edge_key, score in scores.items()
        }
        stage_start = time.perf_counter()
        suspicious_result, predicted_edge_labels = commit_window(
            history,
            graph,
            score_threshold_labels,
            suspicious_observations=suspicious_observations,
        )
        commit_seconds = time.perf_counter() - stage_start
        suspicious_counts["promoted_attack"] += len(
            suspicious_result.promoted_attack_edges
        )
        suspicious_counts["released_normal"] += len(
            suspicious_result.released_normal_edges
        )
        suspicious_counts["active_suspicious_total"] += len(
            suspicious_result.active_suspicious_edges
        )
        suspicious_counts["peak_active_suspicious"] = max(
            suspicious_counts["peak_active_suspicious"],
            len(suspicious_result.active_suspicious_edges),
        )
        for decision in suspicious_result.decisions.values():
            suspicious_reason_counts[decision.reason] += 1

        evaluation_start = time.perf_counter()
        if absolute_index >= TEST_START_INDEX:
            for edge_key in graph.edges:
                decision = suspicious_result.decisions.get(edge_key)
                if decision is not None:
                    evaluation_suspicious_reason_counts[decision.reason] += 1
                    true_label = true_edge_labels.get(edge_key, "normal")
                    evaluation_reason_label_counts[
                        f"{decision.reason}:{true_label}"
                    ] += 1
            for edge_key, score in scores.items():
                true_label = true_edge_labels.get(edge_key, "normal")
                predicted_label = predicted_edge_labels[edge_key]
                group = confusion_group(true_label, predicted_label)
                observation = suspicious_observations[edge_key]
                diagnostic_features = {
                    "local_anomaly_score": score,
                    "attack_similarity_score": observation[
                        "attack_similarity_score"
                    ],
                    "structural_expansion_score": observation[
                        "structural_expansion_score"
                    ],
                    **observation["sub_scores"],
                    **observation["feature_vector"],
                }
                for feature_name, feature_value in diagnostic_features.items():
                    if isinstance(feature_value, (int, float)):
                        confusion_feature_stats[group][feature_name].add(
                            feature_value
                        )
                score_stats[true_label].add(score)
                predicted_counts[f"true_{true_label}"] += 1
                predicted_counts[f"predicted_{predicted_label}"] += 1
                predicted_counts[f"{true_label}_as_{predicted_label}"] += 1
        evaluation_seconds = time.perf_counter() - evaluation_start
        total_seconds = time.perf_counter() - window_start

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
            "window_total": total_seconds,
        }
        timing_target = (
            evaluation_window_timings
            if absolute_index >= TEST_START_INDEX
            else warmup_window_timings
        )
        for timing_name, timing_value in window_timings.items():
            all_window_timings[timing_name].append(timing_value)
            timing_target[timing_name].append(timing_value)
        all_window_packet_counts.append(len(graph.packets))
        all_window_edge_counts.append(len(graph.edges))

        if absolute_index % 10 == 0 or absolute_index == TEST_END_INDEX:
            message = (
                f"idx={absolute_index}/{TEST_END_INDEX} file={path.name} "
                f"predicted_attack={predicted_counts['predicted_attack']} "
                f"active_suspicious={len(suspicious_result.active_suspicious_edges)} "
                f"elapsed={time.perf_counter() - started_at:.1f}s"
            )
            with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    result = {
        "test_day": "Wednesday",
        "test_window_index_range": [TEST_START_INDEX, TEST_END_INDEX],
        "warmup_window_index_range": [warmup_start, TEST_START_INDEX - 1],
        "attack_threshold": ATTACK_THRESHOLD,
        "normal_threshold": NORMAL_THRESHOLD,
        "elapsed_seconds": time.perf_counter() - started_at,
        "window_performance": {
            "all_windows": {
                name: timing_summary(values)
                for name, values in all_window_timings.items()
            },
            "warmup_windows": {
                name: timing_summary(values)
                for name, values in warmup_window_timings.items()
            },
            "evaluation_windows": {
                name: timing_summary(values)
                for name, values in evaluation_window_timings.items()
            },
            "traffic_volume": {
                "total_packets": sum(all_window_packet_counts),
                "mean_packets_per_window": (
                    sum(all_window_packet_counts) / len(all_window_packet_counts)
                ),
                "max_packets_per_window": max(all_window_packet_counts),
                "total_edges": sum(all_window_edge_counts),
                "mean_edges_per_window": (
                    sum(all_window_edge_counts) / len(all_window_edge_counts)
                ),
                "max_edges_per_window": max(all_window_edge_counts),
            },
        },
        "predicted_counts": dict(predicted_counts),
        "final_label_metrics": final_label_metrics(predicted_counts),
        "suspicious_edge_counts": dict(suspicious_counts),
        "suspicious_decision_reasons_all_windows": dict(
            suspicious_reason_counts
        ),
        "suspicious_decision_reasons_evaluation_windows": dict(
            evaluation_suspicious_reason_counts
        ),
        "suspicious_reason_true_label_counts": dict(
            evaluation_reason_label_counts
        ),
        "final_active_suspicious_count": len(
            history.suspicious_edge_history.edges
        ),
        "suspicious_parameters": {
            "theta_suspicious": (
                history.suspicious_edge_history.theta_suspicious
            ),
            "theta_attack": history.suspicious_edge_history.theta_attack,
            "ttl_windows": history.suspicious_edge_history.ttl_windows,
            "evidence_decay": history.suspicious_edge_history.evidence_decay,
            "release_threshold": (
                history.suspicious_edge_history.release_threshold
            ),
            "promotion_evidence_threshold": (
                history.suspicious_edge_history.promotion_evidence_threshold
            ),
            "min_consecutive_windows": (
                history.suspicious_edge_history.min_consecutive_windows
            ),
            "min_reinforcing_signals": (
                history.suspicious_edge_history.min_reinforcing_signals
            ),
            "attack_chain_threshold": (
                history.suspicious_edge_history.attack_chain_threshold
            ),
            "suspicious_diffusion_weight": (
                history.suspicious_edge_history.suspicious_diffusion_weight
            ),
            "attack_diffusion_weight": (
                history.suspicious_edge_history.attack_diffusion_weight
            ),
        },
        "confusion_feature_stats": {
            group: {
                feature_name: compact_stats(stats)
                for feature_name, stats in feature_stats.items()
            }
            for group, feature_stats in confusion_feature_stats.items()
        },
        "score_distribution": {
            label: stats.as_dict() for label, stats in score_stats.items()
        },
        "approximate_auc": approximate_auc(
            score_stats["normal"],
            score_stats["attack"],
        ),
        "threshold_metrics": {
            f"{threshold:.2f}": metrics_at_threshold(score_stats, threshold)
            for threshold in THRESHOLDS
        },
        "label_usage": (
            "CSV 真实标签仅用于 score_distribution 和评估统计；"
            "写入攻击历史的标签全部由异常分数阈值产生。"
        ),
        "evaluation_warning": (
            "当前阈值和攻击链公式曾依据 Wednesday 标签分析进行调整；"
            "因此本结果不存在运行期标签注入，但存在测试集调参导致的评估泄露，"
            "必须使用未参与调参的其他日期重新报告最终泛化指标。"
        ),
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
