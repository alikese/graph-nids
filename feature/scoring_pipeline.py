import importlib.util
import time
from pathlib import Path

from feature.attack_similar.previous_attack_edge import build_recent_attack_edge_index
from feature.auth_bruteforce import compute_auth_bruteforce_scores
from feature.edge.edgeClass import CURRENT_BEHAVIOR_WEIGHTS
from feature.history.historyClass import BEHAVIOR_ROLE_WEIGHTS
from feature.history.history_feature.active_edge_features import EdgeActiveHistoryFeature
from feature.score_profile import (
    active_profile_enabled,
    get_active_score_profile,
    protocol_decision_settings,
    protocol_group_from_edge_key,
    protocol_internal_weights,
    protocol_signal_thresholds,
    protocol_top_level_weights,
    weighted_sum,
)
from feature.sum_score import (
    LOCAL_ANOMALY_WEIGHTS,
    attack_chain_evidence_score,
    local_anomaly_score,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_recent_new_edge_module():
    module_path = ROOT / "feature" / "new score" / "recent_new_edge.py"
    spec = importlib.util.spec_from_file_location("recent_new_edge", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recent_new_edge = _load_recent_new_edge_module()


def score_window(graph, history, return_timing=False):
    timings = {}
    score_profile = get_active_score_profile()
    use_profile = active_profile_enabled()

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
        protocol_group = protocol_group_from_edge_key(edge_key)
        components = {
            "small_packet_ratio": edge_obj.small_packet_ratio(),
            "zero_payload_ratio": edge_obj.zero_payload_ratio(),
            "syn_without_ack_ratio": edge_obj.syn_without_ack_ratio(),
            "handshake_failure_score": edge_obj.handshake_failure_score(),
            "rst_ratio": edge_obj.rst_ratio(),
            "burstiness_score": edge_obj.burstiness_score(),
            "flags_entropy_score": edge_obj.flags_entropy_score(),
        }
        if use_profile:
            current_weights = protocol_internal_weights(
                score_profile,
                protocol_group,
                "current_behavior_anomaly_score",
                CURRENT_BEHAVIOR_WEIGHTS,
            )
            current_score = weighted_sum(components, current_weights)
        else:
            current_score = edge_obj.current_behavior_anomaly_score()
        components["current_behavior_anomaly_score"] = current_score
        current_components[edge_key] = components
        current_scores[edge_key] = current_score
    timings["current_behavior"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    auth_components_by_edge = compute_auth_bruteforce_scores(
        graph,
        history=history,
        return_components=True,
    )
    timings["auth_bruteforce"] = time.perf_counter() - stage_start

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
    has_attack_context = bool(
        history.get_previous_attack_destination_ips()
        or history.recent_attack_edge_history.summary().get(
            "retained_exact_attack_edge_count",
            0,
        )
    )

    scores = {}
    observations = {}
    for edge_key, edge_obj in graph.edges.items():
        protocol_group = protocol_group_from_edge_key(edge_key)
        finite_components = dict(finite_scores.get(id(edge_obj), {}))
        novelty_components = dict(novelty_scores.get(edge_key, {}))
        role_components = dict(role_scores.get(edge_key, {}))
        auth_components = auth_components_by_edge.get(edge_key, {})

        current_behavior_score = current_scores.get(edge_key, 0.0)
        finite_score = finite_components.get(
            "finite_history_offset_anomaly_score",
            0.0,
        )
        novelty_score = novelty_components.get(
            "approximate_novelty_anomaly_score",
            0.0,
        )
        role_score = role_components.get("behavior_role_anomaly_score", 0.0)
        if use_profile:
            finite_score = weighted_sum(
                finite_components,
                protocol_internal_weights(
                    score_profile,
                    protocol_group,
                    "finite_history_offset_anomaly_score",
                    EdgeActiveHistoryFeature.FINITE_HISTORY_OFFSET_WEIGHTS,
                ),
            )
            novelty_score = weighted_sum(
                novelty_components,
                protocol_internal_weights(
                    score_profile,
                    protocol_group,
                    "approximate_novelty_anomaly_score",
                    recent_new_edge.APPROXIMATE_NOVELTY_WEIGHTS,
                ),
            )
            role_score = weighted_sum(
                role_components,
                protocol_internal_weights(
                    score_profile,
                    protocol_group,
                    "behavior_role_anomaly_score",
                    BEHAVIOR_ROLE_WEIGHTS,
                ),
            )
            finite_components["finite_history_offset_anomaly_score"] = finite_score
            novelty_components["approximate_novelty_anomaly_score"] = novelty_score
            role_components["behavior_role_anomaly_score"] = role_score

        score_components = local_anomaly_score(
            current_behavior_score,
            finite_score,
            novelty_score,
            role_score,
            return_components=True,
            edge_or_key=edge_key,
            history=history,
            previous_attack_index=attack_index,
        )
        if use_profile:
            top_weights = protocol_top_level_weights(
                score_profile,
                protocol_group,
                LOCAL_ANOMALY_WEIGHTS,
            )
            base_score = weighted_sum(score_components, top_weights)
            score_components["local_anomaly_score"] = max(
                base_score,
                float(score_components["previous_attack_similarity_score"]),
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
            novelty_components.get("source_destination_diversity_burst_score", 0.0),
            novelty_components.get("source_port_diversity_burst_score", 0.0),
        )
        attack_chain_score = attack_chain_evidence_score(
            previous_attack_dst_to_src_mark=role_components.get(
                "previous_attack_dst_to_src_mark",
                0.0,
            ),
            edge_low_activity_score=novelty_components.get(
                "edge_low_activity_score",
                0.0,
            ),
            approximate_rare_edge_score=novelty_components.get(
                "approximate_rare_edge_score",
                0.0,
            ),
            zero_payload_ratio=current_components[edge_key]["zero_payload_ratio"],
            recent_new_edge_mark=novelty_components.get(
                "recent_new_edge_mark",
                0.0,
            ),
        )
        feature_vector = {
            **current_components[edge_key],
            **finite_components,
            **novelty_components,
            **role_components,
            **auth_components,
            "suspicious_diffusion_score": suspicious_diffusion_score,
            "attack_chain_score": attack_chain_score,
            "has_attack_context": has_attack_context,
            "protocol_group": protocol_group,
        }
        observations[edge_key] = {
            "score": score,
            "protocol_group": protocol_group,
            "sub_scores": {
                "local_anomaly_score": score,
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
                "auth_bruteforce_score": auth_components.get(
                    "auth_bruteforce_score",
                    0.0,
                ),
            },
            "feature_vector": feature_vector,
            "attack_similarity_score": attack_similarity_score,
            "structural_expansion_score": structural_expansion_score,
            "attack_chain_score": attack_chain_score,
            "decision": protocol_decision_settings(
                score_profile,
                protocol_group,
            ) if use_profile else {},
            "signal_thresholds": protocol_signal_thresholds(
                score_profile,
                protocol_group,
            ) if use_profile else None,
        }

    timings["assemble_and_score"] = time.perf_counter() - stage_start
    timings["score_total"] = sum(timings.values())
    if return_timing:
        return scores, observations, timings
    return scores, observations


def commit_window(
    history,
    graph,
    predicted_edge_labels,
    suspicious_observations=None,
):
    final_edge_labels = dict(predicted_edge_labels)
    suspicious_result = None
    if suspicious_observations is not None:
        suspicious_result = history.suspicious_edge_history.update_window(
            suspicious_observations
        )
        for edge_key, decision in suspicious_result.decisions.items():
            if edge_key in graph.edges:
                final_edge_labels[edge_key] = decision.label
        for edge_key in suspicious_result.active_suspicious_edges:
            if edge_key in graph.edges:
                final_edge_labels[edge_key] = "suspicious"
        for edge_key in suspicious_result.promoted_attack_edges:
            final_edge_labels[edge_key] = "attack"
        for edge_key in suspicious_result.released_normal_edges:
            final_edge_labels[edge_key] = "normal"

        history.set_baseline_excluded_edges(
            suspicious_result.active_suspicious_edges
            | suspicious_result.promoted_attack_edges
        )
    else:
        history.set_baseline_excluded_edges(
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        )

    history.update_with_graph(graph)
    history.advance_window()
    history.update_node_behavior_role_vectors(current_window_only=True)
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        ],
        "attack",
    )
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "normal"
        ],
        "normal",
    )
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label in {"unknown", "suspicious"}
        ],
        "unknown",
    )
    if suspicious_result is None:
        strong_attack_edges = {
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        }
    else:
        strong_attack_edges = {
            edge_key
            for edge_key, decision in suspicious_result.decisions.items()
            if decision.reason
            in {
                "score_reached_attack_threshold",
                "auth_bruteforce_evidence",
                "multi_window_evidence_strengthened",
            }
        }
    history.record_previous_attack_destinations(
        attack_edge_keys=strong_attack_edges
    )
    return suspicious_result, final_edge_labels


__all__ = ["commit_window", "recent_new_edge", "score_window"]
