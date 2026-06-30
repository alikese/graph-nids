from typing import Any, Dict, Hashable, Mapping, Optional

from feature.attack_similar.previous_attack_edge import (
    RecentAttackEdgeIndex,
    build_recent_attack_edge_index,
    previous_attack_edge_similarity,
)


LOCAL_ANOMALY_WEIGHTS = {
    "current_behavior_anomaly_score": 0.35,
    "finite_history_offset_anomaly_score": 0.30,
    "approximate_novelty_anomaly_score": 0.15,
    "behavior_role_anomaly_score": 0.20,
}
DEFAULT_LOCAL_ANOMALY_WEIGHTS = dict(LOCAL_ANOMALY_WEIGHTS)

ATTACK_CHAIN_WEIGHTS = {
    "edge_low_activity_score": 0.15,
    "approximate_rare_edge_score": 0.10,
    "zero_payload_ratio": 0.10,
    "recent_new_edge_mark": 0.10,
}
ATTACK_CHAIN_BASE_SCORE = 0.55


def _clamp_score(score: Any) -> float:
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    return min(max(score_value, 0.0), 1.0)


def _normalized_local_anomaly_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    cleaned = {
        name: max(float(weights.get(name, 0.0)), 0.0)
        for name in DEFAULT_LOCAL_ANOMALY_WEIGHTS
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(DEFAULT_LOCAL_ANOMALY_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def set_local_anomaly_weights(weights: Mapping[str, Any]):
    LOCAL_ANOMALY_WEIGHTS.clear()
    LOCAL_ANOMALY_WEIGHTS.update(_normalized_local_anomaly_weights(weights))


def reset_local_anomaly_weights():
    LOCAL_ANOMALY_WEIGHTS.clear()
    LOCAL_ANOMALY_WEIGHTS.update(DEFAULT_LOCAL_ANOMALY_WEIGHTS)


def local_anomaly_score(
    current_behavior_anomaly_score: float,
    finite_history_offset_anomaly_score: float,
    approximate_novelty_anomaly_score: float,
    behavior_role_anomaly_score: float,
    return_components: bool = False,
    edge_or_key: Any = None,
    history: Any = None,
    require_previous_window_visible: bool = True,
    previous_attack_index: Optional[RecentAttackEdgeIndex] = None,
):
    """
    计算 LocalAnomaly 总分。

    LocalAnomaly =
        0.35 * 当前行为异常分数
        + 0.30 * 有限历史偏移异常分数
        + 0.15 * 近似新颖性异常分数
        + 0.20 * 行为角色异常分数

    最近五窗口攻击边相似度与基础异常分数取最大值:
        - 五元组一致: 1.00
        - 忽略源端口后一致: 0.85
        - 源/目的 IP 与目的端口一致: 0.80
        - 仅源/目的 IP 一致: 0.70

    上一个窗口不衰减；每多间隔一个窗口乘以 0.90。
    """
    components = {
        "current_behavior_anomaly_score": _clamp_score(
            current_behavior_anomaly_score
        ),
        "finite_history_offset_anomaly_score": _clamp_score(
            finite_history_offset_anomaly_score
        ),
        "approximate_novelty_anomaly_score": _clamp_score(
            approximate_novelty_anomaly_score
        ),
        "behavior_role_anomaly_score": _clamp_score(behavior_role_anomaly_score),
    }
    similarity = previous_attack_edge_similarity(
        edge_or_key=edge_or_key,
        history=history,
        attack_index=previous_attack_index,
        require_previous_window_visible=require_previous_window_visible,
    )
    score = sum(
        LOCAL_ANOMALY_WEIGHTS[name] * components[name]
        for name in LOCAL_ANOMALY_WEIGHTS
    )
    score = max(
        min(max(score, 0.0), 1.0),
        float(similarity["score"]),
    )

    if return_components:
        components["local_anomaly_score"] = score
        components["forced_by_previous_attack"] = similarity["score"] > 0.0
        components["previous_attack_similarity_score"] = similarity["score"]
        components["previous_attack_similarity_base_score"] = similarity["base_score"]
        components["previous_attack_match_level"] = similarity["match_level"]
        components["previous_attack_window_distance"] = similarity["window_distance"]
        return components
    return score


def attack_chain_evidence_score(
    previous_attack_dst_to_src_mark: float,
    edge_low_activity_score: float,
    approximate_rare_edge_score: float,
    zero_payload_ratio: float,
    recent_new_edge_mark: float,
) -> float:
    """
    计算攻击阶段传播证据。

    仅当上一窗口攻击目的 IP 成为当前源 IP 时启用；其他特征用于确认该边
    是否同时表现为低活跃、罕见、零负载或新出现通信。
    """
    if _clamp_score(previous_attack_dst_to_src_mark) <= 0.0:
        return 0.0

    components = {
        "edge_low_activity_score": _clamp_score(edge_low_activity_score),
        "approximate_rare_edge_score": _clamp_score(
            approximate_rare_edge_score
        ),
        "zero_payload_ratio": _clamp_score(zero_payload_ratio),
        "recent_new_edge_mark": _clamp_score(recent_new_edge_mark),
    }
    score = ATTACK_CHAIN_BASE_SCORE + sum(
        ATTACK_CHAIN_WEIGHTS[name] * value
        for name, value in components.items()
    )
    return _clamp_score(score)


def local_anomaly_score_from_components(
    components: Mapping[str, Any],
    return_components: bool = False,
    edge_or_key: Any = None,
    history: Any = None,
    require_previous_window_visible: bool = True,
    previous_attack_index: Optional[RecentAttackEdgeIndex] = None,
):
    """从组件字典计算 LocalAnomaly 总分。"""
    return local_anomaly_score(
        current_behavior_anomaly_score=components.get(
            "current_behavior_anomaly_score", 0.0
        ),
        finite_history_offset_anomaly_score=components.get(
            "finite_history_offset_anomaly_score", 0.0
        ),
        approximate_novelty_anomaly_score=components.get(
            "approximate_novelty_anomaly_score", 0.0
        ),
        behavior_role_anomaly_score=components.get("behavior_role_anomaly_score", 0.0),
        return_components=return_components,
        edge_or_key=edge_or_key,
        history=history,
        require_previous_window_visible=require_previous_window_visible,
        previous_attack_index=previous_attack_index,
    )


def compute_local_anomaly_scores(
    components_by_key: Mapping[Hashable, Mapping[str, Any]],
    return_components: bool = False,
    history: Any = None,
    require_previous_window_visible: bool = True,
) -> Dict[Hashable, Any]:
    """批量计算 LocalAnomaly，并应用最近五窗口攻击相似度增强。"""
    previous_attack_index = build_recent_attack_edge_index(
        history,
        require_previous_window_visible=require_previous_window_visible,
    )
    return {
        key: local_anomaly_score_from_components(
            components,
            return_components=return_components,
            edge_or_key=key,
            history=history,
            require_previous_window_visible=require_previous_window_visible,
            previous_attack_index=previous_attack_index,
        )
        for key, components in components_by_key.items()
    }


__all__ = [
    "ATTACK_CHAIN_BASE_SCORE",
    "ATTACK_CHAIN_WEIGHTS",
    "DEFAULT_LOCAL_ANOMALY_WEIGHTS",
    "LOCAL_ANOMALY_WEIGHTS",
    "attack_chain_evidence_score",
    "compute_local_anomaly_scores",
    "local_anomaly_score",
    "local_anomaly_score_from_components",
    "reset_local_anomaly_weights",
    "set_local_anomaly_weights",
]
