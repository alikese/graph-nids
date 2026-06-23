from typing import Any, Mapping, Optional


DEFAULT_SIGNAL_THRESHOLDS = {
    "current_behavior_anomaly_score": 0.45,
    "approximate_novelty_anomaly_score": 0.65,
    "behavior_role_anomaly_score": 0.70,
    "attack_similarity_score": 0.70,
    "attack_chain_score": 0.85,
    "suspicious_diffusion_score": 0.45,
}


def _score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, 0.0), 1.0)


def _threshold(
    thresholds: Optional[Mapping[str, Any]],
    name: str,
) -> float:
    source = thresholds or DEFAULT_SIGNAL_THRESHOLDS
    return _score(source.get(name, DEFAULT_SIGNAL_THRESHOLDS[name]))


def strong_general_signal_count(
    observation: Mapping[str, Any],
    signal_thresholds: Optional[Mapping[str, Any]] = None,
) -> int:
    sub_scores = observation.get("sub_scores", {}) or {}
    feature_vector = observation.get("feature_vector", {}) or {}
    attack_similarity_score = _score(observation.get("attack_similarity_score", 0.0))
    attack_chain_score = _score(observation.get("attack_chain_score", 0.0))

    signals = (
        _score(sub_scores.get("current_behavior_anomaly_score", 0.0))
        >= _threshold(signal_thresholds, "current_behavior_anomaly_score"),
        _score(sub_scores.get("approximate_novelty_anomaly_score", 0.0))
        >= _threshold(signal_thresholds, "approximate_novelty_anomaly_score"),
        _score(sub_scores.get("behavior_role_anomaly_score", 0.0))
        >= _threshold(signal_thresholds, "behavior_role_anomaly_score"),
        attack_similarity_score >= _threshold(
            signal_thresholds,
            "attack_similarity_score",
        ),
        attack_chain_score >= _threshold(
            signal_thresholds,
            "attack_chain_score",
        ),
        _score(feature_vector.get("suspicious_diffusion_score", 0.0))
        >= _threshold(signal_thresholds, "suspicious_diffusion_score"),
    )
    return sum(1 for value in signals if value)


def evidence_decision(
    observation: Mapping[str, Any],
    attack_threshold: float = 0.70,
    suspicious_threshold: float = 0.30,
    normal_threshold: float = 0.30,
    min_strong_signals: int = 2,
    auth_attack_threshold: float = 0.80,
    auth_suspicious_threshold: float = 0.60,
    signal_thresholds: Optional[Mapping[str, Any]] = None,
) -> dict:
    sub_scores = observation.get("sub_scores", {}) or {}
    local_score = _score(
        sub_scores.get(
            "local_anomaly_score",
            observation.get("local_anomaly_score", observation.get("score", 0.0)),
        )
    )
    display_score = _score(observation.get("score", local_score))
    auth_score = _score(sub_scores.get("auth_bruteforce_score", 0.0))
    strong_count = strong_general_signal_count(
        observation,
        signal_thresholds=signal_thresholds,
    )

    if auth_score >= auth_attack_threshold:
        return {
            "label": "attack",
            "reason": "auth_bruteforce_high_confidence",
            "strong_general_signal_count": strong_count,
        }

    if local_score >= attack_threshold and strong_count >= min_strong_signals:
        return {
            "label": "attack",
            "reason": "general_multi_signal",
            "strong_general_signal_count": strong_count,
        }

    if auth_score >= auth_suspicious_threshold:
        return {
            "label": "suspicious",
            "reason": "auth_bruteforce_suspicious",
            "strong_general_signal_count": strong_count,
        }

    if local_score >= suspicious_threshold or strong_count >= min_strong_signals:
        return {
            "label": "suspicious",
            "reason": "general_suspicious",
            "strong_general_signal_count": strong_count,
        }

    if display_score <= normal_threshold:
        return {
            "label": "normal",
            "reason": "below_normal_threshold",
            "strong_general_signal_count": strong_count,
        }

    return {
        "label": "unknown",
        "reason": "insufficient_evidence",
        "strong_general_signal_count": strong_count,
    }


def predicted_label_from_observation(
    observation: Mapping[str, Any],
    attack_threshold: float = 0.70,
    suspicious_threshold: float = 0.30,
    normal_threshold: float = 0.30,
    min_strong_signals: int = 2,
    auth_attack_threshold: float = 0.80,
    auth_suspicious_threshold: float = 0.60,
    signal_thresholds: Optional[Mapping[str, Any]] = None,
) -> str:
    return evidence_decision(
        observation,
        attack_threshold=attack_threshold,
        suspicious_threshold=suspicious_threshold,
        normal_threshold=normal_threshold,
        min_strong_signals=min_strong_signals,
        auth_attack_threshold=auth_attack_threshold,
        auth_suspicious_threshold=auth_suspicious_threshold,
        signal_thresholds=signal_thresholds,
    )["label"]


__all__ = [
    "DEFAULT_SIGNAL_THRESHOLDS",
    "evidence_decision",
    "predicted_label_from_observation",
    "strong_general_signal_count",
]
