import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional


PROTOCOL_GROUPS = ("TCP", "UDP", "ICMP", "OTHER")
DEFAULT_DECISION_THRESHOLDS = {
    "attack_threshold": 0.70,
    "suspicious_threshold": 0.30,
    "normal_threshold": 0.30,
    "min_strong_signals": 2,
    "auth_attack_threshold": 0.80,
    "auth_suspicious_threshold": 0.60,
}

_ACTIVE_SCORE_PROFILE: Optional[dict] = None


def protocol_group_from_number(protocol: Any) -> str:
    try:
        protocol_number = int(protocol)
    except (TypeError, ValueError):
        return "OTHER"
    if protocol_number == 6:
        return "TCP"
    if protocol_number == 17:
        return "UDP"
    if protocol_number in {1, 58}:
        return "ICMP"
    return "OTHER"


def protocol_group_from_edge_key(edge_key: Any) -> str:
    if isinstance(edge_key, tuple) and len(edge_key) >= 5:
        return protocol_group_from_number(edge_key[4])
    return "OTHER"


def normalize_weights(weights: Mapping[str, Any], defaults: Mapping[str, float]) -> dict:
    cleaned = {
        name: max(float(weights.get(name, defaults.get(name, 0.0))), 0.0)
        for name in defaults
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(defaults)
    return {name: value / total for name, value in cleaned.items()}


def weighted_sum(components: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    score = 0.0
    for name, weight in weights.items():
        try:
            value = float(components.get(name, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        score += float(weight) * min(max(value, 0.0), 1.0)
    return min(max(score, 0.0), 1.0)


def load_score_profile(path: Any) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_active_score_profile(profile: Optional[Mapping[str, Any]]):
    global _ACTIVE_SCORE_PROFILE
    _ACTIVE_SCORE_PROFILE = deepcopy(dict(profile)) if profile is not None else None


def load_active_score_profile(path: Any) -> dict:
    profile = load_score_profile(path)
    set_active_score_profile(profile)
    return profile


def reset_active_score_profile():
    set_active_score_profile(None)


def get_active_score_profile() -> Optional[dict]:
    return _ACTIVE_SCORE_PROFILE


def active_profile_enabled() -> bool:
    return _ACTIVE_SCORE_PROFILE is not None


def protocol_profile(profile: Optional[Mapping[str, Any]], group: str) -> dict:
    if not profile:
        return {}
    profiles = profile.get("protocol_profiles", {}) or {}
    return dict(profiles.get(group) or profiles.get("OTHER") or {})


def protocol_internal_weights(
    profile: Optional[Mapping[str, Any]],
    group: str,
    component_name: str,
    defaults: Mapping[str, float],
) -> dict:
    group_profile = protocol_profile(profile, group)
    internal = group_profile.get("internal_weights", {}) or {}
    return normalize_weights(internal.get(component_name, {}), defaults)


def protocol_top_level_weights(
    profile: Optional[Mapping[str, Any]],
    group: str,
    defaults: Mapping[str, float],
) -> dict:
    group_profile = protocol_profile(profile, group)
    return normalize_weights(group_profile.get("top_level_weights", {}), defaults)


def protocol_decision_settings(
    profile: Optional[Mapping[str, Any]],
    group: str,
) -> dict:
    settings = dict(DEFAULT_DECISION_THRESHOLDS)
    group_profile = protocol_profile(profile, group)
    settings.update(group_profile.get("decision", {}) or {})
    settings.update(group_profile.get("thresholds", {}) or {})
    return settings


def protocol_signal_thresholds(
    profile: Optional[Mapping[str, Any]],
    group: str,
) -> Optional[dict]:
    group_profile = protocol_profile(profile, group)
    thresholds = group_profile.get("signal_thresholds")
    return dict(thresholds) if thresholds else None


__all__ = [
    "DEFAULT_DECISION_THRESHOLDS",
    "PROTOCOL_GROUPS",
    "active_profile_enabled",
    "get_active_score_profile",
    "load_active_score_profile",
    "load_score_profile",
    "normalize_weights",
    "protocol_decision_settings",
    "protocol_group_from_edge_key",
    "protocol_group_from_number",
    "protocol_internal_weights",
    "protocol_profile",
    "protocol_signal_thresholds",
    "protocol_top_level_weights",
    "reset_active_score_profile",
    "set_active_score_profile",
    "weighted_sum",
]
