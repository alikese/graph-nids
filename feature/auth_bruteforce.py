from bisect import bisect_right
from typing import Any, Dict, Hashable, Mapping, Optional


AUTH_BRUTEFORCE_PORTS = {21, 22}
TCP_PROTOCOL = 6


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, 0.0), 1.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quantile_maps_by_port(
    stats_by_key: Mapping[Hashable, Mapping[str, Any]],
) -> Dict[int, Dict[str, Dict[Hashable, float]]]:
    values_by_port: Dict[int, Dict[str, Dict[Hashable, float]]] = {}
    for auth_key, stats in stats_by_key.items():
        service_port = int(stats.get("service_port", -1))
        sessions = stats.get("sessions", {}) or {}
        session_count = len(sessions)
        client_packets = int(stats.get("client_packets", 0) or 0)
        values_by_port.setdefault(
            service_port,
            {
                "session_count": {},
                "client_packets": {},
            },
        )
        values_by_port[service_port]["session_count"][auth_key] = session_count
        values_by_port[service_port]["client_packets"][auth_key] = client_packets

    quantiles_by_port: Dict[int, Dict[str, Dict[Hashable, float]]] = {}
    for service_port, metric_values in values_by_port.items():
        quantiles_by_port[service_port] = {}
        for metric_name, values in metric_values.items():
            ordered = sorted(values.values())
            total = len(ordered)
            quantiles_by_port[service_port][metric_name] = {
                key: bisect_right(ordered, value) / total
                for key, value in values.items()
            }
    return quantiles_by_port


def _recent_window_score(edge_key: Hashable, history: Any, max_windows: int = 3) -> float:
    if history is None:
        return 0.0
    recent_details = getattr(history, "recent_edge_details", None)
    if recent_details is None or not hasattr(
        recent_details,
        "recent_edge_occurrence_count",
    ):
        return 0.0
    try:
        count = recent_details.recent_edge_occurrence_count(edge_key, max_windows)
    except (TypeError, ValueError):
        count = 0
    return _clamp(count / max_windows)


def _session_components(stats: Mapping[str, Any], history: Any) -> Dict[str, float]:
    sessions = list((stats.get("sessions", {}) or {}).values())
    session_count = len(sessions)
    packet_count = max(
        int(stats.get("client_packets", 0) or 0)
        + int(stats.get("server_packets", 0) or 0),
        1,
    )
    if session_count <= 0:
        return {
            "auth_session_count": 0.0,
            "auth_attempt_count_score": 0.0,
            "auth_unique_src_port_score": 0.0,
            "auth_short_session_ratio": 0.0,
            "auth_low_bytes_per_attempt_score": 0.0,
            "auth_repeated_window_score": 0.0,
            "auth_attempt_burstiness_score": 0.0,
            "auth_rst_or_fin_score": 0.0,
        }

    durations = [
        max(
            _safe_float(session.get("last_ts")) - _safe_float(session.get("first_ts")),
            0.0,
        )
        for session in sessions
    ]
    bytes_per_session = [
        _safe_float(session.get("byte_count", 0.0))
        for session in sessions
    ]
    first_timestamps = sorted(
        _safe_float(session.get("first_ts"))
        for session in sessions
    )
    short_sessions = sum(
        1
        for session, duration in zip(sessions, durations)
        if duration <= 2.0 or int(session.get("packet_count", 0) or 0) <= 8
    )
    rst_or_fin_sessions = sum(
        1
        for session in sessions
        if int(session.get("rst_packets", 0) or 0) > 0
        or int(session.get("fin_packets", 0) or 0) > 0
    )

    if len(first_timestamps) <= 1:
        burstiness_score = 0.0
    else:
        burst_gaps = sum(
            1
            for idx in range(1, len(first_timestamps))
            if first_timestamps[idx] - first_timestamps[idx - 1] <= 1.0
        )
        burstiness_score = burst_gaps / (len(first_timestamps) - 1)

    mean_bytes_per_session = sum(bytes_per_session) / session_count
    low_bytes_score = 1.0 - min(mean_bytes_per_session / 4000.0, 1.0)
    edge_key = stats.get("service_edge_key", stats.get("auth_key"))

    return {
        "auth_session_count": float(session_count),
        "auth_attempt_count_score": _clamp(session_count / 20.0),
        "auth_unique_src_port_score": _clamp(session_count / 20.0),
        "auth_short_session_ratio": _clamp(short_sessions / session_count),
        "auth_low_bytes_per_attempt_score": _clamp(low_bytes_score),
        "auth_repeated_window_score": _recent_window_score(edge_key, history),
        "auth_attempt_burstiness_score": _clamp(burstiness_score),
        "auth_rst_or_fin_score": _clamp(rst_or_fin_sessions / session_count),
        "auth_zero_payload_ratio": _clamp(
            int(stats.get("zero_payload_packets", 0) or 0) / packet_count
        ),
        "auth_small_payload_ratio": _clamp(
            int(stats.get("small_payload_packets", 0) or 0) / packet_count
        ),
    }


def auth_bruteforce_score_from_components(
    components: Mapping[str, Any],
    service_port: Optional[int] = None,
) -> float:
    if service_port not in AUTH_BRUTEFORCE_PORTS:
        return 0.0
    if float(components.get("auth_session_count", 0.0)) < 3.0:
        return 0.0

    score = 0.0
    score += 0.25 * _clamp(components.get("auth_attempt_count_score", 0.0))
    score += 0.20 * _clamp(components.get("auth_service_attempt_quantile", 0.0))
    score += 0.15 * _clamp(components.get("auth_short_session_ratio", 0.0))
    score += 0.15 * _clamp(
        components.get("auth_low_bytes_per_attempt_score", 0.0)
    )
    score += 0.15 * _clamp(components.get("auth_repeated_window_score", 0.0))
    score += 0.10 * max(
        _clamp(components.get("auth_attempt_burstiness_score", 0.0)),
        _clamp(components.get("auth_rst_or_fin_score", 0.0)),
    )
    return _clamp(score)


def compute_auth_bruteforce_scores(
    graph_or_stats: Any,
    history: Any = None,
    return_components: bool = False,
) -> Dict[Hashable, Any]:
    stats_by_key = getattr(graph_or_stats, "auth_service_stats", graph_or_stats)
    if not stats_by_key:
        return {}

    quantiles_by_port = _quantile_maps_by_port(stats_by_key)
    results: Dict[Hashable, Any] = {}
    for auth_key, stats in stats_by_key.items():
        service_port = int(stats.get("service_port", -1))
        protocol = int(stats.get("protocol", -1))
        edge_key = stats.get("service_edge_key", auth_key)
        if protocol != TCP_PROTOCOL or service_port not in AUTH_BRUTEFORCE_PORTS:
            continue

        components = _session_components(stats, history)
        port_quantiles = quantiles_by_port.get(service_port, {})
        components["auth_service_attempt_quantile"] = port_quantiles.get(
            "session_count",
            {},
        ).get(auth_key, 0.0)
        components["auth_service_client_packet_quantile"] = port_quantiles.get(
            "client_packets",
            {},
        ).get(auth_key, 0.0)
        components["auth_service_port_mark"] = 1.0
        score = auth_bruteforce_score_from_components(
            components,
            service_port=service_port,
        )
        components["auth_bruteforce_score"] = score

        result_value = components if return_components else score
        edge_keys = stats.get("edge_keys")
        if edge_keys:
            for concrete_edge_key in edge_keys:
                results[concrete_edge_key] = result_value
        else:
            results[edge_key] = result_value
    return results


__all__ = [
    "AUTH_BRUTEFORCE_PORTS",
    "auth_bruteforce_score_from_components",
    "compute_auth_bruteforce_scores",
]
