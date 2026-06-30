import math
from collections.abc import Hashable as HashableABC
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Set

from feature.attack_similar.previous_attack_edge import edge_key_from_edge_obj


APPROXIMATE_NOVELTY_WEIGHTS = {
    "recent_new_edge_mark": 0.25,
    "edge_low_activity_score": 0.25,
    "approximate_rare_edge_score": 0.25,
    "source_destination_diversity_burst_score": 0.15,
    "source_port_diversity_burst_score": 0.10,
}
DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS = dict(APPROXIMATE_NOVELTY_WEIGHTS)


def _normalized_approximate_novelty_weights(weights: Mapping[str, Any]) -> Dict[str, float]:
    cleaned = {
        name: max(float(weights.get(name, 0.0)), 0.0)
        for name in DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def set_approximate_novelty_weights(weights: Mapping[str, Any]):
    APPROXIMATE_NOVELTY_WEIGHTS.clear()
    APPROXIMATE_NOVELTY_WEIGHTS.update(_normalized_approximate_novelty_weights(weights))


def reset_approximate_novelty_weights():
    APPROXIMATE_NOVELTY_WEIGHTS.clear()
    APPROXIMATE_NOVELTY_WEIGHTS.update(DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS)


def _resolve_edge_key(edge_or_key: Any) -> Optional[Hashable]:
    if isinstance(edge_or_key, tuple):
        return edge_or_key
    return edge_key_from_edge_obj(edge_or_key)


def _resolve_node_key(node_or_key: Any) -> Optional[Hashable]:
    if node_or_key is None:
        return None
    node_ip = getattr(node_or_key, "ip", None)
    if node_ip is not None:
        return str(node_ip)
    if isinstance(node_or_key, (str, int)):
        return str(node_or_key)
    return node_or_key if isinstance(node_or_key, HashableABC) else None


def _iter_edge_inputs(graph_or_edges: Any) -> Iterable[tuple[Any, Any]]:
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return []
    if isinstance(edges, Mapping):
        return edges.items()
    return ((None, edge_obj) for edge_obj in edges)


def _iter_node_inputs(graph_or_nodes: Any) -> Iterable[tuple[Any, Any]]:
    nodes = getattr(graph_or_nodes, "nodes", None)
    if nodes is None:
        return []
    if isinstance(nodes, Mapping):
        return nodes.items()
    return ((None, node_obj) for node_obj in nodes)


def _edge_source_and_destination_port(
    edge_key: Any,
    edge_obj: Any,
) -> tuple[Optional[str], Optional[Any]]:
    src_ip = getattr(edge_obj, "src_ip", None)
    dst_port = getattr(edge_obj, "dst_port", None)
    if src_ip is not None and dst_port is not None:
        return str(src_ip), dst_port

    resolved_key = edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
    if isinstance(resolved_key, tuple):
        if len(resolved_key) >= 5:
            return str(resolved_key[0]), resolved_key[3]
        if len(resolved_key) >= 4:
            return str(resolved_key[0]), resolved_key[2]

    return None, None


def _edge_source_and_destination_ip(
    edge_key: Any,
    edge_obj: Any,
) -> tuple[Optional[str], Optional[str]]:
    src_ip = getattr(edge_obj, "src_ip", None)
    dst_ip = getattr(edge_obj, "dst_ip", None)
    if src_ip is not None and dst_ip is not None:
        return str(src_ip), str(dst_ip)

    resolved_key = edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
    if isinstance(resolved_key, tuple) and len(resolved_key) >= 2:
        return str(resolved_key[0]), str(resolved_key[1])

    return None, None


def _edge_source_key(edge_or_key: Any) -> Optional[str]:
    if isinstance(edge_or_key, tuple):
        source_key, _ = _edge_source_and_destination_ip(edge_or_key, None)
    else:
        source_key, _ = _edge_source_and_destination_ip(None, edge_or_key)
    return source_key


def _source_port_sets_from_edges(graph_or_edges: Any) -> Dict[Hashable, Set[Any]]:
    source_ports: Dict[Hashable, Set[Any]] = {}
    for edge_key, edge_obj in _iter_edge_inputs(graph_or_edges):
        src_ip, dst_port = _edge_source_and_destination_port(edge_key, edge_obj)
        if src_ip is None or dst_port is None:
            continue
        source_ports.setdefault(src_ip, set()).add(dst_port)
    return source_ports


def _source_destination_sets_from_edges(graph_or_edges: Any) -> Dict[Hashable, Set[str]]:
    source_destinations: Dict[Hashable, Set[str]] = {}
    for edge_key, edge_obj in _iter_edge_inputs(graph_or_edges):
        src_ip, dst_ip = _edge_source_and_destination_ip(edge_key, edge_obj)
        if src_ip is None or dst_ip is None:
            continue
        source_destinations.setdefault(src_ip, set()).add(dst_ip)
    return source_destinations


def _source_out_degree_counts(graph_or_edges: Any) -> Dict[Hashable, int]:
    out_degree_counts: Dict[Hashable, int] = {}
    for edge_key, edge_obj in _iter_edge_inputs(graph_or_edges):
        src_ip, _ = _edge_source_and_destination_ip(edge_key, edge_obj)
        if src_ip is None:
            continue
        out_degree_counts[src_ip] = out_degree_counts.get(src_ip, 0) + 1

    for node_key, node_obj in _iter_node_inputs(graph_or_edges):
        resolved_key = (
            _resolve_node_key(node_key)
            if node_key is not None
            else _resolve_node_key(node_obj)
        )
        if resolved_key is None:
            continue
        out_degree = getattr(node_obj, "out_degree", None)
        if callable(out_degree):
            out_degree = out_degree()
        if out_degree is not None:
            try:
                out_degree_counts[resolved_key] = int(out_degree)
            except (TypeError, ValueError):
                continue

    return out_degree_counts


def _current_source_node_keys(graph_or_edges: Any) -> Set[Hashable]:
    source_keys = set(_source_port_sets_from_edges(graph_or_edges).keys())
    source_keys.update(_source_destination_sets_from_edges(graph_or_edges).keys())
    for node_key, node_obj in _iter_node_inputs(graph_or_edges):
        resolved_key = (
            _resolve_node_key(node_key)
            if node_key is not None
            else _resolve_node_key(node_obj)
        )
        if resolved_key is not None:
            source_keys.add(resolved_key)
    return source_keys


def _source_port_count_quantile(
    source_key: Hashable,
    graph_or_edges: Any,
) -> float:
    source_ports = _source_port_sets_from_edges(graph_or_edges)
    source_keys = _current_source_node_keys(graph_or_edges)
    source_keys.add(source_key)
    if not source_keys:
        source_keys = set(source_ports.keys())

    current_count = len(source_ports.get(source_key, set()))
    counts = [len(source_ports.get(key, set())) for key in source_keys]
    if not counts:
        return 0.0

    less_or_equal_count = sum(1 for count in counts if count <= current_count)
    return less_or_equal_count / len(counts)


def _source_out_degree_quantile(
    source_key: Hashable,
    graph_or_edges: Any,
) -> float:
    out_degree_counts = _source_out_degree_counts(graph_or_edges)
    source_keys = _current_source_node_keys(graph_or_edges)
    source_keys.add(source_key)

    current_count = out_degree_counts.get(source_key, 0)
    counts = [out_degree_counts.get(key, 0) for key in source_keys]
    if not counts:
        return 0.0

    less_or_equal_count = sum(1 for count in counts if count <= current_count)
    return less_or_equal_count / len(counts)


def _window_contains_node(
    window_nodes: Mapping[Hashable, Any],
    source_key: Hashable,
) -> bool:
    for node_key, node_obj in window_nodes.items():
        resolved_key = (
            _resolve_node_key(node_key)
            if node_key is not None
            else _resolve_node_key(node_obj)
        )
        if resolved_key == source_key:
            return True
    return False


def _recent_source_port_count_samples(
    source_key: Hashable,
    history: Any,
    max_detail_windows: int,
) -> list[int]:
    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is None:
        return []

    edge_windows = list(getattr(recent_edge_details, "windows", []))[
        -max_detail_windows:
    ]
    node_windows = list(getattr(recent_edge_details, "node_windows", []))[
        -max_detail_windows:
    ]
    window_count = max(len(edge_windows), len(node_windows))
    samples = []

    for index in range(window_count):
        edge_index = index - (window_count - len(edge_windows))
        node_index = index - (window_count - len(node_windows))
        window_edges = edge_windows[edge_index] if edge_index >= 0 else {}
        window_nodes = node_windows[node_index] if node_index >= 0 else {}
        source_ports = _source_port_sets_from_edges(window_edges)

        if source_key not in source_ports and not _window_contains_node(
            window_nodes,
            source_key,
        ):
            continue

        samples.append(len(source_ports.get(source_key, set())))

    return samples


def _recent_source_destination_count_samples(
    source_key: Hashable,
    history: Any,
    max_detail_windows: int,
) -> list[int]:
    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is None:
        return []

    edge_windows = list(getattr(recent_edge_details, "windows", []))[
        -max_detail_windows:
    ]
    node_windows = list(getattr(recent_edge_details, "node_windows", []))[
        -max_detail_windows:
    ]
    window_count = max(len(edge_windows), len(node_windows))
    samples = []

    for index in range(window_count):
        edge_index = index - (window_count - len(edge_windows))
        node_index = index - (window_count - len(node_windows))
        window_edges = edge_windows[edge_index] if edge_index >= 0 else {}
        window_nodes = node_windows[node_index] if node_index >= 0 else {}
        source_destinations = _source_destination_sets_from_edges(window_edges)

        if source_key not in source_destinations and not _window_contains_node(
            window_nodes,
            source_key,
        ):
            continue

        samples.append(len(source_destinations.get(source_key, set())))

    return samples


def _recent_source_diversity_samples(
    history: Any,
    max_detail_windows: int,
) -> tuple[Dict[Hashable, list[int]], Dict[Hashable, list[int]]]:
    """一次遍历近窗明细，生成所有源节点的目的 IP 数和目的端口数样本。"""
    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is None:
        return {}, {}

    edge_windows = list(getattr(recent_edge_details, "windows", []))[
        -max_detail_windows:
    ]
    node_windows = list(getattr(recent_edge_details, "node_windows", []))[
        -max_detail_windows:
    ]
    window_count = max(len(edge_windows), len(node_windows))
    destination_samples: Dict[Hashable, list[int]] = {}
    port_samples: Dict[Hashable, list[int]] = {}

    for index in range(window_count):
        edge_index = index - (window_count - len(edge_windows))
        node_index = index - (window_count - len(node_windows))
        window_edges = edge_windows[edge_index] if edge_index >= 0 else {}
        window_nodes = node_windows[node_index] if node_index >= 0 else {}
        source_destinations = _source_destination_sets_from_edges(window_edges)
        source_ports = _source_port_sets_from_edges(window_edges)
        source_keys = set(source_destinations)
        source_keys.update(source_ports)

        for node_key, node_obj in window_nodes.items():
            resolved_key = (
                _resolve_node_key(node_key)
                if node_key is not None
                else _resolve_node_key(node_obj)
            )
            if resolved_key is not None:
                source_keys.add(resolved_key)

        for source_key in source_keys:
            destination_samples.setdefault(source_key, []).append(
                len(source_destinations.get(source_key, set()))
            )
            port_samples.setdefault(source_key, []).append(
                len(source_ports.get(source_key, set()))
            )

    return destination_samples, port_samples


def _ema(values: list[float], alpha: float) -> float:
    ema_value = float(values[0])
    for value in values[1:]:
        ema_value = alpha * float(value) + (1.0 - alpha) * ema_value
    return ema_value


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _positive_z_norm(z_score: float) -> float:
    z_score = max(float(z_score), 0.0)
    return z_score / (1.0 + z_score)


def _segment_has_details(segment: Any) -> bool:
    if not isinstance(segment, dict):
        return bool(segment)

    details = segment.get("details", segment)
    if not isinstance(details, dict):
        return bool(details)

    return any(bool(value) for value in details.values())


def _edge_has_recent_detail(edge_obj: Any, max_detail_windows: int) -> bool:
    segments = getattr(edge_obj, "_detail_window_segments", None)
    if segments is None:
        return bool(getattr(edge_obj, "timestamp_list", None))

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if int(segment.get("age", 0)) < max_detail_windows and _segment_has_details(
            segment
        ):
            return True

    return False


def _edge_recent_detail_frequency(edge_obj: Any, max_detail_windows: int) -> int:
    segments = getattr(edge_obj, "_detail_window_segments", None)
    if segments is None:
        return int(bool(getattr(edge_obj, "timestamp_list", None)))

    frequency = 0
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if int(segment.get("age", 0)) < max_detail_windows and _segment_has_details(
            segment
        ):
            frequency += 1
    return frequency


def recent_detail_edge_keys(
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> Set[Hashable]:
    """
    返回最近 max_detail_windows 个已提交窗口保留的边 key 集合。

    优先读取 History.recent_edge_details；旧版 History/Edge 的明细片段仅作为兼容回退。
    默认窗口数来自 recent_edge_details.max_detail_windows 或 history.detail_windows。
    """
    if history is None:
        return set()

    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is not None and hasattr(
        recent_edge_details, "recent_edge_keys"
    ):
        return set(recent_edge_details.recent_edge_keys(max_detail_windows))

    if hasattr(history, "recent_edge_keys"):
        return set(history.recent_edge_keys(max_detail_windows))

    if max_detail_windows is None:
        max_detail_windows = int(getattr(history, "detail_windows", 5))
    if max_detail_windows <= 0:
        raise ValueError("max_detail_windows must be greater than 0.")

    recent_keys = set()
    for key, record in getattr(history, "edges", {}).items():
        edge_obj = getattr(record, "obj", None)
        if edge_obj is None:
            continue
        if _edge_has_recent_detail(edge_obj, max_detail_windows):
            recent_keys.add(key)

    return recent_keys


def recent_edge_frequency_estimate(
    edge_or_key: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> float:
    """
    估计边在长期历史保留窗口中的出现比例。

    freq_est(e) = appeared_window_count(e) / retained_window_time
    其中 retained_window_time 来自 History 当前有效保留的窗口时间，最大不超过 life_windows。
    max_detail_windows 仅作为旧接口兼容参数保留，不参与该计算。
    """
    _ = max_detail_windows
    edge_key = _resolve_edge_key(edge_or_key)
    if edge_key is None or history is None:
        return 0.0

    record = getattr(history, "edges", {}).get(edge_key)
    if record is None:
        return 0.0

    retained_window_time = int(getattr(history, "retained_window_time", 0) or 0)
    if retained_window_time <= 0:
        retained_window_time = int(
            getattr(history, "total_committed_window_count", 0) or 0
        )
    if retained_window_time <= 0:
        return 0.0

    appeared_window_count = int(getattr(record, "appeared_window_count", 0) or 0)
    frequency = appeared_window_count / retained_window_time
    return min(max(float(frequency), 0.0), 1.0)


def approximate_rare_edge_score(
    edge_or_key: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> float:
    """
    计算近似罕见边得分。

    score(e) = 1 / (1 + log(1 + freq_est(e)))
    freq_est(e) 为长期历史中的出现比例；越少见分数越接近 1。
    """
    frequency = recent_edge_frequency_estimate(
        edge_or_key,
        history,
        max_detail_windows,
    )
    return 1.0 / (1.0 + math.log1p(frequency))


def _resolve_recent_window_count(
    history: Any,
    recent_window_count: Optional[int] = None,
) -> int:
    if recent_window_count is None:
        recent_edge_details = getattr(history, "recent_edge_details", None)
        recent_window_count = int(
            getattr(
                recent_edge_details,
                "max_detail_windows",
                getattr(history, "detail_windows", 5),
            )
        )
    if recent_window_count <= 0:
        raise ValueError("recent_window_count must be greater than 0.")
    return recent_window_count


def edge_recent_appearance_rate(
    edge_or_key: Any,
    history: Any,
    recent_window_count: Optional[int] = None,
) -> float:
    """
    计算边最近出现率。

    边最近出现率(e) = 最近 K 个详细窗口中出现 e 的窗口数 / K。
    若历史中没有该边或没有近期明细，则出现窗口数按 0 处理。
    """
    edge_key = _resolve_edge_key(edge_or_key)
    if edge_key is None or history is None:
        return 0.0

    recent_window_count = _resolve_recent_window_count(history, recent_window_count)
    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is not None and hasattr(
        recent_edge_details,
        "recent_edge_occurrence_count",
    ):
        appeared_count = recent_edge_details.recent_edge_occurrence_count(
            edge_key,
            recent_window_count,
        )
        return min(max(appeared_count / recent_window_count, 0.0), 1.0)

    if recent_edge_details is not None:
        edge_windows = list(getattr(recent_edge_details, "windows", []))[
            -recent_window_count:
        ]
        appeared_count = sum(
            1 for window_edges in edge_windows if edge_key in window_edges
        )
        return min(max(appeared_count / recent_window_count, 0.0), 1.0)

    record = getattr(history, "edges", {}).get(edge_key)
    edge_obj = getattr(record, "obj", None)
    appeared_count = (
        0
        if edge_obj is None
        else _edge_recent_detail_frequency(edge_obj, recent_window_count)
    )
    return min(max(appeared_count / recent_window_count, 0.0), 1.0)


def edge_low_activity_score(
    edge_or_key: Any,
    history: Any,
    recent_window_count: Optional[int] = None,
) -> float:
    """
    计算边低活跃度得分。

    score(e) = 1 - 边最近出现率(e)。
    因此最近 K 个窗口中完全未出现的边，低活跃度得分为 1。
    """
    appearance_rate = edge_recent_appearance_rate(
        edge_or_key,
        history,
        recent_window_count,
    )
    return min(max(1.0 - appearance_rate, 0.0), 1.0)


def source_port_diversity_burst_score(
    source_or_node: Any,
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
) -> float:
    """
    计算源节点端口多样性突增得分。

    若源节点在最近详细窗口中有样本：
        z = (当前目的端口数(s) - ema_unique_dst_port_count(s))
            / (sqrt(var_unique_dst_port_count(s)) + eps)
        score = Z_norm(z)，其中 Z_norm 只保留正向突增。

    若最近详细窗口中没有该源节点样本：
        score = 当前源节点目的端口数分位数。
    """
    source_key = _resolve_node_key(source_or_node)
    if source_key is None or history is None:
        return 0.0
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    if max_detail_windows is None:
        recent_edge_details = getattr(history, "recent_edge_details", None)
        max_detail_windows = int(
            getattr(
                recent_edge_details,
                "max_detail_windows",
                getattr(history, "detail_windows", 5),
            )
        )
    if max_detail_windows <= 0:
        raise ValueError("max_detail_windows must be greater than 0.")

    current_port_count = len(
        _source_port_sets_from_edges(graph_or_edges).get(source_key, set())
    )
    samples = _recent_source_port_count_samples(
        source_key,
        history,
        max_detail_windows,
    )
    if not samples:
        return _source_port_count_quantile(source_key, graph_or_edges)

    sample_values = [float(value) for value in samples]
    ema_unique_dst_port_count = _ema(sample_values, alpha)
    var_unique_dst_port_count = _variance(sample_values)
    z_port_diversity = (
        float(current_port_count) - ema_unique_dst_port_count
    ) / (math.sqrt(var_unique_dst_port_count) + eps)
    return _positive_z_norm(z_port_diversity)


def port_diversity_burst_score(*args, **kwargs):
    """source_port_diversity_burst_score 的短别名。"""
    return source_port_diversity_burst_score(*args, **kwargs)


def source_destination_diversity_burst_score(
    source_or_node: Any,
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
) -> float:
    """
    计算源节点目的多样性突增得分。

    若源节点在最近详细窗口中有样本：
        z = (当前目的 IP 数(s) - ema_unique_dst_count(s))
            / (sqrt(var_unique_dst_count(s)) + eps)
        score = Z_norm(z)，其中 Z_norm 只保留正向突增。

    若最近详细窗口中没有该源节点样本：
        score = 当前源节点出度分位数。
    """
    source_key = _resolve_node_key(source_or_node)
    if source_key is None or history is None:
        return 0.0
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1].")
    if eps <= 0:
        raise ValueError("eps must be greater than 0.")

    if max_detail_windows is None:
        recent_edge_details = getattr(history, "recent_edge_details", None)
        max_detail_windows = int(
            getattr(
                recent_edge_details,
                "max_detail_windows",
                getattr(history, "detail_windows", 5),
            )
        )
    if max_detail_windows <= 0:
        raise ValueError("max_detail_windows must be greater than 0.")

    current_destination_count = len(
        _source_destination_sets_from_edges(graph_or_edges).get(source_key, set())
    )
    samples = _recent_source_destination_count_samples(
        source_key,
        history,
        max_detail_windows,
    )
    if not samples:
        return _source_out_degree_quantile(source_key, graph_or_edges)

    sample_values = [float(value) for value in samples]
    ema_unique_dst_count = _ema(sample_values, alpha)
    var_unique_dst_count = _variance(sample_values)
    z_dst_diversity = (float(current_destination_count) - ema_unique_dst_count) / (
        math.sqrt(var_unique_dst_count) + eps
    )
    return _positive_z_norm(z_dst_diversity)


def destination_diversity_burst_score(*args, **kwargs):
    """source_destination_diversity_burst_score 的短别名。"""
    return source_destination_diversity_burst_score(*args, **kwargs)


def compute_source_port_diversity_burst_scores(
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
) -> Dict[Hashable, float]:
    """批量计算当前窗口源节点的端口多样性突增得分。"""
    return {
        source_key: source_port_diversity_burst_score(
            source_key,
            graph_or_edges,
            history,
            max_detail_windows,
            alpha,
            eps,
        )
        for source_key in _current_source_node_keys(graph_or_edges)
    }


def compute_source_destination_diversity_burst_scores(
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
) -> Dict[Hashable, float]:
    """批量计算当前窗口源节点的目的多样性突增得分。"""
    return {
        source_key: source_destination_diversity_burst_score(
            source_key,
            graph_or_edges,
            history,
            max_detail_windows,
            alpha,
            eps,
        )
        for source_key in _current_source_node_keys(graph_or_edges)
    }


def recent_new_edge_mark(
    edge_or_key: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> int:
    """
    近期新边标记。

    当前边不在 RecentEdgeDetails 保留的最近窗口边明细中，返回 1；
    已在最近窗口出现过则返回 0。该判断不受 History TTL 刷新影响。
    """
    edge_key = _resolve_edge_key(edge_or_key)
    if edge_key is None:
        return 1

    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is not None and hasattr(
        recent_edge_details, "recent_new_edge_mark"
    ):
        return int(recent_edge_details.recent_new_edge_mark(edge_key, max_detail_windows))

    if hasattr(history, "recent_new_edge_mark"):
        return int(history.recent_new_edge_mark(edge_key, max_detail_windows))

    return int(edge_key not in recent_detail_edge_keys(history, max_detail_windows))


def approximate_novelty_anomaly_score(
    edge_or_key: Any,
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    recent_window_count: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
    return_components: bool = False,
):
    """
    计算边的近似新颖性异常总分。

    近期新服务组标记已取消，其原权重 0.15 均分到前三项：
        score =
            0.25 * 近期新边标记
            + 0.25 * 边低活跃度得分
            + 0.25 * 近似罕见边得分
            + 0.15 * 目的多样性突增得分
            + 0.10 * 端口多样性突增得分
    """
    source_key = _edge_source_key(edge_or_key)
    destination_diversity_score = 0.0
    port_diversity_score = 0.0
    if source_key is not None:
        destination_diversity_score = source_destination_diversity_burst_score(
            source_key,
            graph_or_edges,
            history,
            max_detail_windows,
            alpha,
            eps,
        )
        port_diversity_score = source_port_diversity_burst_score(
            source_key,
            graph_or_edges,
            history,
            max_detail_windows,
            alpha,
            eps,
        )

    components = {
        "recent_new_edge_mark": float(
            recent_new_edge_mark(edge_or_key, history, max_detail_windows)
        ),
        "edge_low_activity_score": edge_low_activity_score(
            edge_or_key,
            history,
            recent_window_count,
        ),
        "approximate_rare_edge_score": approximate_rare_edge_score(
            edge_or_key,
            history,
            max_detail_windows,
        ),
        "source_destination_diversity_burst_score": destination_diversity_score,
        "source_port_diversity_burst_score": port_diversity_score,
    }
    score = sum(
        APPROXIMATE_NOVELTY_WEIGHTS[name] * components[name]
        for name in APPROXIMATE_NOVELTY_WEIGHTS
    )
    score = min(max(float(score), 0.0), 1.0)

    if return_components:
        components["approximate_novelty_anomaly_score"] = score
        return components
    return score


def compute_approximate_rare_edge_scores(
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> Dict[Hashable, float]:
    """
    批量计算当前窗口边的近似罕见得分。

    参数可传 TrafficGraph，或直接传 graph.edges 这样的 key->edge 映射。
    计算基准为 History.edges 中的 appeared_window_count / retained_window_time。
    max_detail_windows 仅作为旧接口兼容参数保留。
    """
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return {}

    scores = {}
    if isinstance(edges, Mapping):
        for edge_key, edge_obj in edges.items():
            resolved_key = (
                edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
            )
            if resolved_key is None:
                continue
            scores[resolved_key] = approximate_rare_edge_score(
                resolved_key,
                history,
                max_detail_windows,
            )
    else:
        for edge_obj in edges:
            resolved_key = _resolve_edge_key(edge_obj)
            if resolved_key is None:
                continue
            scores[resolved_key] = approximate_rare_edge_score(
                resolved_key,
                history,
                max_detail_windows,
            )

    return scores


def compute_approximate_novelty_anomaly_scores(
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
    recent_window_count: Optional[int] = None,
    alpha: float = 0.3,
    eps: float = 1e-8,
    return_components: bool = False,
) -> Dict[Hashable, Any]:
    """
    批量计算当前窗口边的近似新颖性异常总分。

    参数可传 TrafficGraph，或直接传 graph.edges 这样的 key->edge 映射。
    """
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return {}

    if max_detail_windows is None:
        recent_edge_details = getattr(history, "recent_edge_details", None)
        max_detail_windows = int(
            getattr(
                recent_edge_details,
                "max_detail_windows",
                getattr(history, "detail_windows", 5),
            )
        )
    if max_detail_windows <= 0:
        raise ValueError("max_detail_windows must be greater than 0.")

    if recent_window_count is None:
        recent_window_count = max_detail_windows
    recent_window_count = _resolve_recent_window_count(history, recent_window_count)

    current_destination_sets = _source_destination_sets_from_edges(graph_or_edges)
    current_port_sets = _source_port_sets_from_edges(graph_or_edges)
    source_keys = _current_source_node_keys(graph_or_edges)
    destination_samples, port_samples = _recent_source_diversity_samples(
        history,
        max_detail_windows,
    )
    source_out_degree_counts = _source_out_degree_counts(graph_or_edges)
    source_port_counts = {
        source_key: len(current_port_sets.get(source_key, set()))
        for source_key in source_keys
    }

    def quantile_from_counts(
        source_key: Hashable,
        counts_by_source: Mapping[Hashable, int],
    ) -> float:
        keys = set(source_keys)
        keys.add(source_key)
        counts = [counts_by_source.get(key, 0) for key in keys]
        if not counts:
            return 0.0
        current_count = counts_by_source.get(source_key, 0)
        less_or_equal_count = sum(1 for count in counts if count <= current_count)
        return less_or_equal_count / len(counts)

    destination_diversity_by_source = {}
    port_diversity_by_source = {}
    for source_key in source_keys:
        current_destination_count = len(current_destination_sets.get(source_key, set()))
        destination_history = destination_samples.get(source_key, [])
        if destination_history:
            values = [float(value) for value in destination_history]
            ema_value = _ema(values, alpha)
            variance = _variance(values)
            z_score = (float(current_destination_count) - ema_value) / (
                math.sqrt(variance) + eps
            )
            destination_diversity_by_source[source_key] = _positive_z_norm(z_score)
        else:
            destination_diversity_by_source[source_key] = quantile_from_counts(
                source_key,
                source_out_degree_counts,
            )

        current_port_count = len(current_port_sets.get(source_key, set()))
        port_history = port_samples.get(source_key, [])
        if port_history:
            values = [float(value) for value in port_history]
            ema_value = _ema(values, alpha)
            variance = _variance(values)
            z_score = (float(current_port_count) - ema_value) / (
                math.sqrt(variance) + eps
            )
            port_diversity_by_source[source_key] = _positive_z_norm(z_score)
        else:
            port_diversity_by_source[source_key] = quantile_from_counts(
                source_key,
                source_port_counts,
            )

    scores: Dict[Hashable, Any] = {}
    if isinstance(edges, Mapping):
        for edge_key, edge_obj in edges.items():
            resolved_key = (
                edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
            )
            if resolved_key is None:
                continue
            source_key = _edge_source_key(resolved_key if edge_key is not None else edge_obj)
            components = {
                "recent_new_edge_mark": float(
                    recent_new_edge_mark(resolved_key, history, max_detail_windows)
                ),
                "edge_low_activity_score": edge_low_activity_score(
                    resolved_key,
                    history,
                    recent_window_count,
                ),
                "approximate_rare_edge_score": approximate_rare_edge_score(
                    resolved_key,
                    history,
                    max_detail_windows,
                ),
                "source_destination_diversity_burst_score": (
                    destination_diversity_by_source.get(source_key, 0.0)
                ),
                "source_port_diversity_burst_score": port_diversity_by_source.get(
                    source_key,
                    0.0,
                ),
            }
            score = sum(
                APPROXIMATE_NOVELTY_WEIGHTS[name] * components[name]
                for name in APPROXIMATE_NOVELTY_WEIGHTS
            )
            score = min(max(float(score), 0.0), 1.0)
            if return_components:
                components["approximate_novelty_anomaly_score"] = score
                scores[resolved_key] = components
            else:
                scores[resolved_key] = score
    else:
        for edge_obj in edges:
            resolved_key = _resolve_edge_key(edge_obj)
            if resolved_key is None:
                continue
            source_key = _edge_source_key(edge_obj)
            components = {
                "recent_new_edge_mark": float(
                    recent_new_edge_mark(resolved_key, history, max_detail_windows)
                ),
                "edge_low_activity_score": edge_low_activity_score(
                    resolved_key,
                    history,
                    recent_window_count,
                ),
                "approximate_rare_edge_score": approximate_rare_edge_score(
                    resolved_key,
                    history,
                    max_detail_windows,
                ),
                "source_destination_diversity_burst_score": (
                    destination_diversity_by_source.get(source_key, 0.0)
                ),
                "source_port_diversity_burst_score": port_diversity_by_source.get(
                    source_key,
                    0.0,
                ),
            }
            score = sum(
                APPROXIMATE_NOVELTY_WEIGHTS[name] * components[name]
                for name in APPROXIMATE_NOVELTY_WEIGHTS
            )
            score = min(max(float(score), 0.0), 1.0)
            if return_components:
                components["approximate_novelty_anomaly_score"] = score
                scores[resolved_key] = components
            else:
                scores[resolved_key] = score

    return scores


def compute_edge_low_activity_scores(
    graph_or_edges: Any,
    history: Any,
    recent_window_count: Optional[int] = None,
) -> Dict[Hashable, float]:
    """
    批量计算当前窗口边的低活跃度得分。

    参数可传 TrafficGraph，或直接传 graph.edges 这样的 key->edge 映射。
    score(e) = 1 - 最近 K 个详细窗口中出现 e 的窗口数 / K。
    """
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return {}

    scores = {}
    if isinstance(edges, Mapping):
        for edge_key, edge_obj in edges.items():
            resolved_key = (
                edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
            )
            if resolved_key is None:
                continue
            scores[resolved_key] = edge_low_activity_score(
                resolved_key,
                history,
                recent_window_count,
            )
    else:
        for edge_obj in edges:
            resolved_key = _resolve_edge_key(edge_obj)
            if resolved_key is None:
                continue
            scores[resolved_key] = edge_low_activity_score(
                resolved_key,
                history,
                recent_window_count,
            )

    return scores


def compute_recent_new_edge_marks(
    graph_or_edges: Any,
    history: Any,
    max_detail_windows: Optional[int] = None,
) -> Dict[Hashable, int]:
    """
    批量计算当前窗口边的近期新边标记。

    参数可传 TrafficGraph，或直接传 graph.edges 这样的 key->edge 映射。
    应针对尚未进入 RecentEdgeDetails 的当前窗口计算；若使用
    History.process_new_window(graph)，可在调用后立刻对同一个 graph 计算，
    因为该 graph 此时只是暂存，还未写入近窗明细队列。
    """
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return {}

    recent_edge_details = getattr(history, "recent_edge_details", None)
    if recent_edge_details is not None and hasattr(
        recent_edge_details, "compute_recent_new_edge_marks"
    ):
        return recent_edge_details.compute_recent_new_edge_marks(
            graph_or_edges,
            max_detail_windows,
        )

    if hasattr(history, "compute_recent_new_edge_marks"):
        return history.compute_recent_new_edge_marks(graph_or_edges, max_detail_windows)

    recent_keys = recent_detail_edge_keys(history, max_detail_windows)
    marks = {}
    if isinstance(edges, Mapping):
        for edge_key, edge_obj in edges.items():
            resolved_key = (
                edge_key if edge_key is not None else _resolve_edge_key(edge_obj)
            )
            if resolved_key is None:
                continue
            marks[resolved_key] = int(resolved_key not in recent_keys)
    else:
        for edge_obj in edges:
            resolved_key = _resolve_edge_key(edge_obj)
            if resolved_key is None:
                continue
            marks[resolved_key] = int(resolved_key not in recent_keys)

    return marks


__all__ = [
    "APPROXIMATE_NOVELTY_WEIGHTS",
    "DEFAULT_APPROXIMATE_NOVELTY_WEIGHTS",
    "approximate_novelty_anomaly_score",
    "approximate_rare_edge_score",
    "compute_approximate_novelty_anomaly_scores",
    "compute_approximate_rare_edge_scores",
    "compute_edge_low_activity_scores",
    "compute_recent_new_edge_marks",
    "compute_source_destination_diversity_burst_scores",
    "compute_source_port_diversity_burst_scores",
    "destination_diversity_burst_score",
    "edge_low_activity_score",
    "edge_recent_appearance_rate",
    "port_diversity_burst_score",
    "recent_detail_edge_keys",
    "recent_edge_frequency_estimate",
    "recent_new_edge_mark",
    "reset_approximate_novelty_weights",
    "set_approximate_novelty_weights",
    "source_destination_diversity_burst_score",
    "source_port_diversity_burst_score",
]
