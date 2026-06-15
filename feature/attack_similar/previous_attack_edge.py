from collections import deque
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)


EdgeKey = Tuple[str, str, int, int, int]
EdgeInput = Union[Hashable, Any]
EdgeWithoutSourcePort = Tuple[str, str, int, int]
EndpointServiceKey = Tuple[str, str, int]
EndpointPair = Tuple[str, str]

ATTACK_SIMILARITY_BASE_SCORES = {
    "exact_five_tuple": 1.00,
    "without_source_port": 0.85,
    "source_destination_service": 0.80,
    "source_destination": 0.70,
}
DEFAULT_ATTACK_HISTORY_WINDOWS = 5
DEFAULT_WINDOW_DISTANCE_DECAY = 0.90


@dataclass(frozen=True)
class PreviousAttackEdgeIndex:
    """单个历史窗口攻击边的分级匹配索引。"""

    exact_edges: Set[EdgeKey]
    edges_without_source_port: Set[EdgeWithoutSourcePort]
    endpoint_services: Set[EndpointServiceKey]
    endpoint_pairs: Set[EndpointPair]


@dataclass(frozen=True)
class RecentAttackEdgeIndex:
    """最近多个窗口的攻击边索引，窗口按最近到最远排列。"""

    window_indexes: Tuple[PreviousAttackEdgeIndex, ...]
    distance_decay: float = DEFAULT_WINDOW_DISTANCE_DECAY


def _dominant_protocol(edge_obj: Any) -> Optional[int]:
    protocol = getattr(edge_obj, "protocol", None)
    if protocol is not None:
        return int(protocol)

    protocol_list = getattr(edge_obj, "protocol_list", None)
    if protocol_list:
        return int(protocol_list[0])

    protocol_count_map = getattr(edge_obj, "protocol_count_map", None)
    if protocol_count_map:
        return int(max(protocol_count_map.items(), key=lambda item: item[1])[0])

    return None


def edge_key_from_edge_obj(edge_obj: Any) -> Optional[EdgeKey]:
    """从边对象恢复标准五元组 edge_key。"""
    if edge_obj is None:
        return None

    try:
        protocol = _dominant_protocol(edge_obj)
        if protocol is None:
            return None
        return (
            str(edge_obj.src_ip),
            str(edge_obj.dst_ip),
            int(edge_obj.src_port),
            int(edge_obj.dst_port),
            protocol,
        )
    except (AttributeError, TypeError, ValueError):
        return None


def _resolve_edge_key(edge_or_key: EdgeInput) -> Optional[Hashable]:
    if isinstance(edge_or_key, tuple):
        return edge_or_key
    return edge_key_from_edge_obj(edge_or_key)


def _normalize_edge_key(edge_or_key: EdgeInput) -> Optional[EdgeKey]:
    edge_key = _resolve_edge_key(edge_or_key)
    if not isinstance(edge_key, tuple) or len(edge_key) != 5:
        return None
    try:
        return (
            str(edge_key[0]),
            str(edge_key[1]),
            int(edge_key[2]),
            int(edge_key[3]),
            int(edge_key[4]),
        )
    except (TypeError, ValueError):
        return None


def _empty_window_index() -> PreviousAttackEdgeIndex:
    return PreviousAttackEdgeIndex(set(), set(), set(), set())


def build_attack_window_index(
    attack_edges: Iterable[EdgeInput],
) -> PreviousAttackEdgeIndex:
    """从单个窗口攻击边构建分级索引。"""
    exact_edges: Set[EdgeKey] = set()
    edges_without_source_port: Set[EdgeWithoutSourcePort] = set()
    endpoint_services: Set[EndpointServiceKey] = set()
    endpoint_pairs: Set[EndpointPair] = set()

    for raw_edge in attack_edges:
        edge_key = _normalize_edge_key(raw_edge)
        if edge_key is None:
            continue
        src_ip, dst_ip, _, dst_port, protocol = edge_key
        exact_edges.add(edge_key)
        edges_without_source_port.add((src_ip, dst_ip, dst_port, protocol))
        endpoint_services.add((src_ip, dst_ip, dst_port))
        endpoint_pairs.add((src_ip, dst_ip))

    return PreviousAttackEdgeIndex(
        exact_edges=exact_edges,
        edges_without_source_port=edges_without_source_port,
        endpoint_services=endpoint_services,
        endpoint_pairs=endpoint_pairs,
    )


class RecentAttackEdgeHistory:
    """独立保留最近五个窗口攻击边，不使用长期 History TTL。"""

    def __init__(
        self,
        max_windows: int = DEFAULT_ATTACK_HISTORY_WINDOWS,
        distance_decay: float = DEFAULT_WINDOW_DISTANCE_DECAY,
    ):
        if max_windows <= 0:
            raise ValueError("max_windows 必须大于 0。")
        if not 0.0 < distance_decay <= 1.0:
            raise ValueError("distance_decay 必须在 (0, 1] 范围内。")
        self.max_windows = max_windows
        self.distance_decay = distance_decay
        self.windows: Deque[PreviousAttackEdgeIndex] = deque(maxlen=max_windows)

    def record_window(self, attack_edges: Iterable[EdgeInput]):
        """提交一个窗口；空攻击窗口也会占用一个时间位置。"""
        self.windows.appendleft(build_attack_window_index(attack_edges))

    def build_index(self) -> RecentAttackEdgeIndex:
        return RecentAttackEdgeIndex(
            window_indexes=tuple(self.windows),
            distance_decay=self.distance_decay,
        )

    def clear(self):
        self.windows.clear()

    def summary(self) -> Dict[str, int]:
        return {
            "retained_window_count": len(self.windows),
            "max_windows": self.max_windows,
            "retained_exact_attack_edge_count": sum(
                len(index.exact_edges) for index in self.windows
            ),
        }


def build_previous_attack_edge_index(
    history: Any,
    require_previous_window_visible: bool = True,
) -> PreviousAttackEdgeIndex:
    """兼容入口：仅构建刚提交的上一个窗口攻击边索引。"""
    attack_edges = []
    for raw_edge_key, record in getattr(history, "edges", {}).items():
        if require_previous_window_visible and not getattr(
            record, "in_current_graph", False
        ):
            continue
        if getattr(record, "edge_label", "unknown") == "attack":
            attack_edges.append(raw_edge_key)
    return build_attack_window_index(attack_edges)


def build_recent_attack_edge_index(
    history: Any,
    require_previous_window_visible: bool = True,
) -> RecentAttackEdgeIndex:
    """构建最近五窗口攻击边索引；旧 History 自动回退到上一窗口。"""
    recent_history = getattr(history, "recent_attack_edge_history", None)
    if recent_history is not None:
        return recent_history.build_index()

    previous_index = build_previous_attack_edge_index(
        history,
        require_previous_window_visible=require_previous_window_visible,
    )
    return RecentAttackEdgeIndex((previous_index,))


def _window_match(
    edge_key: EdgeKey,
    attack_index: PreviousAttackEdgeIndex,
) -> Tuple[float, str]:
    src_ip, dst_ip, _, dst_port, protocol = edge_key
    if edge_key in attack_index.exact_edges:
        return 1.00, "exact_five_tuple"
    if (
        src_ip,
        dst_ip,
        dst_port,
        protocol,
    ) in attack_index.edges_without_source_port:
        return 0.85, "without_source_port"
    if (src_ip, dst_ip, dst_port) in attack_index.endpoint_services:
        return 0.80, "source_destination_service"
    if (src_ip, dst_ip) in attack_index.endpoint_pairs:
        return 0.70, "source_destination"
    return 0.0, "none"


def previous_attack_edge_similarity(
    edge_or_key: EdgeInput,
    history: Any = None,
    attack_index: Optional[
        Union[PreviousAttackEdgeIndex, RecentAttackEdgeIndex]
    ] = None,
    require_previous_window_visible: bool = True,
) -> Dict[str, Any]:
    """
    返回当前边在最近五个攻击窗口索引中的最高衰减相似度。

    距离为 1 的上一个窗口不衰减；每多间隔一个窗口乘以 0.9。
    最终相似度 = 匹配基础分 × 0.9^(窗口距离 - 1)。
    """
    edge_key = _normalize_edge_key(edge_or_key)
    if edge_key is None:
        return {
            "score": 0.0,
            "base_score": 0.0,
            "match_level": "none",
            "window_distance": None,
        }

    if attack_index is None:
        attack_index = build_recent_attack_edge_index(
            history,
            require_previous_window_visible=require_previous_window_visible,
        )
    if isinstance(attack_index, PreviousAttackEdgeIndex):
        attack_index = RecentAttackEdgeIndex((attack_index,))

    best_score = 0.0
    best_base_score = 0.0
    best_match_level = "none"
    best_window_distance = None
    for window_distance, window_index in enumerate(
        attack_index.window_indexes,
        start=1,
    ):
        base_score, match_level = _window_match(edge_key, window_index)
        if base_score <= 0.0:
            continue
        decayed_score = base_score * (
            attack_index.distance_decay ** (window_distance - 1)
        )
        if decayed_score > best_score:
            best_score = min(max(decayed_score, 0.0), 1.0)
            best_base_score = base_score
            best_match_level = match_level
            best_window_distance = window_distance
    return {
        "score": best_score,
        "base_score": best_base_score,
        "match_level": best_match_level,
        "window_distance": best_window_distance,
    }


def previous_attack_edge_similarity_score(
    edge_or_key: EdgeInput,
    history: Any = None,
    attack_index: Optional[
        Union[PreviousAttackEdgeIndex, RecentAttackEdgeIndex]
    ] = None,
    require_previous_window_visible: bool = True,
) -> float:
    """返回最近五窗口攻击相似度得分。"""
    return float(
        previous_attack_edge_similarity(
            edge_or_key=edge_or_key,
            history=history,
            attack_index=attack_index,
            require_previous_window_visible=require_previous_window_visible,
        )["score"]
    )


def was_attack_edge_in_previous_window(
    edge_or_key: EdgeInput,
    history: Any,
    require_previous_window_visible: bool = True,
) -> bool:
    """兼容入口：只判断五元组是否出现在上一个攻击窗口。"""
    previous_index = build_previous_attack_edge_index(
        history,
        require_previous_window_visible=require_previous_window_visible,
    )
    edge_key = _normalize_edge_key(edge_or_key)
    return edge_key in previous_index.exact_edges if edge_key is not None else False


def _score_to_label(
    score: Optional[float],
    attack_threshold: float,
    normal_threshold: float,
) -> str:
    if score is None:
        return "unknown"
    score = min(max(float(score), 0.0), 1.0)
    if score >= attack_threshold:
        return "attack"
    if score <= normal_threshold:
        return "normal"
    return "unknown"


def classify_edge_with_previous_attack_override(
    edge_or_key: EdgeInput,
    history: Any,
    scorer: Optional[Callable[[Any], Optional[float]]] = None,
    edge_obj: Any = None,
    attack_threshold: float = 0.7,
    normal_threshold: float = 0.3,
    require_previous_window_visible: bool = True,
    attack_index: Optional[
        Union[PreviousAttackEdgeIndex, RecentAttackEdgeIndex]
    ] = None,
) -> Dict[str, Any]:
    """取基础异常分数和最近五窗口攻击相似度得分中的较大值。"""
    if normal_threshold > attack_threshold:
        raise ValueError("normal_threshold 不能大于 attack_threshold。")
    if edge_obj is None and not isinstance(edge_or_key, tuple):
        edge_obj = edge_or_key

    similarity = previous_attack_edge_similarity(
        edge_or_key=edge_or_key,
        history=history,
        attack_index=attack_index,
        require_previous_window_visible=require_previous_window_visible,
    )
    raw_score = scorer(edge_obj) if scorer is not None else None
    score = (
        float(similarity["score"])
        if raw_score is None
        else max(
            min(max(float(raw_score), 0.0), 1.0),
            float(similarity["score"]),
        )
    )
    return {
        "label": _score_to_label(score, attack_threshold, normal_threshold),
        "score": score,
        "forced_by_previous_attack": similarity["score"] > 0.0,
        "previous_attack_similarity_score": similarity["score"],
        "previous_attack_similarity_base_score": similarity["base_score"],
        "previous_attack_match_level": similarity["match_level"],
        "previous_attack_window_distance": similarity["window_distance"],
    }


def classify_graph_edges_with_previous_attack_override(
    graph_or_edges: Any,
    history: Any,
    scorer: Optional[Callable[[Any], Optional[float]]] = None,
    attack_threshold: float = 0.7,
    normal_threshold: float = 0.3,
    require_previous_window_visible: bool = True,
) -> Dict[Hashable, Dict[str, Any]]:
    """批量评分，并为整个当前窗口复用同一个五窗口攻击索引。"""
    edges = getattr(graph_or_edges, "edges", graph_or_edges)
    if edges is None:
        return {}
    if not isinstance(edges, Mapping):
        raise TypeError("graph_or_edges 必须是 graph 对象或 edge 映射。")

    attack_index = build_recent_attack_edge_index(
        history,
        require_previous_window_visible=require_previous_window_visible,
    )
    return {
        edge_key: classify_edge_with_previous_attack_override(
            edge_or_key=edge_key,
            history=history,
            scorer=scorer,
            edge_obj=edge_obj,
            attack_threshold=attack_threshold,
            normal_threshold=normal_threshold,
            require_previous_window_visible=require_previous_window_visible,
            attack_index=attack_index,
        )
        for edge_key, edge_obj in edges.items()
    }
