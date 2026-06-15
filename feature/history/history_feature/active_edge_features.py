import math
from typing import Any, ClassVar, Dict, Iterable, List, Optional

from feature.attack_similar.previous_attack_edge import edge_key_from_edge_obj


class EdgeActiveHistoryFeature:
    """
    边活跃历史特征计算工具。

    说明:
        - 标量偏移与协议/标志漂移默认基于 history.edges 中仍在 TTL 内的边。
        - 时间行为漂移优先基于 history.recent_edge_details 中最近窗口的边快照。
        - current_window_only=True 时，各项基准均只使用刚提交窗口中的对象。
        - 当前边计算偏移/漂移时，默认从活跃基准中排除自身，避免自身参与基准导致分数被稀释。

    TODO:
        如果后续希望使用跨窗口 EMA 作为基准，需要在 HistoryRecord 中新增边特征 EMA 字段；
        这里暂时只提供基于活跃历史边和近期明细边快照的无侵入计算。
    """

    FLAG_NAMES = ("CWR", "ECE", "URG", "ACK", "PSH", "RST", "SYN", "FIN")
    FINITE_HISTORY_OFFSET_WEIGHTS: ClassVar[Dict[str, float]] = {
        "edge_packet_count_active_offset": 0.15,
        "edge_byte_count_active_offset": 0.15,
        "edge_small_packet_ratio_active_offset": 0.15,
        "edge_handshake_failure_active_offset": 0.15,
        "edge_protocol_flags_active_drift_distance": 0.20,
        "edge_time_behavior_active_drift_distance": 0.20,
    }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _mean(values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @classmethod
    def _std(cls, values: List[float], mean_value: Optional[float] = None) -> float:
        if not values:
            return 0.0
        if mean_value is None:
            mean_value = cls._mean(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    @staticmethod
    def _normalize_z_score(z_score: float) -> float:
        """
        将非负 z-score 压缩到 0~1。

        z 越大代表越偏离活跃基准，score 越接近 1。
        """
        z_score = abs(float(z_score))
        return z_score / (1.0 + z_score)

    @staticmethod
    def _clamp_score(score: float) -> float:
        return min(max(float(score), 0.0), 1.0)

    @classmethod
    def _history_record_for_edge(cls, edge_obj: Any, history: Any):
        """通过对象身份或标准边 key 查找 history 记录。"""
        history_edges = getattr(history, "edges", {})
        for record in history_edges.values():
            if (
                getattr(record, "obj", None) is edge_obj
                or getattr(record, "current_window_obj", None) is edge_obj
            ):
                return record

        edge_key = edge_key_from_edge_obj(edge_obj)
        if edge_key is not None:
            return history_edges.get(edge_key)
        return None

    @classmethod
    def _is_new_edge(cls, edge_obj: Any, history: Any) -> bool:
        """
        判断当前边是否为首次进入活跃历史表的新边。

        当前窗口写入 History 后，新边也会存在于 history.edges，
        因此使用 appeared_window_count<=1 近似表示“此前不在活跃表”。
        """
        record = cls._history_record_for_edge(edge_obj, history)
        if record is None:
            return True
        return (
            getattr(record, "obj", None) is edge_obj
            and int(getattr(record, "appeared_window_count", 1)) <= 1
        )

    @classmethod
    def _active_edges(
        cls,
        history: Any,
        current_window_only: bool = False,
        exclude_edge: Optional[Any] = None,
    ) -> List[Any]:
        """从 History.edges 提取活跃边对象；默认使用 TTL 内仍存在的长期聚合边。"""
        active_edges = []
        exclude_key = edge_key_from_edge_obj(exclude_edge)
        for edge_key, record in getattr(history, "edges", {}).items():
            if current_window_only and not getattr(record, "in_current_graph", False):
                continue
            if hasattr(history, "is_edge_baseline_eligible"):
                if not history.is_edge_baseline_eligible(edge_key):
                    continue

            edge_obj = getattr(
                record,
                "current_window_obj" if current_window_only else "obj",
                None,
            )
            if edge_obj is None:
                continue

            if exclude_edge is not None and (
                edge_obj is exclude_edge
                or (
                    exclude_key is not None
                    and edge_key_from_edge_obj(edge_obj) == exclude_key
                )
            ):
                continue

            active_edges.append(edge_obj)

        return active_edges

    @classmethod
    def _time_baseline_edges(
        cls,
        history: Any,
        current_window_only: bool = False,
    ) -> List[Any]:
        """
        提取时间行为漂移基准边。

        默认优先使用 RecentEdgeDetails 中最近窗口的原始边快照，避免 History.edges
        的长期累计 timestamp_list 影响近期时间行为；当前窗口模式仍使用刚提交窗口对象。
        """
        if current_window_only:
            return cls._active_edges(history, True, None)

        recent_edge_details = getattr(history, "recent_edge_details", None)
        if recent_edge_details is not None and hasattr(
            recent_edge_details,
            "recent_edge_items",
        ):
            recent_items = recent_edge_details.recent_edge_items()
            if hasattr(history, "is_edge_baseline_eligible"):
                return [
                    edge_obj
                    for edge_key, edge_obj in recent_items.items()
                    if history.is_edge_baseline_eligible(edge_key)
                ]
            return list(recent_items.values())

        return cls._active_edges(history, False, None)

    @classmethod
    def _scalar_active_offset(
        cls,
        current_value: float,
        active_values: Iterable[float],
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """
        标量活跃偏移:
            z = abs(current - mean(active)) / (std(active) + eps)
            score = z / (1 + z)
        """
        values = [cls._safe_float(value) for value in active_values]
        if not values:
            detail = {
                "score": 0.0,
                "current": float(current_value),
                "active_mean": 0.0,
                "active_std": 0.0,
                "raw_z_offset": 0.0,
                "active_count": 0,
            }
            return detail if return_detail else 0.0

        mean_value = cls._mean(values)
        std_value = cls._std(values, mean_value)
        raw_z_offset = (float(current_value) - mean_value) / (std_value + eps)
        score = cls._normalize_z_score(raw_z_offset)

        detail = {
            "score": score,
            "current": float(current_value),
            "active_mean": mean_value,
            "active_std": std_value,
            "raw_z_offset": raw_z_offset,
            "active_count": len(values),
        }
        return detail if return_detail else score

    @classmethod
    def _edge_packet_count(cls, edge_obj: Any) -> float:
        return cls._safe_float(getattr(edge_obj, "edgepacketnum", 0.0))

    @classmethod
    def _edge_byte_count(cls, edge_obj: Any) -> float:
        return cls._safe_float(getattr(edge_obj, "payload_len", 0.0))

    @classmethod
    def _edge_small_packet_ratio(cls, edge_obj: Any, eps: float = 1e-8) -> float:
        if hasattr(edge_obj, "small_packet_ratio"):
            return cls._safe_float(edge_obj.small_packet_ratio(eps=eps))

        smallpacket = cls._safe_float(getattr(edge_obj, "smallpacket", 0.0))
        packet_count = cls._edge_packet_count(edge_obj)
        return smallpacket / (packet_count + eps)

    @classmethod
    def _edge_handshake_failure_score(cls, edge_obj: Any, eps: float = 1e-8) -> float:
        if hasattr(edge_obj, "handshake_failure_score"):
            return cls._safe_float(edge_obj.handshake_failure_score(eps=eps))

        syn_count = cls._safe_float(getattr(edge_obj, "syn_count", 0.0))
        syn_ack_count = cls._safe_float(getattr(edge_obj, "syn_ack_count", 0.0))
        ack_count = cls._safe_float(getattr(edge_obj, "ack_count", 0.0))
        return max(syn_count - syn_ack_count - ack_count, 0.0) / (syn_count + eps)

    @classmethod
    def edge_packet_count_active_offset(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """边包数量活跃偏移。"""
        if cls._is_new_edge(edge_obj, history):
            return {"score": 1.0, "is_new_edge": True} if return_detail else 1.0

        active_edges = cls._active_edges(history, current_window_only, edge_obj)
        return cls._scalar_active_offset(
            current_value=cls._edge_packet_count(edge_obj),
            active_values=[cls._edge_packet_count(edge) for edge in active_edges],
            eps=eps,
            return_detail=return_detail,
        )

    @classmethod
    def edge_byte_count_active_offset(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """边字节数活跃偏移。"""
        if cls._is_new_edge(edge_obj, history):
            return {"score": 1.0, "is_new_edge": True} if return_detail else 1.0

        active_edges = cls._active_edges(history, current_window_only, edge_obj)
        return cls._scalar_active_offset(
            current_value=cls._edge_byte_count(edge_obj),
            active_values=[cls._edge_byte_count(edge) for edge in active_edges],
            eps=eps,
            return_detail=return_detail,
        )

    @classmethod
    def edge_small_packet_ratio_active_offset(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """边小包比例活跃偏移。"""
        current_ratio = cls._edge_small_packet_ratio(edge_obj, eps)
        if cls._is_new_edge(edge_obj, history):
            detail = {"score": current_ratio, "current": current_ratio, "is_new_edge": True}
            return detail if return_detail else current_ratio

        active_edges = cls._active_edges(history, current_window_only, edge_obj)
        active_values = [cls._edge_small_packet_ratio(edge, eps) for edge in active_edges]
        if not active_values:
            detail = {
                "score": 0.0,
                "current": current_ratio,
                "active_mean": 0.0,
                "active_count": 0,
                "is_new_edge": False,
            }
            return detail if return_detail else 0.0

        active_mean = cls._mean(active_values)
        score = cls._clamp_score(abs(current_ratio - active_mean))
        detail = {
            "score": score,
            "current": current_ratio,
            "active_mean": active_mean,
            "active_count": len(active_values),
            "is_new_edge": False,
        }
        return detail if return_detail else score

    @classmethod
    def edge_handshake_failure_active_offset(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """边握手失败活跃偏移。"""
        current_score = cls._edge_handshake_failure_score(edge_obj, eps)
        if cls._is_new_edge(edge_obj, history):
            detail = {
                "score": current_score,
                "current": current_score,
                "is_new_edge": True,
            }
            return detail if return_detail else current_score

        active_edges = cls._active_edges(history, current_window_only, edge_obj)
        active_values = [
            cls._edge_handshake_failure_score(edge, eps) for edge in active_edges
        ]
        if not active_values:
            detail = {
                "score": 0.0,
                "current": current_score,
                "active_mean": 0.0,
                "active_count": 0,
                "is_new_edge": False,
            }
            return detail if return_detail else 0.0

        active_mean = cls._mean(active_values)
        score = cls._clamp_score(abs(current_score - active_mean))
        detail = {
            "score": score,
            "current": current_score,
            "active_mean": active_mean,
            "active_count": len(active_values),
            "is_new_edge": False,
        }
        return detail if return_detail else score

    @classmethod
    def _protocol_flags_vector(cls, edge_obj: Any) -> Dict[str, float]:
        """
        协议/标志位向量。

        组成:
            - TCP比例
            - UDP比例
            - ICMP比例
            - SYN比例
            - ACK比例
            - RST比例
            - FIN比例
            - SYN无ACK近似比例
            - 握手失败近似得分
        """
        protocol_count_map = getattr(edge_obj, "protocol_count_map", {}) or {}
        flag_name_count_map = getattr(edge_obj, "flag_name_count_map", {}) or {}
        packet_count = cls._edge_packet_count(edge_obj)

        def protocol_ratio(protocol_number):
            protocol_count = cls._safe_float(
                protocol_count_map.get(protocol_number, 0.0)
            )
            return protocol_count / (packet_count + 1e-8)

        def flag_ratio(flag_name):
            flag_count = cls._safe_float(flag_name_count_map.get(flag_name, 0.0))
            return flag_count / (packet_count + 1e-8)

        if hasattr(edge_obj, "syn_without_ack_ratio"):
            syn_without_ack_ratio = cls._safe_float(
                edge_obj.syn_without_ack_ratio(eps=1e-8)
            )
        else:
            syn_count = cls._safe_float(getattr(edge_obj, "syn_count", 0.0))
            ack_count = cls._safe_float(getattr(edge_obj, "ack_count", 0.0))
            syn_without_ack_ratio = max(syn_count - ack_count, 0.0) / (
                syn_count + 1e-8
            )

        return {
            "tcp_ratio": protocol_ratio(6),
            "udp_ratio": protocol_ratio(17),
            "icmp_ratio": protocol_ratio(1),
            "syn_ratio": flag_ratio("SYN"),
            "ack_ratio": flag_ratio("ACK"),
            "rst_ratio": flag_ratio("RST"),
            "fin_ratio": flag_ratio("FIN"),
            "syn_without_ack_ratio": syn_without_ack_ratio,
            "handshake_failure_score": cls._edge_handshake_failure_score(
                edge_obj, eps=1e-8
            ),
        }

    @classmethod
    def _time_behavior_vector(
        cls,
        edge_obj: Any,
        burst_gap_threshold: float = 0.01,
        active_duration_quantile: float = 0.0,
    ) -> Dict[str, float]:
        """
        时间行为向量。

        组成:
            - 边包间隔变异系数归一化
            - 边突发性得分
            - 边活跃持续时间分位数
        """
        timestamps = []
        for value in getattr(edge_obj, "timestamp_list", []) or []:
            try:
                timestamps.append(float(value))
            except (TypeError, ValueError):
                continue

        timestamps = sorted(timestamps)
        if len(timestamps) <= 1:
            gaps = []
        else:
            gaps = [
                max(timestamps[idx] - timestamps[idx - 1], 0.0)
                for idx in range(1, len(timestamps))
            ]

        mean_gap = cls._mean(gaps)
        std_gap = cls._std(gaps, mean_gap) if gaps else 0.0
        gap_cv = std_gap / (mean_gap + 1e-8)
        gap_cv_normalized = gap_cv / (1.0 + gap_cv)

        if hasattr(edge_obj, "burstiness_score"):
            burstiness = cls._safe_float(
                edge_obj.burstiness_score(burst_gap_threshold=burst_gap_threshold)
            )
        else:
            burstiness = 0.0

        return {
            "gap_cv_normalized": gap_cv_normalized,
            "burstiness": burstiness,
            "active_duration_quantile": active_duration_quantile,
        }

    @classmethod
    def _edge_active_duration(cls, edge_obj: Any) -> float:
        timestamps = []
        for value in getattr(edge_obj, "timestamp_list", []) or []:
            try:
                timestamps.append(float(value))
            except (TypeError, ValueError):
                continue
        if len(timestamps) <= 1:
            return 0.0
        return max(timestamps) - min(timestamps)

    @classmethod
    def _duration_quantile_map(cls, edge_objs: Iterable[Any]) -> Dict[int, float]:
        edge_list = list(edge_objs)
        if not edge_list:
            return {}

        durations = [cls._edge_active_duration(edge_obj) for edge_obj in edge_list]
        sorted_durations = sorted(durations)
        total = len(sorted_durations)

        quantile_map = {}
        for edge_obj, duration in zip(edge_list, durations):
            less_or_equal_count = sum(
                1 for value in sorted_durations if value <= duration
            )
            quantile_map[id(edge_obj)] = less_or_equal_count / total
        return quantile_map

    @classmethod
    def _vector_mean_abs_drift(
        cls,
        current_vector: Dict[Any, float],
        active_vectors: List[Dict[Any, float]],
        return_detail: bool = False,
    ):
        """
        向量平均绝对漂移:
            drift = mean(|V_now_i - mean(V_active_i)|)
        """
        if not active_vectors:
            detail = {
                "score": 0.0,
                "raw_drift": 0.0,
                "dimension_count": len(current_vector),
                "active_count": 0,
            }
            return detail if return_detail else 0.0

        all_keys = set(current_vector.keys())
        for vector in active_vectors:
            all_keys.update(vector.keys())

        if not all_keys:
            detail = {
                "score": 0.0,
                "raw_drift": 0.0,
                "dimension_count": 0,
                "active_count": len(active_vectors),
            }
            return detail if return_detail else 0.0

        offsets = []
        for key in all_keys:
            active_values = [
                cls._safe_float(vector.get(key, 0.0)) for vector in active_vectors
            ]
            active_mean = cls._mean(active_values)
            current_value = cls._safe_float(current_vector.get(key, 0.0))
            offsets.append(abs(current_value - active_mean))

        raw_drift = cls._mean(offsets)
        score = cls._clamp_score(raw_drift)
        detail = {
            "score": score,
            "raw_drift": raw_drift,
            "dimension_count": len(all_keys),
            "active_count": len(active_vectors),
        }
        return detail if return_detail else score

    @classmethod
    def edge_protocol_flags_active_drift_distance(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        eps: float = 1e-8,
        return_detail: bool = False,
    ):
        """边协议标志活跃漂移距离。"""
        if cls._is_new_edge(edge_obj, history):
            return {"score": 1.0, "is_new_edge": True} if return_detail else 1.0

        active_edges = cls._active_edges(history, current_window_only, edge_obj)
        return cls._vector_mean_abs_drift(
            current_vector=cls._protocol_flags_vector(edge_obj),
            active_vectors=[cls._protocol_flags_vector(edge) for edge in active_edges],
            return_detail=return_detail,
        )

    @classmethod
    def edge_time_behavior_active_drift_distance(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        burst_gap_threshold: float = 0.01,
        eps: float = 1e-8,
        return_detail: bool = False,
        current_window_edges: Optional[Iterable[Any]] = None,
    ):
        """边时间行为活跃漂移距离；默认基于最近窗口明细快照而非长期累计边。"""
        if cls._is_new_edge(edge_obj, history):
            return {"score": 1.0, "is_new_edge": True} if return_detail else 1.0

        all_active_edges = cls._time_baseline_edges(history, current_window_only)
        if current_window_edges is None:
            current_window_edges = cls._active_edges(history, True, None)
        edge_key = edge_key_from_edge_obj(edge_obj)
        active_edges = [
            edge
            for edge in all_active_edges
            if edge_key_from_edge_obj(edge) != edge_key
        ]
        current_duration_edges = [
            edge
            for edge in current_window_edges
            if edge_key_from_edge_obj(edge) != edge_key
        ]
        current_duration_edges.append(edge_obj)
        current_duration_map = cls._duration_quantile_map(current_duration_edges)
        active_duration_map = cls._duration_quantile_map(all_active_edges)

        return cls._vector_mean_abs_drift(
            current_vector=cls._time_behavior_vector(
                edge_obj,
                burst_gap_threshold,
                current_duration_map.get(id(edge_obj), 0.0),
            ),
            active_vectors=[
                cls._time_behavior_vector(
                    edge,
                    burst_gap_threshold,
                    active_duration_map.get(id(edge), 0.0),
                )
                for edge in active_edges
            ],
            return_detail=return_detail,
        )

    @classmethod
    def compute_all_active_edge_features(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        burst_gap_threshold: float = 0.01,
        eps: float = 1e-8,
        return_detail: bool = False,
        current_window_edges: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """一次性计算当前边的全部活跃偏移/漂移特征。"""
        return {
            "edge_packet_count_active_offset": cls.edge_packet_count_active_offset(
                edge_obj, history, current_window_only, eps, return_detail
            ),
            "edge_byte_count_active_offset": cls.edge_byte_count_active_offset(
                edge_obj, history, current_window_only, eps, return_detail
            ),
            "edge_small_packet_ratio_active_offset": (
                cls.edge_small_packet_ratio_active_offset(
                    edge_obj, history, current_window_only, eps, return_detail
                )
            ),
            "edge_handshake_failure_active_offset": (
                cls.edge_handshake_failure_active_offset(
                    edge_obj, history, current_window_only, eps, return_detail
                )
            ),
            "edge_protocol_flags_active_drift_distance": (
                cls.edge_protocol_flags_active_drift_distance(
                    edge_obj, history, current_window_only, eps, return_detail
                )
            ),
            "edge_time_behavior_active_drift_distance": (
                cls.edge_time_behavior_active_drift_distance(
                    edge_obj,
                    history,
                    current_window_only,
                    burst_gap_threshold,
                    eps,
                    return_detail,
                    current_window_edges,
                )
            ),
        }

    @classmethod
    def finite_history_offset_anomaly_score(
        cls,
        edge_obj: Any,
        history: Any,
        current_window_only: bool = False,
        burst_gap_threshold: float = 0.01,
        eps: float = 1e-8,
        return_components: bool = False,
        current_window_edges: Optional[Iterable[Any]] = None,
    ):
        """
        有限历史偏移异常分数:
            0.15 * 边包数量活跃偏移
            + 0.15 * 边字节数活跃偏移
            + 0.15 * 边小包比例活跃偏移
            + 0.15 * 边握手失败活跃偏移
            + 0.20 * 边协议标志活跃漂移距离
            + 0.20 * 边时间行为活跃漂移距离

        时间行为分量优先使用 RecentEdgeDetails 的最近窗口边快照作为基准；
        其他分量仍使用 History.edges 中 TTL 内的长期历史边。
        """
        components = cls.compute_all_active_edge_features(
            edge_obj=edge_obj,
            history=history,
            current_window_only=current_window_only,
            burst_gap_threshold=burst_gap_threshold,
            eps=eps,
            return_detail=False,
            current_window_edges=current_window_edges,
        )
        score = sum(
            cls.FINITE_HISTORY_OFFSET_WEIGHTS[name] * float(value)
            for name, value in components.items()
        )
        score = min(max(score, 0.0), 1.0)

        if return_components:
            components["finite_history_offset_anomaly_score"] = score
            return components
        return score

    @classmethod
    def _scalar_score_from_summary(
        cls,
        current_value: float,
        total_sum: float,
        total_square_sum: float,
        total_count: int,
        exclude_value: Optional[float] = None,
        eps: float = 1e-8,
    ) -> float:
        """基于预统计结果快速计算单个标量活跃偏移。"""
        count = total_count
        value_sum = total_sum
        square_sum = total_square_sum

        if exclude_value is not None and count > 0:
            count -= 1
            value_sum -= exclude_value
            square_sum -= exclude_value * exclude_value

        if count <= 0:
            return 0.0

        mean_value = value_sum / count
        variance = max(square_sum / count - mean_value * mean_value, 0.0)
        std_value = math.sqrt(variance)
        raw_z_offset = abs(float(current_value) - mean_value) / (std_value + eps)
        return cls._normalize_z_score(raw_z_offset)

    @classmethod
    def _scalar_abs_diff_from_summary(
        cls,
        current_value: float,
        total_sum: float,
        total_count: int,
        exclude_value: Optional[float] = None,
    ) -> float:
        """基于预统计结果快速计算 abs(当前值 - 活跃均值)。"""
        count = total_count
        value_sum = total_sum

        if exclude_value is not None and count > 0:
            count -= 1
            value_sum -= exclude_value

        if count <= 0:
            return 0.0

        active_mean = value_sum / count
        return cls._clamp_score(abs(float(current_value) - active_mean))

    @classmethod
    def _vector_mean_abs_from_summary(
        cls,
        current_vector: Dict[Any, float],
        all_keys: Iterable[Any],
        sums: Dict[Any, float],
        total_count: int,
        exclude_vector: Optional[Dict[Any, float]] = None,
    ) -> float:
        """基于预统计结果快速计算 mean(|V_now_i - mean(V_active_i)|)。"""
        keys = list(all_keys)
        if not keys or total_count <= 0:
            return 0.0

        count = total_count - 1 if exclude_vector is not None else total_count
        if count <= 0:
            return 0.0

        offsets = []
        for key in keys:
            value_sum = cls._safe_float(sums.get(key, 0.0))
            if exclude_vector is not None:
                value_sum -= cls._safe_float(exclude_vector.get(key, 0.0))
            active_mean = value_sum / count
            current_value = cls._safe_float(current_vector.get(key, 0.0))
            offsets.append(abs(current_value - active_mean))

        return cls._clamp_score(cls._mean(offsets))

    @classmethod
    def compute_finite_history_offset_anomaly_scores(
        cls,
        edge_objs: Iterable[Any],
        history: Any,
        current_window_only: bool = False,
        burst_gap_threshold: float = 0.01,
        eps: float = 1e-8,
        return_components: bool = False,
    ) -> Dict[int, Any]:
        """
        批量计算多条边的有限历史偏移异常分数。

        返回:
            id(edge_obj) -> score 或 components

        说明:
            该函数会一次性统计活跃历史边基准，适合按窗口批量计算。
            时间行为分量优先使用 RecentEdgeDetails 的最近窗口快照；
            标量与协议/标志分量仍使用 History.edges 的 TTL 内长期历史边。
        """
        edge_list = list(edge_objs)
        active_edges = cls._active_edges(
            history=history,
            current_window_only=current_window_only,
            exclude_edge=None,
        )
        active_count = len(active_edges)
        time_active_edges = cls._time_baseline_edges(history, current_window_only)
        time_active_count = len(time_active_edges)
        current_duration_map = cls._duration_quantile_map(edge_list)
        active_duration_map = cls._duration_quantile_map(time_active_edges)
        active_edge_ids_by_key = {
            edge_key: id(edge_obj)
            for edge_obj in active_edges
            if (edge_key := edge_key_from_edge_obj(edge_obj)) is not None
        }
        time_edge_ids_by_key = {
            edge_key: id(edge_obj)
            for edge_obj in time_active_edges
            if (edge_key := edge_key_from_edge_obj(edge_obj)) is not None
        }

        scalar_getters = {
            "edge_packet_count_active_offset": lambda edge: cls._edge_packet_count(edge),
            "edge_byte_count_active_offset": lambda edge: cls._edge_byte_count(edge),
            "edge_small_packet_ratio_active_offset": lambda edge: (
                cls._edge_small_packet_ratio(edge, eps)
            ),
            "edge_handshake_failure_active_offset": lambda edge: (
                cls._edge_handshake_failure_score(edge, eps)
            ),
        }

        scalar_values = {}
        scalar_sums = {}
        scalar_square_sums = {}
        for name, getter in scalar_getters.items():
            values_by_edge_id = {}
            values = []
            for edge_obj in active_edges:
                value = cls._safe_float(getter(edge_obj))
                values_by_edge_id[id(edge_obj)] = value
                values.append(value)
            scalar_values[name] = values_by_edge_id
            scalar_sums[name] = sum(values)
            scalar_square_sums[name] = sum(value * value for value in values)

        protocol_vectors = {}
        time_vectors = {}
        protocol_keys: set[str] = set()
        time_keys: set[str] = set()
        for edge_obj in active_edges:
            edge_id = id(edge_obj)
            protocol_vector = cls._protocol_flags_vector(edge_obj)
            protocol_vectors[edge_id] = protocol_vector
            protocol_keys.update(protocol_vector.keys())

        for edge_obj in time_active_edges:
            edge_id = id(edge_obj)
            time_vector = cls._time_behavior_vector(
                edge_obj,
                burst_gap_threshold,
                active_duration_map.get(edge_id, 0.0),
            )
            time_vectors[edge_id] = time_vector
            time_keys.update(time_vector.keys())

        def vector_summaries(vectors, keys):
            sums = {key: 0.0 for key in keys}
            square_sums = {key: 0.0 for key in keys}
            for vector in vectors.values():
                for key in keys:
                    value = cls._safe_float(vector.get(key, 0.0))
                    sums[key] += value
                    square_sums[key] += value * value
            return sums, square_sums

        protocol_sums, protocol_square_sums = vector_summaries(
            protocol_vectors, protocol_keys
        )
        time_sums, time_square_sums = vector_summaries(time_vectors, time_keys)

        results: Dict[int, Any] = {}
        for edge_obj in edge_list:
            edge_id = id(edge_obj)
            edge_key = edge_key_from_edge_obj(edge_obj)
            baseline_edge_id = (
                active_edge_ids_by_key.get(edge_key, edge_id)
                if edge_key is not None
                else edge_id
            )
            time_baseline_edge_id = (
                time_edge_ids_by_key.get(edge_key, edge_id)
                if edge_key is not None
                else edge_id
            )
            components = {}
            is_new_edge = cls._is_new_edge(edge_obj, history)

            packet_count = cls._safe_float(
                scalar_getters["edge_packet_count_active_offset"](edge_obj)
            )
            byte_count = cls._safe_float(
                scalar_getters["edge_byte_count_active_offset"](edge_obj)
            )
            small_ratio = cls._safe_float(
                scalar_getters["edge_small_packet_ratio_active_offset"](edge_obj)
            )
            fail_score = cls._safe_float(
                scalar_getters["edge_handshake_failure_active_offset"](edge_obj)
            )

            if is_new_edge:
                components["edge_packet_count_active_offset"] = 1.0
                components["edge_byte_count_active_offset"] = 1.0
                components["edge_small_packet_ratio_active_offset"] = small_ratio
                components["edge_handshake_failure_active_offset"] = fail_score
            else:
                components["edge_packet_count_active_offset"] = (
                    cls._scalar_score_from_summary(
                        current_value=packet_count,
                        total_sum=scalar_sums["edge_packet_count_active_offset"],
                        total_square_sum=scalar_square_sums[
                            "edge_packet_count_active_offset"
                        ],
                        total_count=active_count,
                        exclude_value=scalar_values[
                            "edge_packet_count_active_offset"
                        ].get(baseline_edge_id),
                        eps=eps,
                    )
                )
                components["edge_byte_count_active_offset"] = (
                    cls._scalar_score_from_summary(
                        current_value=byte_count,
                        total_sum=scalar_sums["edge_byte_count_active_offset"],
                        total_square_sum=scalar_square_sums[
                            "edge_byte_count_active_offset"
                        ],
                        total_count=active_count,
                        exclude_value=scalar_values[
                            "edge_byte_count_active_offset"
                        ].get(baseline_edge_id),
                        eps=eps,
                    )
                )
                components["edge_small_packet_ratio_active_offset"] = (
                    cls._scalar_abs_diff_from_summary(
                        current_value=small_ratio,
                        total_sum=scalar_sums[
                            "edge_small_packet_ratio_active_offset"
                        ],
                        total_count=active_count,
                        exclude_value=scalar_values[
                            "edge_small_packet_ratio_active_offset"
                        ].get(baseline_edge_id),
                    )
                )
                components["edge_handshake_failure_active_offset"] = (
                    cls._scalar_abs_diff_from_summary(
                        current_value=fail_score,
                        total_sum=scalar_sums[
                            "edge_handshake_failure_active_offset"
                        ],
                        total_count=active_count,
                        exclude_value=scalar_values[
                            "edge_handshake_failure_active_offset"
                        ].get(baseline_edge_id),
                    )
                )

            current_protocol_vector = cls._protocol_flags_vector(edge_obj)
            exclude_protocol_vector = protocol_vectors.get(baseline_edge_id)
            if is_new_edge:
                components["edge_protocol_flags_active_drift_distance"] = 1.0
            else:
                components["edge_protocol_flags_active_drift_distance"] = (
                    cls._vector_mean_abs_from_summary(
                        current_vector=current_protocol_vector,
                        all_keys=protocol_keys.union(current_protocol_vector.keys()),
                        sums=protocol_sums,
                        total_count=active_count,
                        exclude_vector=exclude_protocol_vector,
                    )
                )

            current_time_vector = cls._time_behavior_vector(
                edge_obj,
                burst_gap_threshold,
                current_duration_map.get(edge_id, 0.0),
            )
            exclude_time_vector = time_vectors.get(time_baseline_edge_id)
            if is_new_edge:
                components["edge_time_behavior_active_drift_distance"] = 1.0
            else:
                components["edge_time_behavior_active_drift_distance"] = (
                    cls._vector_mean_abs_from_summary(
                        current_vector=current_time_vector,
                        all_keys=time_keys.union(current_time_vector.keys()),
                        sums=time_sums,
                        total_count=time_active_count,
                        exclude_vector=exclude_time_vector,
                    )
                )

            score = sum(
                cls.FINITE_HISTORY_OFFSET_WEIGHTS[name] * components[name]
                for name in cls.FINITE_HISTORY_OFFSET_WEIGHTS
            )
            score = min(max(score, 0.0), 1.0)

            if return_components:
                components["finite_history_offset_anomaly_score"] = score
                results[edge_id] = components
            else:
                results[edge_id] = score

        return results


def edge_packet_count_active_offset(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_packet_count_active_offset(*args, **kwargs)


def edge_byte_count_active_offset(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_byte_count_active_offset(*args, **kwargs)


def edge_small_packet_ratio_active_offset(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_small_packet_ratio_active_offset(
        *args, **kwargs
    )


def edge_handshake_failure_active_offset(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_handshake_failure_active_offset(
        *args, **kwargs
    )


def edge_protocol_flags_active_drift_distance(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_protocol_flags_active_drift_distance(
        *args, **kwargs
    )


def edge_time_behavior_active_drift_distance(*args, **kwargs):
    return EdgeActiveHistoryFeature.edge_time_behavior_active_drift_distance(
        *args, **kwargs
    )


def compute_all_active_edge_features(*args, **kwargs):
    return EdgeActiveHistoryFeature.compute_all_active_edge_features(*args, **kwargs)


def finite_history_offset_anomaly_score(*args, **kwargs):
    return EdgeActiveHistoryFeature.finite_history_offset_anomaly_score(
        *args, **kwargs
    )


def compute_finite_history_offset_anomaly_scores(*args, **kwargs):
    return EdgeActiveHistoryFeature.compute_finite_history_offset_anomaly_scores(
        *args, **kwargs
    )
