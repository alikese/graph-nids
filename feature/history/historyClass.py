from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union

from feature.edge.edgeClass import EdgeWindowBuffer
from feature.attack_similar.previous_attack_edge import RecentAttackEdgeHistory
from feature.history.recentEdgeDetails import RecentEdgeDetails
from feature.history.suspiciousEdgeHistory import SuspiciousEdgeHistory


DEFAULT_BEHAVIOR_ROLE_WEIGHTS = {
    "source_out_degree_quantile": 0.20,
    "destination_in_degree_quantile": 0.15,
    "normalized_byte_ratio": 0.15,
    "source_role_drift": 0.20,
    "destination_role_drift": 0.15,
    "previous_attack_dst_to_src_mark": 0.15,
}
BEHAVIOR_ROLE_WEIGHTS = dict(DEFAULT_BEHAVIOR_ROLE_WEIGHTS)


def _normalized_behavior_role_weights(weights):
    cleaned = {
        name: max(float(weights.get(name, 0.0)), 0.0)
        for name in DEFAULT_BEHAVIOR_ROLE_WEIGHTS
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(DEFAULT_BEHAVIOR_ROLE_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def set_behavior_role_weights(weights):
    BEHAVIOR_ROLE_WEIGHTS.clear()
    BEHAVIOR_ROLE_WEIGHTS.update(_normalized_behavior_role_weights(weights))


def reset_behavior_role_weights():
    BEHAVIOR_ROLE_WEIGHTS.clear()
    BEHAVIOR_ROLE_WEIGHTS.update(DEFAULT_BEHAVIOR_ROLE_WEIGHTS)


@dataclass
class HistoryRecord:
    """
    历史记录项。

    属性:
        obj: 长期历史中的节点/边聚合对象
        current_window_obj: 刚提交窗口中的原始对象快照；仅当前窗口有效
        in_current_graph: 是否出现在刚提交的窗口中
        edge_label: 边类型标识（attack/normal/unknown）
        edge_score: 刚提交窗口边评分（节点记录可忽略该值）
        appeared_window_count: 在未删除前累计出现的窗口次数
        source_behavior_role_vector: 源节点当前行为角色向量
        source_behavior_role_ema: 源节点历史行为角色 EMA 向量
        source_behavior_role_drift: 源节点行为角色漂移距离
        destination_behavior_role_vector: 目的节点当前行为角色向量
        destination_behavior_role_ema: 目的节点历史行为角色 EMA 向量
        destination_behavior_role_drift: 目的节点行为角色漂移距离
        ttl: 生命周期计数（降到 0 删除）
    """

    obj: Any
    in_current_graph: bool
    edge_label: str
    edge_score: float
    appeared_window_count: int
    source_behavior_role_vector: Optional[list]
    source_behavior_role_ema: Optional[list]
    source_behavior_role_drift: float
    destination_behavior_role_vector: Optional[list]
    destination_behavior_role_ema: Optional[list]
    destination_behavior_role_drift: float
    ttl: int
    current_window_obj: Any = None


class History:
    """
    历史类：维护仍在 TTL 生命周期内的节点和边长期状态。

    生命周期机制:
    1) 默认生存周期 life_windows=30
    2) 在新窗口中再次出现 -> TTL 重置为 life_windows
    3) 在新窗口中未出现 -> TTL 减 1
    4) TTL 降到 0 -> 从历史中删除
    5) retained_window_time 记录当前历史有效保留的窗口时间，最大不超过 life_windows

    近期节点/边明细机制:
    - 最近 detail_windows 个窗口的节点和边对象快照由 RecentEdgeDetails 单独维护
    - 该明细队列只随窗口推进滑动，不因同 key 节点/边再次出现而刷新旧快照年龄
    - History.nodes/edges 负责长期状态；同 key 节点/边再次出现会刷新 TTL

    推荐时序:
    1) advance_window() 提交上一窗口暂存并推进历史状态
    2) update_with_graph(current_graph) 暂存当前窗口图数据
    或直接调用 process_new_window(current_graph)。
    """

    def __init__(
        self,
        life_windows: int = 30,
        detail_windows: int = 5,
        edge_scorer: Optional[Callable[[Any], Optional[float]]] = None,
    ):
        if life_windows <= 0:
            raise ValueError("life_windows 必须大于 0。")
        if detail_windows <= 0:
            raise ValueError("detail_windows 必须大于 0。")

        self.life_windows = life_windows
        self.detail_windows = detail_windows
        self.total_committed_window_count = 0
        self.retained_window_time = 0
        # 预留边评分入口；可注入自定义异常分数函数
        self.edge_scorer = edge_scorer or self._edge_score_placeholder
        # key -> HistoryRecord(obj=节点长期对象, in_current_graph=是否刚提交窗口出现)
        self.nodes: Dict[Hashable, HistoryRecord] = {}
        # key -> HistoryRecord(obj=边长期聚合对象, in_current_graph=是否刚提交窗口出现)
        self.edges: Dict[Hashable, HistoryRecord] = {}
        # 上一窗口攻击边的目的 IP 集合，用于判断“攻击目的转源”
        self.previous_attack_dst_ips: Set[str] = set()
        # 最近五个窗口攻击边快照；独立滑动，不使用长期边 TTL
        self.recent_attack_edge_history = RecentAttackEdgeHistory(max_windows=5)
        self._recent_attack_recorded_window_count = 0
        self.suspicious_edge_history = SuspiciousEdgeHistory(
            theta_suspicious=0.55,
            theta_attack=0.70,
            evidence_decay=0.85,
            release_threshold=0.40,
            min_reinforcing_signals=3,
        )
        self.baseline_excluded_edge_keys: Set[Hashable] = set()
        self.current_node_keys: Set[Hashable] = set()
        self.current_edge_keys: Set[Hashable] = set()
        # 当前窗口节点行为角色得分缓存，避免逐边重复计算节点分位数
        self.behavior_role_score_cache: Dict[str, Dict[str, float]] = {}
        self._behavior_role_score_cache_options: Optional[Tuple[bool, float]] = None
        # 当前窗口暂存区：进入下一个窗口时再提交到历史
        self.pending_nodes: Dict[Hashable, Any] = {}
        self.pending_edges = EdgeWindowBuffer()
        self._pending_window_staged = False
        # 最近窗口节点/边明细保留器；不参与 TTL，只供近期明细类分数使用
        self.recent_edge_details = RecentEdgeDetails(max_detail_windows=detail_windows)

    @staticmethod
    def _new_record(obj: Any, ttl: int) -> HistoryRecord:
        return HistoryRecord(
            obj=obj,
            current_window_obj=obj,
            in_current_graph=True,
            edge_label="unknown",
            edge_score=0.0,
            appeared_window_count=1,
            source_behavior_role_vector=None,
            source_behavior_role_ema=None,
            source_behavior_role_drift=0.0,
            destination_behavior_role_vector=None,
            destination_behavior_role_ema=None,
            destination_behavior_role_drift=0.0,
            ttl=ttl,
        )

    def _clear_behavior_role_score_cache(self):
        self.behavior_role_score_cache = {}
        self._behavior_role_score_cache_options = None

    @staticmethod
    def _to_mapping(
        items: Optional[Union[Iterable[Hashable], Mapping[Hashable, Any]]]
    ) -> Dict[Hashable, Any]:
        """
        将输入统一为字典格式:
        - 若传入 Mapping，直接转为 dict
        - 若传入 Iterable（只包含 key），值使用 None 占位
        - 若传入 None，返回空 dict
        """
        if items is None:
            return {}
        if isinstance(items, Mapping):
            return dict(items)
        return {key: None for key in items}

    def _upsert_group(
        self,
        current_items: Dict[Hashable, Any],
        store: Dict[Hashable, HistoryRecord],
    ):
        """
        用当前窗口中出现的对象更新单一分组（节点或边）。

        规则:
        1) 已存在: 更新当前窗口快照、ttl 重置为 life_windows、in_current_graph=1，出现窗口次数+1
        2) 新出现: 新建记录，in_current_graph=1，ttl=life_windows，出现窗口次数=1
        说明:
            不在这里处理“未出现对象”的 in_current_graph 和 ttl 下降，
            这些在 advance_window() 中统一处理。
        """
        for key, value in current_items.items():
            if key in store:
                record = store[key]
                if value is not None:
                    record.obj = value
                record.current_window_obj = value
                record.in_current_graph = True
                record.ttl = self.life_windows
                record.appeared_window_count += 1
            else:
                store[key] = self._new_record(value, self.life_windows)

    def update_with_window(
        self,
        window_nodes: Optional[Union[Iterable[Hashable], Mapping[Hashable, Any]]] = None,
        window_edges: Optional[Union[Iterable[Hashable], Mapping[Hashable, Any]]] = None,
    ):
        """
        暂存单个窗口的数据，等待进入下一个窗口时提交到历史。

        参数:
            window_nodes: 本窗口节点（可传 key 集合，或 key->对象字典）
            window_edges: 本窗口边（可传 key 集合，或 key->对象字典）
        """
        nodes_map = self._to_mapping(window_nodes)
        edges_map = self._to_mapping(window_edges)

        self.pending_nodes = nodes_map
        self.pending_edges.clear()
        self.pending_edges.stage_edges(edges_map)
        self._pending_window_staged = True
        self.recent_edge_details.stage_window(
            window_nodes=nodes_map,
            window_edges=edges_map,
        )
        self._clear_behavior_role_score_cache()

    def update_with_graph(self, graph: Any):
        """
        使用图对象暂存当前窗口数据。
        约定 graph 具有:
            - graph.nodes: dict
            - graph.edges: dict
        """
        graph_nodes = getattr(graph, "nodes", {})
        graph_edges = getattr(graph, "edges", {})
        self.update_with_window(window_nodes=graph_nodes, window_edges=graph_edges)

    def stage_with_window(
        self,
        window_nodes: Optional[Union[Iterable[Hashable], Mapping[Hashable, Any]]] = None,
        window_edges: Optional[Union[Iterable[Hashable], Mapping[Hashable, Any]]] = None,
    ):
        """显式暂存窗口数据；等价于 update_with_window。"""
        self.update_with_window(window_nodes=window_nodes, window_edges=window_edges)

    def stage_with_graph(self, graph: Any):
        """显式暂存图对象；等价于 update_with_graph。"""
        self.update_with_graph(graph)

    def _commit_edge_group(
        self,
        current_edges: Dict[Hashable, Any],
        store: Dict[Hashable, HistoryRecord],
    ):
        """
        将已完成窗口的边提交到历史。

        同 key 边不直接替换长期历史对象，而是累计到历史边对象上；
        current_window_obj 仍保留刚提交窗口的原始边对象，供当前窗口分数使用。
        """
        for key, value in current_edges.items():
            if key in store:
                record = store[key]
                record.current_window_obj = value
                if value is not None:
                    if record.obj is not None and hasattr(
                        record.obj, "merge_from_window_edge"
                    ):
                        record.obj.merge_from_window_edge(
                            value, max_detail_windows=self.detail_windows
                        )
                    else:
                        record.obj = value
                        if hasattr(record.obj, "retain_recent_window_details"):
                            record.obj.retain_recent_window_details(
                                max_detail_windows=self.detail_windows
                            )
                record.in_current_graph = True
                record.ttl = self.life_windows
                record.appeared_window_count += 1
            else:
                if value is not None and hasattr(value, "retain_recent_window_details"):
                    value.retain_recent_window_details(
                        max_detail_windows=self.detail_windows
                    )
                store[key] = self._new_record(value, self.life_windows)

    def _advance_edge_detail_windows(self):
        """兼容旧调用：委托 RecentEdgeDetails 推进近窗明细队列。"""
        self.recent_edge_details.advance_window()

    def commit_pending_window(self):
        """
        将上一窗口暂存数据提交到历史。

        节点更新当前快照；边按 merge_from_window_edge 进行长期累计。
        """
        window_committed = self._pending_window_staged
        committed_node_keys = set(self.pending_nodes)
        committed_edge_keys = set(self.pending_edges.edges)

        if self.pending_nodes:
            self._upsert_group(self.pending_nodes, self.nodes)
            self.pending_nodes = {}

        if not self.pending_edges.is_empty():
            self._commit_edge_group(dict(self.pending_edges.items()), self.edges)
            self.pending_edges.clear()

        if window_committed:
            self.total_committed_window_count += 1
            self.retained_window_time = min(
                self.total_committed_window_count,
                self.life_windows,
            )
            self._pending_window_staged = False
            self.current_node_keys = committed_node_keys
            self.current_edge_keys = committed_edge_keys

        self._clear_behavior_role_score_cache()
        return window_committed

    def process_new_window(self, graph: Any):
        """
        标准窗口处理入口（推荐使用）:
        1) 推进窗口状态，并提交上一窗口暂存数据
        2) 更新已提交窗口节点的行为角色向量与 EMA
        3) 给已提交窗口边评分
        4) 记录已提交窗口攻击边目的 IP 集合
        5) 暂存新窗口图数据，等待进入下一个窗口时提交
        """
        self.advance_window()
        self.update_node_behavior_role_vectors()
        self.score_and_update_current_edges()
        self.record_previous_attack_destinations()
        self.update_with_graph(graph)

    def record_previous_attack_destinations(
        self,
        attack_edge_keys: Optional[Iterable[Hashable]] = None,
    ):
        """
        记录上一窗口攻击边的目的 IP 集合 AttackDst_{t-1}。

        该函数应在提交并评分上一窗口后调用。
        """
        selected_attack_edges = (
            set(attack_edge_keys) if attack_edge_keys is not None else None
        )
        attack_dst_ips = set()
        candidate_edge_keys = (
            self.current_edge_keys
            if self.current_edge_keys
            else set(self.edges)
        )
        for edge_key in candidate_edge_keys:
            record = self.edges.get(edge_key)
            if record is None:
                continue
            if not record.in_current_graph:
                continue
            if selected_attack_edges is None:
                if record.edge_label != "attack":
                    continue
            elif edge_key not in selected_attack_edges:
                continue

            edge_obj = record.obj
            if edge_obj is None:
                continue

            dst_ip = getattr(edge_obj, "dst_ip", None)
            if dst_ip is not None:
                attack_dst_ips.add(str(dst_ip))

        self.previous_attack_dst_ips = attack_dst_ips
        self.record_recent_attack_edges(selected_attack_edges)

    def record_recent_attack_edges(
        self,
        attack_edge_keys: Optional[Iterable[Hashable]] = None,
    ):
        """
        记录刚提交窗口中的攻击边到独立五窗口队列。

        同一已提交窗口重复调用不会重复追加；无攻击边时仍记录空窗口，
        使攻击相似度的窗口距离随真实时间推进。
        """
        if self.total_committed_window_count <= 0:
            return
        if (
            self._recent_attack_recorded_window_count
            == self.total_committed_window_count
        ):
            return

        selected_attack_edges = (
            set(attack_edge_keys) if attack_edge_keys is not None else None
        )
        candidate_edge_keys = (
            self.current_edge_keys
            if self.current_edge_keys
            else set(self.edges)
        )
        current_attack_edge_keys = []
        for edge_key in candidate_edge_keys:
            record = self.edges.get(edge_key)
            if record is None or not record.in_current_graph:
                continue
            if selected_attack_edges is None:
                if record.edge_label != "attack":
                    continue
            elif edge_key not in selected_attack_edges:
                continue
            current_attack_edge_keys.append(edge_key)
        self.recent_attack_edge_history.record_window(current_attack_edge_keys)
        self._recent_attack_recorded_window_count = self.total_committed_window_count

    def get_previous_attack_destination_ips(self) -> Set[str]:
        """返回上一窗口攻击边目的 IP 集合的副本。"""
        return set(self.previous_attack_dst_ips)

    def previous_attack_destination_to_source_mark(self, edge_obj: Any) -> int:
        """
        上一窗口攻击目的转源标记:
            1, 如果 src_ip(edge_obj) 属于 AttackDst_{t-1}
            0, 否则
        """
        src_ip = getattr(edge_obj, "src_ip", None)
        if src_ip is None:
            return 0
        return int(str(src_ip) in self.previous_attack_dst_ips)

    def mark_keep_in_current_graph(
        self,
        node_keys: Optional[Iterable[Hashable]] = None,
        edge_keys: Optional[Iterable[Hashable]] = None,
        keep: bool = True,
    ):
        """
        兼容旧调用的无状态入口。

        keep_in_current_graph 属性已取消；历史保留由全局窗口 life_windows/ttl 控制，
        切换窗口时 in_current_graph 只表示刚提交的上一窗口可见对象。
        """
        _ = node_keys
        _ = edge_keys
        _ = keep

    def set_edge_label(self, edge_keys: Iterable[Hashable], label: str):
        """
        设置边类型标识。

        参数:
            edge_keys: 要设置的边 key 列表
            label: 'attack' / 'normal' / 'unknown'
        """
        valid_labels = {"attack", "normal", "unknown"}
        if label not in valid_labels:
            raise ValueError("label 必须是 'attack'、'normal' 或 'unknown'。")

        for edge_key in edge_keys:
            if edge_key in self.edges:
                self.edges[edge_key].edge_label = label

    def set_baseline_excluded_edges(self, edge_keys: Iterable[Hashable]):
        """设置当前禁止参与正常行为基线更新的边。"""
        self.baseline_excluded_edge_keys = set(edge_keys)
        self._clear_behavior_role_score_cache()

    def is_edge_baseline_eligible(self, edge_key: Hashable) -> bool:
        """可疑边和已判定攻击边不参与正常行为基线。"""
        if edge_key in self.baseline_excluded_edge_keys:
            return False
        record = self.edges.get(edge_key)
        return record is None or record.edge_label != "attack"

    @staticmethod
    def _advance_group(store: Dict[Hashable, HistoryRecord]):
        """
        窗口推进时更新单一分组（节点或边）:
        1) ttl -= 1
        2) ttl <= 0 删除
        3) in_current_graph 置为 0
        """
        for key in list(store.keys()):
            record = store[key]
            record.ttl -= 1
            if record.ttl <= 0:
                store.pop(key, None)
                continue

            record.in_current_graph = False
            record.current_window_obj = None

    @staticmethod
    def _edge_score_placeholder(edge_obj: Any) -> Optional[float]:
        """
        边评分占位函数。
        返回 None 代表“暂不打分”，后续由你替换为真实评估逻辑。
        """
        _ = edge_obj
        return None

    def score_and_update_current_edges(
        self,
        attack_threshold: float = 0.7,
        normal_threshold: float = 0.3,
    ):
        """
        给当前窗口中的边打分，并基于分数更新边类型属性 edge_label。

        规则:
            - score >= attack_threshold -> attack
            - score <= normal_threshold -> normal
            - 其他 -> unknown
        """
        if normal_threshold > attack_threshold:
            raise ValueError("normal_threshold 不能大于 attack_threshold。")

        for record in self.edges.values():
            if not record.in_current_graph:
                continue

            try:
                edge_obj = (
                    record.current_window_obj
                    if record.current_window_obj is not None
                    else record.obj
                )
                score = self.edge_scorer(edge_obj)
            except Exception:
                score = None

            if score is None:
                record.edge_score = 0.0
                record.edge_label = "unknown"
                continue

            score = float(score)
            score = min(max(score, 0.0), 1.0)
            record.edge_score = score

            if score >= attack_threshold:
                record.edge_label = "attack"
            elif score <= normal_threshold:
                record.edge_label = "normal"
            else:
                record.edge_label = "unknown"

    @staticmethod
    def _update_ema_vector(old_vector, current_vector, alpha):
        """更新向量 EMA。"""
        if current_vector is None:
            return old_vector
        current_vector = [float(value) for value in current_vector]
        if not old_vector:
            return current_vector

        length = min(len(old_vector), len(current_vector))
        if length == 0:
            return current_vector

        ema = [
            alpha * current_vector[i] + (1 - alpha) * float(old_vector[i])
            for i in range(length)
        ]
        if len(current_vector) > length:
            ema.extend(current_vector[length:])
        return ema

    @staticmethod
    def _mean_abs_distance(vector_a, vector_b):
        if not vector_a or not vector_b:
            return 0.0
        length = min(len(vector_a), len(vector_b))
        if length == 0:
            return 0.0
        return sum(abs(float(vector_a[i]) - float(vector_b[i])) for i in range(length)) / length

    def update_node_behavior_role_vectors(
        self, alpha: float = 0.3, current_window_only: bool = True
    ):
        """
        更新历史节点中的行为角色向量与 EMA。

        说明:
            - source_behavior_role_drift 使用更新前的 source_behavior_role_ema 计算
            - EMA 更新公式: EMA = alpha * R_now + (1 - alpha) * EMA_old
        """
        if not (0 < alpha <= 1):
            raise ValueError("alpha 必须在 (0, 1] 范围内。")

        baseline_node_ips: Set[str] = set()
        for edge_key, edge_record in self.edges.items():
            if current_window_only and not edge_record.in_current_graph:
                continue
            if not self.is_edge_baseline_eligible(edge_key):
                continue
            edge_obj = (
                edge_record.current_window_obj
                if current_window_only
                else edge_record.obj
            )
            if edge_obj is None:
                continue
            baseline_node_ips.add(str(getattr(edge_obj, "src_ip", "")))
            baseline_node_ips.add(str(getattr(edge_obj, "dst_ip", "")))

        node_items = []
        for key, record in self.nodes.items():
            if current_window_only and not record.in_current_graph:
                continue
            node_obj = (
                record.current_window_obj
                if current_window_only
                else record.obj
            )
            if node_obj is None:
                continue
            node_ip = str(getattr(node_obj, "ip", key))
            if current_window_only and node_ip not in baseline_node_ips:
                continue
            node_items.append((node_ip, record, node_obj))

        out_degrees = {node_ip: 0.0 for node_ip, _, _ in node_items}
        in_degrees = {node_ip: 0.0 for node_ip, _, _ in node_items}

        send_bytes = {node_ip: 0.0 for node_ip, _, _ in node_items}
        receive_bytes = {node_ip: 0.0 for node_ip, _, _ in node_items}
        source_destination_ips: Dict[str, Set[Any]] = {
            node_ip: set() for node_ip, _, _ in node_items
        }
        source_destination_ports: Dict[str, Set[Any]] = {
            node_ip: set() for node_ip, _, _ in node_items
        }
        destination_source_ips: Dict[str, Set[Any]] = {
            node_ip: set() for node_ip, _, _ in node_items
        }
        destination_ports: Dict[str, Set[Any]] = {
            node_ip: set() for node_ip, _, _ in node_items
        }

        source_small_packets = {node_ip: 0.0 for node_ip, _, _ in node_items}
        source_packets = {node_ip: 0.0 for node_ip, _, _ in node_items}
        source_syn = {node_ip: 0.0 for node_ip, _, _ in node_items}
        source_syn_ack = {node_ip: 0.0 for node_ip, _, _ in node_items}
        source_ack = {node_ip: 0.0 for node_ip, _, _ in node_items}

        destination_small_packets = {node_ip: 0.0 for node_ip, _, _ in node_items}
        destination_packets = {node_ip: 0.0 for node_ip, _, _ in node_items}
        destination_syn = {node_ip: 0.0 for node_ip, _, _ in node_items}
        destination_syn_ack = {node_ip: 0.0 for node_ip, _, _ in node_items}
        destination_ack = {node_ip: 0.0 for node_ip, _, _ in node_items}

        for edge_key, record in self.edges.items():
            if current_window_only and not record.in_current_graph:
                continue
            if not self.is_edge_baseline_eligible(edge_key):
                continue

            edge_obj = (
                record.current_window_obj
                if current_window_only
                else record.obj
            )
            if edge_obj is None:
                continue

            src_ip = str(getattr(edge_obj, "src_ip", ""))
            dst_ip = str(getattr(edge_obj, "dst_ip", ""))
            dst_port = getattr(edge_obj, "dst_port", None)
            payload_len = float(getattr(edge_obj, "payload_len", 0.0))
            smallpacket = float(getattr(edge_obj, "smallpacket", 0.0))
            edgepacketnum = float(getattr(edge_obj, "edgepacketnum", 0.0))
            syn_count = float(getattr(edge_obj, "syn_count", 0.0))
            syn_ack_count = float(getattr(edge_obj, "syn_ack_count", 0.0))
            ack_count = float(getattr(edge_obj, "ack_count", 0.0))

            if src_ip in send_bytes:
                out_degrees[src_ip] += 1.0
                send_bytes[src_ip] += payload_len
                source_destination_ips[src_ip].add(dst_ip)
                source_destination_ports[src_ip].add(dst_port)
                source_small_packets[src_ip] += smallpacket
                source_packets[src_ip] += edgepacketnum
                source_syn[src_ip] += syn_count
                source_syn_ack[src_ip] += syn_ack_count
                source_ack[src_ip] += ack_count

            if dst_ip in receive_bytes:
                in_degrees[dst_ip] += 1.0
                receive_bytes[dst_ip] += payload_len
                destination_source_ips[dst_ip].add(src_ip)
                destination_ports[dst_ip].add(dst_port)
                destination_small_packets[dst_ip] += smallpacket
                destination_packets[dst_ip] += edgepacketnum
                destination_syn[dst_ip] += syn_count
                destination_syn_ack[dst_ip] += syn_ack_count
                destination_ack[dst_ip] += ack_count

        out_degree_quantiles = self._quantile_cache(out_degrees)
        in_degree_quantiles = self._quantile_cache(in_degrees)
        send_byte_quantiles = self._quantile_cache(send_bytes)
        receive_byte_quantiles = self._quantile_cache(receive_bytes)
        source_destination_quantiles = self._quantile_cache(
            {key: len(value) for key, value in source_destination_ips.items()}
        )
        source_port_quantiles = self._quantile_cache(
            {key: len(value) for key, value in source_destination_ports.items()}
        )
        destination_source_quantiles = self._quantile_cache(
            {key: len(value) for key, value in destination_source_ips.items()}
        )
        destination_port_quantiles = self._quantile_cache(
            {key: len(value) for key, value in destination_ports.items()}
        )

        eps = 1e-8
        for node_ip, record, _ in node_items:
            old_source_ema = record.source_behavior_role_ema
            old_destination_ema = record.destination_behavior_role_ema

            if old_source_ema and len(old_source_ema) > 4:
                destination_burst = max(
                    source_destination_quantiles.get(node_ip, 0.0)
                    - float(old_source_ema[4]),
                    0.0,
                )
            else:
                destination_burst = 0.0

            if old_source_ema and len(old_source_ema) > 5:
                port_burst = max(
                    source_port_quantiles.get(node_ip, 0.0)
                    - float(old_source_ema[5]),
                    0.0,
                )
            else:
                port_burst = 0.0

            source_vector = [
                out_degree_quantiles.get(node_ip, 0.0),
                in_degree_quantiles.get(node_ip, 0.0),
                send_byte_quantiles.get(node_ip, 0.0),
                receive_byte_quantiles.get(node_ip, 0.0),
                min(destination_burst, 1.0),
                min(port_burst, 1.0),
                source_small_packets.get(node_ip, 0.0)
                / (source_packets.get(node_ip, 0.0) + eps),
                max(
                    source_syn.get(node_ip, 0.0)
                    - source_syn_ack.get(node_ip, 0.0)
                    - source_ack.get(node_ip, 0.0),
                    0.0,
                )
                / (source_syn.get(node_ip, 0.0) + eps),
            ]

            destination_vector = [
                out_degree_quantiles.get(node_ip, 0.0),
                in_degree_quantiles.get(node_ip, 0.0),
                send_byte_quantiles.get(node_ip, 0.0),
                receive_byte_quantiles.get(node_ip, 0.0),
                destination_source_quantiles.get(node_ip, 0.0),
                destination_port_quantiles.get(node_ip, 0.0),
                destination_small_packets.get(node_ip, 0.0)
                / (destination_packets.get(node_ip, 0.0) + eps),
                max(
                    destination_syn.get(node_ip, 0.0)
                    - destination_syn_ack.get(node_ip, 0.0)
                    - destination_ack.get(node_ip, 0.0),
                    0.0,
                )
                / (destination_syn.get(node_ip, 0.0) + eps),
            ]

            record.source_behavior_role_vector = source_vector
            record.source_behavior_role_drift = self._mean_abs_distance(
                source_vector, old_source_ema
            )
            record.source_behavior_role_ema = self._update_ema_vector(
                old_source_ema, source_vector, alpha
            )

            record.destination_behavior_role_vector = destination_vector
            record.destination_behavior_role_drift = self._mean_abs_distance(
                destination_vector, old_destination_ema
            )
            record.destination_behavior_role_ema = self._update_ema_vector(
                old_destination_ema, destination_vector, alpha
            )

        self._clear_behavior_role_score_cache()

    def advance_window(self):
        """
        进入下一个窗口时调用:
        - 遍历当前历史中的所有节点和边，ttl - 1
        - ttl 为 0 时删除
        - 将除“保留标记”外的 in_current_graph 置为 0
        - 将上一窗口暂存数据提交到历史
        - 若成功提交窗口，则更新 retained_window_time
        """
        self._advance_group(self.nodes)
        self._advance_group(self.edges)
        self.recent_edge_details.advance_window()
        self.commit_pending_window()

    @staticmethod
    def _quantile_cache(
        values_by_key: Mapping[str, Union[int, float]],
    ) -> Dict[str, float]:
        """批量计算分位数分数：历史中 <= 当前值 的数量 / 历史总数量。"""
        if not values_by_key:
            return {}

        sorted_values = sorted(values_by_key.values())
        total = len(sorted_values)
        return {
            key: bisect_right(sorted_values, value) / total
            for key, value in values_by_key.items()
        }

    @staticmethod
    def _node_degree_value(node_obj: Any, degree_name: str) -> int:
        degree_value = getattr(node_obj, degree_name, 0)
        if callable(degree_value):
            degree_value = degree_value()
        return int(degree_value)

    def build_behavior_role_score_cache(
        self, current_window_only: bool = True, eps: float = 1e-8
    ) -> Dict[str, Dict[str, float]]:
        """
        批量构建最终行为角色异常得分所需的节点侧缓存。

        缓存内容:
            - 源节点出度分位数
            - 目的节点入度分位数
            - 节点收发字节比归一化
            - 源/目的节点行为角色漂移距离
        """
        node_items = []
        for key, record in self.nodes.items():
            if current_window_only and not record.in_current_graph:
                continue
            node_obj = (
                record.current_window_obj
                if current_window_only
                else record.obj
            )
            if node_obj is None:
                continue
            node_ip = str(getattr(node_obj, "ip", key))
            node_items.append((node_ip, record, node_obj))

        out_degrees = {
            node_ip: self._node_degree_value(node_obj, "out_degree")
            for node_ip, _, node_obj in node_items
        }
        in_degrees = {
            node_ip: self._node_degree_value(node_obj, "in_degree")
            for node_ip, _, node_obj in node_items
        }
        out_degree_quantiles = self._quantile_cache(out_degrees)
        in_degree_quantiles = self._quantile_cache(in_degrees)

        send_bytes = {node_ip: 0.0 for node_ip, _, _ in node_items}
        receive_bytes = {node_ip: 0.0 for node_ip, _, _ in node_items}

        for record in self.edges.values():
            if current_window_only and not record.in_current_graph:
                continue

            edge_obj = (
                record.current_window_obj
                if current_window_only
                else record.obj
            )
            if edge_obj is None:
                continue

            src_ip = str(getattr(edge_obj, "src_ip", ""))
            dst_ip = str(getattr(edge_obj, "dst_ip", ""))
            payload_len = float(getattr(edge_obj, "payload_len", 0.0))

            if src_ip in send_bytes:
                send_bytes[src_ip] += payload_len
            if dst_ip in receive_bytes:
                receive_bytes[dst_ip] += payload_len

        cache = {}
        for node_ip, record, _ in node_items:
            ratio = send_bytes.get(node_ip, 0.0) / (
                receive_bytes.get(node_ip, 0.0) + eps
            )
            ratio_with_eps = ratio + eps
            normalized_ratio = ratio_with_eps / (1.0 + ratio_with_eps)

            cache[node_ip] = {
                "source_out_degree_quantile": float(
                    out_degree_quantiles.get(node_ip, 0.0)
                ),
                "destination_in_degree_quantile": float(
                    in_degree_quantiles.get(node_ip, 0.0)
                ),
                "normalized_byte_ratio": float(normalized_ratio),
                "source_role_drift": float(
                    getattr(record, "source_behavior_role_drift", 0.0)
                ),
                "destination_role_drift": float(
                    getattr(record, "destination_behavior_role_drift", 0.0)
                ),
            }

        self.behavior_role_score_cache = cache
        self._behavior_role_score_cache_options = (current_window_only, eps)
        return cache

    def get_behavior_role_score_components_for_edge(
        self,
        edge_obj: Any,
        current_window_only: bool = True,
        eps: float = 1e-8,
    ) -> Dict[str, float]:
        """返回单条边计算最终行为角色异常得分所需的分项。"""
        cache_options = (current_window_only, eps)
        if (
            not self.behavior_role_score_cache
            or self._behavior_role_score_cache_options != cache_options
        ):
            self.build_behavior_role_score_cache(current_window_only, eps)

        src_ip = str(getattr(edge_obj, "src_ip", ""))
        dst_ip = str(getattr(edge_obj, "dst_ip", ""))
        src_cache = self.behavior_role_score_cache.get(src_ip, {})
        dst_cache = self.behavior_role_score_cache.get(dst_ip, {})

        return {
            "source_out_degree_quantile": float(
                src_cache.get("source_out_degree_quantile", 0.0)
            ),
            "destination_in_degree_quantile": float(
                dst_cache.get("destination_in_degree_quantile", 0.0)
            ),
            "normalized_byte_ratio": float(
                src_cache.get("normalized_byte_ratio", 0.0)
            ),
            "source_role_drift": float(src_cache.get("source_role_drift", 0.0)),
            "destination_role_drift": float(
                dst_cache.get("destination_role_drift", 0.0)
            ),
            "previous_attack_dst_to_src_mark": float(
                self.previous_attack_destination_to_source_mark(edge_obj)
            ),
        }

    @staticmethod
    def _behavior_role_score_from_components(components: Mapping[str, float]) -> float:
        score = sum(
            BEHAVIOR_ROLE_WEIGHTS[name] * float(components.get(name, 0.0))
            for name in BEHAVIOR_ROLE_WEIGHTS
        )
        return min(max(score, 0.0), 1.0)

    def compute_behavior_role_anomaly_scores(
        self,
        graph_or_edges: Any,
        current_window_only: bool = True,
        eps: float = 1e-8,
        return_components: bool = False,
    ) -> Dict[Hashable, Any]:
        """
        批量计算行为角色异常分数。

        固定先调用 build_behavior_role_score_cache()，再按边查缓存，
        避免逐边调用节点版总分时重复遍历历史节点和边。
        """
        edges = getattr(graph_or_edges, "edges", graph_or_edges)
        if edges is None:
            return {}

        self.build_behavior_role_score_cache(current_window_only, eps)

        results: Dict[Hashable, Any] = {}
        if isinstance(edges, Mapping):
            for edge_key, edge_obj in edges.items():
                components = self.get_behavior_role_score_components_for_edge(
                    edge_obj,
                    current_window_only=current_window_only,
                    eps=eps,
                )
                score = self._behavior_role_score_from_components(components)
                if return_components:
                    components["behavior_role_anomaly_score"] = score
                    results[edge_key] = components
                else:
                    results[edge_key] = score
        else:
            for edge_obj in edges:
                edge_key = id(edge_obj)
                components = self.get_behavior_role_score_components_for_edge(
                    edge_obj,
                    current_window_only=current_window_only,
                    eps=eps,
                )
                score = self._behavior_role_score_from_components(components)
                if return_components:
                    components["behavior_role_anomaly_score"] = score
                    results[edge_key] = components
                else:
                    results[edge_key] = score

        return results

    def summary(self) -> Dict[str, int]:
        """返回当前历史中的节点/边总数与当前窗口出现数量。"""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "node_in_current_graph_count": sum(
                1 for record in self.nodes.values() if record.in_current_graph
            ),
            "edge_in_current_graph_count": sum(
                1 for record in self.edges.values() if record.in_current_graph
            ),
            "pending_node_count": len(self.pending_nodes),
            "pending_edge_count": len(self.pending_edges),
            "total_committed_window_count": self.total_committed_window_count,
            "retained_window_time": self.retained_window_time,
            "attack_edge_count": sum(
                1 for record in self.edges.values() if record.edge_label == "attack"
            ),
            "normal_edge_count": sum(
                1 for record in self.edges.values() if record.edge_label == "normal"
            ),
            "unknown_edge_count": sum(
                1 for record in self.edges.values() if record.edge_label == "unknown"
            ),
            "recent_detail_window_count": len(self.recent_edge_details.windows),
            "recent_detail_node_window_count": len(
                self.recent_edge_details.node_windows
            ),
            "recent_detail_node_count": len(self.recent_edge_details.recent_node_keys()),
            "recent_detail_edge_count": len(self.recent_edge_details.recent_edge_keys()),
        }

    def get_ttl_snapshot(self) -> Tuple[Dict[Hashable, int], Dict[Hashable, int]]:
        """返回节点和边的 TTL 快照（副本）。"""
        node_ttl = {key: record.ttl for key, record in self.nodes.items()}
        edge_ttl = {key: record.ttl for key, record in self.edges.items()}
        return node_ttl, edge_ttl
