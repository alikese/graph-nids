from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Set

from feature.attack_similar.previous_attack_edge import edge_key_from_edge_obj


@dataclass
class RecentOccurrenceRecord:
    """
    单个节点/边在最近窗口中的环形出现记录。

    字段含义:
        obj: 最近一次出现时的对象，用于按 key 取最新快照
        a: 长度为 max_windows 的 TTL 环；正数表示该槽位仍在最近窗口内
        b: 最旧有效出现位置
        c: 最新有效出现位置
        sum: 最近窗口内出现的窗口数
    """

    obj: Any
    max_windows: int
    a: list[int]
    b: int = 0
    c: int = -1
    sum: int = 0

    @classmethod
    def create(cls, obj: Any, max_windows: int):
        return cls(obj=obj, max_windows=max_windows, a=[0] * max_windows)

    def advance_window(self):
        """窗口推进时所有有效槽位 TTL -1，并从 b 开始移除过期出现。"""
        for idx, ttl in enumerate(self.a):
            if ttl > 0:
                self.a[idx] = ttl - 1

        while self.sum > 0 and self.a[self.b] <= 0:
            self.a[self.b] = 0
            self.b = (self.b + 1) % self.max_windows
            self.sum -= 1

        if self.sum <= 0:
            self.b = 0
            self.c = -1
            self.sum = 0

    def mark_seen(self, obj: Any):
        """记录当前窗口出现一次，写入 c+1 mod max_windows。"""
        next_index = (self.c + 1) % self.max_windows if self.c >= 0 else 0
        if self.a[next_index] > 0:
            if next_index == self.b:
                self.b = (self.b + 1) % self.max_windows
            else:
                self.sum = max(self.sum - 1, 0)
        else:
            self.sum += 1

        self.obj = obj
        self.a[next_index] = self.max_windows
        if self.sum == 1:
            self.b = next_index
        self.c = next_index
        self.sum = min(self.sum, self.max_windows)

    def occurrence_count(self, max_windows: Optional[int] = None) -> int:
        """返回最近 max_windows 个窗口内的出现窗口数。"""
        if max_windows is None or max_windows >= self.max_windows:
            return self.sum
        if max_windows <= 0:
            raise ValueError("max_windows must be greater than 0.")

        threshold = self.max_windows - max_windows
        return sum(1 for ttl in self.a if ttl > threshold)


class RecentEdgeDetails:
    """
    最近窗口节点/边明细保留器。

    职责只包含最近 max_detail_windows 个已提交窗口的节点和边对象快照。
    它不参与 History 的 TTL 生命周期，也不做长期累计；同一节点或边再次出现时，
    只会进入新的窗口快照，不会刷新旧窗口快照的年龄。
    """

    def __init__(self, max_detail_windows: int = 5):
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows must be greater than 0.")

        self.max_detail_windows = max_detail_windows
        self.edge_occurrences: Dict[Hashable, RecentOccurrenceRecord] = {}
        self.node_occurrences: Dict[Hashable, RecentOccurrenceRecord] = {}
        self.windows: list[Dict[Hashable, Any]] = []
        self.node_windows: list[Dict[Hashable, Any]] = []
        self.pending_nodes: Optional[Dict[Hashable, Any]] = None
        self.pending_edges: Optional[Dict[Hashable, Any]] = None

    @staticmethod
    def _to_mapping(
        items: Optional[Mapping[Hashable, Any] | Iterable[Hashable]],
    ) -> Dict[Hashable, Any]:
        if items is None:
            return {}
        if isinstance(items, Mapping):
            return dict(items)
        return {key: None for key in items}

    @staticmethod
    def _resolve_edge_key(edge_or_key: Any) -> Optional[Hashable]:
        if isinstance(edge_or_key, tuple):
            return edge_or_key
        return edge_key_from_edge_obj(edge_or_key)

    @staticmethod
    def _resolve_node_key(node_or_key: Any) -> Optional[Hashable]:
        if node_or_key is None:
            return None
        node_ip = getattr(node_or_key, "ip", None)
        if node_ip is not None:
            return str(node_ip)
        return node_or_key

    @staticmethod
    def _advance_occurrences(
        records: Dict[Hashable, RecentOccurrenceRecord],
    ):
        for key in list(records.keys()):
            record = records[key]
            record.advance_window()
            if record.sum <= 0:
                records.pop(key, None)

    def _mark_occurrences(
        self,
        records: Dict[Hashable, RecentOccurrenceRecord],
        items: Mapping[Hashable, Any],
    ):
        for key, obj in items.items():
            if key not in records:
                records[key] = RecentOccurrenceRecord.create(
                    obj,
                    self.max_detail_windows,
                )
            records[key].mark_seen(obj)

    def stage_edges(
        self,
        window_edges: Optional[Mapping[Hashable, Any] | Iterable[Hashable]] = None,
    ):
        """暂存当前窗口边明细，等待窗口推进时提交。"""
        self.pending_edges = self._to_mapping(window_edges)

    def stage_nodes(
        self,
        window_nodes: Optional[Mapping[Hashable, Any] | Iterable[Hashable]] = None,
    ):
        """暂存当前窗口节点明细，等待窗口推进时提交。"""
        self.pending_nodes = self._to_mapping(window_nodes)

    def stage_window(
        self,
        window_nodes: Optional[Mapping[Hashable, Any] | Iterable[Hashable]] = None,
        window_edges: Optional[Mapping[Hashable, Any] | Iterable[Hashable]] = None,
    ):
        """同时暂存当前窗口的节点和边明细。"""
        self.stage_nodes(window_nodes)
        self.stage_edges(window_edges)

    def stage_graph(self, graph: Any):
        """从 TrafficGraph 暂存当前窗口节点和边明细。"""
        self.stage_window(
            window_nodes=getattr(graph, "nodes", {}),
            window_edges=getattr(graph, "edges", {}),
        )

    def commit_pending_window(self):
        """提交上一窗口暂存节点/边明细；没有暂存窗口时不改变近窗队列。"""
        if self.pending_nodes is None and self.pending_edges is None:
            return

        pending_nodes = dict(self.pending_nodes or {})
        pending_edges = dict(self.pending_edges or {})

        self._advance_occurrences(self.node_occurrences)
        self._advance_occurrences(self.edge_occurrences)
        self._mark_occurrences(self.node_occurrences, pending_nodes)
        self._mark_occurrences(self.edge_occurrences, pending_edges)

        self.node_windows.append(pending_nodes)
        self.windows.append(pending_edges)
        if len(self.node_windows) > self.max_detail_windows:
            self.node_windows = self.node_windows[-self.max_detail_windows :]
        if len(self.windows) > self.max_detail_windows:
            self.windows = self.windows[-self.max_detail_windows :]
        self.pending_nodes = None
        self.pending_edges = None

    def advance_window(self):
        """推进一个窗口，只滑动近窗队列，不做 TTL 恢复。"""
        self.commit_pending_window()

    def recent_node_keys(self, max_detail_windows: Optional[int] = None) -> Set[Hashable]:
        """返回最近 max_detail_windows 个已提交窗口出现过的节点 key。"""
        if max_detail_windows is None:
            max_detail_windows = self.max_detail_windows
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows must be greater than 0.")

        return {
            key
            for key, record in self.node_occurrences.items()
            if record.occurrence_count(max_detail_windows) > 0
        }

    def recent_node_items(
        self,
        max_detail_windows: Optional[int] = None,
    ) -> Dict[Hashable, Any]:
        """返回最近窗口中的节点对象；同 key 多次出现时使用最新窗口快照。"""
        if max_detail_windows is None:
            max_detail_windows = self.max_detail_windows
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows must be greater than 0.")

        return {
            key: record.obj
            for key, record in self.node_occurrences.items()
            if record.occurrence_count(max_detail_windows) > 0
        }

    def recent_edge_keys(self, max_detail_windows: Optional[int] = None) -> Set[Hashable]:
        """返回最近 max_detail_windows 个已提交窗口出现过的边 key。"""
        if max_detail_windows is None:
            max_detail_windows = self.max_detail_windows
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows must be greater than 0.")

        return {
            key
            for key, record in self.edge_occurrences.items()
            if record.occurrence_count(max_detail_windows) > 0
        }

    def recent_edge_items(
        self,
        max_detail_windows: Optional[int] = None,
    ) -> Dict[Hashable, Any]:
        """返回最近窗口中的边对象；同 key 多次出现时使用最新窗口快照。"""
        if max_detail_windows is None:
            max_detail_windows = self.max_detail_windows
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows must be greater than 0.")

        return {
            key: record.obj
            for key, record in self.edge_occurrences.items()
            if record.occurrence_count(max_detail_windows) > 0
        }

    def recent_node_occurrence_count(
        self,
        node_or_key: Any,
        max_detail_windows: Optional[int] = None,
    ) -> int:
        """返回节点在最近窗口中出现过的窗口数。"""
        node_key = self._resolve_node_key(node_or_key)
        if node_key is None:
            return 0
        record = self.node_occurrences.get(node_key)
        if record is None:
            return 0
        return record.occurrence_count(max_detail_windows)

    def recent_edge_occurrence_count(
        self,
        edge_or_key: Any,
        max_detail_windows: Optional[int] = None,
    ) -> int:
        """返回边在最近窗口中出现过的窗口数。"""
        edge_key = self._resolve_edge_key(edge_or_key)
        if edge_key is None:
            return 0
        record = self.edge_occurrences.get(edge_key)
        if record is None:
            return 0
        return record.occurrence_count(max_detail_windows)

    def recent_new_edge_mark(
        self,
        edge_or_key: Any,
        max_detail_windows: Optional[int] = None,
    ) -> int:
        """当前边不在最近窗口边明细中返回 1，否则返回 0。"""
        edge_key = self._resolve_edge_key(edge_or_key)
        if edge_key is None:
            return 1
        return int(self.recent_edge_occurrence_count(edge_key, max_detail_windows) == 0)

    def compute_recent_new_edge_marks(
        self,
        graph_or_edges: Any,
        max_detail_windows: Optional[int] = None,
    ) -> Dict[Hashable, int]:
        """批量计算当前窗口边的近期新边标记。"""
        edges = getattr(graph_or_edges, "edges", graph_or_edges)
        if edges is None:
            return {}

        marks = {}
        if isinstance(edges, Mapping):
            for edge_key, edge_obj in edges.items():
                resolved_key = (
                    edge_key if edge_key is not None else self._resolve_edge_key(edge_obj)
                )
                if resolved_key is None:
                    continue
                marks[resolved_key] = self.recent_new_edge_mark(
                    resolved_key,
                    max_detail_windows,
                )
        else:
            for edge_obj in edges:
                resolved_key = self._resolve_edge_key(edge_obj)
                if resolved_key is None:
                    continue
                marks[resolved_key] = self.recent_new_edge_mark(
                    resolved_key,
                    max_detail_windows,
                )

        return marks

    def summary(self) -> Dict[str, int]:
        """返回近窗明细保留状态。"""
        return {
            "max_detail_windows": self.max_detail_windows,
            "window_count": max(len(self.node_windows), len(self.windows)),
            "node_window_count": len(self.node_windows),
            "edge_window_count": len(self.windows),
            "recent_node_count": len(self.recent_node_keys()),
            "recent_edge_count": len(self.recent_edge_keys()),
            "node_occurrence_record_count": len(self.node_occurrences),
            "edge_occurrence_record_count": len(self.edge_occurrences),
            "pending_node_count": 0
            if self.pending_nodes is None
            else len(self.pending_nodes),
            "pending_edge_count": 0
            if self.pending_edges is None
            else len(self.pending_edges),
        }


__all__ = ["RecentEdgeDetails", "RecentOccurrenceRecord"]
