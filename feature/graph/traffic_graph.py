from typing import Dict, List, Tuple
import csv

from feature.edge.edgeClass import edge
from feature.node.nodeClass import node
from feature.packet.packetclass import packet


class TrafficGraph:
    """
    从单个时间窗口 CSV 的包数据构建流量图。

    节点: IP
    边: (src_ip, dst_ip, src_port, dst_port, protocol)
    """

    def __init__(self):
        self.nodes: Dict[str, node] = {}
        self.edges: Dict[Tuple[str, str, int, int, int], edge] = {}
        self.packets: List[packet] = []

    def _get_or_create_node(self, ip: str) -> node:
        if ip not in self.nodes:
            self.nodes[ip] = node(ip=ip)
        return self.nodes[ip]

    def add_packet(self, pkt: packet):
        """向图中加入一个包，并更新节点与边。"""
        parsed = pkt.extract_proto_flags_fields()
        protocol = int(parsed["protocol"])
        flags_map = parsed["flags_map"]
        flags_byte = int(parsed["flags_byte"])

        src_ip = str(pkt.src_ip)
        dst_ip = str(pkt.dst_ip)
        src_port = int(pkt.src_port)
        dst_port = int(pkt.dst_port)

        edge_key = (src_ip, dst_ip, src_port, dst_port, protocol)
        if edge_key not in self.edges:
            self.edges[edge_key] = edge(
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                timestamp=float(pkt.timestamp),
                payload_len=float(pkt.payload_len),
                edgepacketnum=1,
                protocol=protocol,
                flags_map=flags_map,
                flags_byte=flags_byte,
            )
        else:
            self.edges[edge_key].add_packet(
                pkt,
                protocol=protocol,
                flags_map=flags_map,
                flags_byte=flags_byte,
            )

        src_node = self._get_or_create_node(src_ip)
        dst_node = self._get_or_create_node(dst_ip)
        src_node.add_out_edge(edge_key)
        dst_node.add_in_edge(edge_key)
        self.packets.append(pkt)

    @classmethod
    def from_window_csv(cls, csv_path: str):
        """
        从 `windows divide` 中的单个窗口 CSV 构建 TrafficGraph。
        期望 CSV 至少包含以下列:
        src_ip,dst_ip,src_port,dst_port,timestamp,proto_flags_mask,payload_len,...
        """
        graph = cls()
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pkt = packet(
                    src_ip=row["src_ip"],
                    dst_ip=row["dst_ip"],
                    src_port=int(row["src_port"]),
                    dst_port=int(row["dst_port"]),
                    timestamp=float(row["timestamp"]),
                    proto_flags_mask=row["proto_flags_mask"],
                    payload_len=float(row["payload_len"]),
                )
                graph.add_packet(pkt)
        return graph

    def summary(self) -> Dict[str, int]:
        """返回图的基础统计信息。"""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "packet_count": len(self.packets),
        }


def build_graph_from_window_csv(csv_path: str) -> TrafficGraph:
    """便捷封装：从窗口 CSV 构建流量图。"""
    return TrafficGraph.from_window_csv(csv_path)
