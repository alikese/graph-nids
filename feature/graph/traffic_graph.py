from typing import Dict, List, Tuple
import csv

from feature.edge.edgeClass import edge
from feature.node.nodeClass import node
from feature.packet.packetclass import packet


AUTH_BRUTEFORCE_PORTS = {21, 22}
TCP_PROTOCOL = 6


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
        self.auth_service_stats = {}

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
        self._add_auth_service_stat(
            edge_key=edge_key,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            timestamp=float(pkt.timestamp),
            payload_len=float(pkt.payload_len),
            flags_map=flags_map,
        )
        self.packets.append(pkt)

    def _add_auth_service_stat(
        self,
        edge_key,
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol,
        timestamp,
        payload_len,
        flags_map,
    ):
        if protocol != TCP_PROTOCOL:
            return

        if dst_port in AUTH_BRUTEFORCE_PORTS:
            service_key = (src_ip, dst_ip, dst_port, protocol)
            service_port = dst_port
            client_port = src_port
            direction = "client"
        elif src_port in AUTH_BRUTEFORCE_PORTS:
            service_key = (dst_ip, src_ip, src_port, protocol)
            service_port = src_port
            client_port = dst_port
            direction = "server"
        else:
            return

        stats = self.auth_service_stats.setdefault(
            service_key,
            {
                "auth_key": service_key,
                "service_edge_key": service_key,
                "service_port": service_port,
                "protocol": protocol,
                "sessions": {},
                "client_packets": 0,
                "server_packets": 0,
                "zero_payload_packets": 0,
                "small_payload_packets": 0,
                "edge_keys": set(),
            },
        )
        if direction == "client":
            stats["edge_keys"].add(edge_key)
            stats["client_packets"] += 1
        else:
            stats["server_packets"] += 1
        if payload_len == 0:
            stats["zero_payload_packets"] += 1
        if payload_len <= 64:
            stats["small_payload_packets"] += 1

        session = stats["sessions"].setdefault(
            client_port,
            {
                "first_ts": timestamp,
                "last_ts": timestamp,
                "byte_count": 0.0,
                "packet_count": 0,
                "rst_packets": 0,
                "fin_packets": 0,
            },
        )
        session["first_ts"] = min(float(session["first_ts"]), timestamp)
        session["last_ts"] = max(float(session["last_ts"]), timestamp)
        session["byte_count"] += payload_len
        session["packet_count"] += 1
        if flags_map.get("RST", False):
            session["rst_packets"] += 1
        if flags_map.get("FIN", False):
            session["fin_packets"] += 1

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
