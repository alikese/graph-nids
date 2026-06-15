import math


class node:
    """
    流量图节点类（以 IP 作为节点标识）。
    """

    def __init__(self, ip):
        self.ip = str(ip)
        self.in_edge_keys = set()
        self.out_edge_keys = set()

    def add_in_edge(self, edge_key):
        """新增一条入边 key。"""
        self.in_edge_keys.add(edge_key)

    def add_out_edge(self, edge_key):
        """新增一条出边 key。"""
        self.out_edge_keys.add(edge_key)

    def in_degree(self):
        """返回入度。"""
        return len(self.in_edge_keys)

    def out_degree(self):
        """返回出度。"""
        return len(self.out_edge_keys)

    def degree(self):
        """返回总度（入度 + 出度）。"""
        return self.in_degree() + self.out_degree()

    @staticmethod
    def _get_degree_value(node_obj, degree_name):
        """兼容方法形式和属性形式的度数读取。"""
        degree_value = getattr(node_obj, degree_name, 0)
        if callable(degree_value):
            degree_value = degree_value()
        return int(degree_value)

    @staticmethod
    def _quantile_score(current_value, values):
        """
        分位数分数:
        历史中 <= 当前值 的数量 / 历史总数量
        """
        if not values:
            return 0.0
        less_or_equal_count = sum(1 for value in values if value <= current_value)
        return less_or_equal_count / len(values)

    @staticmethod
    def _iter_history_node_objs(history, current_window_only=True):
        """从 history 中取节点对象。"""
        for record in getattr(history, "nodes", {}).values():
            if current_window_only and not getattr(record, "in_current_graph", False):
                continue
            node_obj = getattr(
                record,
                "current_window_obj" if current_window_only else "obj",
                None,
            )
            if node_obj is not None:
                yield node_obj

    @staticmethod
    def _iter_history_edge_objs(history, current_window_only=True):
        """从 history 中取边对象。"""
        for record in getattr(history, "edges", {}).values():
            if current_window_only and not getattr(record, "in_current_graph", False):
                continue
            edge_obj = getattr(
                record,
                "current_window_obj" if current_window_only else "obj",
                None,
            )
            if edge_obj is not None:
                yield edge_obj

    def source_out_degree_quantile_score(self, history, current_window_only=True):
        """
        源节点出度分位数。

        用当前节点的出度，与 history 中节点出度分布比较。
        """
        out_degrees = [
            self._get_degree_value(node_obj, "out_degree")
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        current_out_degree = self.out_degree()
        return self._quantile_score(current_out_degree, out_degrees)

    def destination_in_degree_quantile_score(self, history, current_window_only=True):
        """
        目的节点入度分位数。

        用当前节点的入度，与 history 中节点入度分布比较。
        """
        in_degrees = [
            self._get_degree_value(node_obj, "in_degree")
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        current_in_degree = self.in_degree()
        return self._quantile_score(current_in_degree, in_degrees)

    def send_receive_byte_ratio(self, history, current_window_only=True, eps=1e-8):
        """
        节点收发字节比:
            发送字节数 / (接收字节数 + eps)

        发送/接收字节数从 history 中保存的边对象提取。
        """
        send_bytes = 0.0
        receive_bytes = 0.0

        for edge_obj in self._iter_history_edge_objs(history, current_window_only):
            src_ip = str(getattr(edge_obj, "src_ip", ""))
            dst_ip = str(getattr(edge_obj, "dst_ip", ""))
            payload_len = float(getattr(edge_obj, "payload_len", 0.0))

            if src_ip == self.ip:
                send_bytes += payload_len
            if dst_ip == self.ip:
                receive_bytes += payload_len

        return send_bytes / (receive_bytes + eps)

    def normalized_send_receive_byte_ratio(
        self, history, current_window_only=True, eps=1e-8
    ):
        """
        节点收发字节比归一化:
            sigmoid(log(节点收发字节比 + eps))
        """
        ratio = self.send_receive_byte_ratio(
            history=history,
            current_window_only=current_window_only,
            eps=eps,
        )
        ratio_log = math.log(ratio + eps)
        return 1 / (1 + math.exp(-ratio_log))

    def source_behavior_role_vector(self, history, current_window_only=True, eps=1e-8):
        """
        源节点行为角色向量:
        [
            出度分位数,
            入度分位数,
            发送字节分位数,
            接收字节分位数,
            目的多样性突增得分,
            端口多样性突增得分,
            小包比例,
            握手失败近似得分
        ]
        """
        return [
            self.source_out_degree_quantile_score(history, current_window_only),
            self.destination_in_degree_quantile_score(history, current_window_only),
            self.send_bytes_quantile_score(history, current_window_only),
            self.receive_bytes_quantile_score(history, current_window_only),
            self.source_destination_diversity_burst_score(
                history, current_window_only
            ),
            self.source_port_diversity_burst_score(history, current_window_only),
            self.source_small_packet_ratio(history, current_window_only, eps),
            self.source_handshake_failure_score(history, current_window_only, eps),
        ]

    @staticmethod
    def _node_bytes(node_obj, history, current_window_only=True):
        send_bytes = 0.0
        receive_bytes = 0.0
        node_ip = str(getattr(node_obj, "ip", ""))

        for edge_obj in node._iter_history_edge_objs(history, current_window_only):
            src_ip = str(getattr(edge_obj, "src_ip", ""))
            dst_ip = str(getattr(edge_obj, "dst_ip", ""))
            payload_len = float(getattr(edge_obj, "payload_len", 0.0))

            if src_ip == node_ip:
                send_bytes += payload_len
            if dst_ip == node_ip:
                receive_bytes += payload_len

        return send_bytes, receive_bytes

    @staticmethod
    def _destination_access_stats(node_obj, history, current_window_only=True):
        node_ip = str(getattr(node_obj, "ip", ""))
        source_ips = set()
        destination_ports = set()

        for edge_obj in node._iter_history_edge_objs(history, current_window_only):
            dst_ip = str(getattr(edge_obj, "dst_ip", ""))
            if dst_ip != node_ip:
                continue
            source_ips.add(str(getattr(edge_obj, "src_ip", "")))
            destination_ports.add(getattr(edge_obj, "dst_port", None))

        return len(source_ips), len(destination_ports)

    @staticmethod
    def _source_access_stats(node_obj, history, current_window_only=True):
        """统计源节点通过出边访问过的目的 IP 数和目的端口数。"""
        node_ip = str(getattr(node_obj, "ip", ""))
        destination_ips = set()
        destination_ports = set()

        for edge_obj in node._iter_history_edge_objs(history, current_window_only):
            src_ip = str(getattr(edge_obj, "src_ip", ""))
            if src_ip != node_ip:
                continue
            destination_ips.add(str(getattr(edge_obj, "dst_ip", "")))
            destination_ports.add(getattr(edge_obj, "dst_port", None))

        return len(destination_ips), len(destination_ports)

    @staticmethod
    def _ema_component(history, node_ip, attr_name, index):
        """读取历史 EMA 向量中的指定维度。"""
        record = getattr(history, "nodes", {}).get(str(node_ip))
        if record is None:
            return None

        ema_vector = getattr(record, attr_name, None)
        if not ema_vector or index >= len(ema_vector):
            return None

        return float(ema_vector[index])

    def send_bytes_quantile_score(self, history, current_window_only=True):
        """发送字节分位数。"""
        current_send_bytes, _ = self._node_bytes(self, history, current_window_only)
        send_bytes_list = [
            self._node_bytes(node_obj, history, current_window_only)[0]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_send_bytes, send_bytes_list)

    def receive_bytes_quantile_score(self, history, current_window_only=True):
        """接收字节分位数。"""
        _, current_receive_bytes = self._node_bytes(self, history, current_window_only)
        receive_bytes_list = [
            self._node_bytes(node_obj, history, current_window_only)[1]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_receive_bytes, receive_bytes_list)

    def accessed_source_count_quantile_score(self, history, current_window_only=True):
        """被访问源数量分位数。"""
        current_source_count, _ = self._destination_access_stats(
            self, history, current_window_only
        )
        source_count_list = [
            self._destination_access_stats(node_obj, history, current_window_only)[0]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_source_count, source_count_list)

    def accessed_destination_port_count_quantile_score(
        self, history, current_window_only=True
    ):
        """被访问目的端口数分位数。"""
        _, current_port_count = self._destination_access_stats(
            self, history, current_window_only
        )
        port_count_list = [
            self._destination_access_stats(node_obj, history, current_window_only)[1]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_port_count, port_count_list)

    def source_destination_diversity_quantile_score(
        self, history, current_window_only=True
    ):
        """源节点目的多样性分位数（访问过的不同目的 IP 数）。"""
        current_destination_count, _ = self._source_access_stats(
            self, history, current_window_only
        )
        destination_count_list = [
            self._source_access_stats(node_obj, history, current_window_only)[0]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_destination_count, destination_count_list)

    def source_port_diversity_quantile_score(self, history, current_window_only=True):
        """源节点端口多样性分位数（访问过的不同目的端口数）。"""
        _, current_port_count = self._source_access_stats(
            self, history, current_window_only
        )
        port_count_list = [
            self._source_access_stats(node_obj, history, current_window_only)[1]
            for node_obj in self._iter_history_node_objs(history, current_window_only)
        ]
        return self._quantile_score(current_port_count, port_count_list)

    def source_destination_diversity_burst_score(
        self, history, current_window_only=True
    ):
        """
        源节点目的多样性突增得分。

        用当前目的多样性分位数减去历史源角色 EMA 对应维度，只保留正向突增。
        """
        current_score = self.source_destination_diversity_quantile_score(
            history, current_window_only
        )
        history_score = self._ema_component(
            history, self.ip, "source_behavior_role_ema", 4
        )
        if history_score is None:
            return 0.0
        return min(max(current_score - history_score, 0.0), 1.0)

    def source_port_diversity_burst_score(self, history, current_window_only=True):
        """
        源节点端口多样性突增得分。

        用当前端口多样性分位数减去历史源角色 EMA 对应维度，只保留正向突增。
        """
        current_score = self.source_port_diversity_quantile_score(
            history, current_window_only
        )
        history_score = self._ema_component(
            history, self.ip, "source_behavior_role_ema", 5
        )
        if history_score is None:
            return 0.0
        return min(max(current_score - history_score, 0.0), 1.0)

    @staticmethod
    def _history_node_record(history, node_or_ip):
        node_ip = getattr(node_or_ip, "ip", node_or_ip)
        if node_ip is None:
            return None
        return getattr(history, "nodes", {}).get(str(node_ip))

    @staticmethod
    def _history_node_obj(history, node_or_ip, current_window_only=True):
        if hasattr(node_or_ip, "ip"):
            return node_or_ip

        record = node._history_node_record(history, node_or_ip)
        if record is None:
            return None

        node_obj = getattr(
            record,
            "current_window_obj" if current_window_only else "obj",
            None,
        )
        if node_obj is None:
            node_obj = getattr(record, "obj", None)
        return node_obj

    @staticmethod
    def _record_float(record, attr_name, default=0.0):
        if record is None:
            return float(default)
        try:
            return float(getattr(record, attr_name, default))
        except (TypeError, ValueError):
            return float(default)

    def previous_attack_destination_to_source_mark(self, history=None, attack_dst_ips=None):
        """
        上一窗口攻击目的转源标记:
            1, 如果当前节点 IP 属于 AttackDst_{t-1}
            0, 否则
        """
        if attack_dst_ips is None and history is not None:
            if hasattr(history, "get_previous_attack_destination_ips"):
                attack_dst_ips = history.get_previous_attack_destination_ips()
            else:
                attack_dst_ips = getattr(history, "previous_attack_dst_ips", set())

        if not attack_dst_ips:
            return 0

        return int(self.ip in {str(ip) for ip in attack_dst_ips})

    def behavior_role_anomaly_score(
        self,
        history,
        destination_node,
        current_window_only=True,
        eps=1e-8,
        return_components=False,
    ):
        """
        行为角色异常分数。

        self 表示源节点，destination_node 可传目的节点对象或目的节点 IP。

        公式:
            0.20 * 源节点出度分位数
            + 0.15 * 目的节点入度分位数
            + 0.15 * 节点收发字节比归一化
            + 0.20 * 源节点行为角色漂移距离
            + 0.15 * 目的节点行为角色漂移距离
            + 0.15 * 上一窗口攻击目的转源标记
        """
        destination_obj = self._history_node_obj(
            history,
            destination_node,
            current_window_only,
        )

        source_record = self._history_node_record(history, self.ip)
        destination_record = self._history_node_record(history, destination_node)

        source_out_degree_quantile = self.source_out_degree_quantile_score(
            history,
            current_window_only,
        )
        normalized_byte_ratio = self.normalized_send_receive_byte_ratio(
            history,
            current_window_only,
            eps,
        )
        source_role_drift = self._record_float(
            source_record,
            "source_behavior_role_drift",
            self.source_behavior_role_drift_distance(
                history,
                current_window_only,
                eps,
            ),
        )

        if destination_obj is None:
            destination_in_degree_quantile = 0.0
            destination_role_drift = 0.0
        else:
            destination_in_degree_quantile = (
                destination_obj.destination_in_degree_quantile_score(
                    history,
                    current_window_only,
                )
            )
            destination_role_drift = self._record_float(
                destination_record,
                "destination_behavior_role_drift",
                destination_obj.destination_behavior_role_drift_distance(
                    history,
                    current_window_only,
                    eps,
                ),
            )

        previous_attack_mark = self.previous_attack_destination_to_source_mark(
            history=history
        )

        components = {
            "source_out_degree_quantile": source_out_degree_quantile,
            "destination_in_degree_quantile": destination_in_degree_quantile,
            "normalized_byte_ratio": normalized_byte_ratio,
            "source_role_drift": source_role_drift,
            "destination_role_drift": destination_role_drift,
            "previous_attack_dst_to_src_mark": float(previous_attack_mark),
        }

        score = 0.0
        score += 0.20 * components["source_out_degree_quantile"]
        score += 0.15 * components["destination_in_degree_quantile"]
        score += 0.15 * components["normalized_byte_ratio"]
        score += 0.20 * components["source_role_drift"]
        score += 0.15 * components["destination_role_drift"]
        score += 0.15 * components["previous_attack_dst_to_src_mark"]
        score = min(max(score, 0.0), 1.0)

        if return_components:
            components["behavior_role_anomaly_score"] = score
            return components
        return score

    def source_small_packet_ratio(self, history, current_window_only=True, eps=1e-8):
        """源节点发出的边小包比例。"""
        small_packet_count = 0.0
        packet_count = 0.0

        for edge_obj in self._iter_history_edge_objs(history, current_window_only):
            if str(getattr(edge_obj, "src_ip", "")) != self.ip:
                continue
            small_packet_count += float(getattr(edge_obj, "smallpacket", 0.0))
            packet_count += float(getattr(edge_obj, "edgepacketnum", 0.0))

        return small_packet_count / (packet_count + eps)

    def source_handshake_failure_score(
        self, history, current_window_only=True, eps=1e-8
    ):
        """源节点发出的边的握手失败近似得分。"""
        syn_count = 0.0
        syn_ack_count = 0.0
        ack_count = 0.0

        for edge_obj in self._iter_history_edge_objs(history, current_window_only):
            if str(getattr(edge_obj, "src_ip", "")) != self.ip:
                continue
            syn_count += float(getattr(edge_obj, "syn_count", 0.0))
            syn_ack_count += float(getattr(edge_obj, "syn_ack_count", 0.0))
            ack_count += float(getattr(edge_obj, "ack_count", 0.0))

        return max(syn_count - syn_ack_count - ack_count, 0.0) / (syn_count + eps)

    def destination_small_packet_ratio(self, history, current_window_only=True, eps=1e-8):
        """目的节点收到的边小包比例。"""
        small_packet_count = 0.0
        packet_count = 0.0

        for edge_obj in self._iter_history_edge_objs(history, current_window_only):
            if str(getattr(edge_obj, "dst_ip", "")) != self.ip:
                continue
            small_packet_count += float(getattr(edge_obj, "smallpacket", 0.0))
            packet_count += float(getattr(edge_obj, "edgepacketnum", 0.0))

        return small_packet_count / (packet_count + eps)

    def destination_handshake_failure_score(
        self, history, current_window_only=True, eps=1e-8
    ):
        """目的节点收到的边的握手失败近似得分。"""
        syn_count = 0.0
        syn_ack_count = 0.0
        ack_count = 0.0

        for edge_obj in self._iter_history_edge_objs(history, current_window_only):
            if str(getattr(edge_obj, "dst_ip", "")) != self.ip:
                continue
            syn_count += float(getattr(edge_obj, "syn_count", 0.0))
            syn_ack_count += float(getattr(edge_obj, "syn_ack_count", 0.0))
            ack_count += float(getattr(edge_obj, "ack_count", 0.0))

        return max(syn_count - syn_ack_count - ack_count, 0.0) / (syn_count + eps)

    def destination_behavior_role_vector(
        self, history, current_window_only=True, eps=1e-8
    ):
        """
        目的节点行为角色向量:
        [
            出度分位数,
            入度分位数,
            发送字节分位数,
            接收字节分位数,
            被访问源数量分位数,
            被访问目的端口数分位数,
            小包比例,
            握手失败近似得分
        ]
        """
        return [
            self.source_out_degree_quantile_score(history, current_window_only),
            self.destination_in_degree_quantile_score(history, current_window_only),
            self.send_bytes_quantile_score(history, current_window_only),
            self.receive_bytes_quantile_score(history, current_window_only),
            self.accessed_source_count_quantile_score(history, current_window_only),
            self.accessed_destination_port_count_quantile_score(
                history, current_window_only
            ),
            self.destination_small_packet_ratio(history, current_window_only, eps),
            self.destination_handshake_failure_score(history, current_window_only, eps),
        ]

    @staticmethod
    def _mean_abs_distance(vector_a, vector_b):
        if not vector_a or not vector_b:
            return 0.0
        length = min(len(vector_a), len(vector_b))
        if length == 0:
            return 0.0
        return sum(abs(vector_a[i] - vector_b[i]) for i in range(length)) / length

    def source_behavior_role_drift_distance(
        self, history, current_window_only=True, eps=1e-8
    ):
        """
        源节点行为角色漂移距离:
            mean(|R_now(src) - R_ema(src)|)
        """
        record = getattr(history, "nodes", {}).get(self.ip)
        if record is None:
            return 0.0

        r_ema = getattr(record, "source_behavior_role_ema", None)
        if not r_ema:
            return 0.0

        r_now = self.source_behavior_role_vector(
            history=history,
            current_window_only=current_window_only,
            eps=eps,
        )
        return self._mean_abs_distance(r_now, r_ema)

    def destination_behavior_role_drift_distance(
        self, history, current_window_only=True, eps=1e-8
    ):
        """
        目的节点行为角色漂移距离:
            mean(|R_now(dst) - R_ema(dst)|)
        """
        record = getattr(history, "nodes", {}).get(self.ip)
        if record is None:
            return 0.0

        r_ema = getattr(record, "destination_behavior_role_ema", None)
        if not r_ema:
            return 0.0

        r_now = self.destination_behavior_role_vector(
            history=history,
            current_window_only=current_window_only,
            eps=eps,
        )
        return self._mean_abs_distance(r_now, r_ema)
