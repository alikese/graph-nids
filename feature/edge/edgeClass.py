import math


DEFAULT_CURRENT_BEHAVIOR_WEIGHTS = {
    "small_packet_ratio": 0.15,
    "zero_payload_ratio": 0.10,
    "syn_without_ack_ratio": 0.20,
    "handshake_failure_score": 0.20,
    "rst_ratio": 0.10,
    "burstiness_score": 0.10,
    "flags_entropy_score": 0.15,
}
CURRENT_BEHAVIOR_WEIGHTS = dict(DEFAULT_CURRENT_BEHAVIOR_WEIGHTS)


def _normalized_weights(weights):
    cleaned = {
        name: max(float(weights.get(name, 0.0)), 0.0)
        for name in DEFAULT_CURRENT_BEHAVIOR_WEIGHTS
    }
    total = sum(cleaned.values())
    if total <= 0.0:
        return dict(DEFAULT_CURRENT_BEHAVIOR_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def set_current_behavior_weights(weights):
    CURRENT_BEHAVIOR_WEIGHTS.clear()
    CURRENT_BEHAVIOR_WEIGHTS.update(_normalized_weights(weights))


def reset_current_behavior_weights():
    CURRENT_BEHAVIOR_WEIGHTS.clear()
    CURRENT_BEHAVIOR_WEIGHTS.update(DEFAULT_CURRENT_BEHAVIOR_WEIGHTS)


class EdgeWindowBuffer:
    """
    当前窗口边暂存容器。

    暂存窗口内生成的 edge 对象，等待进入下一个窗口时再提交到历史。
    """

    def __init__(self):
        self.edges = {}

    def stage_edges(self, window_edges=None):
        if not window_edges:
            return
        self.edges.update(dict(window_edges))

    def clear(self):
        self.edges.clear()

    def is_empty(self):
        return not self.edges

    def items(self):
        return self.edges.items()

    def __len__(self):
        return len(self.edges)


class edge:
    """
    边类（流/会话）。
    约定输入到边中的协议与标志位应由 packet 先拆解完成。
    """

    FLAGS_BIT_DEFS = (
        ("CWR", 7),
        ("ECE", 6),
        ("URG", 5),
        ("ACK", 4),
        ("PSH", 3),
        ("RST", 2),
        ("SYN", 1),
        ("FIN", 0),
    )

    DETAIL_LIST_FIELDS = (
        "timestamp_list",
        "payload_len_list",
        "protocol_list",
        "flags_map_list",
        "flags_byte_list",
        "active_flags_list",
    )

    def __init__(
        self,
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        timestamp,
        payload_len,
        edgepacketnum,
        protocol=None,
        flags_map=None,
        flags_byte=None,
    ):
        """
        初始化一条边，并记录首包信息与协议/标志位信息。
        """
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.timestamp = timestamp
        self.timestamp_list = [float(timestamp)]

        self.payload_len = payload_len
        self.payload_len_list = [payload_len]

        self.staywindows = 0
        self.edgepacketnum = edgepacketnum
        self.zeropacket = 0
        self.smallpacket = 0

        if payload_len == 0:
            self.zeropacket = 1
        if payload_len <= 64:
            self.smallpacket = 1

        # 协议与标志位的逐包记录
        self.protocol_list = []
        self.flags_map_list = []
        self.flags_byte_list = []
        self.active_flags_list = []

        # 统计计数（用于熵与比例）
        self.protocol_count_map = {}
        self.flags_count_map = {}
        self.flag_name_count_map = {name: 0 for name, _ in self.FLAGS_BIT_DEFS}

        self.syn_count = 0
        self.ack_count = 0
        self.syn_ack_count = 0
        self.rst_count = 0

        self._add_proto_flags_stat(
            protocol=protocol,
            flags_map=flags_map,
            flags_byte=flags_byte,
        )

    def add_packet(self, packet, protocol=None, flags_map=None, flags_byte=None):
        """
        向当前边中新增一个包，并更新统计信息。
        协议与标志位优先使用外部已拆解输入；若未传入则尝试从 packet 提取。
        """
        payload_len = getattr(packet, "payload_len", packet)
        packet_timestamp = getattr(packet, "timestamp", None)
        self.edgepacketnum += 1
        self.payload_len += payload_len
        self.payload_len_list.append(payload_len)
        if packet_timestamp is not None:
            try:
                self.timestamp_list.append(float(packet_timestamp))
            except (TypeError, ValueError):
                pass

        if payload_len == 0:
            self.zeropacket += 1
        if payload_len <= 64:
            self.smallpacket += 1

        if (
            (protocol is None or flags_map is None)
            and hasattr(packet, "extract_proto_flags_fields")
        ):
            fields = packet.extract_proto_flags_fields()
            if protocol is None:
                protocol = fields.get("protocol")
            if flags_map is None:
                flags_map = fields.get("flags_map")
            if flags_byte is None:
                flags_byte = fields.get("flags_byte")

        self._add_proto_flags_stat(
            protocol=protocol,
            flags_map=flags_map,
            flags_byte=flags_byte,
        )

    def _detail_segment_from_edge(self, edge_obj, age=0):
        return {
            "age": int(age),
            "details": {
                field: list(getattr(edge_obj, field, []) or [])
                for field in self.DETAIL_LIST_FIELDS
            },
        }

    @staticmethod
    def _detail_segment_payload(segment):
        if isinstance(segment, dict) and "details" in segment:
            return segment["details"]
        return segment

    def _ensure_detail_window_segments(self):
        if not hasattr(self, "_detail_window_segments"):
            self._detail_window_segments = [self._detail_segment_from_edge(self)]

    def _rebuild_detail_lists_from_segments(self):
        for field in self.DETAIL_LIST_FIELDS:
            merged = []
            for segment in self._detail_window_segments:
                details = self._detail_segment_payload(segment)
                merged.extend(details.get(field, []))
            setattr(self, field, merged)

    def retain_recent_window_details(self, max_detail_windows=5):
        """
        兼容旧版边内明细片段：只保留最近 max_detail_windows 个窗口的逐包详细数据。

        当前 History 的主要近期边明细由 RecentEdgeDetails 维护；该方法仅用于
        Edge 自身历史聚合对象的明细列表裁剪，不影响 edgepacketnum/payload_len 等累计统计。
        """
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows 必须大于 0。")

        self._ensure_detail_window_segments()
        self._detail_window_segments = [
            segment
            for segment in self._detail_window_segments
            if int(segment.get("age", 0)) < max_detail_windows
        ]
        self._detail_window_segments = self._detail_window_segments[-max_detail_windows:]
        self._rebuild_detail_lists_from_segments()
        return self

    def advance_detail_window(self, max_detail_windows=5):
        """
        兼容旧版边内明细片段：推进一个窗口年龄并移除过期逐包详细数据。

        History.advance_window 不再依赖该方法推进近期明细；只影响本 Edge 的
        timestamp_list/payload_len_list 等详细列表，不影响累计统计。
        """
        if max_detail_windows <= 0:
            raise ValueError("max_detail_windows 必须大于 0。")

        self._ensure_detail_window_segments()
        for segment in self._detail_window_segments:
            segment["age"] = int(segment.get("age", 0)) + 1
        self.retain_recent_window_details(max_detail_windows=max_detail_windows)
        return self

    def merge_from_window_edge(self, window_edge, max_detail_windows=5):
        """
        将一个完整窗口中的同 key 边累计到当前历史边。

        该方法用于 History.edges 的长期聚合对象；RecentEdgeDetails 会另外保存
        最近窗口的原始边对象快照，避免长期累计明细影响近期时间行为分数。

        负载平均大小满足:
            (历史边包数 * 历史平均大小 + 当前窗口边包数 * 当前窗口边平均大小)
            / (历史边包数 + 当前窗口边包数)

        在类内用 payload_len 总量与 edgepacketnum 总包数保存，因此等价于:
            新 payload_len = 历史 payload_len + 当前窗口 payload_len
            新 edgepacketnum = 历史 edgepacketnum + 当前窗口 edgepacketnum
        """
        if window_edge is None:
            return self

        window_packet_num = int(getattr(window_edge, "edgepacketnum", 0) or 0)
        if window_packet_num <= 0:
            return self

        self._ensure_detail_window_segments()

        self.edgepacketnum += window_packet_num
        self.payload_len += float(getattr(window_edge, "payload_len", 0.0) or 0.0)
        self.zeropacket += int(getattr(window_edge, "zeropacket", 0) or 0)
        self.smallpacket += int(getattr(window_edge, "smallpacket", 0) or 0)
        self.syn_count += int(getattr(window_edge, "syn_count", 0) or 0)
        self.ack_count += int(getattr(window_edge, "ack_count", 0) or 0)
        self.syn_ack_count += int(getattr(window_edge, "syn_ack_count", 0) or 0)
        self.rst_count += int(getattr(window_edge, "rst_count", 0) or 0)

        self.timestamp = getattr(window_edge, "timestamp", self.timestamp)
        self._detail_window_segments.append(self._detail_segment_from_edge(window_edge))

        for protocol, count in getattr(window_edge, "protocol_count_map", {}).items():
            self.protocol_count_map[protocol] = (
                self.protocol_count_map.get(protocol, 0) + count
            )

        for flags_byte, count in getattr(window_edge, "flags_count_map", {}).items():
            self.flags_count_map[flags_byte] = (
                self.flags_count_map.get(flags_byte, 0) + count
            )

        for flag_name, count in getattr(window_edge, "flag_name_count_map", {}).items():
            self.flag_name_count_map[flag_name] = (
                self.flag_name_count_map.get(flag_name, 0) + count
            )

        self.staywindows += 1
        self.retain_recent_window_details(max_detail_windows=max_detail_windows)
        return self

    @classmethod
    def get_flag_bit_mapping(cls):
        """返回标志位与比特位映射。"""
        return {name: bit for name, bit in cls.FLAGS_BIT_DEFS}

    @staticmethod
    def _flags_map_to_byte(flags_map):
        value = 0
        for name, bit in edge.FLAGS_BIT_DEFS:
            if flags_map.get(name, False):
                value |= (1 << bit)
        return value & 0xFF

    @staticmethod
    def _byte_to_flags_map(flags_byte):
        return {
            name: ((flags_byte >> bit) & 1) == 1
            for name, bit in edge.FLAGS_BIT_DEFS
        }

    def _add_proto_flags_stat(self, protocol=None, flags_map=None, flags_byte=None):
        """
        记录单个包的协议与标志位拆解结果到边中。
        """
        if protocol is None and flags_map is None and flags_byte is None:
            return

        if flags_map is None and flags_byte is not None:
            flags_map = self._byte_to_flags_map(int(flags_byte))
        elif flags_map is not None:
            flags_map = dict(flags_map)

        if flags_byte is None and flags_map is not None:
            flags_byte = self._flags_map_to_byte(flags_map)

        if protocol is not None:
            protocol = int(protocol)
            self.protocol_list.append(protocol)
            self.protocol_count_map[protocol] = self.protocol_count_map.get(protocol, 0) + 1

        if flags_map is not None:
            self.flags_map_list.append(flags_map)
            active_flags = [name for name, is_set in flags_map.items() if is_set]
            self.active_flags_list.append(active_flags)
            for name, _ in self.FLAGS_BIT_DEFS:
                if flags_map.get(name, False):
                    self.flag_name_count_map[name] = self.flag_name_count_map.get(name, 0) + 1

            if flags_map.get("SYN", False):
                self.syn_count += 1
            if flags_map.get("ACK", False):
                self.ack_count += 1
            if flags_map.get("SYN", False) and flags_map.get("ACK", False):
                self.syn_ack_count += 1
            if flags_map.get("RST", False):
                self.rst_count += 1

        if flags_byte is not None:
            flags_byte = int(flags_byte) & 0xFF
            self.flags_byte_list.append(flags_byte)
            self.flags_count_map[flags_byte] = self.flags_count_map.get(flags_byte, 0) + 1

    def payload_stats(self, eps=1e-8):
        """
        计算边负载统计特征:
        1) 平均负载长度
        2) 负载长度标准差
        3) 负载长度变异系数
        """
        n_e = len(self.payload_len_list)
        if n_e == 0:
            return 0.0, 0.0, 0.0

        mean_payload_len = sum(self.payload_len_list) / n_e
        variance = sum(
            (payload_len_i - mean_payload_len) ** 2
            for payload_len_i in self.payload_len_list
        ) / n_e
        std_payload_len = variance ** 0.5
        cv_payload_len = std_payload_len / (mean_payload_len + eps)
        return mean_payload_len, std_payload_len, cv_payload_len

    def syn_without_ack_ratio(self, syn_count=None, ack_count=None, eps=1e-8):
        """
        SYN 无 ACK 近似比例。
        未提供输入时使用边内累计计数。
        """
        if syn_count is None:
            syn_count = self.syn_count
        if ack_count is None:
            ack_count = self.ack_count
        return max(syn_count - ack_count, 0) / (syn_count + eps)

    def handshake_failure_score(
        self, syn_count=None, syn_ack_count=None, ack_count=None, eps=1e-8
    ):
        """
        握手失败近似得分。
        未提供输入时使用边内累计计数。
        """
        if syn_count is None:
            syn_count = self.syn_count
        if syn_ack_count is None:
            syn_ack_count = self.syn_ack_count
        if ack_count is None:
            ack_count = self.ack_count
        return max(syn_count - syn_ack_count - ack_count, 0) / (syn_count + eps)

    def rst_ratio(self, eps=1e-8):
        """RST 标志位比例。"""
        return self.rst_count / (self.edgepacketnum + eps)

    def small_packet_ratio(self, eps=1e-8):
        """边小包比例（payload_len<=64 的包占比）。"""
        return self.smallpacket / (self.edgepacketnum + eps)

    def zero_payload_ratio(self, eps=1e-8):
        """边零负载包比例（payload_len==0 的包占比）。"""
        return self.zeropacket / (self.edgepacketnum + eps)

    def get_edgepacketnum_list_from_history(self, history, current_window_only=True):
        """
        从 history 中提取边包数量列表。
        """
        history_edges = getattr(history, "edges", {})
        packet_counts = []

        for record in history_edges.values():
            if current_window_only and not getattr(record, "in_current_graph", False):
                continue
            edge_obj = getattr(
                record,
                "current_window_obj" if current_window_only else "obj",
                None,
            )
            if edge_obj is None:
                continue
            edge_packet_num = getattr(edge_obj, "edgepacketnum", None)
            if edge_packet_num is None:
                continue
            try:
                packet_counts.append(float(edge_packet_num))
            except (TypeError, ValueError):
                continue

        return packet_counts

    def edgepacketnum_quantile_score(self, history, current_window_only=True):
        """
        边包数量分位数分数（0~1）:
        历史中边包数量 <= 当前边包数量 的占比。
        """
        packet_counts = self.get_edgepacketnum_list_from_history(
            history=history,
            current_window_only=current_window_only,
        )
        if not packet_counts:
            return 0.0

        current_count = float(self.edgepacketnum)
        less_or_equal_count = sum(1 for value in packet_counts if value <= current_count)
        return less_or_equal_count / len(packet_counts)

    @staticmethod
    def _entropy_from_count_map(count_map):
        total = sum(count_map.values())
        if total <= 0:
            return 0.0

        entropy = 0.0
        for count in count_map.values():
            if count <= 0:
                continue
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _normalized_entropy_score(count_map, eps=1e-12):
        category_count = len([value for value in count_map.values() if value > 0])
        if category_count <= 1:
            return 0.0
        entropy = edge._entropy_from_count_map(count_map)
        max_entropy = math.log2(category_count + eps)
        if max_entropy <= 0:
            return 0.0
        return min(max(entropy / max_entropy, 0.0), 1.0)

    def flags_entropy_score(self):
        """标志位熵归一化分数（0~1）。"""
        return self._normalized_entropy_score(self.flags_count_map)

    def previous_attack_destination_to_source_mark(
        self, history=None, attack_dst_ips=None
    ):
        """
        上一窗口攻击目的转源标记:
            1, 如果 src_ip(e) 属于 AttackDst_{t-1}
            0, 否则

        参数:
            history: 历史类对象，优先读取 history.previous_attack_dst_ips
            attack_dst_ips: 外部直接传入的 AttackDst_{t-1} 集合
        """
        if attack_dst_ips is None and history is not None:
            if hasattr(history, "get_previous_attack_destination_ips"):
                attack_dst_ips = history.get_previous_attack_destination_ips()
            else:
                attack_dst_ips = getattr(history, "previous_attack_dst_ips", set())

        if not attack_dst_ips:
            return 0

        return int(str(self.src_ip) in {str(ip) for ip in attack_dst_ips})

    @staticmethod
    def _get_history_node_record(history, ip):
        if history is None:
            return None
        return getattr(history, "nodes", {}).get(str(ip))

    @classmethod
    def _get_history_node_obj(cls, history, ip):
        record = cls._get_history_node_record(history, ip)
        if record is None:
            return None
        return getattr(record, "obj", None)

    def behavior_role_anomaly_score(
        self,
        history,
        current_window_only=True,
        eps=1e-8,
        return_components=False,
    ):
        """
        最终行为角色异常得分:
            0.20 * 源节点出度分位数
            + 0.15 * 目的节点入度分位数
            + 0.15 * 源节点收发字节比归一化
            + 0.20 * 源节点行为角色漂移距离
            + 0.15 * 目的节点行为角色漂移距离
            + 0.15 * 上一窗口攻击目的转源标记
        """
        if hasattr(history, "get_behavior_role_score_components_for_edge"):
            components = history.get_behavior_role_score_components_for_edge(
                self,
                current_window_only=current_window_only,
                eps=eps,
            )
            score = 0.0
            score += 0.20 * components["source_out_degree_quantile"]
            score += 0.15 * components["destination_in_degree_quantile"]
            score += 0.15 * components["normalized_byte_ratio"]
            score += 0.20 * components["source_role_drift"]
            score += 0.15 * components["destination_role_drift"]
            score += 0.15 * components["previous_attack_dst_to_src_mark"]
            score = min(max(score, 0.0), 1.0)

            if return_components:
                components["score"] = score
                return components
            return score

        src_node = self._get_history_node_obj(history, self.src_ip)
        dst_node = self._get_history_node_obj(history, self.dst_ip)
        src_record = self._get_history_node_record(history, self.src_ip)
        dst_record = self._get_history_node_record(history, self.dst_ip)

        source_out_degree_quantile = 0.0
        destination_in_degree_quantile = 0.0
        normalized_byte_ratio = 0.0
        source_role_drift = 0.0
        destination_role_drift = 0.0

        if src_node is not None:
            source_out_degree_quantile = src_node.source_out_degree_quantile_score(
                history, current_window_only
            )
            normalized_byte_ratio = src_node.normalized_send_receive_byte_ratio(
                history, current_window_only, eps
            )
            if src_record is not None:
                source_role_drift = float(
                    getattr(src_record, "source_behavior_role_drift", 0.0)
                )

        if dst_node is not None:
            destination_in_degree_quantile = (
                dst_node.destination_in_degree_quantile_score(
                    history, current_window_only
                )
            )
            if dst_record is not None:
                destination_role_drift = float(
                    getattr(dst_record, "destination_behavior_role_drift", 0.0)
                )

        attack_dst_to_src_mark = self.previous_attack_destination_to_source_mark(
            history=history
        )

        components = {
            "source_out_degree_quantile": source_out_degree_quantile,
            "destination_in_degree_quantile": destination_in_degree_quantile,
            "normalized_byte_ratio": normalized_byte_ratio,
            "source_role_drift": source_role_drift,
            "destination_role_drift": destination_role_drift,
            "previous_attack_dst_to_src_mark": attack_dst_to_src_mark,
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
            components["score"] = score
            return components
        return score

    def current_behavior_anomaly_score(
        self,
        history=None,
        burstiness_score=None,
        eps=1e-8,
        current_window_only=True,
        burst_gap_threshold=0.01,
    ):
        """
        当前行为异常分数（独立封装）:
        0.15 * 边小包比例
        + 0.10 * 边零负载包比例
        + 0.20 * SYN无ACK近似比例
        + 0.20 * 握手失败近似得分
        + 0.10 * RST标志比例
        + 0.10 * 边突发性得分（外部输入）
        + 0.15 * 标志位熵归一化

        说明:
            - burstiness_score 为 None 时，自动调用 burstiness_score() 计算
            - history 仅在你需要额外结合历史指标时传入；当前公式中未强制依赖
        """
        _ = history
        _ = current_window_only

        if burstiness_score is None:
            burstiness_score = self.burstiness_score(
                burst_gap_threshold=burst_gap_threshold,
                eps=eps,
            )

        components = {
            "small_packet_ratio": self.small_packet_ratio(eps=eps),
            "zero_payload_ratio": self.zero_payload_ratio(eps=eps),
            "syn_without_ack_ratio": self.syn_without_ack_ratio(eps=eps),
            "handshake_failure_score": self.handshake_failure_score(eps=eps),
            "rst_ratio": self.rst_ratio(eps=eps),
            "burstiness_score": min(max(float(burstiness_score), 0.0), 1.0),
            "flags_entropy_score": self.flags_entropy_score(),
        }
        score = sum(
            CURRENT_BEHAVIOR_WEIGHTS[name] * components[name]
            for name in CURRENT_BEHAVIOR_WEIGHTS
        )

        return min(max(score, 0.0), 1.0)

    def burstiness_score(self, burst_gap_threshold=0.01, eps=1e-8):
        """
        计算边突发性得分:
            若连续包时间间隔 < burst_gap_threshold，则视为同一 burst。
            得分 = burst 内包总数 / n_e

        说明:
            - 仅把长度>=2 的连续段计为 burst
            - 结果范围 0~1
        """
        n_e = len(self.timestamp_list)
        if n_e <= 1 or len(self.timestamp_list) <= 1:
            return 0.0

        timestamps = sorted(self.timestamp_list)
        burst_packet_total = 0
        current_burst_len = 1

        for idx in range(1, len(timestamps)):
            gap = timestamps[idx] - timestamps[idx - 1]
            if gap < burst_gap_threshold:
                current_burst_len += 1
            else:
                if current_burst_len >= 2:
                    burst_packet_total += current_burst_len
                current_burst_len = 1

        if current_burst_len >= 2:
            burst_packet_total += current_burst_len

        return burst_packet_total / (n_e + eps)
