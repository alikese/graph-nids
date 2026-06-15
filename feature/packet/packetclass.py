from typing import Dict, Tuple, TypedDict, Union


class ParsedProtoFlags(TypedDict):
    protocol: int
    flags_map: Dict[str, bool]


class ProtoFlagsFields(ParsedProtoFlags):
    flags_byte: int
    active_flags: list[str]


class packet:
    """用于包的数据结构，包含了包的基本信息。"""

    _BIT_DEFS = (
        ("CWR", 7),
        ("ECE", 6),
        ("URG", 5),
        ("ACK", 4),
        ("PSH", 3),
        ("RST", 2),
        ("SYN", 1),
        ("FIN", 0),
    )

    def __init__(
        self,
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        timestamp,
        proto_flags_mask: Union[int, str],
        payload_len,
    ):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.timestamp = timestamp
        self.payload_len = payload_len
        self.proto_flags_mask = proto_flags_mask

    @staticmethod
    def _normalize_mask_value(mask: Union[int, str]) -> int:
        """将 proto_flags_mask 归一化为 16 位整数。"""
        if isinstance(mask, str):
            mask = mask.strip()
            if mask.startswith(("0b", "0B")):
                value = int(mask, 2)
            elif mask.startswith(("0x", "0X")):
                value = int(mask, 16)
            else:
                value = int(mask)
        else:
            value = int(mask)

        return value & 0xFFFF

    def split_proto_flags_mask(self) -> Tuple[int, int]:
        """
        分割 proto_flags_mask:
        - 前 8 位: 协议位
        - 后 8 位: 标识位（ACK/SYN 等）
        """
        value = self._normalize_mask_value(self.proto_flags_mask)
        protocol = (value >> 8) & 0xFF
        tcp_flags = value & 0xFF
        return protocol, tcp_flags

    def parse_proto_flags_mask(self) -> ParsedProtoFlags:
        """
        解析 proto_flags_mask，并返回精简结果。

        返回:
            {
                "protocol": 协议号(0~255),
                "flags_map": 各标志位是否置位的布尔字典
            }
        """
        protocol, tcp_flags = self.split_proto_flags_mask()
        flags_map = {
            name: ((tcp_flags >> bit) & 1) == 1 for name, bit in self._BIT_DEFS
        }
        return {"protocol": protocol, "flags_map": flags_map}

    @staticmethod
    def flags_map_to_byte(flags_map: Dict[str, bool]) -> int:
        """将 flags_map 转换为低 8 位整数。"""
        value = 0
        for name, bit in packet._BIT_DEFS:
            if flags_map.get(name, False):
                value |= (1 << bit)
        return value & 0xFF

    def extract_proto_flags_fields(self) -> ProtoFlagsFields:
        """
        提供给 edge 的标准拆解结果：
        - protocol: 协议号
        - flags_map: 标志位布尔字典
        - flags_byte: 低8位标志整数
        - active_flags: 当前置位的标志列表
        """
        parsed = self.parse_proto_flags_mask()
        protocol = int(parsed["protocol"])
        flags_map = dict(parsed["flags_map"])
        flags_byte = self.flags_map_to_byte(flags_map)
        active_flags = [name for name, v in flags_map.items() if v]
        return {
            "protocol": protocol,
            "flags_map": flags_map,
            "flags_byte": flags_byte,
            "active_flags": active_flags,
        }
