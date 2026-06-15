import csv
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from feature.graph.traffic_graph import TrafficGraph
from feature.attack_similar.previous_attack_edge import (
    build_recent_attack_edge_index,
)
from feature.history.historyClass import History
from feature.history.history_feature.active_edge_features import EdgeActiveHistoryFeature
from feature.packet.packetclass import packet
from feature.sum_score import local_anomaly_score


ROOT = Path(__file__).resolve().parent
WINDOW_DIR = ROOT / "windows divide"
RESULT_PATH = ROOT / "full_week_no_label_leakage_score_distribution.json"
PROGRESS_PATH = ROOT / "full_week_no_label_leakage_score_progress.log"
BIN_COUNT = 1000
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
ATTACK_THRESHOLD = 0.70
NORMAL_THRESHOLD = 0.30


def _load_recent_new_edge_module():
    module_path = ROOT / "feature" / "new score" / "recent_new_edge.py"
    spec = importlib.util.spec_from_file_location("recent_new_edge", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


recent_new_edge = _load_recent_new_edge_module()


class ScoreStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.minimum = 1.0
        self.maximum = 0.0
        self.bins = [0] * (BIN_COUNT + 1)

    def add(self, score):
        score = min(max(float(score), 0.0), 1.0)
        self.count += 1
        self.total += score
        self.total_square += score * score
        self.minimum = min(self.minimum, score)
        self.maximum = max(self.maximum, score)
        self.bins[min(int(score * BIN_COUNT), BIN_COUNT)] += 1

    def quantile(self, q):
        if self.count <= 0:
            return None
        target = max(math.ceil(self.count * q), 1)
        cumulative = 0
        for idx, count in enumerate(self.bins):
            cumulative += count
            if cumulative >= target:
                return idx / BIN_COUNT
        return 1.0

    def histogram(self, width=0.05):
        if self.count <= 0:
            return []
        bucket_count = int(1 / width)
        buckets = [0] * bucket_count
        for idx, count in enumerate(self.bins):
            if count <= 0:
                continue
            score = idx / BIN_COUNT
            bucket_idx = min(int(score / width), bucket_count - 1)
            buckets[bucket_idx] += count
        return [
            {
                "range": f"{idx * width:.2f}-{(idx + 1) * width:.2f}",
                "count": count,
            }
            for idx, count in enumerate(buckets)
        ]

    def as_dict(self):
        if self.count <= 0:
            return {"n": 0}
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "n": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "p25": self.quantile(0.25),
            "p50": self.quantile(0.50),
            "p75": self.quantile(0.75),
            "p90": self.quantile(0.90),
            "p95": self.quantile(0.95),
            "p99": self.quantile(0.99),
            "max": self.maximum,
            "histogram_0.05": self.histogram(0.05),
        }


def ordered_weekday_files():
    day_order = {day: index for index, day in enumerate(DAYS)}
    return sorted(
        [
            path
            for path in WINDOW_DIR.glob("*.csv")
            if path.name.split("-", 1)[0] in day_order
        ],
        key=lambda path: (day_order[path.name.split("-", 1)[0]], path.name),
    )


def is_attack_label(label):
    return str(label).strip() not in {"", "0", "BENIGN", "Normal", "normal"}


def safe_int(value, default=0):
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    if value in {None, ""}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_graph_with_labels(path):
    graph = TrafficGraph()
    edge_labels = {}
    packet_labels = Counter()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pkt = packet(
                src_ip=row["src_ip"],
                dst_ip=row["dst_ip"],
                src_port=safe_int(row.get("src_port")),
                dst_port=safe_int(row.get("dst_port")),
                timestamp=safe_float(row.get("timestamp")),
                proto_flags_mask=row.get("proto_flags_mask") or 0,
                payload_len=safe_float(row.get("payload_len")),
            )
            graph.add_packet(pkt)
            protocol = int(pkt.extract_proto_flags_fields()["protocol"])
            edge_key = (
                str(pkt.src_ip),
                str(pkt.dst_ip),
                int(pkt.src_port),
                int(pkt.dst_port),
                protocol,
            )
            label = "attack" if is_attack_label(row.get("label", "0")) else "normal"
            edge_labels[edge_key] = (
                "attack" if label == "attack" else edge_labels.get(edge_key, "normal")
            )
            packet_labels[label] += 1

    return graph, edge_labels, packet_labels


def predicted_label_from_score(
    score,
    attack_threshold=ATTACK_THRESHOLD,
    normal_threshold=NORMAL_THRESHOLD,
):
    if score >= attack_threshold:
        return "attack"
    if score <= normal_threshold:
        return "normal"
    return "unknown"


def commit_window(
    history,
    graph,
    predicted_edge_labels,
    suspicious_observations=None,
):
    """提交窗口，并且只使用模型预测和可疑证据更新攻击历史。"""
    final_edge_labels = dict(predicted_edge_labels)
    suspicious_result = None
    if suspicious_observations is not None:
        suspicious_result = history.suspicious_edge_history.update_window(
            suspicious_observations
        )
        for edge_key in suspicious_result.active_suspicious_edges:
            if edge_key in graph.edges:
                final_edge_labels[edge_key] = "suspicious"
        for edge_key in suspicious_result.promoted_attack_edges:
            final_edge_labels[edge_key] = "attack"
        for edge_key in suspicious_result.released_normal_edges:
            final_edge_labels[edge_key] = "normal"

        history.set_baseline_excluded_edges(
            suspicious_result.active_suspicious_edges
            | suspicious_result.promoted_attack_edges
        )
    else:
        history.set_baseline_excluded_edges(
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        )

    history.update_with_graph(graph)
    history.advance_window()
    history.update_node_behavior_role_vectors(current_window_only=True)
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        ],
        "attack",
    )
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "normal"
        ],
        "normal",
    )
    history.set_edge_label(
        [
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label in {"unknown", "suspicious"}
        ],
        "unknown",
    )
    if suspicious_result is None:
        strong_attack_edges = {
            edge_key
            for edge_key, label in final_edge_labels.items()
            if label == "attack"
        }
    else:
        strong_attack_edges = {
            edge_key
            for edge_key, decision in suspicious_result.decisions.items()
            if decision.reason == "score_reached_attack_threshold"
        }
    history.record_previous_attack_destinations(
        attack_edge_keys=strong_attack_edges
    )
    return suspicious_result, final_edge_labels


def log_progress(message):
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def main():
    PROGRESS_PATH.write_text("", encoding="utf-8")
    files = ordered_weekday_files()
    history = History(life_windows=30, detail_windows=5)
    score_stats = {
        "overall": {"normal": ScoreStats(), "attack": ScoreStats()},
        **{
            day: {"normal": ScoreStats(), "attack": ScoreStats()}
            for day in DAYS
        },
    }
    component_stats = {
        label: defaultdict(ScoreStats) for label in ["normal", "attack"]
    }
    timing_totals = Counter()
    timing_max = defaultdict(float)
    packet_counts = Counter()
    edge_counts = Counter()
    attack_windows = 0
    start_time = time.perf_counter()

    log_progress(f"start files={len(files)}")
    for idx, path in enumerate(files):
        day = path.name.split("-")[0]
        window_start = time.perf_counter()

        stage_start = time.perf_counter()
        graph, edge_labels, packet_labels = build_graph_with_labels(path)
        build_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        finite_scores = EdgeActiveHistoryFeature.compute_finite_history_offset_anomaly_scores(
            graph.edges.values(),
            history,
            current_window_only=False,
        )
        finite_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        novelty_scores = recent_new_edge.compute_approximate_novelty_anomaly_scores(
            graph,
            history,
        )
        novelty_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        current_scores = {
            edge_key: edge_obj.current_behavior_anomaly_score()
            for edge_key, edge_obj in graph.edges.items()
        }
        current_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        role_scores = history.compute_behavior_role_anomaly_scores(
            graph,
            current_window_only=False,
        )
        role_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        window_edge_labels = Counter()
        predicted_edge_labels = {}
        previous_attack_index = build_recent_attack_edge_index(history)
        for edge_key, edge_obj in graph.edges.items():
            label = edge_labels.get(edge_key, "normal")
            local_score = local_anomaly_score(
                current_scores.get(edge_key, 0.0),
                finite_scores.get(id(edge_obj), 0.0),
                novelty_scores.get(edge_key, 0.0),
                role_scores.get(edge_key, 0.0),
                edge_or_key=edge_key,
                history=history,
                previous_attack_index=previous_attack_index,
            )
            predicted_edge_labels[edge_key] = predicted_label_from_score(local_score)
            score_stats["overall"][label].add(local_score)
            score_stats[day][label].add(local_score)
            component_stats[label]["current_behavior"].add(
                current_scores.get(edge_key, 0.0)
            )
            component_stats[label]["finite_history"].add(
                finite_scores.get(id(edge_obj), 0.0)
            )
            component_stats[label]["novelty"].add(novelty_scores.get(edge_key, 0.0))
            component_stats[label]["behavior_role"].add(role_scores.get(edge_key, 0.0))
            window_edge_labels[label] += 1
        sum_time = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        commit_window(history, graph, predicted_edge_labels)
        commit_time = time.perf_counter() - stage_start
        total_time = time.perf_counter() - window_start

        if window_edge_labels.get("attack", 0) > 0:
            attack_windows += 1
        packet_counts.update({f"{day}_{key}": value for key, value in packet_labels.items()})
        edge_counts.update({f"{day}_{key}": value for key, value in window_edge_labels.items()})

        timings = {
            "build": build_time,
            "finite": finite_time,
            "novelty": novelty_time,
            "current": current_time,
            "role": role_time,
            "sum": sum_time,
            "commit": commit_time,
            "total": total_time,
        }
        for key, value in timings.items():
            timing_totals[key] += value
            timing_max[key] = max(timing_max[key], value)

        if idx % 100 == 0 or idx == len(files) - 1:
            elapsed = time.perf_counter() - start_time
            log_progress(
                "progress "
                f"idx={idx + 1}/{len(files)} file={path.name} "
                f"packets={len(graph.packets)} edges={len(graph.edges)} "
                f"edge_labels={dict(window_edge_labels)} "
                f"total={total_time:.3f}s elapsed={elapsed:.1f}s"
            )

    elapsed = time.perf_counter() - start_time
    timing_avg = {
        key: timing_totals[key] / max(len(files), 1) for key in timing_totals
    }
    result = {
        "file_count": len(files),
        "elapsed_seconds": elapsed,
        "attack_window_count": attack_windows,
        "packet_counts": dict(packet_counts),
        "edge_counts": dict(edge_counts),
        "timing_avg_seconds": timing_avg,
        "timing_max_seconds": dict(timing_max),
        "score_distribution": {
            scope: {
                label: stats.as_dict() for label, stats in label_stats.items()
            }
            for scope, label_stats in score_stats.items()
        },
        "component_distribution": {
            label: {
                name: stats.as_dict() for name, stats in stats_by_component.items()
            }
            for label, stats_by_component in component_stats.items()
        },
    }
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log_progress(f"done elapsed_seconds={elapsed:.3f} result={RESULT_PATH.name}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
