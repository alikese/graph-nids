import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.decision_engine import evidence_decision, predicted_label_from_observation
from feature.edge.edgeClass import CURRENT_BEHAVIOR_WEIGHTS
from feature.history.historyClass import BEHAVIOR_ROLE_WEIGHTS, History
from feature.history.history_feature.active_edge_features import EdgeActiveHistoryFeature
from feature.score_profile import (
    PROTOCOL_GROUPS,
    normalize_weights as normalize_named_weights,
    protocol_group_from_edge_key,
    weighted_sum,
    weighted_sum_with_gains,
)
from feature.scoring_pipeline import commit_window, recent_new_edge, score_window
from feature.sum_score import DEFAULT_LOCAL_ANOMALY_WEIGHTS
from full_week_score_test import build_graph_with_labels
from train.train_score_weights import load_trained_weights, reset_all_weights

try:
    import torch
except ImportError:
    torch = None


VECTOR_NAMES = (
    "current_behavior_anomaly_score",
    "finite_history_offset_anomaly_score",
    "approximate_novelty_anomaly_score",
    "behavior_role_anomaly_score",
)
LABELS = ("normal", "attack")
COMPONENT_DEFAULT_WEIGHTS = {
    "current_behavior_anomaly_score": CURRENT_BEHAVIOR_WEIGHTS,
    "finite_history_offset_anomaly_score": (
        EdgeActiveHistoryFeature.FINITE_HISTORY_OFFSET_WEIGHTS
    ),
    "approximate_novelty_anomaly_score": recent_new_edge.APPROXIMATE_NOVELTY_WEIGHTS,
    "behavior_role_anomaly_score": BEHAVIOR_ROLE_WEIGHTS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--source-window-dir", type=Path, required=True)
    parser.add_argument("--source-days", nargs="+", required=True)
    parser.add_argument("--target-window-dir", type=Path, required=True)
    parser.add_argument("--target-days", nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--source-max-windows", type=int, default=0)
    parser.add_argument("--target-max-windows", type=int, default=0)
    parser.add_argument(
        "--source-windows-per-day",
        type=int,
        default=0,
        help="Use a contiguous slice of this many source windows from each source day.",
    )
    parser.add_argument(
        "--target-windows-per-day",
        type=int,
        default=0,
        help="Use a contiguous slice of this many target windows from each target day.",
    )
    parser.add_argument(
        "--source-window-start",
        type=int,
        default=0,
        help="Zero-based start offset for each source day contiguous slice.",
    )
    parser.add_argument(
        "--target-window-start",
        type=int,
        default=0,
        help="Zero-based start offset for each target day contiguous slice.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-source-samples-per-bucket", type=int, default=50000)
    parser.add_argument("--max-target-samples-per-bucket", type=int, default=20000)
    parser.add_argument("--max-optimization-samples-per-label", type=int, default=5000)
    parser.add_argument(
        "--adaptation-target",
        choices=("internal_weights", "component_gains"),
        default="internal_weights",
    )
    parser.add_argument("--prototype-count", type=int, default=3)
    parser.add_argument("--target-use-labels", action="store_true")
    parser.add_argument("--pseudo-normal-quantile", type=float, default=0.30)
    parser.add_argument("--pseudo-attack-quantile", type=float, default=0.90)
    parser.add_argument("--pseudo-normal-max-strong", type=int, default=1)
    parser.add_argument("--pseudo-attack-min-strong", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--regularization", type=float, default=0.10)
    parser.add_argument("--score-margin", type=float, default=0.05)
    parser.add_argument("--score-margin-weight", type=float, default=0.0)
    parser.add_argument("--sparse-direction-weight", type=float, default=0.0)
    parser.add_argument("--sparse-direction-top-k", type=int, default=3)
    parser.add_argument("--sparse-direction-positive-threshold", type=float, default=0.01)
    parser.add_argument("--sparse-direction-negative-threshold", type=float, default=0.05)
    parser.add_argument("--sparse-direction-positive-target", type=float, default=1.45)
    parser.add_argument("--sparse-direction-negative-target", type=float, default=0.90)
    parser.add_argument(
        "--sparse-direction-protocol-negative-targets",
        default="",
        help="Optional comma-separated overrides, for example TCP=0.80,UDP=0.70,OTHER=0.90.",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--initial-step", type=float, default=0.15)
    parser.add_argument("--step-decay", type=float, default=0.60)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Use cuda for the contrastive scale optimizer when PyTorch is available.",
    )
    parser.add_argument("--torch-steps", type=int, default=400)
    parser.add_argument("--torch-learning-rate", type=float, default=0.05)
    parser.add_argument("--min-scale", type=float, default=0.70)
    parser.add_argument("--max-scale", type=float, default=1.30)
    parser.add_argument("--disable-scale-bounds", action="store_true")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument(
        "--domain-calibrated-thresholds",
        action="store_true",
        help=(
            "Use target-domain normal score quantiles directly for decision "
            "thresholds instead of applying the source-domain minimum floors."
        ),
    )
    parser.add_argument("--suspicious-target-quantile", type=float, default=0.75)
    parser.add_argument("--attack-threshold", type=float)
    parser.add_argument("--suspicious-threshold", type=float)
    parser.add_argument("--min-attack-threshold", type=float, default=0.45)
    parser.add_argument("--min-suspicious-threshold", type=float, default=0.25)
    parser.add_argument("--normal-threshold", type=float, default=0.30)
    parser.add_argument("--min-strong-signals", type=int, default=0)
    parser.add_argument("--use-attack-evidence-gate", action="store_true")
    parser.add_argument("--attack-evidence-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def ordered_files(window_dir, days):
    day_order = {day: index for index, day in enumerate(days)}
    files = [
        path
        for path in window_dir.glob("*.csv")
        if path.name.split("-", 1)[0] in day_order
    ]
    return sorted(
        files,
        key=lambda path: (
            day_order[path.name.split("-", 1)[0]],
            path.name,
        ),
    )


def contiguous_files_per_day(window_dir, days, windows_per_day, start_index):
    if not windows_per_day or windows_per_day <= 0:
        return ordered_files(window_dir, days)
    selected = []
    start_index = max(int(start_index), 0)
    for day in days:
        day_files = sorted(window_dir.glob(f"{day}-ip_*.csv"), key=lambda path: path.name)
        if not day_files:
            continue
        end_index = min(start_index + int(windows_per_day), len(day_files))
        selected.extend(day_files[start_index:end_index])
    return selected


def limit_files_evenly(files, limit):
    if not limit or limit <= 0 or len(files) <= limit:
        return files
    if limit == 1:
        return [files[0]]
    indexes = {
        round(index * (len(files) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [files[index] for index in sorted(indexes)]


def label_key(label):
    return (
        "attack"
        if str(label).strip() not in {"", "0", "BENIGN", "Normal", "normal"}
        else "normal"
    )


def vector_from_observation(observation):
    sub_scores = observation.get("sub_scores", {}) or {}
    return tuple(float(sub_scores.get(name, 0.0)) for name in VECTOR_NAMES)


def log(path, message):
    print(message, flush=True)
    if path:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")


def reservoir_add(bucket, sample, limit, rng):
    if limit <= 0 or len(bucket) < limit:
        bucket.append(sample)
        return
    seen = getattr(bucket, "_seen", None)
    if seen is None:
        seen = len(bucket)
    seen += 1
    setattr(bucket, "_seen", seen)
    index = rng.randrange(seen)
    if index < limit:
        bucket[index] = sample


class SampleBucket(list):
    pass


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    q = min(max(float(q), 0.0), 1.0)
    index = int(round(q * (len(ordered) - 1)))
    return ordered[index]


def collect_samples(
    files,
    weights_path,
    progress_path,
    progress_every,
    rng,
    max_samples_per_bucket,
    use_labels,
    pseudo_normal_quantile,
    pseudo_attack_quantile,
    pseudo_normal_max_strong,
    pseudo_attack_min_strong,
    source_mode,
):
    reset_all_weights()
    load_trained_weights(weights_path)
    history = History(life_windows=30, detail_windows=5)
    buckets = {
        group: {label: SampleBucket() for label in LABELS}
        for group in ("ALL", *PROTOCOL_GROUPS)
    }
    unlabeled_target = []
    score_by_group = defaultdict(list)
    started_at = time.perf_counter()
    label_source = "source" if source_mode else "target"
    log(
        progress_path,
        f"{label_source}_collect_start windows={len(files)} labels={use_labels}",
    )

    for index, path in enumerate(files, start=1):
        graph, true_edge_labels, _ = build_graph_with_labels(path)
        scores, observations = score_window(graph, history)
        predicted = {
            edge_key: predicted_label_from_observation(observations[edge_key])
            for edge_key in scores
        }
        _, final_labels = commit_window(
            history,
            graph,
            predicted,
            suspicious_observations=observations,
        )

        for edge_key, score in scores.items():
            observation = observations[edge_key]
            group = protocol_group_from_edge_key(edge_key)
            vector = vector_from_observation(observation)
            true_label = label_key(true_edge_labels.get(edge_key, "normal"))
            decision = evidence_decision(observation)
            strong_count = int(decision.get("strong_general_signal_count", 0))
            sample = {
                "vector": vector,
                "features": dict(observation.get("feature_vector", {}) or {}),
                "score": float(score),
                "true_label": true_label,
                "final_label": final_labels.get(edge_key, "normal"),
                "strong_count": strong_count,
            }
            if source_mode or use_labels:
                label = true_label
                for target_group in (group, "ALL"):
                    reservoir_add(
                        buckets[target_group][label],
                        sample,
                        max_samples_per_bucket,
                        rng,
                    )
            else:
                unlabeled_target.append((group, sample))
                score_by_group[group].append(float(score))
                score_by_group["ALL"].append(float(score))

        if index % progress_every == 0 or index == len(files):
            log(
                progress_path,
                (
                    f"{label_source}_collect progress={index}/{len(files)} "
                    f"file={path.name} elapsed={time.perf_counter() - started_at:.1f}"
                ),
            )

    if not source_mode and not use_labels:
        thresholds = {}
        for group in PROTOCOL_GROUPS:
            scores = score_by_group.get(group) or score_by_group.get("ALL") or []
            thresholds[group] = {
                "normal": quantile(scores, pseudo_normal_quantile),
                "attack": quantile(scores, pseudo_attack_quantile),
            }
        for group, sample in unlabeled_target:
            limits = thresholds.get(group, {})
            normal_threshold = limits.get("normal")
            attack_threshold = limits.get("attack")
            label = None
            if (
                normal_threshold is not None
                and sample["score"] <= normal_threshold
                and sample["final_label"] != "attack"
                and sample["strong_count"] <= pseudo_normal_max_strong
            ):
                label = "normal"
            elif (
                attack_threshold is not None
                and sample["score"] >= attack_threshold
                and (
                    sample["final_label"] == "attack"
                    or sample["strong_count"] >= pseudo_attack_min_strong
                )
            ):
                label = "attack"
            if label is None:
                continue
            for target_group in (group, "ALL"):
                reservoir_add(
                    buckets[target_group][label],
                    sample,
                    max_samples_per_bucket,
                    rng,
                )

    summary = {
        group: {
            label: len(buckets[group][label])
            for label in LABELS
        }
        for group in ("ALL", *PROTOCOL_GROUPS)
    }
    log(progress_path, f"{label_source}_collect_done samples={summary}")
    return buckets


def normalize_vector(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)


def cosine(vector_a, vector_b):
    a = normalize_vector(vector_a)
    b = normalize_vector(vector_b)
    return sum(x * y for x, y in zip(a, b))


def mean_vector(samples):
    if not samples:
        return tuple(0.0 for _ in VECTOR_NAMES)
    totals = [0.0] * len(VECTOR_NAMES)
    for sample in samples:
        for index, value in enumerate(sample["vector"]):
            totals[index] += float(value)
    return tuple(value / len(samples) for value in totals)


def build_prototypes(samples, prototype_count):
    if not samples:
        return []
    count = max(int(prototype_count), 1)
    ordered = sorted(samples, key=lambda item: item["score"])
    if len(ordered) <= count:
        return [normalize_vector(sample["vector"]) for sample in ordered]
    prototypes = []
    for index in range(count):
        start = round(index * len(ordered) / count)
        end = round((index + 1) * len(ordered) / count)
        chunk = ordered[start:end]
        if chunk:
            prototypes.append(normalize_vector(mean_vector(chunk)))
    return prototypes


def limit_samples_for_optimization(samples, limit, rng):
    limit = int(limit or 0)
    if limit <= 0 or len(samples) <= limit:
        return list(samples)
    return rng.sample(list(samples), limit)


def logsumexp(values):
    if not values:
        return -1e12
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def scaled_vector(vector, scales):
    return tuple(float(value) * float(scale) for value, scale in zip(vector, scales))


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def base_internal_weights_from_payload(payload):
    internal_payload = payload.get("internal_weights", {}) or {}
    return {
        component: normalize_named_weights(
            internal_payload.get(component, {}),
            defaults,
        )
        for component, defaults in COMPONENT_DEFAULT_WEIGHTS.items()
    }


def internal_scale_keys(base_internal_weights):
    keys = []
    for component in VECTOR_NAMES:
        for feature in base_internal_weights.get(component, {}):
            keys.append((component, feature))
    return keys


def internal_feature_gains_from_scales(base_internal_weights, scale_keys, scales):
    gains = {
        component: {
            feature: 1.0
            for feature in weights
        }
        for component, weights in base_internal_weights.items()
    }
    for index, (component, feature) in enumerate(scale_keys):
        if component not in gains or feature not in gains[component]:
            continue
        gains[component][feature] = float(scales[index])
    return gains


def vector_from_features(sample, internal_weights, internal_feature_gains=None):
    features = sample.get("features", {}) or {}
    values = []
    internal_feature_gains = internal_feature_gains or {}
    for component in VECTOR_NAMES:
        weights = internal_weights.get(component, {})
        gains = internal_feature_gains.get(component)
        if gains:
            values.append(weighted_sum_with_gains(features, weights, gains))
        else:
            values.append(weighted_sum(features, weights))
    return tuple(values)


def total_score_from_features(
    sample,
    internal_weights,
    top_level_weights,
    internal_feature_gains=None,
):
    vector = vector_from_features(sample, internal_weights, internal_feature_gains)
    components = {
        name: vector[index]
        for index, name in enumerate(VECTOR_NAMES)
    }
    return weighted_sum(components, top_level_weights)


def vector_with_gains(sample, gains):
    vector = sample.get("vector", ())
    return tuple(
        clamp01(float(vector[index]) * float(gains[index]))
        for index in range(len(VECTOR_NAMES))
    )


def total_score_with_gains(sample, gains, top_level_weights):
    vector = vector_with_gains(sample, gains)
    components = {
        name: vector[index]
        for index, name in enumerate(VECTOR_NAMES)
    }
    return weighted_sum(components, top_level_weights)


def score_for_sample(
    sample,
    adaptation_target,
    base_internal_weights,
    scale_keys,
    scales,
    top_level_weights,
):
    if adaptation_target == "component_gains":
        return total_score_with_gains(sample, scales, top_level_weights)
    internal_feature_gains = internal_feature_gains_from_scales(
        base_internal_weights,
        scale_keys,
        scales,
    )
    return total_score_from_features(
        sample,
        base_internal_weights,
        top_level_weights,
        internal_feature_gains=internal_feature_gains,
    )


def mean_score_gap(
    samples_by_label,
    adaptation_target,
    base_internal_weights,
    scale_keys,
    scales,
    top_level_weights,
):
    normal_samples = samples_by_label.get("normal", [])
    attack_samples = samples_by_label.get("attack", [])
    if not normal_samples or not attack_samples:
        return None
    normal_mean = sum(
        score_for_sample(
            sample,
            adaptation_target,
            base_internal_weights,
            scale_keys,
            scales,
            top_level_weights,
        )
        for sample in normal_samples
    ) / len(normal_samples)
    attack_mean = sum(
        score_for_sample(
            sample,
            adaptation_target,
            base_internal_weights,
            scale_keys,
            scales,
            top_level_weights,
        )
        for sample in attack_samples
    ) / len(attack_samples)
    return attack_mean - normal_mean


def parse_protocol_float_overrides(raw_value):
    overrides = {}
    for item in str(raw_value or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid protocol override: {item}")
        group, value = item.split("=", 1)
        group = group.strip().upper()
        if group not in PROTOCOL_GROUPS:
            raise ValueError(f"unknown protocol override group: {group}")
        overrides[group] = float(value)
    return overrides


def bounded_scale(value, args):
    value = float(value)
    if args.disable_scale_bounds:
        return value
    return min(max(value, args.min_scale), args.max_scale)


def sparse_direction_prior(samples_by_label, scale_keys, args, protocol_group=None):
    normal_samples = samples_by_label.get("normal", [])
    attack_samples = samples_by_label.get("attack", [])
    if (
        args.sparse_direction_weight <= 0.0
        or not normal_samples
        or not attack_samples
        or not scale_keys
    ):
        return None, None

    positive_threshold = float(args.sparse_direction_positive_threshold)
    negative_threshold = float(args.sparse_direction_negative_threshold)
    top_k = max(int(args.sparse_direction_top_k), 0)
    positive_target = bounded_scale(args.sparse_direction_positive_target, args)
    negative_target = bounded_scale(args.sparse_direction_negative_target, args)
    protocol_negative_targets = parse_protocol_float_overrides(
        args.sparse_direction_protocol_negative_targets
    )
    if protocol_group in protocol_negative_targets:
        negative_target = bounded_scale(protocol_negative_targets[protocol_group], args)

    stats = []
    positive_candidates = []
    for index, (component, feature) in enumerate(scale_keys):
        normal_mean = sum(
            float(sample.get("features", {}).get(feature, 0.0))
            for sample in normal_samples
        ) / len(normal_samples)
        attack_mean = sum(
            float(sample.get("features", {}).get(feature, 0.0))
            for sample in attack_samples
        ) / len(attack_samples)
        delta = attack_mean - normal_mean
        row = {
            "component": component,
            "feature": feature,
            "normal_mean": normal_mean,
            "attack_mean": attack_mean,
            "delta": delta,
            "target_gain": 1.0,
            "direction_role": "neutral",
        }
        stats.append(row)
        if delta > positive_threshold:
            positive_candidates.append((delta, index))

    targets = [1.0] * len(scale_keys)
    for _, index in sorted(positive_candidates, reverse=True)[:top_k]:
        targets[index] = positive_target
        stats[index]["target_gain"] = positive_target
        stats[index]["direction_role"] = "positive_top"

    for index, row in enumerate(stats):
        if row["direction_role"] == "positive_top":
            continue
        if row["delta"] < -negative_threshold:
            targets[index] = negative_target
            row["target_gain"] = negative_target
            row["protocol_negative_target"] = negative_target
            row["direction_role"] = "negative_soft_down"

    return stats, targets


def sparse_direction_penalty(scales, targets, weight):
    if not targets or weight <= 0.0:
        return 0.0
    return float(weight) * (
        sum(
            (float(scale) - float(target)) ** 2
            for scale, target in zip(scales, targets)
        )
        / max(len(targets), 1)
    )


def selected_torch_device(args):
    if args.device == "cpu":
        return None
    if torch is None:
        if args.device == "cuda":
            raise RuntimeError("PyTorch is required for --device cuda")
        return None
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for PyTorch")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return None


def inverse_softplus(value):
    value = max(float(value), 1e-6)
    return math.log(math.exp(value) - 1.0)


def differentiable_scales(raw_scales, args):
    if args.disable_scale_bounds:
        return torch.nn.functional.softplus(raw_scales) + 1e-6
    min_scale = float(args.min_scale)
    max_scale = float(args.max_scale)
    return min_scale + (max_scale - min_scale) * torch.sigmoid(raw_scales)


def initial_raw_scales(scale_count, args, device):
    if args.disable_scale_bounds:
        value = inverse_softplus(1.0)
        return torch.full((scale_count,), value, dtype=torch.float32, device=device)
    min_scale = float(args.min_scale)
    max_scale = float(args.max_scale)
    ratio = (1.0 - min_scale) / max(max_scale - min_scale, 1e-8)
    ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
    value = math.log(ratio / (1.0 - ratio))
    return torch.full((scale_count,), value, dtype=torch.float32, device=device)


def make_torch_target(samples, adaptation_target, base_internal_weights, scale_keys, device):
    samples = list(samples)
    vectors = torch.tensor(
        [sample.get("vector", (0.0, 0.0, 0.0, 0.0)) for sample in samples],
        dtype=torch.float32,
        device=device,
    )
    if adaptation_target == "component_gains":
        return {"vectors": vectors}

    component_index = {
        name: index
        for index, name in enumerate(VECTOR_NAMES)
    }
    weighted_features = []
    component_ids = []
    for component, feature in scale_keys:
        weight = float(base_internal_weights.get(component, {}).get(feature, 0.0))
        component_ids.append(component_index[component])
        weighted_features.append([
            weight * clamp01(sample.get("features", {}).get(feature, 0.0))
            for sample in samples
        ])
    if weighted_features:
        feature_matrix = torch.tensor(
            list(zip(*weighted_features)),
            dtype=torch.float32,
            device=device,
        )
    else:
        feature_matrix = torch.zeros((len(samples), 0), dtype=torch.float32, device=device)
    component_ids = torch.tensor(component_ids, dtype=torch.long, device=device)
    return {
        "features": feature_matrix,
        "component_ids": component_ids,
    }


def torch_vectors(target, scales, adaptation_target):
    if adaptation_target == "component_gains":
        return torch.clamp(target["vectors"] * scales, min=0.0, max=1.0)
    features = target["features"] * scales
    vectors = torch.zeros(
        (features.shape[0], len(VECTOR_NAMES)),
        dtype=features.dtype,
        device=features.device,
    )
    if features.shape[1] > 0:
        vectors.index_add_(1, target["component_ids"], features)
    return torch.clamp(vectors, min=0.0, max=1.0)


def torch_prototypes(prototypes, fallback, device):
    selected = prototypes or fallback or []
    if not selected:
        return None
    return torch.nn.functional.normalize(
        torch.tensor(selected, dtype=torch.float32, device=device),
        dim=1,
        eps=1e-8,
    )


def optimize_scales_torch(
    samples_by_label,
    source_prototypes,
    fallback_prototypes,
    base_internal_weights,
    scale_keys,
    top_level_weights,
    args,
    protocol_group=None,
):
    device = selected_torch_device(args)
    if device is None:
        return None

    scale_count = (
        len(VECTOR_NAMES)
        if args.adaptation_target == "component_gains"
        else len(scale_keys)
    )
    if scale_count <= 0:
        return None

    raw_scales = initial_raw_scales(scale_count, args, device)
    raw_scales.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [raw_scales],
        lr=max(float(args.torch_learning_rate), 1e-6),
    )
    _, sparse_targets = sparse_direction_prior(
        samples_by_label,
        scale_keys,
        args,
        protocol_group=protocol_group,
    )
    sparse_target_tensor = None
    if sparse_targets:
        sparse_target_tensor = torch.tensor(
            sparse_targets,
            dtype=torch.float32,
            device=device,
        )

    torch_samples = {
        label: make_torch_target(
            samples_by_label.get(label, []),
            args.adaptation_target,
            base_internal_weights,
            scale_keys,
            device,
        )
        for label in LABELS
    }
    proto = {}
    for label in LABELS:
        negative_label = "attack" if label == "normal" else "normal"
        proto[(label, "positive")] = torch_prototypes(
            source_prototypes.get(label),
            fallback_prototypes.get(label),
            device,
        )
        proto[(label, "negative")] = torch_prototypes(
            source_prototypes.get(negative_label),
            fallback_prototypes.get(negative_label),
            device,
        )

    top_weights = torch.tensor(
        [float(top_level_weights.get(name, 0.0)) for name in VECTOR_NAMES],
        dtype=torch.float32,
        device=device,
    )
    best_loss = None
    best_scales = None
    steps = max(int(args.torch_steps), 0)
    temperature = max(float(args.temperature), 1e-6)
    for _ in range(steps):
        optimizer.zero_grad()
        scales = differentiable_scales(raw_scales, args)
        losses = []
        score_by_label = {}
        for label in LABELS:
            positive = proto[(label, "positive")]
            negative = proto[(label, "negative")]
            if positive is None or negative is None:
                continue
            vectors = torch_vectors(
                torch_samples[label],
                scales,
                args.adaptation_target,
            )
            if vectors.shape[0] <= 0:
                continue
            score_by_label[label] = torch.sum(vectors * top_weights, dim=1)
            normalized = torch.nn.functional.normalize(vectors, dim=1, eps=1e-8)
            pos_terms = normalized @ positive.T / temperature
            neg_terms = normalized @ negative.T / temperature
            losses.append(
                -(
                    torch.logsumexp(pos_terms, dim=1)
                    - torch.logsumexp(
                        torch.cat([pos_terms, neg_terms], dim=1),
                        dim=1,
                    )
                ).mean()
            )
        if not losses:
            return None
        loss = torch.stack(losses).mean()
        loss = loss + float(args.regularization) * torch.mean((scales - 1.0) ** 2)
        if (
            float(args.score_margin_weight) > 0.0
            and "normal" in score_by_label
            and "attack" in score_by_label
        ):
            gap = score_by_label["attack"].mean() - score_by_label["normal"].mean()
            loss = loss + float(args.score_margin_weight) * torch.relu(
                torch.tensor(float(args.score_margin), dtype=torch.float32, device=device)
                - gap
            )
        if (
            args.adaptation_target == "internal_weights"
            and sparse_target_tensor is not None
            and float(args.sparse_direction_weight) > 0.0
        ):
            loss = loss + float(args.sparse_direction_weight) * torch.mean(
                (scales - sparse_target_tensor) ** 2
            )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_scales = scales.detach().cpu().tolist()

    if best_scales is None:
        return None
    return [bounded_scale(value, args) for value in best_scales], best_loss


def contrastive_loss(
    samples_by_label,
    source_prototypes,
    fallback_prototypes,
    base_internal_weights,
    scale_keys,
    scales,
    temperature,
    regularization,
    top_level_weights,
    score_margin,
    score_margin_weight,
    sparse_direction_targets=None,
    sparse_direction_weight=0.0,
    adaptation_target="internal_weights",
):
    losses = []
    internal_feature_gains = None
    if adaptation_target == "internal_weights":
        internal_feature_gains = internal_feature_gains_from_scales(
            base_internal_weights,
            scale_keys,
            scales,
        )
    for label in LABELS:
        positive = source_prototypes.get(label) or fallback_prototypes.get(label) or []
        negative_label = "attack" if label == "normal" else "normal"
        negative = (
            source_prototypes.get(negative_label)
            or fallback_prototypes.get(negative_label)
            or []
        )
        if not positive or not negative:
            continue
        for sample in samples_by_label.get(label, []):
            if adaptation_target == "component_gains":
                vector = vector_with_gains(sample, scales)
            else:
                vector = vector_from_features(
                    sample,
                    base_internal_weights,
                    internal_feature_gains,
                )
            pos_terms = [cosine(vector, proto) / temperature for proto in positive]
            neg_terms = [cosine(vector, proto) / temperature for proto in negative]
            losses.append(-(logsumexp(pos_terms) - logsumexp(pos_terms + neg_terms)))
    if not losses:
        return None
    base = sum(losses) / len(losses)
    penalty = regularization * (
        sum((scale - 1.0) ** 2 for scale in scales) / max(len(scales), 1)
    )
    margin_penalty = 0.0
    if score_margin_weight > 0.0:
        gap = mean_score_gap(
            samples_by_label,
            adaptation_target,
            base_internal_weights,
            scale_keys,
            scales,
            top_level_weights,
        )
        if gap is not None:
            margin_penalty = score_margin_weight * max(
                0.0,
                float(score_margin) - float(gap),
            )
    sparse_penalty = 0.0
    if adaptation_target == "internal_weights":
        sparse_penalty = sparse_direction_penalty(
            scales,
            sparse_direction_targets,
            sparse_direction_weight,
        )
    return base + penalty + margin_penalty + sparse_penalty


def optimize_scales(
    samples_by_label,
    source_prototypes,
    fallback_prototypes,
    base_internal_weights,
    scale_keys,
    top_level_weights,
    args,
    protocol_group=None,
):
    torch_result = optimize_scales_torch(
        samples_by_label,
        source_prototypes,
        fallback_prototypes,
        base_internal_weights,
        scale_keys,
        top_level_weights,
        args,
        protocol_group=protocol_group,
    )
    if torch_result is not None:
        return torch_result

    scale_count = len(VECTOR_NAMES) if args.adaptation_target == "component_gains" else len(scale_keys)
    scales = [1.0] * scale_count
    _, sparse_targets = sparse_direction_prior(
        samples_by_label,
        scale_keys,
        args,
        protocol_group=protocol_group,
    )
    best_loss = contrastive_loss(
        samples_by_label,
        source_prototypes,
        fallback_prototypes,
        base_internal_weights,
        scale_keys,
        scales,
        args.temperature,
        args.regularization,
        top_level_weights,
        args.score_margin,
        args.score_margin_weight,
        sparse_targets,
        args.sparse_direction_weight,
        args.adaptation_target,
    )
    if best_loss is None:
        return scales, None

    step = float(args.initial_step)
    for _ in range(max(int(args.iterations), 0)):
        for dimension in range(len(scales)):
            candidates = []
            for delta in (-step, 0.0, step):
                candidate = list(scales)
                candidate[dimension] = bounded_scale(
                    candidate[dimension] + delta,
                    args,
                )
                loss = contrastive_loss(
                    samples_by_label,
                    source_prototypes,
                    fallback_prototypes,
                    base_internal_weights,
                    scale_keys,
                    candidate,
                    args.temperature,
                    args.regularization,
                    top_level_weights,
                    args.score_margin,
                    args.score_margin_weight,
                    sparse_targets,
                    args.sparse_direction_weight,
                    args.adaptation_target,
                )
                if loss is not None:
                    candidates.append((loss, candidate))
            if candidates:
                best_loss, scales = min(candidates, key=lambda item: item[0])
        step *= float(args.step_decay)
    return scales, best_loss


def normalize_weights(weights):
    total = sum(max(float(value), 0.0) for value in weights.values())
    if total <= 0.0:
        return normalize_named_weights({}, DEFAULT_LOCAL_ANOMALY_WEIGHTS)
    return {
        name: max(float(value), 0.0) / total
        for name, value in weights.items()
    }


def adapted_top_weights(base_weights, scales):
    adjusted = {
        name: float(base_weights.get(name, DEFAULT_LOCAL_ANOMALY_WEIGHTS[name]))
        * float(scales[index])
        for index, name in enumerate(VECTOR_NAMES)
    }
    return normalize_weights(adjusted)


def protocol_thresholds(
    target_samples,
    group,
    args,
    internal_weights=None,
    internal_feature_gains=None,
    top_level_weights=None,
    component_gains=None,
):
    def scores_for(samples):
        if component_gains is not None and top_level_weights is not None:
            return [
                total_score_with_gains(sample, component_gains, top_level_weights)
                for sample in samples
            ]
        if internal_weights is None or top_level_weights is None:
            return [sample["score"] for sample in samples]
        return [
            total_score_from_features(
                sample,
                internal_weights,
                top_level_weights,
                internal_feature_gains=internal_feature_gains,
            )
            for sample in samples
        ]

    normal_scores = [
        *scores_for(target_samples[group]["normal"])
    ] or [*scores_for(target_samples["ALL"]["normal"])]
    if args.attack_threshold is not None:
        attack_threshold = args.attack_threshold
        attack_threshold_source = "override"
    else:
        calibrated = quantile(normal_scores, 1.0 - args.target_fpr)
        fallback = args.min_attack_threshold
        calibrated = calibrated if calibrated is not None else fallback
        if args.domain_calibrated_thresholds:
            attack_threshold = calibrated
            attack_threshold_source = "target_normal_quantile"
        else:
            attack_threshold = max(fallback, calibrated)
            attack_threshold_source = "target_normal_quantile_with_source_floor"
    if args.suspicious_threshold is not None:
        suspicious_threshold = args.suspicious_threshold
        suspicious_threshold_source = "override"
    else:
        calibrated = quantile(normal_scores, args.suspicious_target_quantile)
        fallback = args.min_suspicious_threshold
        calibrated = calibrated if calibrated is not None else fallback
        if args.domain_calibrated_thresholds:
            suspicious_threshold = calibrated
            suspicious_threshold_source = "target_normal_quantile"
        else:
            suspicious_threshold = max(fallback, calibrated)
            suspicious_threshold_source = "target_normal_quantile_with_source_floor"
    return {
        "attack_threshold": min(max(float(attack_threshold), 0.0), 1.0),
        "suspicious_threshold": min(max(float(suspicious_threshold), 0.0), 1.0),
        "normal_threshold": min(max(float(args.normal_threshold), 0.0), 1.0),
        "min_strong_signals": int(args.min_strong_signals),
        "auth_attack_threshold": 0.80,
        "auth_suspicious_threshold": 0.60,
        "target_fpr": args.target_fpr,
        "domain_calibrated_thresholds": bool(args.domain_calibrated_thresholds),
        "attack_threshold_source": attack_threshold_source,
        "suspicious_threshold_source": suspicious_threshold_source,
        "suspicious_target_quantile": args.suspicious_target_quantile,
        "allow_score_only_attack": bool(args.domain_calibrated_thresholds),
        "use_attack_evidence_gate": bool(args.use_attack_evidence_gate),
        "attack_evidence_threshold": args.attack_evidence_threshold,
    }


def main():
    args = parse_args()
    if args.progress_path:
        args.progress_path.parent.mkdir(parents=True, exist_ok=True)
        args.progress_path.write_text("", encoding="utf-8")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    source_files = limit_files_evenly(
        contiguous_files_per_day(
            args.source_window_dir,
            args.source_days,
            args.source_windows_per_day,
            args.source_window_start,
        ),
        args.source_max_windows,
    )
    target_files = limit_files_evenly(
        contiguous_files_per_day(
            args.target_window_dir,
            args.target_days,
            args.target_windows_per_day,
            args.target_window_start,
        ),
        args.target_max_windows,
    )
    if not source_files:
        raise FileNotFoundError("no source windows found")
    if not target_files:
        raise FileNotFoundError("no target windows found")

    weights_payload = json.loads(args.weights_path.read_text(encoding="utf-8"))
    base_top_weights = normalize_named_weights(
        weights_payload.get("top_level_weights", {}),
        DEFAULT_LOCAL_ANOMALY_WEIGHTS,
    )
    base_internal_weights = base_internal_weights_from_payload(weights_payload)
    scale_keys = internal_scale_keys(base_internal_weights)

    source_samples = collect_samples(
        source_files,
        args.weights_path,
        args.progress_path,
        args.progress_every,
        rng,
        args.max_source_samples_per_bucket,
        use_labels=True,
        pseudo_normal_quantile=args.pseudo_normal_quantile,
        pseudo_attack_quantile=args.pseudo_attack_quantile,
        pseudo_normal_max_strong=args.pseudo_normal_max_strong,
        pseudo_attack_min_strong=args.pseudo_attack_min_strong,
        source_mode=True,
    )
    target_samples = collect_samples(
        target_files,
        args.weights_path,
        args.progress_path,
        args.progress_every,
        rng,
        args.max_target_samples_per_bucket,
        use_labels=args.target_use_labels,
        pseudo_normal_quantile=args.pseudo_normal_quantile,
        pseudo_attack_quantile=args.pseudo_attack_quantile,
        pseudo_normal_max_strong=args.pseudo_normal_max_strong,
        pseudo_attack_min_strong=args.pseudo_attack_min_strong,
        source_mode=False,
    )

    prototypes = {
        group: {
            label: build_prototypes(source_samples[group][label], args.prototype_count)
            for label in LABELS
        }
        for group in ("ALL", *PROTOCOL_GROUPS)
    }

    protocol_profiles = {}
    alignment_summary = {}
    for group in PROTOCOL_GROUPS:
        fallback_reason = None
        group_targets = {
            label: list(target_samples[group][label])
            for label in LABELS
        }
        if not group_targets["normal"] or not group_targets["attack"]:
            fallback_reason = "missing_target_label_sample"
            group_targets = {
                label: list(target_samples["ALL"][label])
                for label in LABELS
            }
        optimization_targets = {
            label: limit_samples_for_optimization(
                group_targets[label],
                args.max_optimization_samples_per_label,
                rng,
            )
            for label in LABELS
        }
        sparse_stats, sparse_targets = sparse_direction_prior(
            optimization_targets,
            scale_keys,
            args,
            protocol_group=group,
        )
        scales, loss = optimize_scales(
            optimization_targets,
            prototypes[group],
            prototypes["ALL"],
            base_internal_weights,
            scale_keys,
            base_top_weights,
            args,
            protocol_group=group,
        )
        score_gap = mean_score_gap(
            optimization_targets,
            args.adaptation_target,
            base_internal_weights,
            scale_keys,
            scales,
            base_top_weights,
        )
        profile_entry = {
            "top_level_weights": dict(base_top_weights),
        }
        if args.adaptation_target == "component_gains":
            component_gains = {
                name: scales[index]
                for index, name in enumerate(VECTOR_NAMES)
            }
            internal_weights = None
            profile_entry["component_gains"] = component_gains
            profile_entry["decision"] = protocol_thresholds(
                target_samples,
                group,
                args,
                component_gains=scales,
                top_level_weights=base_top_weights,
            )
            profile_entry["alignment_weights"] = dict(component_gains)
        else:
            internal_feature_gains = internal_feature_gains_from_scales(
                base_internal_weights,
                scale_keys,
                scales,
            )
            profile_entry["internal_weights"] = base_internal_weights
            profile_entry["internal_feature_gains"] = internal_feature_gains
            profile_entry["decision"] = protocol_thresholds(
                target_samples,
                group,
                args,
                internal_weights=base_internal_weights,
                internal_feature_gains=internal_feature_gains,
                top_level_weights=base_top_weights,
            )
            profile_entry["alignment_weights"] = {
                f"{component}.{feature}": scales[index]
                for index, (component, feature) in enumerate(scale_keys)
            }
            profile_entry["internal_feature_gain_scales"] = internal_feature_gains
        profile_entry.update(
            {
                "alignment_loss": loss,
                "optimization_score_gap": score_gap,
                "sparse_direction_weight": args.sparse_direction_weight,
                "sparse_direction_targets": sparse_targets,
                "sparse_direction_stats": sparse_stats,
                "sparse_direction_protocol_negative_target": (
                    parse_protocol_float_overrides(
                        args.sparse_direction_protocol_negative_targets
                    ).get(group, args.sparse_direction_negative_target)
                ),
                "target_sample_counts": {
                    label: len(group_targets[label])
                    for label in LABELS
                },
                "target_sample_fallback_reason": fallback_reason,
                "optimization_sample_counts": {
                    label: len(optimization_targets[label])
                    for label in LABELS
                },
                "source_prototype_counts": {
                    label: len(prototypes[group][label])
                    for label in LABELS
                },
            }
        )
        protocol_profiles[group] = profile_entry
        alignment_summary[group] = {
            "scale_count": len(scales),
            "loss": loss,
            "top_level_weights": dict(base_top_weights),
            "decision": protocol_profiles[group]["decision"],
            "component_gains": protocol_profiles[group].get("component_gains"),
            "optimization_score_gap": score_gap,
            "optimization_sample_counts": protocol_profiles[group][
                "optimization_sample_counts"
            ],
        }
        log(args.progress_path, f"aligned group={group} summary={alignment_summary[group]}")

    result = {
        "schema_version": 1,
        "kind": "score_profile",
        "method": (
            "Protocol-conditioned asymmetric multi-prototype contrastive "
            "calibration. Source protocol/class prototypes are fixed; target "
            "samples train protocol-specific internal feature weight scales."
        ),
        "weights_path": str(args.weights_path),
        "source_window_dir": str(args.source_window_dir),
        "source_days": args.source_days,
        "source_window_count": len(source_files),
        "source_windows_per_day": args.source_windows_per_day,
        "source_window_start": args.source_window_start,
        "target_window_dir": str(args.target_window_dir),
        "target_days": args.target_days,
        "target_window_count": len(target_files),
        "target_windows_per_day": args.target_windows_per_day,
        "target_window_start": args.target_window_start,
        "target_use_labels": bool(args.target_use_labels),
        "adaptation_target": args.adaptation_target,
        "optimizer_device_request": args.device,
        "torch_available": torch is not None,
        "torch_cuda_available": bool(torch is not None and torch.cuda.is_available()),
        "torch_steps": args.torch_steps,
        "torch_learning_rate": args.torch_learning_rate,
        "vector_names": VECTOR_NAMES,
        "internal_scale_keys": [
            {"component": component, "feature": feature}
            for component, feature in scale_keys
        ],
        "prototype_count": args.prototype_count,
        "max_optimization_samples_per_label": args.max_optimization_samples_per_label,
        "temperature": args.temperature,
        "regularization": args.regularization,
        "score_margin": args.score_margin,
        "score_margin_weight": args.score_margin_weight,
        "sparse_direction_weight": args.sparse_direction_weight,
        "sparse_direction_top_k": args.sparse_direction_top_k,
        "sparse_direction_positive_threshold": (
            args.sparse_direction_positive_threshold
        ),
        "sparse_direction_negative_threshold": (
            args.sparse_direction_negative_threshold
        ),
        "sparse_direction_positive_target": args.sparse_direction_positive_target,
        "sparse_direction_negative_target": args.sparse_direction_negative_target,
        "sparse_direction_protocol_negative_targets": (
            parse_protocol_float_overrides(
                args.sparse_direction_protocol_negative_targets
            )
        ),
        "scale_bounds": None if args.disable_scale_bounds else [args.min_scale, args.max_scale],
        "scale_bounds_disabled": bool(args.disable_scale_bounds),
        "protocol_profiles": protocol_profiles,
    }
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(args.progress_path, f"done result={args.output_path}")


if __name__ == "__main__":
    main()
