import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature.edge.edgeClass import DEFAULT_CURRENT_BEHAVIOR_WEIGHTS
from feature.history.historyClass import DEFAULT_BEHAVIOR_ROLE_WEIGHTS
from feature.history.history_feature.active_edge_features import EdgeActiveHistoryFeature
from feature.score_profile import PROTOCOL_GROUPS, normalize_weights
from feature.sum_score import DEFAULT_LOCAL_ANOMALY_WEIGHTS


INTERNAL_DEFAULTS = {
    "current_behavior_anomaly_score": DEFAULT_CURRENT_BEHAVIOR_WEIGHTS,
    "finite_history_offset_anomaly_score": (
        EdgeActiveHistoryFeature.DEFAULT_FINITE_HISTORY_OFFSET_WEIGHTS
    ),
    "approximate_novelty_anomaly_score": {
        "recent_new_edge_mark": 0.25,
        "edge_low_activity_score": 0.25,
        "approximate_rare_edge_score": 0.25,
        "source_destination_diversity_burst_score": 0.15,
        "source_port_diversity_burst_score": 0.10,
    },
    "behavior_role_anomaly_score": DEFAULT_BEHAVIOR_ROLE_WEIGHTS,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--source-env-profile", type=Path, required=True)
    parser.add_argument("--target-env-profile", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--min-attack-threshold", type=float, default=0.70)
    parser.add_argument("--min-suspicious-threshold", type=float, default=0.55)
    parser.add_argument("--drift-strength", type=float, default=2.0)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested_get(mapping, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def stats_mean(profile, group, section, component, label, feature):
    if section == "top":
        value = nested_get(
            profile,
            "protocol_profiles",
            group,
            "top_level_feature_stats",
            label,
            feature,
            "mean",
            default=None,
        )
    else:
        value = nested_get(
            profile,
            "protocol_profiles",
            group,
            "internal_feature_stats",
            component,
            label,
            feature,
            "mean",
            default=None,
        )
    if value is None:
        value = nested_get(
            profile,
            "protocol_profiles",
            group,
            "score_stats",
            label,
            "mean",
            default=0.0,
        )
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def detail_signal(profile, group, section, component, feature):
    if section == "top":
        value = nested_get(
            profile,
            "protocol_profiles",
            group,
            "top_level_feature_details",
            feature,
            "discrimination_signal",
            default=0.0,
        )
    else:
        value = nested_get(
            profile,
            "protocol_profiles",
            group,
            "internal_feature_details",
            component,
            feature,
            "discrimination_signal",
            default=0.0,
        )
    try:
        return max(float(value), 0.0)
    except (TypeError, ValueError):
        return 0.0


def adapt_weights(
    base_weights,
    defaults,
    source_profile,
    target_profile,
    group,
    section,
    component=None,
    drift_strength=2.0,
):
    base_weights = normalize_weights(base_weights or {}, defaults)
    max_signal = max(
        detail_signal(source_profile, group, section, component, name)
        for name in defaults
    )
    adjusted = {}
    for name, base_weight in base_weights.items():
        signal = detail_signal(source_profile, group, section, component, name)
        if max_signal > 0.0:
            reliability = (signal + 0.02) / (max_signal + 0.02)
        else:
            reliability = 1.0
        source_normal_mean = stats_mean(
            source_profile,
            group,
            section,
            component,
            "normal",
            name,
        )
        target_all_mean = stats_mean(
            target_profile,
            group,
            section,
            component,
            "all",
            name,
        )
        drift = abs(target_all_mean - source_normal_mean)
        similarity = 1.0 / (1.0 + drift_strength * drift)
        adjusted[name] = base_weight * reliability * similarity
    return normalize_weights(adjusted, defaults)


def protocol_thresholds(target_profile, group, args):
    recommendation = nested_get(
        target_profile,
        "protocol_profiles",
        group,
        "decision_recommendation",
        default={},
    ) or {}
    attack_threshold = max(
        args.min_attack_threshold,
        float(recommendation.get("attack_threshold", args.min_attack_threshold)),
    )
    suspicious_threshold = max(
        args.min_suspicious_threshold,
        float(
            recommendation.get(
                "suspicious_threshold",
                args.min_suspicious_threshold,
            )
        ),
    )
    return {
        "attack_threshold": min(max(attack_threshold, 0.0), 1.0),
        "suspicious_threshold": min(max(suspicious_threshold, 0.0), 1.0),
        "normal_threshold": 0.30,
        "min_strong_signals": 2,
        "auth_attack_threshold": 0.80,
        "auth_suspicious_threshold": 0.60,
        "target_fpr": args.target_fpr,
    }


def main():
    args = parse_args()
    weights_payload = load_json(args.weights_path)
    source_profile = load_json(args.source_env_profile)
    target_profile = load_json(args.target_env_profile)
    top_base = weights_payload.get("top_level_weights", DEFAULT_LOCAL_ANOMALY_WEIGHTS)
    internal_base = weights_payload.get("internal_weights", {})

    protocol_profiles = {}
    for group in PROTOCOL_GROUPS:
        top_level_weights = adapt_weights(
            top_base,
            DEFAULT_LOCAL_ANOMALY_WEIGHTS,
            source_profile,
            target_profile,
            group,
            "top",
            drift_strength=args.drift_strength,
        )
        internal_weights = {}
        for component, defaults in INTERNAL_DEFAULTS.items():
            internal_weights[component] = adapt_weights(
                internal_base.get(component, defaults),
                defaults,
                source_profile,
                target_profile,
                group,
                "internal",
                component=component,
                drift_strength=args.drift_strength,
            )
        protocol_profiles[group] = {
            "top_level_weights": top_level_weights,
            "internal_weights": internal_weights,
            "decision": protocol_thresholds(target_profile, group, args),
            "source_traffic": nested_get(
                source_profile,
                "protocol_profiles",
                group,
                "traffic",
                default={},
            ),
            "target_traffic": nested_get(
                target_profile,
                "protocol_profiles",
                group,
                "traffic",
                default={},
            ),
        }

    result = {
        "schema_version": 1,
        "kind": "score_profile",
        "method": (
            "Protocol-aware transfer calibration. Base trained weights are "
            "attenuated by source feature reliability and source-normal to "
            "target-environment drift; decision thresholds come from target "
            "environment score quantiles."
        ),
        "weights_path": str(args.weights_path),
        "source_env_profile": str(args.source_env_profile),
        "target_env_profile": str(args.target_env_profile),
        "drift_strength": args.drift_strength,
        "target_fpr": args.target_fpr,
        "protocol_profiles": protocol_profiles,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"done result={args.output_path}")


if __name__ == "__main__":
    main()
