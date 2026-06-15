from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Set, Union


FeatureVector = Union[Mapping[str, Any], Sequence[Any]]


def _clamp(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, 0.0), 1.0)


def _copy_feature_vector(feature_vector: Optional[FeatureVector]) -> Any:
    if feature_vector is None:
        return None
    if isinstance(feature_vector, Mapping):
        return dict(feature_vector)
    if isinstance(feature_vector, (str, bytes)):
        return feature_vector
    return list(feature_vector)


@dataclass(frozen=True)
class SuspiciousEdgeObservation:
    """一个窗口内用于更新可疑边状态的证据。"""

    score: float
    sub_scores: Mapping[str, Any] = field(default_factory=dict)
    feature_vector: Optional[FeatureVector] = None
    attack_similarity_score: float = 0.0
    structural_expansion_score: float = 0.0
    attack_chain_score: float = 0.0


@dataclass
class SuspiciousEdgeRecord:
    """跨窗口保存的可疑边状态。"""

    edge_key: Hashable
    evidence_score: float
    max_score: float
    consecutive_suspicious_count: int
    sub_scores: Dict[str, Any]
    feature_vector: Any
    current_score: float
    previous_score: float
    attack_similarity_score: float
    structural_expansion_score: float
    attack_chain_score: float
    ttl: int
    first_seen_window: int
    last_seen_window: int
    suspicious_window_count: int = 1
    windows_since_seen: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "edge_key": (
                list(self.edge_key)
                if isinstance(self.edge_key, tuple)
                else self.edge_key
            ),
            "evidence_score": self.evidence_score,
            "max_score": self.max_score,
            "consecutive_suspicious_count": self.consecutive_suspicious_count,
            "sub_scores": dict(self.sub_scores),
            "feature_vector": _copy_feature_vector(self.feature_vector),
            "current_score": self.current_score,
            "previous_score": self.previous_score,
            "attack_similarity_score": self.attack_similarity_score,
            "structural_expansion_score": self.structural_expansion_score,
            "attack_chain_score": self.attack_chain_score,
            "ttl": self.ttl,
            "first_seen_window": self.first_seen_window,
            "last_seen_window": self.last_seen_window,
            "suspicious_window_count": self.suspicious_window_count,
            "windows_since_seen": self.windows_since_seen,
        }


@dataclass(frozen=True)
class SuspiciousEdgeDecision:
    """单条边在当前窗口结束后的分类结果。"""

    label: str
    reason: str
    evidence_score: float = 0.0
    reinforcing_signals: tuple = ()


@dataclass
class SuspiciousWindowResult:
    """一次窗口更新产生的状态变化。"""

    decisions: Dict[Hashable, SuspiciousEdgeDecision] = field(default_factory=dict)
    promoted_attack_edges: Set[Hashable] = field(default_factory=set)
    released_normal_edges: Set[Hashable] = field(default_factory=set)
    active_suspicious_edges: Set[Hashable] = field(default_factory=set)


class SuspiciousEdgeHistory:
    """
    可疑边的短期证据缓冲区。

    使用方式:
    1) 当前分数位于 [theta_suspicious, theta_attack) 时进入可疑集合。
    2) 每个窗口调用 update_window()，未延续的记录会降低 TTL 并衰减证据。
    3) 同边延续且分数、攻击相似度或结构扩张增强时，可升级为攻击。
    4) 可疑边通过 diffusion_weight() 以低于攻击边的权重参与扩散。
    5) should_update_behavior_baseline() 对可疑边返回 False，避免污染基线。
    6) TTL 到期且 evidence_score 下降到 release_threshold 后释放为正常。

    升级后的攻击边会从本缓冲区移除，应由现有 History.edge_label 接管。
    """

    def __init__(
        self,
        theta_suspicious: float = 0.55,
        theta_attack: float = 0.80,
        ttl_windows: int = 3,
        evidence_decay: float = 0.65,
        release_threshold: float = 0.35,
        promotion_evidence_threshold: Optional[float] = None,
        min_consecutive_windows: int = 2,
        min_reinforcing_signals: int = 2,
        min_score_rise: float = 0.05,
        min_similarity_rise: float = 0.05,
        min_structural_rise: float = 0.05,
        attack_chain_threshold: float = 0.85,
        reinforcing_signal_bonus: float = 0.04,
        suspicious_diffusion_weight: float = 0.25,
        attack_diffusion_weight: float = 1.0,
    ):
        if not 0.0 <= theta_suspicious < theta_attack <= 1.0:
            raise ValueError(
                "阈值必须满足 0 <= theta_suspicious < theta_attack <= 1。"
            )
        if ttl_windows <= 0:
            raise ValueError("ttl_windows 必须大于 0。")
        if not 0.0 < evidence_decay < 1.0:
            raise ValueError("evidence_decay 必须位于 (0, 1)。")
        if not 0.0 <= release_threshold < theta_suspicious:
            raise ValueError("release_threshold 必须低于 theta_suspicious。")
        if min_consecutive_windows <= 0 or min_reinforcing_signals <= 0:
            raise ValueError("连续窗口数和增强信号数必须大于 0。")
        if not 0.0 <= suspicious_diffusion_weight < attack_diffusion_weight:
            raise ValueError("可疑边扩散权重必须低于攻击边扩散权重。")

        self.theta_suspicious = float(theta_suspicious)
        self.theta_attack = float(theta_attack)
        self.ttl_windows = int(ttl_windows)
        self.evidence_decay = float(evidence_decay)
        self.release_threshold = float(release_threshold)
        self.promotion_evidence_threshold = float(
            promotion_evidence_threshold
            if promotion_evidence_threshold is not None
            else (theta_suspicious + theta_attack) / 2.0
        )
        self.min_consecutive_windows = int(min_consecutive_windows)
        self.min_reinforcing_signals = int(min_reinforcing_signals)
        self.min_score_rise = float(min_score_rise)
        self.min_similarity_rise = float(min_similarity_rise)
        self.min_structural_rise = float(min_structural_rise)
        self.attack_chain_threshold = float(attack_chain_threshold)
        self.reinforcing_signal_bonus = float(reinforcing_signal_bonus)
        self.suspicious_diffusion_weight = float(suspicious_diffusion_weight)
        self.attack_diffusion_weight = float(attack_diffusion_weight)
        self.window_index = 0
        self.edges: Dict[Hashable, SuspiciousEdgeRecord] = {}

    @staticmethod
    def _coerce_observation(
        observation: Union[SuspiciousEdgeObservation, Mapping[str, Any]],
    ) -> SuspiciousEdgeObservation:
        if isinstance(observation, SuspiciousEdgeObservation):
            return observation
        if not isinstance(observation, Mapping):
            raise TypeError("observation 必须是 SuspiciousEdgeObservation 或 Mapping。")
        return SuspiciousEdgeObservation(
            score=observation.get("score", observation.get("local_anomaly_score", 0.0)),
            sub_scores=observation.get("sub_scores", {}),
            feature_vector=observation.get("feature_vector"),
            attack_similarity_score=observation.get(
                "attack_similarity_score",
                observation.get("previous_attack_similarity_score", 0.0),
            ),
            structural_expansion_score=observation.get(
                "structural_expansion_score", 0.0
            ),
            attack_chain_score=observation.get("attack_chain_score", 0.0),
        )

    @staticmethod
    def _continuity_key(edge_key: Hashable) -> Hashable:
        """
        返回跨窗口延续匹配 key。

        标准五元组允许源端口变化，使用
        (源 IP, 目的 IP, 目的端口, 协议) 识别同一持续通信关系。
        非标准 key 保持精确匹配。
        """
        if isinstance(edge_key, tuple) and len(edge_key) == 5:
            return edge_key[0], edge_key[1], edge_key[3], edge_key[4]
        return edge_key

    @staticmethod
    def _window_evidence(observation: SuspiciousEdgeObservation) -> float:
        return _clamp(
            0.70 * _clamp(observation.score)
            + 0.18 * _clamp(observation.attack_similarity_score)
            + 0.12 * _clamp(observation.structural_expansion_score)
        )

    def _age_record(self, record: SuspiciousEdgeRecord) -> bool:
        record.ttl -= 1
        record.windows_since_seen += 1
        record.consecutive_suspicious_count = 0
        record.evidence_score = _clamp(
            record.evidence_score * self.evidence_decay
        )
        return record.ttl <= 0 and record.evidence_score <= self.release_threshold

    def _create_record(
        self,
        edge_key: Hashable,
        observation: SuspiciousEdgeObservation,
    ) -> SuspiciousEdgeRecord:
        score = _clamp(observation.score)
        return SuspiciousEdgeRecord(
            edge_key=edge_key,
            evidence_score=self._window_evidence(observation),
            max_score=score,
            consecutive_suspicious_count=1,
            sub_scores=dict(observation.sub_scores),
            feature_vector=_copy_feature_vector(observation.feature_vector),
            current_score=score,
            previous_score=score,
            attack_similarity_score=_clamp(observation.attack_similarity_score),
            structural_expansion_score=_clamp(
                observation.structural_expansion_score
            ),
            attack_chain_score=_clamp(observation.attack_chain_score),
            ttl=self.ttl_windows,
            first_seen_window=self.window_index,
            last_seen_window=self.window_index,
        )

    def _update_record(
        self,
        record: SuspiciousEdgeRecord,
        observation: SuspiciousEdgeObservation,
    ) -> tuple:
        score = _clamp(observation.score)
        similarity_score = _clamp(observation.attack_similarity_score)
        structural_score = _clamp(observation.structural_expansion_score)
        previous_score = record.current_score
        previous_similarity = record.attack_similarity_score
        previous_structural = record.structural_expansion_score

        reinforcing_signals = ["same_edge_continuation"]
        if score - previous_score >= self.min_score_rise:
            reinforcing_signals.append("score_rise")
        if (
            similarity_score - previous_similarity >= self.min_similarity_rise
        ):
            reinforcing_signals.append("attack_similarity_enhanced")
        if (
            structural_score - previous_structural >= self.min_structural_rise
        ):
            reinforcing_signals.append("structural_expansion_enhanced")

        current_evidence = self._window_evidence(observation)
        bonus = self.reinforcing_signal_bonus * len(reinforcing_signals)
        record.evidence_score = _clamp(
            self.evidence_decay * record.evidence_score
            + (1.0 - self.evidence_decay) * current_evidence
            + bonus
        )
        record.previous_score = previous_score
        record.current_score = score
        record.max_score = max(record.max_score, score)
        record.consecutive_suspicious_count += 1
        record.suspicious_window_count += 1
        record.sub_scores = dict(observation.sub_scores)
        record.feature_vector = _copy_feature_vector(observation.feature_vector)
        record.attack_similarity_score = similarity_score
        record.structural_expansion_score = structural_score
        record.attack_chain_score = _clamp(observation.attack_chain_score)
        record.ttl = self.ttl_windows
        record.last_seen_window = self.window_index
        record.windows_since_seen = 0
        return tuple(reinforcing_signals)

    def _should_promote(
        self,
        record: SuspiciousEdgeRecord,
        reinforcing_signals: tuple,
    ) -> bool:
        return (
            record.consecutive_suspicious_count >= self.min_consecutive_windows
            and len(reinforcing_signals) >= self.min_reinforcing_signals
            and record.evidence_score >= self.promotion_evidence_threshold
        )

    def update_window(
        self,
        observations: Mapping[
            Hashable, Union[SuspiciousEdgeObservation, Mapping[str, Any]]
        ],
    ) -> SuspiciousWindowResult:
        """推进一个窗口并返回每条边的可疑、攻击或正常决策。"""
        self.window_index += 1
        result = SuspiciousWindowResult()
        normalized_observations = {
            edge_key: self._coerce_observation(raw_observation)
            for edge_key, raw_observation in observations.items()
        }
        observed_keys = set(normalized_observations)

        continuity_candidates: Dict[Hashable, list] = {}
        for existing_key, record in self.edges.items():
            if existing_key in observed_keys:
                continue
            continuity_candidates.setdefault(
                self._continuity_key(existing_key), []
            ).append((existing_key, record))
        for candidates in continuity_candidates.values():
            candidates.sort(
                key=lambda item: (
                    item[1].last_seen_window,
                    item[1].evidence_score,
                ),
                reverse=True,
            )

        migrated_from: Dict[Hashable, Hashable] = {}
        protected_existing_keys: Set[Hashable] = set()
        for edge_key, observation in normalized_observations.items():
            if edge_key in self.edges or _clamp(observation.score) < self.theta_suspicious:
                continue
            candidates = continuity_candidates.get(
                self._continuity_key(edge_key), []
            )
            while candidates and candidates[0][0] in protected_existing_keys:
                candidates.pop(0)
            if not candidates:
                continue
            previous_key, _ = candidates.pop(0)
            migrated_from[edge_key] = previous_key
            protected_existing_keys.add(previous_key)

        for edge_key in list(self.edges):
            if edge_key in observed_keys or edge_key in protected_existing_keys:
                continue
            record = self.edges[edge_key]
            if self._age_record(record):
                self.edges.pop(edge_key, None)
                result.released_normal_edges.add(edge_key)
                result.decisions[edge_key] = SuspiciousEdgeDecision(
                    label="normal",
                    reason="ttl_expired_and_evidence_decayed",
                    evidence_score=record.evidence_score,
                )

        for edge_key, observation in normalized_observations.items():
            score = _clamp(observation.score)
            attack_chain_score = _clamp(observation.attack_chain_score)
            previous_key = migrated_from.get(edge_key)

            if (
                score >= self.theta_attack
                or attack_chain_score >= self.attack_chain_threshold
            ):
                previous_record = self.edges.pop(edge_key, None)
                if previous_record is None and previous_key is not None:
                    previous_record = self.edges.pop(previous_key, None)
                evidence_score = (
                    max(score, attack_chain_score, previous_record.evidence_score)
                    if previous_record is not None
                    else max(score, attack_chain_score)
                )
                result.promoted_attack_edges.add(edge_key)
                result.decisions[edge_key] = SuspiciousEdgeDecision(
                    label="attack",
                    reason=(
                        "score_reached_attack_threshold"
                        if score >= self.theta_attack
                        else "attack_chain_evidence"
                    ),
                    evidence_score=evidence_score,
                )
                continue

            if score >= self.theta_suspicious:
                record = self.edges.get(edge_key)
                if record is None and previous_key is not None:
                    record = self.edges.pop(previous_key, None)
                    if record is not None:
                        record.edge_key = edge_key
                        self.edges[edge_key] = record
                if record is None:
                    record = self._create_record(edge_key, observation)
                    self.edges[edge_key] = record
                    reinforcing_signals = ()
                else:
                    reinforcing_signals = self._update_record(record, observation)

                if self._should_promote(record, reinforcing_signals):
                    self.edges.pop(edge_key, None)
                    result.promoted_attack_edges.add(edge_key)
                    result.decisions[edge_key] = SuspiciousEdgeDecision(
                        label="attack",
                        reason="multi_window_evidence_strengthened",
                        evidence_score=record.evidence_score,
                        reinforcing_signals=reinforcing_signals,
                    )
                else:
                    result.decisions[edge_key] = SuspiciousEdgeDecision(
                        label="suspicious",
                        reason="score_in_suspicious_interval",
                        evidence_score=record.evidence_score,
                        reinforcing_signals=reinforcing_signals,
                    )
                continue

            record = self.edges.get(edge_key)
            if record is None:
                result.decisions[edge_key] = SuspiciousEdgeDecision(
                    label="normal",
                    reason="score_below_suspicious_threshold",
                )
                continue

            record.previous_score = record.current_score
            record.current_score = score
            record.sub_scores = dict(observation.sub_scores)
            record.feature_vector = _copy_feature_vector(
                observation.feature_vector
            )
            if self._age_record(record):
                self.edges.pop(edge_key, None)
                result.released_normal_edges.add(edge_key)
                result.decisions[edge_key] = SuspiciousEdgeDecision(
                    label="normal",
                    reason="ttl_expired_and_evidence_decayed",
                    evidence_score=record.evidence_score,
                )
            else:
                result.decisions[edge_key] = SuspiciousEdgeDecision(
                    label="suspicious",
                    reason="cooling_down_before_release",
                    evidence_score=record.evidence_score,
                )

        result.active_suspicious_edges = set(self.edges)
        return result

    def should_update_behavior_baseline(self, edge_key: Hashable) -> bool:
        """可疑边仍在缓冲区时禁止正常更新行为基线。"""
        return edge_key not in self.edges

    def baseline_excluded_edges(self) -> Set[Hashable]:
        return set(self.edges)

    def diffusion_weight(
        self,
        edge_key: Hashable,
        is_attack: bool = False,
    ) -> float:
        """攻击边使用完整权重，可疑边使用较低权重，普通边不参与异常扩散。"""
        if is_attack:
            return self.attack_diffusion_weight
        if edge_key in self.edges:
            return self.suspicious_diffusion_weight
        return 0.0

    def weighted_diffusion_score(
        self,
        edge_key: Hashable,
        score: float,
        is_attack: bool = False,
    ) -> float:
        return _clamp(score) * self.diffusion_weight(edge_key, is_attack=is_attack)

    def build_destination_to_source_diffusion_index(self) -> Dict[str, float]:
        """
        构建可疑边的低权重目的转源扩散索引。

        若上一窗口可疑边 A->B，则当前以 B 为源的边可以获得低权重证据；
        可疑边不会把证据直接扩散给自身，避免形成自反馈升级。
        """
        diffusion_index: Dict[str, float] = {}
        for edge_key, record in self.edges.items():
            if not isinstance(edge_key, tuple) or len(edge_key) != 5:
                continue
            destination_ip = str(edge_key[1])
            weighted_score = self.weighted_diffusion_score(
                edge_key,
                record.evidence_score,
            )
            diffusion_index[destination_ip] = max(
                diffusion_index.get(destination_ip, 0.0),
                weighted_score,
            )
        return diffusion_index

    def get(self, edge_key: Hashable) -> Optional[SuspiciousEdgeRecord]:
        return self.edges.get(edge_key)

    def snapshot(self) -> Dict[Hashable, Dict[str, Any]]:
        return {
            edge_key: record.as_dict()
            for edge_key, record in self.edges.items()
        }

    def clear(self):
        self.edges.clear()
        self.window_index = 0


__all__ = [
    "SuspiciousEdgeDecision",
    "SuspiciousEdgeHistory",
    "SuspiciousEdgeObservation",
    "SuspiciousEdgeRecord",
    "SuspiciousWindowResult",
]
