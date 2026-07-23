"""Few-shot 元任务采样器 (支持原始样本区间不重叠约束)。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..utils.logger import get_logger
from .dataset import IntrusionDataset
from .leakage import check_support_query_overlap, window_indices_overlap

logger = get_logger(__name__)


@dataclass
class MetaTask:
    """一个元任务（episode）。"""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    global_classes: List[int]
    support_window_ids: List[int] = field(default_factory=list)
    query_window_ids: List[int] = field(default_factory=list)
    shot: Optional[int] = None

    def to(self, device: torch.device) -> "MetaTask":
        return MetaTask(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            global_classes=self.global_classes,
            support_window_ids=list(self.support_window_ids),
            query_window_ids=list(self.query_window_ids),
            shot=self.shot,
        )

    @property
    def n_way(self) -> int:
        return len(self.global_classes)


def _pick_non_overlapping_windows(
    dataset: IntrusionDataset,
    pool: List[int],
    k: int,
    forbidden: Set[int],
    rng: np.random.Generator,
    disallow_overlap: bool,
    disallow_internal_overlap: bool = True,
) -> List[int]:
    """从 pool 中无放回选取 k 个窗口, 不与 forbidden 中窗口共享原始样本。"""
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked: List[int] = []
    for wid in shuffled:
        if len(picked) >= k:
            break
        if disallow_overlap:
            conflict = False
            for f in forbidden:
                if window_indices_overlap(dataset, wid, f):
                    conflict = True
                    break
            if disallow_internal_overlap:
                for p in picked:
                    if window_indices_overlap(dataset, wid, p):
                        conflict = True
                        break
            if conflict:
                continue
        picked.append(wid)
    return picked


class FewShotTaskSampler:
    """从 IntrusionDataset 采样 N-way K-shot 元任务。"""

    def __init__(self, dataset: IntrusionDataset, n_way: int, k_shot: int,
                 q_query: int, allowed_classes: Optional[List[int]] = None,
                 seed: Optional[int] = None,
                 disallow_support_query_overlap: bool = True,
                 disallow_internal_overlap: bool = True,
                 binary_pair_mode: str = "any_two",
                 benign_class: Optional[int] = None) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.disallow_overlap = disallow_support_query_overlap
        self.disallow_internal_overlap = disallow_internal_overlap
        self.binary_pair_mode = str(binary_pair_mode).lower()
        self.benign_class = benign_class
        self.rng = np.random.default_rng(seed)
        self._perm_seed_max = np.iinfo(np.int32).max
        logger.info(
            "FewShotTaskSampler overlap policy: support_query_disallowed=%s "
            "internal_disallowed=%s",
            disallow_support_query_overlap,
            disallow_internal_overlap,
        )

        if not disallow_support_query_overlap:
            logger.warning("disallow_support_query_overlap=false: support/query 可能共享原始样本(旧行为)。")

        available = allowed_classes if allowed_classes is not None else dataset.classes
        self.class_pool: List[int] = []
        need = k_shot + q_query
        for cls in available:
            count = len(dataset.class_to_indices.get(cls, []))
            if count >= need:
                self.class_pool.append(cls)
            else:
                logger.warning("类别 %d 窗口不足 (%d < K+Q=%d), 已从任务池移除。", cls, count, need)
        if len(self.class_pool) < n_way:
            raise ValueError(
                f"可用类别数 {len(self.class_pool)} < n_way {n_way}; "
                f"请减小 n_way/k_shot/q_query/stride 或增大数据。")

        if self.n_way == 2 and self.binary_pair_mode == "benign_vs_attack":
            if self.benign_class is None:
                raise ValueError("binary_pair_mode=benign_vs_attack requires benign_class")
            if self.benign_class not in self.class_pool:
                raise ValueError(
                    "binary_pair_mode=benign_vs_attack requires benign_class "
                    f"{self.benign_class} in class_pool={self.class_pool}"
                )
            attack_pool = [cls for cls in self.class_pool if cls != self.benign_class]
            if not attack_pool:
                raise ValueError(
                    "binary_pair_mode=benign_vs_attack requires at least one "
                    "non-benign class in class_pool"
                )
        elif self.binary_pair_mode not in {"any_two", "benign_vs_attack"}:
            raise ValueError(
                "Unsupported binary_pair_mode=%r; expected any_two/benign_vs_attack"
                % self.binary_pair_mode
            )

        logger.info(
            "FewShotTaskSampler class_pool=%s | window_counts=%s | K=%d Q=%d "
            "binary_pair_mode=%s benign_class=%s",
            self.class_pool,
            {int(cls): len(dataset.class_to_indices.get(int(cls), [])) for cls in available},
            self.k_shot,
            self.q_query,
            self.binary_pair_mode,
            self.benign_class,
        )

    def sample_task(self) -> MetaTask:
        if self.n_way == 2 and self.binary_pair_mode == "benign_vs_attack":
            attack_pool = [cls for cls in self.class_pool if cls != self.benign_class]
            attack = int(self.rng.choice(attack_pool))
            chosen = np.array([int(self.benign_class), attack], dtype=int)
        else:
            chosen = self.rng.choice(self.class_pool, size=self.n_way, replace=False)
        support_wids: List[int] = []
        query_wids: List[int] = []
        support_local: List[int] = []
        query_local: List[int] = []
        forbidden: Set[int] = set()

        for local_label, cls in enumerate(chosen):
            pool = list(self.dataset.class_to_indices[int(cls)])
            picked_s = _pick_non_overlapping_windows(
                self.dataset, pool, self.k_shot, set(), self.rng,
                self.disallow_overlap,
                disallow_internal_overlap=self.disallow_internal_overlap)
            if len(picked_s) < self.k_shot:
                logger.error(
                    "Insufficient support windows for class=%s: picked=%d needed=%d "
                    "pool=%d internal_overlap_disallowed=%s",
                    cls, len(picked_s), self.k_shot, len(pool),
                    self.disallow_internal_overlap,
                )
                raise ValueError(
                    f"类 {cls} 无法在 disallow_overlap={self.disallow_overlap} 下采满 "
                    f"{self.k_shot} 个 support 窗口; 请增大 stride 或减小 k_shot。")
            support_wids.extend(picked_s)
            support_local.extend([local_label] * len(picked_s))

            forbidden.update(picked_s)
            picked_q = _pick_non_overlapping_windows(
                self.dataset, pool, self.q_query, forbidden, self.rng,
                self.disallow_overlap,
                disallow_internal_overlap=self.disallow_internal_overlap)
            if len(picked_q) < self.q_query:
                logger.error(
                    "Insufficient query windows for class=%s: picked=%d needed=%d "
                    "pool=%d support_picked=%d internal_overlap_disallowed=%s",
                    cls, len(picked_q), self.q_query, len(pool), len(picked_s),
                    self.disallow_internal_overlap,
                )
                raise ValueError(
                    f"类 {cls} 无法在 disallow_overlap={self.disallow_overlap} 下采满 "
                    f"{self.q_query} 个 query 窗口。")
            query_wids.extend(picked_q)
            query_local.extend([local_label] * len(picked_q))

        if self.disallow_overlap:
            overlaps = check_support_query_overlap(support_wids, query_wids, self.dataset)
            logger.debug("support/query overlap audit pairs=%d", len(overlaps))
            if overlaps:
                raise RuntimeError(f"support/query 原始样本重叠: {overlaps[:5]}...")

        support_x = self.dataset.features[support_wids]
        query_x = self.dataset.features[query_wids]
        support_y = torch.tensor(support_local, dtype=torch.long)
        query_y = torch.tensor(query_local, dtype=torch.long)

        s_gen = torch.Generator().manual_seed(int(self.rng.integers(0, self._perm_seed_max)))
        q_gen = torch.Generator().manual_seed(int(self.rng.integers(0, self._perm_seed_max)))
        s_perm = torch.randperm(support_x.shape[0], generator=s_gen)
        q_perm = torch.randperm(query_x.shape[0], generator=q_gen)
        return MetaTask(
            support_x=support_x[s_perm],
            support_y=support_y[s_perm],
            query_x=query_x[q_perm],
            query_y=query_y[q_perm],
            global_classes=[int(c) for c in chosen],
            support_window_ids=[support_wids[i] for i in s_perm.tolist()],
            query_window_ids=[query_wids[i] for i in q_perm.tolist()],
            shot=self.k_shot,
        )

    def sample_batch(self, meta_batch_size: int) -> List[MetaTask]:
        return [self.sample_task() for _ in range(meta_batch_size)]


class MixedShotTaskSampler:
    """Seeded wrapper that samples a shot value before sampling each task."""

    def __init__(
        self,
        prototype: FewShotTaskSampler,
        shots: List[int],
        seed: int,
    ) -> None:
        unique_shots = sorted({int(shot) for shot in shots if int(shot) > 0})
        if not unique_shots:
            raise ValueError("mixed-shot training requires at least one positive shot")
        self.shots = unique_shots
        self.rng = np.random.default_rng(int(seed))
        self.samplers = {
            shot: FewShotTaskSampler(
                dataset=prototype.dataset,
                n_way=prototype.n_way,
                k_shot=shot,
                q_query=prototype.q_query,
                allowed_classes=list(prototype.class_pool),
                seed=int(seed) + 1009 * index,
                disallow_support_query_overlap=prototype.disallow_overlap,
                disallow_internal_overlap=prototype.disallow_internal_overlap,
                binary_pair_mode=prototype.binary_pair_mode,
                benign_class=prototype.benign_class,
            )
            for index, shot in enumerate(unique_shots, start=1)
        }
        self.dataset = prototype.dataset
        self.n_way = prototype.n_way
        self.q_query = prototype.q_query
        self.k_shot = prototype.k_shot

    def sample_task(self) -> MetaTask:
        shot = int(self.rng.choice(self.shots))
        return self.samplers[shot].sample_task()

    def sample_batch(self, meta_batch_size: int) -> List[MetaTask]:
        return [self.sample_task() for _ in range(meta_batch_size)]
