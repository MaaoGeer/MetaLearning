"""Windowed dataset construction and few-shot task samplers."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from ..utils.config import Config
from ..utils.logger import get_logger
from .dataset import IntrusionDataset
from .loao import SplitArrays
from .task_sampler import FewShotTaskSampler, MetaTask
from .windowing import build_windows

logger = get_logger(__name__)


def _windowing_kwargs(cfg: Config) -> dict:
    d = cfg.data
    return {
        "windowing_mode": str(d.get("windowing_mode", "temporal")).lower(),
        "label_strategy": str(d.get("window_label_strategy", "last")).lower(),
        "majority_tie_break": str(d.get("majority_tie_break", "last")).lower(),
    }


def _labels_for_windowing(split: SplitArrays, class_to_idx: Dict[str, int],
                          label_strategy: str) -> tuple[np.ndarray, Dict[str, int]]:
    """Return integer labels and the class index used by windowing.

    For ``any_attack`` every non-benign raw label is collapsed into the same
    attack class. This makes the label-strategy ablation runnable for datasets
    with many known attack families without leaking the held-out unknown class
    into meta-train.
    """
    if label_strategy == "any_attack":
        c2i = {"benign": 0, "attack": 1}
        labels_idx = np.array([
            0 if str(label).lower() in {"benign", "normal"} else 1
            for label in split.labels
        ], dtype=np.int64)
        return labels_idx, c2i
    labels_idx = np.array([class_to_idx[str(label)] for label in split.labels], dtype=np.int64)
    return labels_idx, dict(class_to_idx)


def build_windowed_dataset(
    split: SplitArrays,
    class_to_idx: Dict[str, int],
    window_size: int,
    stride: int,
    cfg: Optional[Config] = None,
    windowing_mode: str = "temporal",
    label_strategy: str = "last",
    majority_tie_break: str = "last",
) -> IntrusionDataset:
    """Window one already-isolated raw split."""
    if cfg is not None:
        wk = _windowing_kwargs(cfg)
        windowing_mode = wk["windowing_mode"]
        label_strategy = wk["label_strategy"]
        majority_tie_break = wk["majority_tie_break"]
    else:
        windowing_mode = str(windowing_mode).lower()
        label_strategy = str(label_strategy).lower()
        majority_tie_break = str(majority_tie_break).lower()

    labels_idx, c2i = _labels_for_windowing(split, class_to_idx, label_strategy)
    result = build_windows(
        split.features,
        labels_idx,
        split.order,
        window_size,
        stride,
        windowing_mode=windowing_mode,
        label_strategy=label_strategy,
        label_names=split.labels,
        class_to_idx=c2i,
        row_id=split.row_id,
        segment_id=split.segment_id,
        majority_tie_break=majority_tie_break,
    )
    ds = IntrusionDataset.from_window_result(result, window_size)
    logger.info(
        "Windowed[%s/%s]: samples=%d, L=%d, F=%d, class_dist=%s",
        windowing_mode, label_strategy, len(ds), window_size, ds.feature_dim,
        {k: len(v) for k, v in ds.class_to_indices.items()},
    )
    return ds


def make_meta_sampler(
    dataset: IntrusionDataset,
    allowed_classes: List[int],
    n_way: int,
    k_shot: int,
    q_query: int,
    seed: Optional[int] = None,
    disallow_support_query_overlap: bool = True,
    disallow_internal_overlap: bool = True,
    binary_pair_mode: str = "any_two",
    benign_class: Optional[int] = None,
) -> FewShotTaskSampler:
    return FewShotTaskSampler(
        dataset=dataset,
        n_way=min(n_way, len(allowed_classes)),
        k_shot=k_shot,
        q_query=q_query,
        allowed_classes=allowed_classes,
        seed=seed,
        disallow_support_query_overlap=disallow_support_query_overlap,
        disallow_internal_overlap=disallow_internal_overlap,
        binary_pair_mode=binary_pair_mode,
        benign_class=benign_class,
    )


def validate_unknown_window_sufficiency(
    dataset: IntrusionDataset,
    unknown_idx: int,
    k_shot: int,
    q_query: int,
    context: str = "adaptation",
) -> int:
    """Require enough held-out attack windows before building adaptation tasks."""
    need = int(k_shot) + int(q_query)
    count = len(dataset.class_to_indices.get(int(unknown_idx), []))
    if count < need:
        msg = (
            f"{context}: unknown class index {unknown_idx} has {count} windows, "
            f"but K+Q={need} are required. Skip this unknown or reduce "
            "window_size/stride/K/Q."
        )
        logger.error(msg)
        raise ValueError(msg)
    logger.info(
        "%s unknown window sufficiency PASS: unknown_idx=%s windows=%d K+Q=%d",
        context,
        unknown_idx,
        count,
        need,
    )
    return count


class AdaptationTaskSampler:
    """Unknown-attack adaptation sampler that forces the unknown class in task."""

    def __init__(
        self,
        dataset: IntrusionDataset,
        unknown_idx: int,
        ref_indices: List[int],
        mode: str = "binary",
        n_way: int = 2,
        k_shot: int = 5,
        q_query: int = 15,
        benign_idx: Optional[int] = None,
        seed: Optional[int] = None,
        disallow_support_query_overlap: bool = True,
        disallow_internal_overlap: bool = True,
    ) -> None:
        self.dataset = dataset
        self.unknown_idx = unknown_idx
        self.ref_indices = [i for i in ref_indices if i != unknown_idx]
        self.mode = mode
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.benign_idx = benign_idx
        self.disallow_overlap = disallow_support_query_overlap
        self.disallow_internal_overlap = disallow_internal_overlap
        self.rng = np.random.default_rng(seed)
        self._perm_seed_max = np.iinfo(np.int32).max
        if not disallow_support_query_overlap:
            logger.warning("AdaptationTaskSampler allows support/query raw overlap.")
        self._validate()
        logger.info(
            "AdaptationTaskSampler ref_pool=%s | unknown_idx=%s | window_counts=%s "
            "K=%d Q=%d internal_overlap_disallowed=%s",
            self.ref_indices,
            self.unknown_idx,
            {int(cls): len(dataset.class_to_indices.get(int(cls), []))
             for cls in sorted(dataset.class_to_indices)},
            self.k_shot,
            self.q_query,
            self.disallow_internal_overlap,
        )

    def _validate(self) -> None:
        validate_unknown_window_sufficiency(
            self.dataset,
            self.unknown_idx,
            self.k_shot,
            self.q_query,
            context="AdaptationTaskSampler",
        )
        if self.mode == "binary" and self.benign_idx is None:
            raise ValueError("binary mode requires benign_idx")

    def _classes_for_task(self) -> List[int]:
        if self.mode == "binary":
            return [self.benign_idx, self.unknown_idx]  # type: ignore[list-item]
        n_other = max(1, self.n_way - 1)
        pool = [
            i for i in self.ref_indices
            if len(self.dataset.class_to_indices.get(i, [])) >= self.k_shot + self.q_query
        ]
        others = list(self.rng.choice(pool, size=min(n_other, len(pool)), replace=False))
        return others + [self.unknown_idx]

    def sample_task(self) -> MetaTask:
        from .leakage import check_support_query_overlap
        from .task_sampler import _pick_non_overlapping_windows

        classes = self._classes_for_task()
        support_wids: List[int] = []
        query_wids: List[int] = []
        sx, sy, qx, qy = [], [], [], []
        forbidden: set = set()

        for local, cls in enumerate(classes):
            pool = list(self.dataset.class_to_indices[cls])
            ps = _pick_non_overlapping_windows(
                self.dataset, pool, self.k_shot, forbidden, self.rng,
                self.disallow_overlap,
                disallow_internal_overlap=self.disallow_internal_overlap)
            pq = _pick_non_overlapping_windows(
                self.dataset, pool, self.q_query, forbidden | set(ps), self.rng,
                self.disallow_overlap,
                disallow_internal_overlap=self.disallow_internal_overlap)
            if len(ps) < self.k_shot or len(pq) < self.q_query:
                logger.error(
                    "Adaptation sampling failed for class=%s: support=%d/%d "
                    "query=%d/%d pool=%d internal_overlap_disallowed=%s",
                    cls, len(ps), self.k_shot, len(pq), self.q_query, len(pool),
                    self.disallow_internal_overlap,
                )
                raise ValueError(f"class {cls} has too few non-overlapping windows")
            forbidden.update(ps)
            forbidden.update(pq)
            support_wids.extend(ps)
            query_wids.extend(pq)
            sx.append(self.dataset.features[ps])
            qx.append(self.dataset.features[pq])
            sy.append(torch.full((len(ps),), local, dtype=torch.long))
            qy.append(torch.full((len(pq),), local, dtype=torch.long))

        if self.disallow_overlap:
            ov = check_support_query_overlap(support_wids, query_wids, self.dataset)
            if ov:
                raise RuntimeError(f"adaptation task raw overlap: {ov[:3]}")

        support_x = torch.cat(sx, dim=0)
        query_x = torch.cat(qx, dim=0)
        support_y = torch.cat(sy, dim=0)
        query_y = torch.cat(qy, dim=0)
        s_gen = torch.Generator().manual_seed(int(self.rng.integers(0, self._perm_seed_max)))
        q_gen = torch.Generator().manual_seed(int(self.rng.integers(0, self._perm_seed_max)))
        sp = torch.randperm(support_x.shape[0], generator=s_gen)
        qp = torch.randperm(query_x.shape[0], generator=q_gen)
        return MetaTask(
            support_x=support_x[sp],
            support_y=support_y[sp],
            query_x=query_x[qp],
            query_y=query_y[qp],
            global_classes=classes,
            support_window_ids=[support_wids[i] for i in sp.tolist()],
            query_window_ids=[query_wids[i] for i in qp.tolist()],
            shot=self.k_shot,
        )

    @property
    def task_n_way(self) -> int:
        return 2 if self.mode == "binary" else self.n_way
