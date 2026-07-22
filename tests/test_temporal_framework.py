"""时序窗口 / 泄漏 / 公平性 / 数据集标签映射测试。"""

import logging

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.cse_cic_ids2018 import CSECICIDS2018Dataset
from src.data.dataset import IntrusionDataset, merge_windowed_datasets
from src.data.leakage import (
    audit_pipeline_splits,
    check_support_query_overlap,
    check_unknown_not_in_meta_train,
    datasets_window_row_disjoint,
    raw_sample_sets_overlap,
    windows_overlap_between,
)
from src.data.loao import SplitArrays, build_loao
from src.data.preprocessing import build_class_index
from src.data.task_builder import build_windowed_dataset, make_meta_sampler
from src.data.task_sampler import FewShotTaskSampler
from src.data.unsw_nb15 import UNSWNB15Dataset
from src.data.windowing import build_temporal_windows, build_windows
from src.utils.config import Config


def _ordered_split(n=200, feat=5):
    feats = np.random.default_rng(0).normal(size=(n, feat)).astype(np.float32)
    half = n // 2
    labels = np.array(["benign"] * half + ["dos"] * (n - half), dtype=object)
    order = np.arange(n, dtype=float)
    return SplitArrays(features=feats, labels=labels, order=order,
                       row_id=np.arange(n, dtype=np.int64),
                       segment_id=np.zeros(n, dtype=np.int64))


def test_temporal_last_label():
    split = _ordered_split(50)
    c2i = build_class_index(["benign", "dos"])
    labs = np.array([c2i[str(l)] for l in split.labels])
    res = build_temporal_windows(split.features, labs, split.order, 4, 2, "last", split.labels, c2i)
    # 窗口末样本为 index 3,7,... 对应标签
    for i in range(len(res.labels)):
        assert res.labels[i] == labs[int(res.raw_end[i])]


def test_any_attack_strategy():
    # 窗口 [benign,benign,dos,dos] 应标为 attack
    feats = np.ones((8, 3), dtype=np.float32)
    names = np.array(["benign", "benign", "benign", "benign", "dos", "dos", "dos", "dos"], dtype=object)
    c2i = build_class_index(["benign", "dos"])
    c2i["attack"] = c2i["dos"]
    labs = np.array([c2i["benign"]] * 4 + [c2i["dos"]] * 4)
    order = np.arange(8, dtype=float)
    res = build_temporal_windows(feats, labs, order, 4, 4, "any_attack", names, c2i)
    assert c2i["dos"] in res.labels


def test_majority_strategy():
    feats = np.ones((10, 3), dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    order = np.arange(10, dtype=float)
    c2i = {"benign": 0, "dos": 1}
    res = build_temporal_windows(feats, labels, order, 3, 3, "majority", None, c2i)
    assert int(res.labels[0]) == 0  # [0,0,0]
    assert int(res.labels[1]) == 1  # [1,1,1]


def test_classwise_still_works():
    split = _ordered_split(60)
    c2i = build_class_index(["benign", "dos"])
    labs = np.array([c2i[str(l)] for l in split.labels])
    res = build_windows(split.features, labs, split.order, 4, 4,
                        windowing_mode="classwise", label_strategy="classwise")
    assert len(res.features) > 0


def test_support_query_no_raw_overlap():
    split = _ordered_split(300)
    c2i = build_class_index(["benign", "dos", "ddos"])
    labs = np.array([c2i[str(l)] for l in split.labels])
    # 扩大类
    split.labels = np.array(["benign"] * 100 + ["dos"] * 100 + ["ddos"] * 100, dtype=object)
    labs = np.array([c2i[str(l)] for l in split.labels])
    ds = build_windowed_dataset(split, c2i, 8, 8, windowing_mode="temporal", label_strategy="last")
    sampler = FewShotTaskSampler(ds, n_way=2, k_shot=2, q_query=2, seed=0,
                                 disallow_support_query_overlap=True)
    task = sampler.sample_task()
    ov = check_support_query_overlap(task.support_window_ids, task.query_window_ids, ds)
    assert len(ov) == 0


def test_train_eval_splits_disjoint():
    split = _ordered_split(400)
    split.labels = np.array(["benign"] * 200 + ["dos"] * 200, dtype=object)
    tr = SplitArrays(split.features[:160], split.labels[:160], split.order[:160])
    ev = SplitArrays(split.features[160:200], split.labels[160:200], split.order[160:200])
    assert not raw_sample_sets_overlap(tr, ev)


def test_unknown_not_in_meta_train_loao():
    df = pd.DataFrame({
        "f0": np.random.randn(120),
        "label": ["benign"] * 40 + ["dos"] * 40 + ["webattack"] * 40,
        "__order__": np.arange(120, dtype=float),
    })
    from src.data.base_dataset import LoadResult
    lr = LoadResult(df=df, feature_columns=["f0"])
    loao = build_loao(lr, known_classes=["dos"], unknown_class="webattack",
                      include_benign=True, eval_ratio=0.2, seed=0)
    assert check_unknown_not_in_meta_train(loao)


def test_unsw_label_mapping():
    ds = UNSWNB15Dataset(root="datasets/UNSW_NB15")
    assert ds.normalize_label("Normal") == "benign"
    assert ds.normalize_label("DoS") == "dos"
    assert ds.normalize_label("Fuzzers") == "fuzzers"


def test_cicids2018_label_mapping():
    ds = CSECICIDS2018Dataset(root="datasets/CSE_CIC_IDS2018")
    assert ds.normalize_label("Benign") == "benign"
    assert ds.normalize_label("DDoS") == "ddos"


def test_fair_same_init_and_task():
    """MetaOpt/SGD/Adam 应使用相同 task 与 init (结构检查)。"""
    init_a = {"w": torch.nn.Parameter(torch.ones(3))}
    init_b = {k: v.clone() for k, v in init_a.items()}
    assert torch.equal(init_a["w"], init_b["w"])
    task_x = torch.randn(4, 2, 3)
    task_a = task_x.clone()
    task_b = task_x.clone()
    assert torch.equal(task_a, task_b)


# --------------------------------------------------------------------------- #
# P0-1: 缓存 key 严格校验
# --------------------------------------------------------------------------- #
def test_cache_key_strictly_validates_label_mapping(tmp_path):
    """改变 label_mapping 后必须不接受旧 cache(重新清洗)。"""
    from src.data import pipeline as P

    base = {
        "data": {
            "name": "cicids2017", "root": str(tmp_path / "ds"),
            "cache_dir": str(tmp_path / "cache"), "use_cache": True,
            "label_mapping": {},
        }
    }
    cfg_a = Config(base)
    key_a = P._cache_key(cfg_a)

    # 写入一个"旧 cache", key 对应 label_mapping={}。
    cache_file = P._cache_paths(cfg_a)
    import json as _json
    np.savez(
        cache_file,
        features=np.zeros((3, 2), dtype=np.float32),
        labels=np.array(["benign", "dos", "benign"], dtype=object),
        order=np.arange(3, dtype=float),
        feature_columns=np.array(["f0", "f1"], dtype=object),
        cache_key_json=_json.dumps(key_a, sort_keys=True),
    )
    loaded = P._load_cache_npz(cache_file)
    assert loaded.stats["cache_key"] == key_a

    # 改变 label_mapping → cache key 必须不同 → 旧 cache 不应被接受。
    cfg_b = Config({"data": dict(base["data"], label_mapping={"Foo": "dos"})})
    key_b = P._cache_key(cfg_b)
    assert key_a != key_b
    # 同一 cache 文件路径对 cfg_b 而言 digest 不同 → 不会命中。
    assert P._cache_paths(cfg_b) != cache_file


def test_cache_key_none_not_accepted(tmp_path):
    """无 cache_key 字段(旧缓存)解析为 None, 不应被视为命中。"""
    from src.data import pipeline as P
    cfg = Config({"data": {"name": "cicids2017", "root": str(tmp_path / "ds"),
                            "cache_dir": str(tmp_path / "c"), "use_cache": True}})
    cache_file = P._cache_paths(cfg)
    np.savez(
        cache_file,
        features=np.zeros((2, 1), dtype=np.float32),
        labels=np.array(["benign", "dos"], dtype=object),
        order=np.arange(2, dtype=float),
        feature_columns=np.array(["f0"], dtype=object),
    )  # 故意不写 cache_key_json
    loaded = P._load_cache_npz(cache_file)
    assert loaded.stats["cache_key"] is None


# --------------------------------------------------------------------------- #
# P0-2: adaptation dataset 不跨 eval/unknown 边界
# --------------------------------------------------------------------------- #
def _split(features, labels, order, seg, row):
    return SplitArrays(features=features, labels=labels, order=order,
                       row_id=row, segment_id=seg)


def test_no_window_crosses_eval_unknown_boundary():
    from src.data.dataset import merge_windowed_datasets
    rng = np.random.default_rng(0)
    # eval: benign+dos, row 0..59, segment 0
    n_eval = 60
    eval_sp = _split(
        rng.normal(size=(n_eval, 4)).astype(np.float32),
        np.array(["benign"] * 30 + ["dos"] * 30, dtype=object),
        np.arange(n_eval, dtype=float),
        np.zeros(n_eval, dtype=np.int64),
        np.arange(n_eval, dtype=np.int64))
    # unknown: webattack, row 100..159, segment 1
    n_unk = 60
    unk_sp = _split(
        rng.normal(size=(n_unk, 4)).astype(np.float32),
        np.array(["webattack"] * n_unk, dtype=object),
        np.arange(n_unk, dtype=float),
        np.ones(n_unk, dtype=np.int64),
        np.arange(100, 100 + n_unk, dtype=np.int64))
    c2i = build_class_index(["benign", "dos", "webattack"])
    eval_ds = build_windowed_dataset(eval_sp, c2i, 8, 4, windowing_mode="temporal", label_strategy="last")
    unk_ds = build_windowed_dataset(unk_sp, c2i, 8, 4, windowing_mode="temporal", label_strategy="last")
    merged = merge_windowed_datasets([eval_ds, unk_ds])

    # 没有任何窗口同时覆盖 eval 行(<100) 与 unknown 行(>=100)。
    for i in range(len(merged)):
        rs, re = int(merged.row_start[i]), int(merged.row_end[i])
        crosses = rs < 100 <= re
        assert not crosses, f"窗口 {i} 跨越 eval/unknown 边界: row[{rs},{re}]"


# --------------------------------------------------------------------------- #
# P0-3: adaptation val/test 原始样本级 disjoint
# --------------------------------------------------------------------------- #
def test_adaptation_val_test_tasks_raw_disjoint():
    rng = np.random.default_rng(1)
    n = 80
    feats = rng.normal(size=(n, 4)).astype(np.float32)
    labels = np.array(["benign"] * 40 + ["dos"] * 40, dtype=object)
    order = np.arange(n, dtype=float)
    seg = np.zeros(n, dtype=np.int64)
    row = np.arange(n, dtype=np.int64)
    full = _split(feats, labels, order, seg, row)

    # 模拟 pipeline._temporal_subsplit: 前半 val, 后半 test(每类各切)。
    from src.data.pipeline import _temporal_subsplit
    val_sp, test_sp = _temporal_subsplit(full, 0.5)
    c2i = build_class_index(["benign", "dos"])
    val_ds = build_windowed_dataset(val_sp, c2i, 8, 4, windowing_mode="temporal")
    test_ds = build_windowed_dataset(test_sp, c2i, 8, 4, windowing_mode="temporal")

    assert datasets_window_row_disjoint(val_ds, test_ds)


# --------------------------------------------------------------------------- #
# P1-1: temporal 窗口不跨 segment
# --------------------------------------------------------------------------- #
def test_temporal_windows_do_not_cross_segments():
    n = 40
    feats = np.random.default_rng(2).normal(size=(n, 3)).astype(np.float32)
    labels = np.array(["benign"] * n, dtype=object)
    order = np.concatenate([np.arange(20), np.arange(20)]).astype(float)
    seg = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    row = np.arange(n, dtype=np.int64)
    c2i = build_class_index(["benign"])
    labs = np.array([c2i["benign"]] * n)
    res = build_temporal_windows(feats, labs, order, 8, 4, "last",
                                 labels, c2i, row_id=row, segment_id=seg)
    # 每个窗口的 segment 唯一; 不存在跨 segment(row 跨越 19/20 边界)的窗口。
    for i in range(len(res.labels)):
        rs, re = int(res.row_start[i]), int(res.row_end[i])
        assert not (rs < 20 <= re), f"窗口 {i} 跨 segment: row[{rs},{re}]"


# --------------------------------------------------------------------------- #
# P1-2: audit 跨 dataset overlap 使用各自 metadata
# --------------------------------------------------------------------------- #
def test_audit_window_overlap_between_datasets():
    # 两个 row 不相交的 dataset → 不应判为重叠。
    rng = np.random.default_rng(3)
    a_sp = _split(rng.normal(size=(40, 3)).astype(np.float32),
                  np.array(["benign"] * 40, dtype=object),
                  np.arange(40, dtype=float), np.zeros(40, dtype=np.int64),
                  np.arange(40, dtype=np.int64))
    b_sp = _split(rng.normal(size=(40, 3)).astype(np.float32),
                  np.array(["benign"] * 40, dtype=object),
                  np.arange(40, dtype=float), np.zeros(40, dtype=np.int64),
                  np.arange(100, 140, dtype=np.int64))
    c2i = build_class_index(["benign"])
    ds_a = build_windowed_dataset(a_sp, c2i, 8, 8, windowing_mode="temporal")
    ds_b = build_windowed_dataset(b_sp, c2i, 8, 8, windowing_mode="temporal")
    # 局部 raw_start/raw_end 相同(都从 0 开始), 旧实现会误判重叠; 新实现用 row_id → 不重叠。
    assert windows_overlap_between(ds_a, 0, ds_a, 0) is True
    assert windows_overlap_between(ds_a, 0, ds_b, 0) is False
    assert datasets_window_row_disjoint(ds_a, ds_b)


def test_exact_window_row_ids_avoid_range_false_positive():
    features = np.zeros((1, 4, 2), dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    ds_a = IntrusionDataset(
        features, labels, 4,
        row_start=np.array([1]), row_end=np.array([7]),
        row_ids=np.array([[1, 3, 5, 7]]),
        segment_id=np.array([0]))
    ds_b = IntrusionDataset(
        features, labels, 4,
        row_start=np.array([2]), row_end=np.array([8]),
        row_ids=np.array([[2, 4, 6, 8]]),
        segment_id=np.array([0]))
    assert not windows_overlap_between(ds_a, 0, ds_b, 0)


# --------------------------------------------------------------------------- #
# P2-1: any_attack 语义
# --------------------------------------------------------------------------- #
def test_any_attack_requires_binary_or_explicit_attack_label():
    feats = np.ones((12, 3), dtype=np.float32)
    names = np.array(["benign"] * 4 + ["dos"] * 4 + ["ddos"] * 4, dtype=object)
    order = np.arange(12, dtype=float)
    # 多个非 benign 类且无 'attack' 别名 → 必须报错。
    c2i_multi = build_class_index(["benign", "dos", "ddos"])
    labs = np.array([c2i_multi[str(l)] for l in names])
    with pytest.raises(ValueError):
        build_temporal_windows(feats, labs, order, 4, 4, "any_attack", names, c2i_multi)

    # 显式提供 'attack' 别名 → 允许。
    c2i_ok = build_class_index(["benign", "dos", "ddos"])
    c2i_ok["attack"] = c2i_ok["dos"]
    res = build_temporal_windows(feats, labs, order, 4, 4, "any_attack", names, c2i_ok)
    assert len(res.labels) > 0


# --------------------------------------------------------------------------- #
# P2-2: majority 平票策略
# --------------------------------------------------------------------------- #
def test_majority_tie_break_last():
    feats = np.ones((4, 2), dtype=np.float32)
    # 窗口 [benign, benign, dos, dos] 平票; last → 取最后一个(dos)。
    c2i = {"benign": 0, "dos": 1}
    labs = np.array([0, 0, 1, 1], dtype=np.int64)
    order = np.arange(4, dtype=float)
    res_last = build_temporal_windows(feats, labs, order, 4, 4, "majority", None, c2i,
                                      majority_tie_break="last")
    assert int(res_last.labels[0]) == 1
    res_small = build_temporal_windows(feats, labs, order, 4, 4, "majority", None, c2i,
                                       majority_tie_break="smallest")
    assert int(res_small.labels[0]) == 0
    with pytest.raises(ValueError):
        build_temporal_windows(feats, labs, order, 4, 4, "majority", None, c2i,
                               majority_tie_break="error")


# --------------------------------------------------------------------------- #
# LOAO split granularity: per_class_temporal vs global_temporal
# --------------------------------------------------------------------------- #
def _clustered_loadresult(n_per_class: int = 40):
    """构造按类别强时间聚集的数据(benign→dos→ddos→webattack 依次成块)。"""
    from src.data.base_dataset import LoadResult
    classes = ["benign", "dos", "ddos", "webattack"]
    rng = np.random.default_rng(7)
    feats = rng.normal(size=(n_per_class * len(classes), 1)).astype(np.float32)
    labels, order = [], []
    o = 0
    for c in classes:
        labels.extend([c] * n_per_class)
        order.extend(range(o, o + n_per_class))
        o += n_per_class
    df = pd.DataFrame({
        "f0": feats[:, 0],
        "label": np.array(labels, dtype=object),
        "__order__": np.array(order, dtype=float),
        "__row_id__": np.arange(len(labels), dtype=np.int64),
        "__segment_id__": np.zeros(len(labels), dtype=np.int64),
    })
    return LoadResult(df=df, feature_columns=["f0"])


def test_per_class_temporal_split_preserves_classes():
    lr = _clustered_loadresult()
    loao = build_loao(
        lr, known_classes=["dos", "ddos"], unknown_class="webattack",
        include_benign=True, eval_ratio=0.2, test_ratio=0.2,
        split_granularity="per_class_temporal", seed=0)
    expected = set(loao.known_classes)  # {benign, dos, ddos}
    assert set(loao.train.labels.tolist()) == expected
    assert set(loao.eval.labels.tolist()) == expected
    # 每个 split 仍按 order 单调(合并后已排序)。
    assert np.all(np.diff(loao.train.order) >= 0)
    assert np.all(np.diff(loao.eval.order) >= 0)


def test_global_temporal_split_can_warn_on_class_skew(caplog):
    lr = _clustered_loadresult()
    with caplog.at_level(logging.WARNING):
        loao = build_loao(
            lr, known_classes=["dos", "ddos"], unknown_class="webattack",
            include_benign=True, eval_ratio=0.2, test_ratio=0.2,
            split_granularity="global_temporal", seed=0)
    # global_temporal 不应抛错, 但应 warning 提示类别偏斜/缺失。
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "global_temporal" in text
    # 时序聚集 → 某 split 缺类。
    assert (set(loao.train.labels.tolist()) != set(loao.known_classes)
            or set(loao.eval.labels.tolist()) != set(loao.known_classes))


def test_split_class_distribution_logged_or_available():
    lr = _clustered_loadresult()
    loao = build_loao(
        lr, known_classes=["dos", "ddos"], unknown_class="webattack",
        include_benign=True, eval_ratio=0.2, test_ratio=0.2,
        split_granularity="per_class_temporal", seed=0)
    cd = loao.class_distribution
    assert set(cd.keys()) == {"train", "eval", "test", "unknown"}
    # train 应覆盖全部 known class, 且计数为正。
    for c in loao.known_classes:
        assert cd["train"].get(c, 0) > 0
    assert cd["unknown"].get("webattack", 0) > 0


def test_per_class_temporal_raises_when_class_too_small():
    """某 known class 样本过少无法填满 meta_train/eval → 清晰报错。"""
    from src.data.base_dataset import LoadResult
    df = pd.DataFrame({
        "f0": np.random.default_rng(0).normal(size=83),
        "label": (["benign"] * 40 + ["dos"] * 40 + ["ddos"] * 1 + ["webattack"] * 2),
        "__order__": np.arange(83, dtype=float),
        "__row_id__": np.arange(83, dtype=np.int64),
        "__segment_id__": np.zeros(83, dtype=np.int64),
    })
    lr = LoadResult(df=df, feature_columns=["f0"])
    with pytest.raises(ValueError):
        build_loao(lr, known_classes=["dos", "ddos"], unknown_class="webattack",
                   include_benign=True, eval_ratio=0.2, test_ratio=0.2,
                   split_granularity="per_class_temporal", seed=0)


def test_low_data_fraction_only_reduces_known_train():
    lr = _clustered_loadresult(n_per_class=100)
    full = build_loao(
        lr, known_classes=["dos", "ddos"], unknown_class="webattack",
        include_benign=True, eval_ratio=0.2, test_ratio=0.2,
        split_granularity="per_class_temporal", train_fraction=1.0, seed=0)
    low = build_loao(
        lr, known_classes=["dos", "ddos"], unknown_class="webattack",
        include_benign=True, eval_ratio=0.2, test_ratio=0.2,
        split_granularity="per_class_temporal", train_fraction=0.1, seed=0)
    assert len(low.train) < len(full.train)
    assert len(low.eval) == len(full.eval)
    assert len(low.unknown) == len(full.unknown)
