"""CICIDS2017 数据集实现。

特点:
    - 原始为按天分割的多 CSV(MachineLearningCVE / TrafficLabelling 两种变体均兼容)。
    - 递归发现 datasets/CICIDS2017/**/*.csv, 不假设已合并。
    - 标签列为 'Label'; 部分变体含 'Timestamp'(用于时序窗排序), 无则用行序。
    - 攻击命名异构(含特殊字符/破折号), 用规范化 + 子串规则映射到统一 taxonomy。
"""

from __future__ import annotations

import re
from typing import List, Optional

from .base_dataset import BaseDataset


def _norm(s: str) -> str:
    """标签规范化: 转小写, 去非字母数字。"""
    return re.sub(r"[^a-z0-9]", "", s.lower())


class CICIDS2017Dataset(BaseDataset):
    """CICIDS2017。"""

    def file_glob(self) -> str:
        return "**/*.csv"

    def label_column_candidates(self) -> List[str]:
        return ["Label"]

    def timestamp_column_candidates(self) -> List[str]:
        return ["Timestamp"]

    def normalize_label(self, raw: str) -> Optional[str]:
        """原始 CICIDS2017 标签 → 统一 taxonomy（按优先级判定, 顺序重要）。"""
        n = _norm(raw)
        if n in {"benign", "normal"}:
            return "benign"
        # Web Attack 系列必须在 DoS/bruteforce 之前判定
        # (原始名 'Web Attack – Brute Force' 含 'bruteforce' 字样, 应归 webattack)。
        if "webattack" in n or "xss" in n or "sqlinjection" in n:
            return "webattack"
        if n == "ddos":
            return "ddos"
        if n == "heartbleed":
            return "heartbleed"
        # DoS 系列: DoS Hulk / GoldenEye / slowloris / Slowhttptest
        if n.startswith("dos"):
            return "dos"
        if "portscan" in n:
            return "portscan"
        if n == "bot" or "botnet" in n:
            return "botnet"
        # 独立暴力破解(非 Web): FTP-Patator / SSH-Patator
        if "ftppatator" in n or "sshpatator" in n or "patator" in n:
            return "bruteforce"
        if "infiltration" in n or "infilteration" in n:
            return "infiltration"
        return None  # 其它/未知标签丢弃
