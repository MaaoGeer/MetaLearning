"""标签规范化工具 (跨数据集共享)。"""

from __future__ import annotations

import re
from typing import Dict, Optional


def norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_cic_family(raw: str) -> Optional[str]:
    """CICIDS2017/2018 系列通用映射。"""
    n = norm_label(raw)
    if n in {"benign", "normal"}:
        return "benign"
    if "webattack" in n or "xss" in n or "sqlinjection" in n:
        return "webattack"
    if n == "ddos" or "ddos" in n:
        return "ddos"
    if n.startswith("dos"):
        return "dos"
    if "portscan" in n:
        return "portscan"
    if n == "bot" or "botnet" in n:
        return "botnet"
    if "ftppatator" in n or "sshpatator" in n or "patator" in n:
        return "bruteforce"
    if "infiltration" in n or "infilteration" in n:
        return "infiltration"
    if "heartbleed" in n:
        return "heartbleed"
    return None


def map_unsw_attack_cat(raw: str) -> Optional[str]:
    n = norm_label(raw)
    mapping = {
        "normal": "benign",
        "dos": "dos",
        "ddos": "ddos",
        "fuzzers": "fuzzers",
        "analysis": "analysis",
        "backdoors": "backdoor",
        "backdoor": "backdoor",
        "exploits": "exploits",
        "generic": "generic",
        "reconnaissance": "reconnaissance",
        "shellcode": "shellcode",
        "worms": "worms",
    }
    return mapping.get(n)


def apply_custom_mapping(raw: str, custom: Dict[str, str]) -> Optional[str]:
    if raw in custom:
        return custom[raw]
    n = norm_label(raw)
    for k, v in custom.items():
        if norm_label(k) == n:
            return v
    return None
