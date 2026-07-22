"""YAML 配置系统。

设计目的:
    - 单一可序列化的配置对象, 贯穿数据 / 模型 / 元优化器 / 训练 / 实验全流程。
    - 支持点号访问 (cfg.meta.inner_steps) 与字典访问 (cfg["meta"]["inner_steps"])。
    - 支持命令行 override（key=value, 支持嵌套 a.b.c=1）, 便于做对比实验。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

import yaml


class Config(Mapping):
    """递归字典封装, 支持属性访问与字典访问。

    所有嵌套 dict 会被递归封装为 Config, 这样既能 cfg.a.b 也能 cfg["a"]["b"]。
    Config 实现了 Mapping 协议, 因此可以直接 **cfg 解包或 dict(cfg)。
    """

    def __init__(self, data: Dict[str, Any] | None = None) -> None:
        data = data or {}
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            self._data[key] = self._wrap(value)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @classmethod
    def _wrap(cls, value: Any) -> Any:
        """递归地把 dict 封装为 Config, list 内的 dict 同样封装。"""
        if isinstance(value, Config):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    # ------------------------------------------------------------------ #
    # Mapping 协议
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------ #
    # 属性访问
    # ------------------------------------------------------------------ #
    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(f"配置中不存在键: {key}") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = self._wrap(value)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ------------------------------------------------------------------ #
    # 序列化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """递归转换回纯 dict, 便于保存或打印。"""
        out: Dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [v.to_dict() if isinstance(v, Config) else v for v in value]
            else:
                out[key] = value
        return out

    def merge(self, other: Mapping) -> "Config":
        """深度合并另一个 dict / Config, 返回新的 Config（不修改自身）。"""
        merged = copy.deepcopy(self.to_dict())
        _deep_update(merged, dict(other))
        return Config(merged)

    def apply_overrides(self, overrides: List[str]) -> "Config":
        """应用形如 ["meta.inner_steps=5", "train.lr=0.001"] 的命令行覆盖。"""
        merged = copy.deepcopy(self.to_dict())
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"非法 override (需要 key=value): {item}")
            key, raw_value = item.split("=", 1)
            value = _parse_scalar(raw_value)
            _set_nested(merged, key.split("."), value)
        return Config(merged)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()!r})"


def _deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> None:
    """就地深度合并 new 到 base。"""
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _set_nested(base: Dict[str, Any], keys: List[str], value: Any) -> None:
    """按 key 路径写入嵌套字典。"""
    cur = base
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def _parse_scalar(raw: str) -> Any:
    """把命令行字符串解析为合适的标量类型。"""
    if raw.lower() == "none":
        return None
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return parsed


def load_config(path: str, overrides: List[str] | None = None) -> Config:
    """从 YAML 文件加载配置, 并可选地应用命令行覆盖。

    Args:
        path: YAML 文件路径。
        overrides: 形如 ["a.b=1"] 的覆盖列表。

    Returns:
        Config 对象。
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    cfg = Config(raw)
    if overrides:
        cfg = cfg.apply_overrides(overrides)
    return cfg
