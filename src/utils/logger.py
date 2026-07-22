"""统一日志系统。

设计目的:
    - 同时输出到控制台与文件, 便于复盘训练。
    - 避免重复添加 handler（多次 get_logger 不会重复打印）。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED = False


def setup_logging(log_dir: str = "logs", level: int = logging.INFO,
                  run_name: Optional[str] = None) -> str:
    """初始化根日志配置, 返回日志文件路径。

    Args:
        log_dir: 日志目录。
        level: 日志级别。
        run_name: 运行名, 用于区分日志文件; 为空则用时间戳。

    Returns:
        日志文件的完整路径。
    """
    global _CONFIGURED
    os.makedirs(log_dir, exist_ok=True)
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{run_name}.log")

    root = logging.getLogger()
    root.setLevel(level)

    # 清理旧 handler, 避免重复写入。
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)
    root.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root.addHandler(file_handler)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger; 若尚未配置根日志则用默认配置兜底。"""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
