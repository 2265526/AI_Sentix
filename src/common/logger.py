"""
日志配置

提供统一的应用日志初始化入口：
  - setup_logging()：应用入口（api/main.py）调用一次，全局生效；
  - 各模块只需 `logger = logging.getLogger(__name__)` 使用，不再各自配置。
"""
import logging

_DEFAULT_LEVEL = logging.INFO
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def setup_logging(level: int = _DEFAULT_LEVEL, fmt: str = _DEFAULT_FORMAT) -> None:
    """初始化根日志配置。"""
    logging.basicConfig(
        level=level,
        format=fmt,
    )
