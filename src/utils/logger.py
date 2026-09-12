"""
统一日志系统

使用 loguru 替代标准库 logging，提供简洁的日志输出：
- stderr：彩色格式，INFO 级别，供运行时查看
- 文件：DEBUG 级别，自动轮转（10MB），供事后排查
"""

import sys

from loguru import logger

from src.config import PROJECT_ROOT

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 移除 loguru 默认 handler，重新配置
logger.remove()

# 终端输出：彩色、简洁
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | <cyan>{name}</cyan> - {message}",
)

# 文件输出：完整记录，自动轮转
# enqueue=True：多进程/多 worker（uvicorn --workers N）下写入队列化，避免轮转竞态
logger.add(
    str(LOG_DIR / "ragforge.log"),
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    encoding="utf-8",
    enqueue=True,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {name}:{line} - {message}",
)
