"""错误码 E100~E500。作者：晨星"""

from __future__ import annotations


class NeuroSymError(Exception):
    """基类。"""


class E100ConfigError(NeuroSymError):
    """配置错误。"""


class E200DataError(NeuroSymError):
    """数据/DGP 错误（如标签非 0/1）。"""


class E300LanguageError(NeuroSymError):
    """规则语言错误（如变量未约束 / 类型不匹配）。"""


class E400LogicError(NeuroSymError):
    """逻辑引擎错误（如固定点不收敛 / 真值越界）。"""


class E500PipelineError(NeuroSymError):
    """pipeline 编排错误（如缺失模块 / 降级失败）。"""
