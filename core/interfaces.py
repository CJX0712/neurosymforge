"""接口契约（Protocol）。统一语义：score/truth 越大越可能为正类。作者：晨星"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Dataset


@runtime_checkable
class Model(Protocol):
    """任何可学习模型的统一接口。"""

    def fit(self, data: Dataset) -> "Model":
        """在 data.train 上学习。"""
        ...

    def predict(self, queries: list) -> list[float]:
        """返回每个查询的正类软分数 ∈ [0,1]。"""
        ...

    def available(self) -> bool:
        """依赖是否可用；不可用时 pipeline 降级跳过。"""
        ...


@runtime_checkable
class Benchmark(Protocol):
    """基准评测接口。"""

    def run(self, data: Dataset) -> dict:
        ...
