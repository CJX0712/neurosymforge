"""核心类型定义：原子 / 规则 / 数据集。作者：晨星"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# 一个常量是一个字符串 id（节点名），变量以 '?' 前缀表示
Term = str


@dataclass(frozen=True)
class Atom:
    """带元组的谓词原子，例如 parent('a','b')。"""

    pred: str
    args: Tuple[str, ...]

    def __str__(self) -> str:
        inner = ",".join(self.args)
        return f"{self.pred}({inner})"

    def substitute(self, mapping: dict[str, str]) -> "Atom":
        return Atom(self.pred, tuple(mapping.get(a, a) for a in self.args))


@dataclass(frozen=True)
class Rule:
    """单子句规则：head :- body[0], body[1], ...（合取）。"""

    head: Atom
    body: Tuple[Atom, ...] = field(default_factory=tuple)

    @property
    def is_fact(self) -> bool:
        return len(self.body) == 0

    @property
    def head_vars_grounded(self) -> bool:
        """头部每个变量都必须出现在体某字面量中（排除忽略头部变量的'松'规则）。"""
        if not self.body:
            return True
        body_terms = {t for lit in self.body for t in lit.args}
        return all(v in body_terms for v in self.head.args if v.startswith("?"))

    @property
    def variables(self) -> Tuple[str, ...]:
        vs: list[str] = []
        for a in (self.head, *self.body):
            for t in a.args:
                if t.startswith("?") and t not in vs:
                    vs.append(t)
        return tuple(vs)

    def __str__(self) -> str:
        if not self.body:
            return str(self.head)
        body_str = ",".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}"


@dataclass
class Example:
    """带标签的查询原子。label ∈ {0,1}。"""

    query: Atom
    label: int

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("label 必须为 0 或 1")


@dataclass
class Dataset:
    """一个关系世界：背景事实 + 训练/测试查询。"""

    name: str
    facts: list[Atom]  # 已知真事实（背景谓词）
    train: list[Example]
    test: list[Example]
    target_pred: str
    target_arity: int
    constants: list[str] = field(default_factory=list)
