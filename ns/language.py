"""规则语言：候选规则模板枚举（含递归规则）。作者：晨星"""

from __future__ import annotations

import itertools
from typing import Iterable

from core.types import Atom, Rule, Dataset

VAR_NAMES = ("?X", "?Y", "?Z")


def background_predicates(data: Dataset) -> list[str]:
    """从事实中提取背景谓词名（排除目标谓词）。"""
    preds: set[str] = set()
    for f in data.facts:
        if f.pred != data.target_pred:
            preds.add(f.pred)
    return sorted(preds)


def enumerate_rules(data: Dataset, body_len: int = 2, max_rules: int = 400) -> list[Rule]:
    """枚举候选规则（单子句，体 1..body_len 个字面量，变量 ∈ {?X,?Y,?Z}）。

    头部固定 target_pred(?X,?Y)；体字面量谓词取自 {背景谓词, 目标谓词}
    （允许目标谓词出现在体 -> 递归规则，表达任意深度关系）。
    """
    bg = background_predicates(data)
    preds = bg + [data.target_pred]  # 递归可用
    # 体字面量形状：每个位置从 VAR_NAMES 取变量（3^2=9 种）
    pos_choices = [(a, b) for a in VAR_NAMES for b in VAR_NAMES]
    literal_shapes = [(p, pa) for p in preds for pa in pos_choices]

    seen: set[str] = set()
    rules: list[Rule] = []
    head = Atom(data.target_pred, ("?X", "?Y"))

    # 优先级：体越短越靠前（更简洁可解释）
    for L in range(1, body_len + 1):
        for combo in itertools.product(literal_shapes, repeat=L):
            body = tuple(Atom(p, pa) for (p, pa) in combo)
            r = Rule(head, body)
            key = str(r)
            if key in seen:
                continue
            if not _sound_recursive(r, data.target_pred, bg):
                continue
            seen.add(key)
            rules.append(r)
            if len(rules) >= max_rules:
                return rules
    return rules


def _sound_recursive(r: Rule, target_pred: str, bg_preds: list[str]) -> bool:
    """递归规则生产性方向约束：保证 head(X,Y) 通过单向边链式展开，剔除反向/非生产性递归。

    允许的生产性模板：
      target(X,Y) :- edge(X,Z), target(Z,Y)   # X 经边到 Z，再递归到 Y
      target(X,Y) :- target(X,Z), edge(Z,Y)   # X 递归到 Z，再经边到 Y
    """
    tlits = [lit for lit in r.body if lit.pred == target_pred]
    if not tlits:
        return True  # 非递归规则，全部保留
    if len(tlits) > 1:
        return False  # 至多一个递归字面量
    X, Y = r.head.args
    A, B = tlits[0].args
    edges = [lit for lit in r.body if lit.pred in bg_preds]
    if len(edges) != 1:
        return False
    Eu, Ev = edges[0].args
    if A == X and B not in (X, Y):
        # target(X, Z) 形：需 edge(Z, Y)（无向图允许任意方向）
        return (Eu == B and Ev == Y) or (Eu == Y and Ev == B)
    if B == Y and A not in (X, Y):
        # target(Z, Y) 形：需 edge(X, Z)
        return (Eu == X and Ev == A) or (Eu == A and Ev == X)
    return False


def rule_to_vars(r: Rule) -> tuple[str, ...]:
    return r.variables
