"""矩阵化逻辑引擎：前向链接固定点（软真值 / 布尔），查询打分。作者：晨星

谓词均为二元 -> 真值用 N×N 矩阵表示，规则应用 = 矩阵 join（einsum 退化为 reshape 广播），
计算量 O(rounds × rules × N^3)，N 为单世界常量数（<=~14），CPU 友好。
"""

from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import Atom, Rule, Dataset, Example


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _world_key(const: str) -> str:
    return const.split("_")[0]


def group_worlds(facts: list[Atom]) -> dict[str, list[Atom]]:
    out: dict[str, list[Atom]] = {}
    for f in facts:
        k = _world_key(f.args[0])
        out.setdefault(k, []).append(f)
    return out


def build_matrix(facts: list[Atom], pred: str, idx: dict) -> np.ndarray:
    N = len(idx)
    M = np.zeros((N, N), dtype=float)
    for f in facts:
        if f.pred == pred:
            M[idx[f.args[0]], idx[f.args[1]]] = 1.0
    return M


def _raw_firing(r: Rule, mats: dict, C: list) -> np.ndarray:
    """规则体软真值矩阵（去权重），体合取=product，引入变量=存在量词(OR)。

    字面量用 einsum 连接（同变量复用=自环对角广播，跨字面共享变量自动收缩）。
    """
    N = len(C)
    vorder = list(r.variables)
    nax = len(vorder)
    letters = [chr(ord("a") + i) for i in range(nax)]
    all_axes = "".join(letters)
    tensor = np.ones((N,) * nax)
    for lit in r.body:
        M = np.asarray(mats[lit.pred], dtype=float)
        p = letters[vorder.index(lit.args[0])]
        q = letters[vorder.index(lit.args[1])]
        tensor = np.einsum(f"{all_axes},{p}{q}->{all_axes}", tensor, M)
    p = 1.0 - tensor
    head_set = set(r.head.args)
    intro_axes = [i for i in range(nax) if vorder[i] not in head_set]
    if intro_axes:
        for ax in intro_axes:
            p = np.prod(p, axis=ax, keepdims=True)
        body_val = np.squeeze(1.0 - p, axis=tuple(intro_axes))
    else:
        body_val = tensor
    target_axes = tuple(vorder.index(a) for a in r.head.args)
    return np.transpose(body_val, target_axes)


def _apply_rule_soft(r: Rule, w: float, mats: dict, C: list, cfg: Config) -> np.ndarray:
    return _sigmoid(w) * _raw_firing(r, mats, C)


def rule_firing(r: Rule, fact_mats: dict, prev_target: np.ndarray,
                target_pred: str, C: list) -> np.ndarray:
    """单条规则原始 firing 矩阵；递归体字面量用 prev_target（上一轮目标真值）。"""
    mats = dict(fact_mats)
    mats[target_pred] = prev_target
    return _raw_firing(r, mats, C)


def _apply_rule_exact(r: Rule, mats: dict, C: list) -> np.ndarray:
    """精确（布尔）规则应用：复用 einsum 体真值，阈值化。"""
    return (_raw_firing(r, mats, C) > 0.5).astype(float)


def forward_chain(rules, weights, fact_mats: dict, target_pred: str,
                  C: list, cfg: Config, exact: bool = False) -> dict:
    N = len(C)
    mats = {p: m.copy() for p, m in fact_mats.items()}
    mats[target_pred] = np.zeros((N, N))
    if exact:
        for _ in range(cfg.max_rule_depth + 2):
            new = mats[target_pred].copy()
            for r in rules:
                new = np.maximum(new, _apply_rule_exact(r, mats, C))
            if np.array_equal(new, mats[target_pred]):
                break
            mats[target_pred] = new
        return mats
    for _ in range(cfg.max_rule_depth + 2):
        prev = mats[target_pred]
        acc = np.zeros((N, N))
        for r, w in zip(rules, weights):
            contrib = _apply_rule_soft(r, float(w), mats, C, cfg)
            acc = 1.0 - (1.0 - acc) * (1.0 - contrib)  # 多规则 noisy-OR 聚合
        if np.allclose(acc, prev, atol=1e-9):
            mats[target_pred] = acc
            break
        mats[target_pred] = acc
    return mats


class LogicEngine:
    """按世界拆分事实，前向链接，按查询取分。索引同时纳入查询常量。"""

    def __init__(self, data: Dataset, cfg: Config):
        self.cfg = cfg
        self.target_pred = data.target_pred
        self.worlds: dict[str, dict] = {}
        groups = group_worlds(data.facts)
        # 查询常量也纳入世界索引（孤立节点被查询时仍须可解析）
        qconsts: dict[str, set] = {}
        for e in list(data.train) + list(data.test):
            qconsts.setdefault(_world_key(e.query.args[0]), set()).update(e.query.args)
        for wk, wfacts in groups.items():
            consts = sorted({c for f in wfacts for c in f.args} | qconsts.get(wk, set()))
            idx = {c: i for i, c in enumerate(consts)}
            bg_preds = {f.pred for f in wfacts if f.pred != data.target_pred}
            fact_mats = {p: build_matrix(wfacts, p, idx) for p in bg_preds}
            self.worlds[wk] = {"C": consts, "idx": idx, "fact_mats": fact_mats}

    def compute(self, rules, weights, exact: bool = False) -> None:
        for wk, w in self.worlds.items():
            mats = forward_chain(rules, weights, w["fact_mats"], self.target_pred,
                                 w["C"], self.cfg, exact=exact)
            w["target"] = mats[self.target_pred]

    def predict(self, queries: list[Atom]) -> list[float]:
        out: list[float] = []
        for q in queries:
            wk = _world_key(q.args[0])
            w = self.worlds[wk]
            idx = w["idx"]
            i = idx.get(q.args[0])
            j = idx.get(q.args[1])
            if i is None or j is None:
                out.append(0.0)
            else:
                out.append(float(w["target"][i, j]))
        return out

    def predict_examples(self, examples: list[Example]) -> list[float]:
        return self.predict([e.query for e in examples])
