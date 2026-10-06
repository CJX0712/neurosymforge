"""LogicFuse 旗舰：可微分规则权重学习 + 贪心前向规则选择。作者：晨星

方法（神经符号范式，加权规则学习 / Markov Logic 风格）：
- 阶段1 可微分权重学习：逻辑回归在每条规则的 one-shot 体 firing 特征上学得软权重，
  仅用于「规则排序」；
- 阶段2 贪心前向选择：在真实递归前向链目标上，按训练准确率逐条加入规则，
  天然跳过对负例误触发的噪声规则与反向/非生产性递归规则（生产性方向约束）；
- 最终推理：选中规则经布尔精确前向链（收敛固定点）求查询真值，对任意深度组合泛化成立。
"""

from __future__ import annotations

import numpy as np

from core.config import Config, DEFAULT_CONFIG
from core.types import Atom, Rule, Dataset, Example
from core.seed import get_seed
from .language import enumerate_rules
from .logic import LogicEngine, forward_chain, rule_firing, _sigmoid, _world_key


class LogicFuse:
    def __init__(self, cfg: Config | None = None):
        self.cfg = (cfg or DEFAULT_CONFIG).validate()
        self.rules: list[Rule] = []
        self.weights: np.ndarray = np.array([])
        self.kept: list[int] = []
        self.engine: LogicEngine | None = None
        self._target_pred = ""

    # ---------- 训练 ----------
    def fit(self, data: Dataset) -> "LogicFuse":
        self._target_pred = data.target_pred
        self.engine = LogicEngine(data, self.cfg)
        rules = enumerate_rules(data, body_len=self.cfg.body_len, max_rules=self.cfg.max_rules)
        n = len(rules)
        y = np.array([e.label for e in data.train], dtype=float)
        qs = [e.query for e in data.train]
        m = len(y)

        # 阶段1：可微分规则权重学习（逻辑回归，one-shot 体 firing 特征 -> 规则排序）
        rng = np.random.default_rng(get_seed() or 42)
        w = rng.standard_normal(n) * 0.01
        prev_target = {wk: np.zeros((len(wd["C"]), len(wd["C"])))
                       for wk, wd in self.engine.worlds.items()}
        for _ in range(self.cfg.n_epochs):
            X = self._build_X(rules, prev_target, qs)
            p = _sigmoid(X @ w)
            grad = (X.T @ (p - y)) / max(m, 1) + self.cfg.l2 * w
            w = w - self.cfg.lr * grad
            prev_target = self._fixed_target(rules, w)

        # 阶段2：贪心前向选择（在真实递归前向链目标上，按训练准确率挑规则）。
        # 句法偏置：头部变量须全部被体约束（排除 edge(?Z,?Y) 这类忽略 ?X 的噪声规则）；
        # 候选按逻辑回归权重降序，评估用布尔前向链（递归规则被选中、噪声被拒）。
        candidates = [i for i in range(n)
                      if w[i] > 0.0 and rules[i].head_vars_grounded]
        candidates.sort(key=lambda i: -float(w[i]))
        candidates = candidates[: self.cfg.candidate_top]
        selected: list[int] = []
        best = -1.0

        def fwd_acc(idxs: list[int]) -> float:
            if not idxs:
                return 0.0
            self.engine.compute([rules[i] for i in idxs], [1.0] * len(idxs), exact=True)
            pred = np.array(self.engine.predict_examples(data.train)) >= self.cfg.decision_threshold
            return float((pred == y).mean())

        for i in candidates:
            acc = fwd_acc(selected + [i])
            if acc > best + 1e-9:
                selected.append(i)
                best = acc
                if len(selected) >= self.cfg.beam_width:
                    break
        if not selected:
            selected = [candidates[0]] if candidates else [0]

        self.kept = selected
        self.rules = [rules[i] for i in selected]
        self.weights = w[selected]            # 可微分学得的软权重（用于排序/可解释）
        # 最终推理：选中规则经递归前向链接（布尔精确固定点）求查询真值表，
        # 保证任意深度组合泛化（软链未饱和时改用布尔收敛点）。
        self.engine.compute(self.rules, [1.0] * len(self.rules), exact=True)
        return self

    def _fixed_target(self, rules, w) -> dict:
        """阶段1 用：以当前权重跑软前向链，得到每世界目标谓词的软真值固定点。"""
        out = {}
        for wk, wd in self.engine.worlds.items():
            mats = forward_chain(rules, w, wd["fact_mats"], self._target_pred,
                                 wd["C"], self.cfg, exact=False)
            out[wk] = mats[self._target_pred]
        return out

    def _build_X(self, rules, prev_target, qs) -> np.ndarray:
        m = len(qs)
        n = len(rules)
        X = np.zeros((m, n))
        for i, r in enumerate(rules):
            firing_by_world: dict[str, np.ndarray] = {}
            for wk, wd in self.engine.worlds.items():
                firing_by_world[wk] = rule_firing(
                    r, wd["fact_mats"], prev_target[wk], self._target_pred, wd["C"])
            for ei, q in enumerate(qs):
                wk = _world_key(q.args[0])
                wd = self.engine.worlds[wk]
                X[ei, i] = firing_by_world[wk][wd["idx"][q.args[0]], wd["idx"][q.args[1]]]
        return X

    # ---------- 预测 ----------
    def predict(self, queries: list[Atom]) -> list[float]:
        out: list[float] = []
        for q in queries:
            wk = _world_key(q.args[0])
            wd = self.engine.worlds[wk]
            i = wd["idx"].get(q.args[0])
            j = wd["idx"].get(q.args[1])
            if i is None or j is None:
                out.append(0.0)
                continue
            out.append(float(wd["target"][i, j]))
        return out

    def predict_examples(self, examples: list[Example]) -> list[float]:
        return self.predict([e.query for e in examples])

    def available(self) -> bool:
        return True

    def learned_rules(self) -> list[tuple[Rule, float]]:
        return [(r, float(w)) for r, w in zip(self.rules, self.weights)]
