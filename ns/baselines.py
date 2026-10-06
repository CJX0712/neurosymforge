"""基线模型。作者：晨星

- MLPMlpBaseline：纯连接主义，每对查询用 <=2 跳局部特征，无关系结构 -> 深度泛化失败。
- BruteILPBaseline：经典非递归 ILP（精确单子句，体不含目标谓词）-> 无法表达传递闭包。
- MajorityBaseline：多数类。
全部纯 numpy/sklearn，零下载，离线可跑。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from core.config import DEFAULT_CONFIG
from core.types import Atom, Dataset, Example
from core.seed import get_seed
from .language import enumerate_rules
from .logic import LogicEngine, forward_chain, _world_key


def _world_adj(data, target_pred, directed):
    """按世界构建邻接矩阵；索引同时纳入查询常量（孤立节点被查询时仍可被特征化）。"""
    facts = data.facts
    groups: dict[str, list[tuple[str, str]]] = {}
    for f in facts:
        if f.pred == target_pred:
            continue
        k = _world_key(f.args[0])
        groups.setdefault(k, []).append((f.args[0], f.args[1]))
    # 查询常量按世界收集（训练+测试）
    qconsts: dict[str, set] = {}
    for e in list(data.train) + list(data.test):
        qk = _world_key(e.query.args[0])
        qconsts.setdefault(qk, set()).update(e.query.args)
    out = {}
    for k, edges in groups.items():
        consts = sorted({c for e in edges for c in e} | qconsts.get(k, set()))
        idx = {c: i for i, c in enumerate(consts)}
        N = len(consts)
        row = [idx[u] for u, _ in edges]
        col = [idx[v] for _, v in edges]
        arr = np.zeros((N, N))
        for u, v in edges:
            arr[idx[u], idx[v]] = 1.0
        if not directed:
            arr = np.maximum(arr, arr.T)
        out[k] = (consts, idx, arr)
    return out


class MLPMlpBaseline:
    def __init__(self):
        self.model = None
        self.directed = True
        self._adj = None
        self._target = ""

    def available(self) -> bool:
        try:
            from sklearn.neural_network import MLPClassifier  # noqa: F401
            return True
        except Exception:
            return False

    def _feat(self, u, v, consts, idx, adj):
        i, j = idx[u], idx[v]
        N = len(consts)
        direct = float(adj[i, j] > 0)
        if not self.directed:
            direct = max(direct, float(adj[j, i] > 0))
        outdeg = float(adj[i].sum())
        indeg = float(adj[:, j].sum())
        twohop = 0.0
        # exists w: adj[i,w] & adj[w,j]
        prod = adj[i] * adj[:, j]
        twohop = float(np.any(prod > 0))
        if not self.directed:
            prod2 = adj[j] * adj[:, i]
            twohop = max(twohop, float(np.any(prod2 > 0)))
        return np.array([
            direct,
            outdeg / N, indeg / N,
            float(outdeg > 0), float(indeg > 0),
            twohop,
        ])

    def fit(self, data: Dataset) -> "MLPMlpBaseline":
        self._target = data.target_pred
        self.directed = data.target_pred != "connected"
        self._adj = _world_adj(data, data.target_pred, self.directed)
        X, y = [], []
        for e in data.train:
            wk = _world_key(e.query.args[0])
            consts, idx, adj = self._adj[wk]
            X.append(self._feat(e.query.args[0], e.query.args[1], consts, idx, adj))
            y.append(e.label)
        X = np.array(X)
        y = np.array(y)
        from sklearn.neural_network import MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=600,
                                   random_state=get_seed() or 42, early_stopping=False)
        self.model.fit(X, y)
        return self

    def predict(self, queries: list[Atom]) -> list[float]:
        X = []
        for q in queries:
            wk = _world_key(q.args[0])
            consts, idx, adj = self._adj[wk]
            X.append(self._feat(q.args[0], q.args[1], consts, idx, adj))
        proba = self.model.predict_proba(np.array(X))
        pos = list(self.model.classes_).index(1)
        return [float(p[pos]) for p in proba]

    def predict_examples(self, examples: list[Example]) -> list[float]:
        return self.predict([e.query for e in examples])


class BruteILPBaseline:
    """非递归精确 ILP：枚举体不含目标谓词的规则，按训练准确率选前 K 条 OR 组合。"""

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.rules = []
        self.engine: LogicEngine | None = None
        self._target = ""

    def available(self) -> bool:
        return True

    def fit(self, data: Dataset) -> "BruteILPBaseline":
        self._target = data.target_pred
        self.engine = LogicEngine(data, DEFAULT_CONFIG)
        # 仅非递归规则
        cands = [r for r in enumerate_rules(data, body_len=DEFAULT_CONFIG.body_len,
                                            max_rules=DEFAULT_CONFIG.max_rules)
                 if not any(lit.pred == data.target_pred for lit in r.body)]
        scored = []
        y = np.array([e.label for e in data.train])
        for r in cands:
            self.engine.compute([r], [1.0], exact=True)
            p = np.array(self.engine.predict_examples(data.train)) > 0.5
            acc = float((p == y).mean())
            scored.append((acc, r))
        scored.sort(key=lambda x: -x[0])
        self.rules = [r for _, r in scored[: self.top_k]]
        self.engine.compute(self.rules, [1.0] * len(self.rules), exact=True)
        return self

    def predict(self, queries: list[Atom]) -> list[float]:
        p = np.array(self.engine.predict(queries))
        return [float(v) for v in p]

    def predict_examples(self, examples: list[Example]) -> list[float]:
        return self.predict([e.query for e in examples])


class MajorityBaseline:
    def __init__(self):
        self.frac = 0.5

    def available(self) -> bool:
        return True

    def fit(self, data: Dataset) -> "MajorityBaseline":
        self.frac = float(np.mean([e.label for e in data.train])) if data.train else 0.5
        return self

    def predict(self, queries: list[Atom]) -> list[float]:
        pred = 1.0 if self.frac >= 0.5 else 0.0
        return [pred] * len(queries)

    def predict_examples(self, examples: list[Example]) -> list[float]:
        return self.predict([e.query for e in examples])
