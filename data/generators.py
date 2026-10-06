"""合成关系数据集生成器。作者：晨星

设计要点（防数据泄漏 / 泛化甜点）：
- 训练世界：最长路径 <= train_max_depth（浅），测试世界：最长路径可达 test_max_depth（深）。
- 目标谓词为递归关系（reach / ancestor / connected），旗舰可用递归规则表达任意深度；
  MLP 基线只看到 <=2 跳特征，非递归 ILP 基线只能表达固定深度 -> 深度泛化失败。
- seed 固定，可复现；train/test 世界常量空间互不相交（归纳式泛化）。
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from core.types import Atom, Example, Dataset


def _build_world(n_nodes: int, max_depth: int, directed: bool, rng: np.random.Generator, density: float = 0.5):
    """构造一个 DAG / 无向图，返回边列表 [(u,v), ...]。

    density 控制额外随机边数量（相对节点数），越低图越稀、最长路径越深，
    越能暴露纯局部特征基线的深度泛化短板。
    """
    edges: list[tuple[int, int]] = []
    # 主链保证存在深度 max_depth 的路径
    chain_len = min(max_depth + 1, n_nodes)
    order = list(range(n_nodes))
    rng.shuffle(order)
    chain = order[:chain_len]
    for i in range(chain_len - 1):
        a, b = chain[i], chain[i + 1]
        edges.append((a, b) if directed else (min(a, b), max(a, b)))
    # 额外随机边（DAG：只允许小 id -> 大 id 以保持无环，无向则任意）
    extra = max(0, n_nodes - chain_len)
    for _ in range(extra + int(n_nodes * density)):
        u = int(rng.integers(0, n_nodes))
        v = int(rng.integers(0, n_nodes))
        if u == v:
            continue
        if directed:
            if u > v:
                u, v = v, u
            # 避免与主链形成过短短路（仍允许，仅增加正例）
            edges.append((u, v))
        else:
            edges.append((min(u, v), max(u, v)))
    # 去重
    seen = set()
    uniq = []
    for e in edges:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def _shortest_paths(n_nodes: int, edges, directed: bool):
    if n_nodes == 0:
        return np.zeros((0, 0))
    row = np.array([e[0] for e in edges], dtype=int)
    col = np.array([e[1] for e in edges], dtype=int)
    data = np.ones(len(edges), dtype=float)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    dist = shortest_path(adj, directed=directed, unweighted=True)
    return dist


def _make_dataset(name, target_pred, directed, seed,
                  n_train_worlds, n_test_worlds,
                  train_nodes, test_nodes, train_max_depth, test_max_depth,
                  queries_per_world, density: float = 0.5):
    rng = np.random.default_rng(seed)
    facts: list[Atom] = []
    train: list[Example] = []
    test: list[Example] = []
    all_constants: list[str] = []

    def world_const(wid, i):
        return f"w{wid}_{i}"

    # 训练世界（浅）
    for wid in range(n_train_worlds):
        nn = int(rng.integers(train_nodes, train_nodes + 3))
        edges = _build_world(nn, train_max_depth, directed, rng, density)
        for (u, v) in edges:
            cu, cv = world_const(wid, u), world_const(wid, v)
            facts.append(Atom(target_pred if False else ("edge" if target_pred in ("reach", "connected") else "parent"), (cu, cv)))
            if cu not in all_constants:
                all_constants.append(cu)
            if cv not in all_constants:
                all_constants.append(cv)
        dist = _shortest_paths(nn, edges, directed)
        _sample_queries(wid, nn, dist, target_pred, directed, rng,
                        train, lo=1, hi=train_max_depth, n=queries_per_world,
                        world_const=world_const)

    # 测试世界（深）
    for wid in range(n_test_worlds):
        wid2 = wid + 1000
        nn = int(rng.integers(test_nodes, test_nodes + 4))
        edges = _build_world(nn, test_max_depth, directed, rng, density)
        for (u, v) in edges:
            cu, cv = world_const(wid2, u), world_const(wid2, v)
            facts.append(Atom(("edge" if target_pred in ("reach", "connected") else "parent"), (cu, cv)))
            if cu not in all_constants:
                all_constants.append(cu)
            if cv not in all_constants:
                all_constants.append(cv)
        dist = _shortest_paths(nn, edges, directed)
        _sample_queries(wid2, nn, dist, target_pred, directed, rng,
                        test, lo=train_max_depth + 1, hi=test_max_depth, n=queries_per_world,
                        world_const=world_const)

    return Dataset(
        name=name,
        facts=facts,
        train=train,
        test=test,
        target_pred=target_pred,
        target_arity=2,
        constants=all_constants,
    )


def _sample_queries(wid, nn, dist, target_pred, directed, rng, out,
                    lo, hi, n, world_const):
    # 仅保留出现在边中的节点（避免查询引用孤立节点，其在事实索引中不存在）
    present = [i for i in range(nn)
               if np.any(np.isfinite(dist[i, :])) or np.any(np.isfinite(dist[:, i]))]
    present_set = set(present)
    # 收集正例（sp ∈ [lo,hi]）与负例（不可达）
    pos_pairs, neg_pairs = [], []
    for i in present:
        for j in present:
            if i == j:
                continue
            d = dist[i, j]
            if not np.isfinite(d):
                neg_pairs.append((i, j))
            elif lo <= int(d) <= hi:
                pos_pairs.append((i, j))
    if not pos_pairs:
        return
    # 平衡采样
    k = min(n, len(pos_pairs))
    pos = [pos_pairs[int(x)] for x in rng.choice(len(pos_pairs), size=k, replace=False)]
    kn = min(k, len(neg_pairs))
    neg = [neg_pairs[int(x)] for x in rng.choice(len(neg_pairs), size=kn, replace=False)] if neg_pairs else []
    for (i, j) in pos:
        out.append(Example(Atom(target_pred, (world_const(wid, i), world_const(wid, j))), 1))
    for (i, j) in neg:
        out.append(Example(Atom(target_pred, (world_const(wid, i), world_const(wid, j))), 0))


def gen_reach(seed: int = 0, **kw) -> Dataset:
    p = dict(n_train_worlds=6, n_test_worlds=6, train_nodes=5, test_nodes=9,
             train_max_depth=3, test_max_depth=6, queries_per_world=14)
    p.update(kw)
    return _make_dataset("reach", "reach", True, seed, **p)


def gen_ancestor(seed: int = 0, **kw) -> Dataset:
    p = dict(n_train_worlds=6, n_test_worlds=6, train_nodes=5, test_nodes=9,
             train_max_depth=3, test_max_depth=6, queries_per_world=14)
    p.update(kw)
    return _make_dataset("ancestor", "ancestor", True, seed, **p)


def gen_connected(seed: int = 0, **kw) -> Dataset:
    p = dict(n_train_worlds=6, n_test_worlds=6, train_nodes=6, test_nodes=12,
             train_max_depth=3, test_max_depth=7, queries_per_world=16, density=0.10)
    p.update(kw)
    return _make_dataset("connected", "connected", False, seed, **p)


def gen_all(seed: int = 0) -> list[Dataset]:
    """返回全部任务族数据集（每个任务独立 seed 派生）。"""
    return [gen_reach(seed), gen_ancestor(seed + 1), gen_connected(seed + 2)]
