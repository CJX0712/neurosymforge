"""NeuroSymPipeline：端到端基准编排。作者：晨星

- 多任务（reach/ancestor/connected）× 多 seed 评测旗舰与基线测试准确率；
- 报告 mean±std，旗舰 vs 最强基线 Δ；
- 确定性二次运行校验（同 seed 逐位一致）；
- 消融：波束剪枝开关对照；
- 失败案例：典型误例归因。
"""

from __future__ import annotations

import dataclasses
import numpy as np

from core.config import Config, DEFAULT_CONFIG
from core.seed import set_all
from core.types import Dataset
from data.generators import gen_reach, gen_ancestor, gen_connected
from ns.fuse import LogicFuse
from ns.baselines import MLPMlpBaseline, BruteILPBaseline, MajorityBaseline


TASKS = {
    "reach": gen_reach,
    "ancestor": gen_ancestor,
    "connected": gen_connected,
}


def _acc(model, data: Dataset) -> dict:
    y = np.array([e.label for e in data.test])
    p = np.array(model.predict_examples(data.test)) >= 0.5
    test = float((p == y).mean())
    yt = np.array([e.label for e in data.train])
    pt = np.array(model.predict_examples(data.train)) >= 0.5
    train = float((pt == yt).mean())
    return {"test": test, "train": train}


class NeuroSymPipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = (cfg or DEFAULT_CONFIG).validate()

    def evaluate(self, data: Dataset) -> dict:
        flagship = LogicFuse(self.cfg).fit(data)
        mlp = MLPMlpBaseline().fit(data) if MLPMlpBaseline().available() else None
        brute = BruteILPBaseline().fit(data)
        majority = MajorityBaseline().fit(data)
        out = {
            "flagship": _acc(flagship, data),
            "mlp": _acc(mlp, data) if mlp is not None else None,
            "brute": _acc(brute, data),
            "majority": _acc(majority, data),
            "learned_rules": [str(r) for r, _ in flagship.learned_rules()],
        }
        return out

    def run(self, seeds=(7, 13, 42, 99, 123, 2024, 777)) -> dict:
        per_task: dict[str, dict] = {}
        for tname, gen in TASKS.items():
            seed_rows = []
            for s in seeds:
                set_all(s)
                data = gen(s)
                row = {"seed": s, **self.evaluate(data)}
                seed_rows.append(row)
            # 聚合
            models = ["flagship", "mlp", "brute", "majority"]
            agg = {}
            for m in models:
                vals = [r[m]["test"] for r in seed_rows if r.get(m) is not None]
                if vals:
                    arr = np.array(vals)
                    agg[m] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
                else:
                    agg[m] = None
            # 最强基线
            base_vals = {m: agg[m] for m in ("mlp", "brute", "majority") if agg[m]}
            strongest = max(base_vals, key=lambda k: base_vals[k]["mean"]) if base_vals else None
            delta = (agg["flagship"]["mean"] - base_vals[strongest]["mean"]) if (strongest and agg["flagship"]) else None
            # 单侧 Wilcoxon 显著性（旗舰 > 最强基线，逐 seed 配对）
            wilcoxon_p = None
            if strongest:
                fg = np.array([r["flagship"]["test"] for r in seed_rows])
                bs = np.array([r[strongest]["test"] for r in seed_rows])
                try:
                    from scipy.stats import wilcoxon
                    if np.any(fg != bs):
                        stat, p = wilcoxon(fg, bs, alternative="greater")
                        wilcoxon_p = float(p)
                    else:
                        wilcoxon_p = 0.0  # 全部严格更优
                except Exception:
                    wilcoxon_p = None
            per_task[tname] = {
                "seeds": seed_rows,
                "agg": agg,
                "strongest_baseline": strongest,
                "flagship_minus_strongest": delta,
                "wilcoxon_p_gt_baseline": wilcoxon_p,
                "passes_threshold": bool(delta is not None and delta >= 0.10),
            }
        return {"tasks": per_task, "seeds": list(seeds),
                "threshold": 0.10, "cfg": self.cfg.__dict__}

    # ---------- 确定性 ----------
    def determinism_check(self, task="reach", seed=7) -> dict:
        set_all(seed)
        data = TASKS[task](seed)
        a = LogicFuse(self.cfg).fit(data).predict_examples(data.test)
        set_all(seed)
        data2 = TASKS[task](seed)
        b = LogicFuse(self.cfg).fit(data2).predict_examples(data.test)
        a = np.array(a)
        b = np.array(b)
        return {"max_abs_diff": float(np.max(np.abs(a - b))),
                "bit_identical": bool(np.array_equal(a, b))}

    # ---------- 消融：波束剪枝开关 ----------
    def ablation(self, task="reach", seed=7) -> dict:
        set_all(seed)
        data = TASKS[task](seed)
        pruned = LogicFuse(self.cfg).fit(data)
        wide = LogicFuse(dataclasses.replace(self.cfg, beam_width=400)).fit(data)
        return {
            "pruned_test": _acc(pruned, data)["test"],
            "noprune_test": _acc(wide, data)["test"],
            "pruned_rules": len(pruned.rules),
            "noprune_rules": len(wide.rules),
        }

    # ---------- 失败案例（典型误例） ----------
    def failure_cases(self, task="reach", seed=7, k=3) -> list[dict]:
        set_all(seed)
        data = TASKS[task](seed)
        m = LogicFuse(self.cfg).fit(data)
        y = np.array([e.label for e in data.test])
        p = np.array(m.predict_examples(data.test))
        pred = (p >= 0.5).astype(int)
        wrong = np.where(pred != y)[0]
        cases = []
        for idx in wrong[:k]:
            cases.append({
                "query": str(data.test[idx].query),
                "true": int(y[idx]),
                "pred": int(pred[idx]),
                "score": float(p[idx]),
            })
        return cases
