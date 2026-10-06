"""端到端演示：跑基准 + 确定性 + 消融 + 失败案例，落盘 benchmark.json。作者：晨星

运行：python examples/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import NeuroSymPipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    t0 = time.time()
    cfg = Config()
    pipe = NeuroSymPipeline(cfg)
    rep = pipe.run(seeds=(7, 13, 42, 99, 123, 2024, 777))
    det = pipe.determinism_check()
    abl = pipe.ablation()
    fail = pipe.failure_cases()

    # 二次运行确定性校验（再跑一遍）
    det2 = pipe.determinism_check()

    out = {
        "system": "NeuroSymForge",
        "domain": "Neuro-Symbolic Inductive Logic Programming",
        "benchmark": rep,
        "determinism": det,
        "determinism_repeat": det2,
        "ablation": abl,
        "failure_cases": fail,
        "config": cfg.__dict__,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    path = os.path.join(REPO, "benchmark.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    # 控制台表格
    print("NeuroSymForge · demo 完成")
    print(f"耗时 {out['elapsed_sec']}s | 确定性逐位一致 = {det['bit_identical']}")
    print("task        flagship      mlp          brute        majority      Δ(max baseline)")
    for t, v in rep["tasks"].items():
        agg = v["agg"]
        def s(m):
            a = agg.get(m)
            return f"{a['mean']:.3f}±{a['std']:.3f}" if a else "skipped"
        d = v["flagship_minus_strongest"]
        print(f"{t:<11} {s('flagship'):<13} {s('mlp'):<13} {s('brute'):<13} {s('majority'):<13} {d:+.3f}" if d is not None else f"{t:<11} {s('flagship'):<13} {s('mlp'):<13} {s('brute'):<13} {s('majority'):<13} n/a")
    print(f"benchmark.json 已写入: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
