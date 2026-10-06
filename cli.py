"""NeuroSymForge CLI 入口。作者：晨星

用法：
  python cli.py demo               # 跑完整基准 + 确定性 + 消融 + 失败案例，打印表格
  python cli.py bench --seeds 7,13,42
  python cli.py eval --task reach --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys

from core.config import Config
from core.seed import set_all
from pipeline.pipeline import NeuroSymPipeline, TASKS


def _print_table(rep: dict) -> None:
    print("\n=== NeuroSymForge 基准（测试准确率 mean±std，多 seed）===")
    hdr = f"{'task':<10} | {'flagship':>10} | {'mlp':>10} | {'brute':>10} | {'majority':>10} | {'Δ旗舰-最强基线':>16}"
    print(hdr)
    print("-" * len(hdr))
    for t, v in rep["tasks"].items():
        agg = v["agg"]
        def f(m):
            a = agg.get(m)
            return f"{a['mean']:.3f}±{a['std']:.3f}" if a else "  skipped"
        d = v["flagship_minus_strongest"]
        ds = f"{d:+.3f}" if d is not None else "  n/a"
        print(f"{t:<10} | {f('flagship'):>10} | {f('mlp'):>10} | {f('brute'):>10} | {f('majority'):>10} | {ds:>16}")
        print(f"           最强基线 = {v['strongest_baseline']}")


def cmd_demo(args) -> int:
    cfg = Config()
    pipe = NeuroSymPipeline(cfg)
    rep = pipe.run(seeds=(7, 13, 42, 99, 123, 2024, 777))
    det = pipe.determinism_check()
    abl = pipe.ablation()
    fail = pipe.failure_cases()
    _print_table(rep)
    print("\n=== 确定性校验 ===")
    print(f"同 seed 两次运行 max|Δ| = {det['max_abs_diff']:.2e}  逐位一致 = {det['bit_identical']}")
    print("\n=== 消融：波束剪枝 ===")
    print(f"剪枝({abl['pruned_rules']}条) test={abl['pruned_test']:.3f}  | 不剪枝({abl['noprune_rules']}条) test={abl['noprune_test']:.3f}")
    print("\n=== 失败案例（典型误例）===")
    for c in fail:
        print(f"  {c['query']}  真={c['true']} 预测={c['pred']} 分={c['score']:.3f}")
    out = {"benchmark": rep, "determinism": det, "ablation": abl, "failure_cases": fail}
    with open("benchmark.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n已落盘 benchmark.json")
    return 0


def cmd_bench(args) -> int:
    seeds = tuple(int(s) for s in args.seeds.split(","))
    pipe = NeuroSymPipeline(Config())
    rep = pipe.run(seeds=seeds)
    _print_table(rep)
    return 0


def cmd_eval(args) -> int:
    set_all(args.seed)
    from data.generators import gen_reach, gen_ancestor, gen_connected
    gens = {"reach": gen_reach, "ancestor": gen_ancestor, "connected": gen_connected}
    data = gens[args.task](args.seed)
    pipe = NeuroSymPipeline(Config())
    res = pipe.evaluate(data)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NeuroSymForge · 神经符号归纳逻辑编程系统")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(func=cmd_demo)
    b = sub.add_parser("bench")
    b.add_argument("--seeds", default="7,13,42")
    b.set_defaults(func=cmd_bench)
    e = sub.add_parser("eval")
    e.add_argument("--task", default="reach", choices=list(TASKS.keys()))
    e.add_argument("--seed", type=int, default=7)
    e.set_defaults(func=cmd_eval)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
