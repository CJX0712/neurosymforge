# NeuroSymForge 使用文档

> 作者：晨星

## 1. 环境准备

```bash
# Python 3.13（已在隔离 venv 验证）
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`numpy==2.5.3` / `scipy==1.18.1` / `scikit-learn==1.9.1`（纯 numpy 栈，无需 torch，无需联网）。

## 2. 快速开始

```python
from core.seed import set_all
from core.config import Config
from data.generators import gen_reach
from ns.fuse import LogicFuse

set_all(7)
data = gen_reach(7)
model = LogicFuse(Config()).fit(data)
scores = model.predict_examples(data.test)   # 软真值 [0,1]
preds  = [1 if s >= 0.5 else 0 for s in scores]
print("learned rules:", [str(r) for r, _ in model.learned_rules()])
```

## 3. 命令行

```bash
python cli.py demo                 # 完整基准 + 确定性 + 消融 + 失败案例，打印表格并落盘 benchmark.json
python cli.py bench --seeds 7,13,42,99,123,2024,777
python cli.py eval --task reach --seed 7
```

或直接：

```bash
python examples/run_demo.py
```

## 4. 自定义数据集

`data/generators.py` 提供三个合成关系任务族：

| 任务 | 关系 | 方向性 | 训练深度 | 测试深度 |
|------|------|--------|----------|----------|
| `reach` | 可达性 | 有向 | ≤3 | ≤6 |
| `ancestor` | 祖先 | 有向 | ≤3 | ≤6 |
| `connected` | 连通性 | 无向 | ≤3 | ≤7 |

也可自行构造 `Dataset` 喂给 `LogicFuse.fit`：
`facts` 为背景谓词原子列表，`train`/`test` 为带标签查询 `Example(query, label)`。

## 5. 配置覆盖

通过环境变量覆盖默认配置（前缀 `NEUROSYM_`）：

```bash
NEUROSYM_SEED=42 NEUROSYM_BEAM_WIDTH=8 python cli.py eval --task reach
```

关键配置项：`seed` / `body_len` / `max_rules` / `beam_width` / `candidate_top` /
`n_epochs` / `lr` / `l2` / `max_rule_depth` / `decision_threshold`。

## 6. 输出物

- `benchmark.json`：每任务每种子的旗舰/基线测试与训练准确率、mean±std、Δ、Wilcoxon p、
  确定性校验、消融对照、典型失败案例。
- `docs/ARCHITECTURE.md`：架构设计。
- `README.md`：总览与结果摘要。
