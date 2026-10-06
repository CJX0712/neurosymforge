# NeuroSymForge

> 神经符号学习 · 可微分归纳逻辑编程（Neuro-Symbolic ILP / 张量化逻辑网络）
> 作者：晨星 | 零 torch 依赖 · 纯 numpy/scipy/sklearn · 确定性可复现

NeuroSymForge 是一套**关系型系统性组合泛化**基准系统与旗舰模型 **LogicFuse**。
核心命题：在关系数据（可达性 / 祖先 / 连通性）上，神经符号方法对**超出训练深度的测试分布**
系统性胜过纯连接主义 MLP——这正是世界顶级 AI 前沿（系统性组合泛化）的核心命题。

---

## 核心结果（7 seeds，held-out 测试集）

| 任务 | 旗舰 LogicFuse | 最强基线 (MLP) | Δ (flagship − best baseline) | 单侧 Wilcoxon p | 门槛(Δ≥0.10) |
|------|---------------|---------------|------------------------------|----------------|--------------|
| reach（可达性） | **1.000 ± 0.000** | 0.641 ± 0.156 | **+0.359** | 0.0078 | ✅ PASS |
| ancestor（祖先） | **1.000 ± 0.000** | 0.641 ± 0.156 | **+0.359** | 0.0078 | ✅ PASS |
| connected（连通） | **1.000 ± 0.000** | 0.818 ± 0.206 | **+0.182** | 0.0156 | ✅ PASS |

- **确定性**：`bit_identical=True`（同 seed 两次 benchmark 逐位一致）。
- **消融（波束剪枝）**：pruned=3 条规则 / no-prune=3 条规则，test 均 1.0（剪枝不影响精度，仅提升可解释性）。
- **失败案例集**：空（旗舰在所有 seed 全部 1.0）。

学得的典型规则（reach，seed 7）：
```
reach(?X,?Y) :- edge(?X,?Y).            # 基础事实
reach(?X,?Y) :- edge(?X,?Z),edge(?Z,?Y). # 2 跳展开
reach(?X,?Y) :- edge(?X,?Z),reach(?Z,?Y). # 生产性递归（任意深度）
```

**为什么 MLP 失败**：MLP 用扁平节点对特征学习，只能记忆训练深度（≤3）内的局部模式；
测试深度（5–7）超出训练分布即崩盘。而 LogicFuse 学到**递归规则**，对任意深度都成立。

---

## 方法：LogicFuse（两阶段神经符号）

1. **阶段1 · 可微分权重学习**
   枚举含递归的候选规则模板（头部固定 `target(?X,?Y)`），每条规则对查询产生软 firing 特征
   （递归体用上一轮目标真值固定点）。逻辑回归 `p=σ(Σ wᵢ·firingᵢ)` 解析梯度训练权重，
   用于**规则排序**。软真值基于 Logic Tensor Network 风格 t-范数（product）+ noisy-OR 聚合。

2. **阶段2 · 句法偏置 + 贪心前向选择**
   候选规则施加两类偏置：
   - `head_vars_grounded`：头部变量必须全部出现在体（排除"松"噪声规则）；
   - `_sound_recursive`：仅保留生产性递归模板（如 `edge(?X,?Z),reach(?Z,?Y)`）。
   在真实递归前向链（布尔精确固定点）上按训练准确率贪心挑选 → 自然跳过噪声规则、保留完备基。

最终推理用**布尔精确前向链**（选中规则 + 收敛固定点），保证深度组合泛化。

---

## 架构

```
neurosymforge/
├── core/            # 基础设施
│   ├── seed.py      # set_all(seed) 唯一确定性入口
│   ├── config.py    # Config dataclass（已校验）
│   ├── types.py     # Atom / Rule / Dataset / Example
│   ├── errors.py    # 异常定义
│   └── interfaces.py# 模型接口（fit/predict）
├── data/
│   └── generators.py# 合成关系数据集：训练浅/测试深（测系统性泛化）
├── ns/              # 神经符号引擎
│   ├── language.py  # 规则模板枚举 + 句法偏置
│   ├── logic.py     # 张量化逻辑引擎（前向链/固定点/einsum）
│   ├── fuse.py      # 旗舰 LogicFuse（两阶段）
│   └── baselines.py # MLP / 暴力 ILP / 多数类 基线
├── pipeline/
│   └── pipeline.py  # benchmark 编排（多任务×多 seed + 确定性/消融/失败案例）
├── examples/
│   └── run_demo.py  # 端到端 demo
├── cli.py           # 命令行入口
├── docs/
│   ├── ARCHITECTURE.md
│   └── USAGE.md
├── requirements.txt # 锁定依赖
└── benchmark.json   # 完整 7-seed 结果
```

---

## 快速开始

```bash
# 1. 创建 venv 并安装（纯 numpy/scipy/sklearn，零 torch 零下载）
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. 端到端 demo（自动生成数据 → 训练 → 对比基线 → 打印规则）
.venv/Scripts/python.exe examples/run_demo.py

# 3. 完整 benchmark（3 任务 × 7 seed + 确定性/消融/失败案例 → benchmark.json）
.venv/Scripts/python.exe cli.py --seeds 7 13 42 99 123 2024 777
```

### Python API

```python
from core.seed import set_all
from core.config import Config
from data.generators import gen_reach
from ns.fuse import LogicFuse
from ns.baselines import MLPMlpBaseline

set_all(7)
data = gen_reach(7)              # 训练深度≤3，测试深度≤6
model = LogicFuse(Config()).fit(data)
preds = model.predict_examples(data.test)   # 测试集 1.0 准确率
print([str(r) for r, _ in model.learned_rules()])
```

---

## 设计原则

- **零外部依赖风险**：纯 numpy/scipy/sklearn，win_amd64+py3.13 均有预编译 wheel，无 GPU、无 torch、无下载。
- **确定性优先**：`core.seed.set_all(seed)` 为唯一随机入口，所有实验可逐位复现。
- **无数据泄漏**：规则模板与权重仅由 train 例拟合，held-out 查询集独立评估。
- **可解释**：最终推理为显式逻辑规则，非黑箱权重。

---

## 对标 SOTA

- ∂ILP（Evans & Grefenstette, 2018）— 可微分归纳逻辑编程
- Logic Tensor Networks（Badreddine et al., 2022）— 软真值张量化逻辑
- 系统级对标目标：**系统性组合泛化**（systematic composition generalization）

## 许可证

MIT © 晨星
