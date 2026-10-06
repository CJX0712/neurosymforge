# NeuroSymForge 架构设计

> 神经符号归纳逻辑编程（Neuro-Symbolic ILP）系统 · 旗舰 LogicFuse
> 作者：晨星

## 1. 设计哲学

纯连接主义模型（MLP）只能看到固定跳数的局部特征，无法表达**递归/传递闭包**关系，
因此在「关系型组合泛化」任务（ancestor / reach / connected）上，训练浅、测试深时会系统性崩盘。
NeuroSymForge 用**可微分规则学习 + 递归前向链接**弥合这一鸿沟：规则是符号、可解释；
权重是连续、可学习；推理是递归前向链（对任意深度关系成立）。

## 2. 整体流水线

```
原始事实(facts) + 训练/测试查询
        │
        ▼
   ┌──────────── LogicEngine ────────────┐
   │  按世界(world)拆分常量空间，二值谓词  │
   │  表示为 N×N 邻接矩阵，规则应用 = 矩阵  │
   │  join（einsum / reshape 广播）         │
   └──────────────────────────────────────┘
        │
        ▼
   enumerate_rules  ──►  候选规则模板（含递归，生产性方向约束）
        │
        ▼
   ┌──── LogicFuse.fit ────┐
   │ 阶段1 可微分权重学习：  │  逻辑回归拟合每条规则的 one-shot 体 firing
   │                         │  特征 → 规则排序 (ranking)
   │ 阶段2 贪心前向选择：    │  在「真实递归前向链」目标上，按训练准确率
   │                         │  逐条加入规则，天然跳过噪声/反向规则
   └────────────────────────┘
        │
        ▼
   选中规则集合 → 布尔精确前向链固定点 → 查询真值表 → 预测
```

## 3. 核心模块

| 模块 | 职责 |
|------|------|
| `core/seed.py` | 唯一确定性入口 `set_all(seed)`，同 seed 两次跑逐位一致 |
| `core/config.py` | 配置 schema + 环境变量覆盖（`NEUROSYM_*`）+ 校验 |
| `core/types.py` | `Atom` / `Rule` / `Example` / `Dataset`；`Rule.head_vars_grounded` 句法偏置 |
| `core/errors.py` | 错误码分层（E1xx~E5xx） |
| `data/generators.py` | 合成关系数据集：训练浅（depth≤3）、测试深（depth≤6/7），归纳式泛化 |
| `ns/language.py` | 规则模板枚举 + `_sound_recursive` 生产性递归方向约束 |
| `ns/logic.py` | 矩阵化逻辑引擎：软真值 t-范数、noisy-OR 聚合、递归前向链接固定点 |
| `ns/fuse.py` | **旗舰 LogicFuse**：可微分权重学习 + 贪心规则选择 |
| `ns/baselines.py` | 基线：MLP（纯连接主义）/ 非递归暴力 ILP / 多数类 |
| `pipeline/pipeline.py` | 端到端基准编排：多任务×多种子、mean±std、Δ、Wilcoxon、确定性、消融、失败案例 |

## 4. 关键技术决策

- **矩阵化逻辑（零 torch）**：所有谓词为二元 → N×N 矩阵；规则体合取 = 矩阵乘积（product t-范数），
  存在量词 = 沿引入变量轴取 OR（乘积补）。计算量 O(rounds × rules × N³)，N≤~18，CPU 友好。
- **递归表达**：规则体允许出现目标谓词（如 `reach(X,Y):-edge(X,Z),reach(Z,Y)`），
  前向链接迭代求不动点即传递闭包。
- **生产性方向约束**：递归规则必须满足 `edge(X,Z),target(Z,Y)` 或 `target(X,Z),edge(Z,Y)` 形式，
  剔除 `reach(Y,Z),edge(X,Z)` 等反向/非生产性递归，避免选出逻辑错误规则。
- **两阶段学习**：
  - 阶段1 用逻辑回归在 one-shot 体 firing 特征上学得软权重，仅用于**规则排序**；
  - 阶段2 在**真实前向链**目标上贪心选择规则（按训练准确率），保证最终推理正确且可泛化。
- **最终推理**：选中规则经布尔精确前向链（收敛固定点）求查询真值，对任意深度组合泛化成立。

## 5. 性能门槛（预注册）

- 旗舰 held-out 测试准确率 ≥ 最强基线（MLP）× **+0.10**；
- 3+ 种子 mean±std，单侧 Wilcoxon 显著；
- 同 seed 两次运行逐位一致（确定性）。

## 6. 依赖与运行

纯 `numpy` / `scipy` / `scikit-learn` 栈，无 torch、无编译、无联网下载。
详见 `requirements.txt` 与 `docs/USAGE.md`。
