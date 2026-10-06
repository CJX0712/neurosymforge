"""配置：ENV_XXX_* 覆盖 + schema 校验。作者：晨星"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import E100ConfigError


@dataclass
class Config:
    """系统配置，支持环境变量覆盖与 schema 校验。"""

    seed: int = 42
    max_rule_depth: int = 6          # 递归规则最大展开深度（固定点迭代轮数）
    body_len: int = 2                # 规则体最大字面量数
    max_rules: int = 400             # 候选规则上限（模板枚举后截断）
    beam_width: int = 12             # 波束剪枝保留规则数
    candidate_top: int = 60          # 贪心选择候选上限（按 |w| 排序截断）
    n_epochs: int = 30               # 权重学习轮数
    lr: float = 0.3                  # 学习率
    l2: float = 1e-3                 # 权重 L2 正则
    decision_threshold: float = 0.5  # 预测阈值
    tnorm: str = "product"           # 软逻辑 t-范数：product / lukasiewicz
    agg: str = "noisy_or"            # 多规则聚合：noisy_or / max

    def validate(self) -> "Config":
        if self.max_rule_depth < 1:
            raise E100ConfigError("max_rule_depth 必须 >= 1")
        if self.body_len < 1 or self.body_len > 3:
            raise E100ConfigError("body_len 必须 ∈ [1,3]")
        if self.beam_width < 1:
            raise E100ConfigError("beam_width 必须 >= 1")
        if self.lr <= 0:
            raise E100ConfigError("lr 必须 > 0")
        if self.tnorm not in ("product", "lukasiewicz"):
            raise E100ConfigError("tnorm 必须是 product / lukasiewicz")
        if self.agg not in ("noisy_or", "max"):
            raise E100ConfigError("agg 必须是 noisy_or / max")
        return self

    @classmethod
    def from_env(cls, prefix: str = "NEUROSYM_") -> "Config":
        """环境变量覆盖，如 NEUROSYM_SEED=7。"""
        kv: dict[str, str] = {}
        for k, v in os.environ.items():
            if k.startswith(prefix):
                kv[k[len(prefix):].lower()] = v
        ints = {"seed", "max_rule_depth", "body_len", "max_rules", "beam_width", "candidate_top", "n_epochs"}
        floats = {"lr", "l2", "decision_threshold"}
        out = cls()
        for key, val in kv.items():
            if key in ints:
                setattr(out, key, int(val))
            elif key in floats:
                setattr(out, key, float(val))
            elif hasattr(out, key):
                setattr(out, key, val)
        return out.validate()


# 默认全局配置（不可变引用）
DEFAULT_CONFIG = Config()
