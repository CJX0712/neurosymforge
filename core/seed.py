"""全局确定性入口。唯一 seed 来源，numpy/random 一次设齐。作者：晨星"""

from __future__ import annotations

import random as _random
import numpy as np

_CURRENT: int | None = None


def set_all(seed: int) -> None:
    """设置全局所有随机源，保证可复现。"""
    global _CURRENT
    _CURRENT = seed
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_seed() -> int | None:
    return _CURRENT
