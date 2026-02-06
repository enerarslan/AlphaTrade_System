"""Reproducibility utilities for institutional research workflows."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(
    seed: int | None,
    deterministic_torch: bool = True,
) -> int | None:
    """Set process-wide random seeds across common ML stacks.

    Args:
        seed: Seed value. If ``None``, no changes are applied.
        deterministic_torch: When True, configures deterministic Torch behavior
            when Torch is available.

    Returns:
        The seed that was applied (or ``None``).
    """
    if seed is None:
        return None

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Some operations/backends do not support strict determinism.
                pass
    except Exception:
        # Torch is optional; keep utility lightweight and fail-safe.
        pass

    return seed


def child_seed(parent_seed: int | None, offset: int) -> int | None:
    """Derive a deterministic child seed from a parent seed."""
    if parent_seed is None:
        return None
    return int(parent_seed) + int(offset)

