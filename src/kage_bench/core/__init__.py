"""Core environment components for JAX platformer.

This package contains the pure functional core of the environment:
- State representations (EnvState, StepOutput)
- Configuration dataclasses (EnvConfig)
- Main environment logic
"""

from __future__ import annotations

from .config import EnvConfig
from .environment import KAGE_Env
from .state import EnvState

__all__ = [
    "EnvState",

    "EnvConfig",
    "KAGE_Env",
]

