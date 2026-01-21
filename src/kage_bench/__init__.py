"""KAGE-Bench: Known-Axis Generalization Evaluation Benchmark.

A pure functional, JIT-compilable 2D platformer environment with:
- Immutable state (Flax struct dataclasses)
- Deterministic rendering
- Procedural level generation
- Visual diversity axes (backgrounds, filters, effects)
- Gymnasium-compatible wrapper

Quickstart
----------
```python
import jax
from kage_bench import KAGE_Env, EnvConfig

# Create environment
config = EnvConfig()
env = KAGE_Env(config)

# Reset and step
key = jax.random.PRNGKey(0)
obs, info = env.reset(key)
state = info["state"]
obs, reward, terminated, truncated, info = env.step(state, action=0)

# JIT compile for performance
reset_jit = jax.jit(env.reset)
step_jit = jax.jit(env.step)
render_jit = jax.jit(env.render)
```

Gymnasium API
-------------
```python
from kage_bench import KAGE_Env_Gymnasium

env = KAGE_Env_Gymnasium(render_mode="rgb_array")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

Modules
-------
core
    Pure functional environment core (KAGE_Env, EnvState, EnvConfig)
systems
    Subsystems (physics, camera, layout, rendering, generation)
entities
    Game entities (character, NPCs, distractors)
utils
    Utilities (shapes, helpers)
wrappers
    API wrappers (Gymnasium)
"""

from __future__ import annotations

# Core API
from .core import EnvConfig, EnvState, KAGE_Env
from .utils.config_loader import load_config_from_yaml

# Gymnasium wrapper
from .wrappers import KAGE_Env_Gymnasium, AutoResetWrapper, LogWrapper


__version__ = "0.2.0"

__all__ = [
    # Core API
    "KAGE_Env",
    "EnvState",

    "EnvConfig",
    "load_config_from_yaml",
    # Wrappers
    "KAGE_Env_Gymnasium",
    "AutoResetWrapper",
    "LogWrapper",
]
