"""Game entities (character, NPCs, distractors)."""

from __future__ import annotations

from .character import CharacterConfig
from .distractors import DistractorConfig
from .npc import NPCConfig

__all__ = [
    "CharacterConfig",
    "NPCConfig",
    "DistractorConfig",
]

