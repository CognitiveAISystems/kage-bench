"""Environment configuration composition.

EnvConfig composes all subsystem configurations into a single dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..entities.character import CharacterConfig
from ..entities.distractors import DistractorConfig
from ..entities.npc import NPCConfig
from ..systems.camera import CameraConfig
from ..systems.generation.background import BackgroundConfig
from ..systems.generation.effects import EffectConfig
from ..systems.generation.filters import FilterConfig
from ..systems.layout import LayoutConfig
from ..systems.physics import PhysicsConfig


@dataclass
class EnvConfig:
    """Complete environment configuration.
    
    Composes all subsystem configurations for easy parameter passing.
    Immutable after construction.
    
    Attributes
    ----------
    H : int
        Screen height in pixels (default: 128)
    W : int
        Screen width in pixels (default: 128)
    episode_length : int
        Maximum timesteps per episode (default: 500)
    forward_reward_scale : float
        Reward scaling for forward progress (default: 0.1)
    jump_penalty : float
        Penalty for jumping (default: 1.0)
    timestep_penalty : float
        Per-step reward penalty (default: 0.1)
    idle_penalty : float
        Penalty per step when x position does not change (default: 0.0)
    dist_to_success : float
        Passed distance threshold for success flag (default: 0.0)
    layout : LayoutConfig
        Level layout generation config
    camera : CameraConfig
        Camera following config
    physics : PhysicsConfig
        Physics simulation config
    character : CharacterConfig
        Player character config (sprites, shapes, animation)
    background : BackgroundConfig
        Background generation config
    filters : FilterConfig
        Visual filter effects config
    effects : EffectConfig
        Lighting and other effects config
    npc : NPCConfig
        NPC system config (world-fixed and camera-relative)
    distractors : DistractorConfig
        Visual distractor config
    
    Examples
    --------
    Create minimal config:
    
    >>> cfg = EnvConfig()
    
    Create config with custom layout:
    
    >>> from kage_bench.systems.layout import LayoutConfig
    >>> layout_cfg = LayoutConfig(length=256, height_px=128)
    >>> cfg = EnvConfig(layout=layout_cfg)
    
    Create config with sprites:
    
    >>> from kage_bench.entities.character import CharacterConfig
    >>> char_cfg = CharacterConfig(use_sprites=True, sprite_dir="assets/sprites/skelet/")
    >>> cfg = EnvConfig(character=char_cfg)
    """
    
    # Screen dimensions
    H: int = 128
    W: int = 128
    
    # Episode settings
    episode_length: int = 500
    
    # Reward shaping
    forward_reward_scale: float = 0.2
    jump_penalty: float = 10.0
    timestep_penalty: float = 0.1
    idle_penalty: float = 5.0
    dist_to_success: float = 490.0
    
    # Subsystem configs
    layout: LayoutConfig = None
    camera: CameraConfig = None
    physics: PhysicsConfig = None
    character: CharacterConfig = None
    background: BackgroundConfig = None
    filters: FilterConfig = None
    effects: EffectConfig = None
    npc: NPCConfig = None
    distractors: DistractorConfig = None
    
    def __post_init__(self):
        """Initialize default sub-configs if not provided."""
        if self.layout is None:
            self.layout = LayoutConfig(
                length=self.W * 16,  # 16x screen width
                height_px=self.H,
            )
        
        if self.camera is None:
            self.camera = CameraConfig()
        
        if self.physics is None:
            self.physics = PhysicsConfig()
        
        if self.character is None:
            self.character = CharacterConfig()
        
        if self.background is None:
            self.background = BackgroundConfig()
        
        if self.filters is None:
            self.filters = FilterConfig()
        
        if self.effects is None:
            self.effects = EffectConfig()
        
        if self.npc is None:
            self.npc = NPCConfig()
        
        if self.distractors is None:
            self.distractors = DistractorConfig()
