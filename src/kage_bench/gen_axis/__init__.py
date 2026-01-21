"""Background generation, image filters, and visual effects module for platformer."""
from .image_bg import (
    BackgroundConfig,
    load_background_image,
    apply_background,
    generate_noise_background,
    generate_color_background,
    select_color_by_index,
    COLOR_PALETTE,
)
from .filters import (
    FilterConfig,
    apply_filters,
    select_filter_by_index,
)
from ..systems.generation.effects import (
    EffectConfig,
    apply_effects,
    generate_light_properties,
    LIGHT_COLORS,
)

__all__ = [
    "BackgroundConfig",
    "load_background_image",
    "apply_background",
    "generate_noise_background",
    "generate_color_background",
    "select_color_by_index",
    "COLOR_PALETTE",
    "FilterConfig",
    "apply_filters",
    "select_filter_by_index",
    "EffectConfig",
    "apply_effects",
    "LIGHT_COLORS",
]

