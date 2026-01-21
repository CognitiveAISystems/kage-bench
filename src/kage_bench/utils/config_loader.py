import yaml
from typing import Type, TypeVar, Any, Dict
from dataclasses import is_dataclass, fields
from pathlib import Path
from termcolor import colored, cprint
from ..core.config import EnvConfig
from ..utils.shapes import SHAPE_COLORS, SHAPE_TYPES

T = TypeVar("T")

def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    Values in override take precedence over base.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

from typing import get_type_hints

def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Convert dictionary to dataclass recursively.
    """
    if not is_dataclass(cls):
        return data

    # Normalize legacy physics keys to current PhysicsConfig fields.
    if cls.__name__ == "PhysicsConfig":
        data = dict(data)
        if "move_speed" in data:
            move_speed = data["move_speed"]
            assert isinstance(move_speed, int), (
                f"{colored('PHYSICS CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid move_speed type\n"
                f"{colored('Required:', 'cyan')} move_speed must be int and > 1\n"
                f"{colored('Provided:', 'yellow')} move_speed={move_speed} ({type(move_speed).__name__})"
            )
            assert move_speed >= 1, (
                f"{colored('PHYSICS CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid move_speed value\n"
                f"{colored('Required:', 'cyan')} move_speed must be int and > 1\n"
                f"{colored('Provided:', 'yellow')} move_speed={move_speed}"
            )
        if "move_speed" in data and "max_speed_x" not in data:
            data["max_speed_x"] = data.pop("move_speed")
        if "jump_force" in data and "jump_velocity" not in data:
            data["jump_velocity"] = data.pop("jump_force")
        if "max_fall_speed" in data and "terminal_velocity" not in data:
            data["terminal_velocity"] = data.pop("max_fall_speed")

    # Special validation for BackgroundConfig
    if cls.__name__ == "BackgroundConfig":
        image_options = ['image_path', 'image_dir', 'image_paths']
        provided_options = [key for key in image_options if key in data and data[key] is not None]
        assert len(provided_options) <= 1, (
            f"{colored('BACKGROUND CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
            f"{colored('Problem:', 'red', attrs=['bold'])} Multiple background image source options specified\n"
            f"{colored('Allowed:', 'cyan')} Only one of {image_options}\n"
            f"{colored('Provided:', 'yellow')} {provided_options}"
        )

        # Validate switch_frequency is in [0, 1]
        if 'switch_frequency' in data:
            freq = data['switch_frequency']
            assert 0.0 <= freq <= 1.0, (
                f"{colored('BACKGROUND CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid background switch_frequency value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.0, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {freq}"
            )

    # Special validation for CharacterConfig
    if cls.__name__ == "CharacterConfig":
        sprite_options = ['sprite_dir', 'sprite_paths', 'sprite_path']
        provided_options = [key for key in sprite_options if key in data and data[key] is not None]
        assert len(provided_options) <= 1, (
            f"{colored('CHARACTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
            f"{colored('Problem:', 'red', attrs=['bold'])} Multiple agent sprite source options specified\n"
            f"{colored('Allowed:', 'cyan')} Only one of {sprite_options}\n"
            f"{colored('Provided:', 'yellow')} {provided_options}"
        )

        # Validate animation_fps >= 0
        if 'animation_fps' in data:
            fps = data['animation_fps']
            assert fps >= 0, (
                f"{colored('CHARACTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid agent animation_fps value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {fps}"
            )

        # Validate use_shape and use_sprites are not both True
        if 'use_shape' in data and 'use_sprites' in data:
            use_shape = data['use_shape']
            use_sprites = data['use_sprites']
            assert not ((use_shape and use_sprites) or (not use_shape and not use_sprites)), (
                f"{colored('CHARACTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid agent visualization method combination\n"
                f"{colored('use_shape:', 'cyan')} {use_shape} | {colored('use_sprites:', 'cyan')} {use_sprites}\n"
                f"{colored('Solution:', 'green', attrs=['bold'])} Set exactly ONE of use_shape or use_sprites to True"
            )

        # Validate shape_types contains only valid shape names
        if 'shape_types' in data:
            shape_types = data['shape_types']
            if isinstance(shape_types, str):
                shape_list = [shape_types]
            else:
                shape_list = shape_types
            
            invalid_shapes = [shape for shape in shape_list if shape not in SHAPE_TYPES]
            if invalid_shapes:
                assert False, (
                    f"{colored('CHARACTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid shape names in character shape_types\n"
                    f"{colored('Invalid shapes:', 'yellow')} {invalid_shapes}\n"
                    f"{colored('Valid shapes:', 'cyan')} {sorted(SHAPE_TYPES)}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid shape names from the list above"
                )

        # Validate shape_colors contains only valid color names
        if 'shape_colors' in data:
            shape_colors = data['shape_colors']
            if isinstance(shape_colors, str):
                color_list = [shape_colors]
            else:
                color_list = shape_colors
            
            invalid_colors = [color for color in color_list if color not in SHAPE_COLORS]
            if invalid_colors:
                assert False, (
                    f"{colored('CHARACTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color names in character shape_colors\n"
                    f"{colored('Invalid colors:', 'yellow')} {invalid_colors}\n"
                    f"{colored('Valid colors:', 'cyan')} {list(SHAPE_COLORS.keys())}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid color names from the list above"
                )

    # Special validation for NPCConfig
    if cls.__name__ == "NPCConfig":
        # Validate world-fixed NPC sprite source options (only one should be specified)
        sprite_options = ['sprite_dir', 'sprite_paths', 'sprite_path']
        provided_options = [key for key in sprite_options if key in data and data[key] is not None and (key != 'sprite_paths' or data[key])]
        assert len(provided_options) <= 1, (
            f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
            f"{colored('Problem:', 'red', attrs=['bold'])} Multiple NPC sprite source options specified\n"
            f"{colored('Allowed:', 'cyan')} Only one of {sprite_options}\n"
            f"{colored('Provided:', 'yellow')} {provided_options}"
        )

        # Validate sticky NPC sprite source options (only one should be specified)
        sticky_sprite_options = ['sticky_sprite_dir', 'sticky_sprite_dirs', 'sticky_sprite_path']
        provided_sticky_options = [key for key in sticky_sprite_options if key in data and data[key] is not None and (key != 'sticky_sprite_dirs' or data[key])]
        assert len(provided_sticky_options) <= 1, (
            f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
            f"{colored('Problem:', 'red', attrs=['bold'])} Multiple sticky NPC sprite source options specified\n"
            f"{colored('Allowed:', 'cyan')} Only one of {sticky_sprite_options}\n"
            f"{colored('Provided:', 'yellow')} {provided_sticky_options}"
        )

        # Validate min_npc_count >= 0
        if 'min_npc_count' in data:
            min_count = data['min_npc_count']
            assert min_count >= 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid min_npc_count value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {min_count}"
            )

        # Validate max_npc_count >= 0
        if 'max_npc_count' in data:
            max_count = data['max_npc_count']
            assert max_count >= 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid max_npc_count value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {max_count}"
            )

        # Validate animation_fps >= 0
        if 'animation_fps' in data:
            fps = data['animation_fps']
            assert fps >= 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid animation_fps value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {fps}"
            )

        # Validate min_npc_count <= max_npc_count when min_npc_count > 0
        if 'min_npc_count' in data and 'max_npc_count' in data:
            min_count = data['min_npc_count']
            max_count = data['max_npc_count']
            if min_count > 0:
                assert max_count >= min_count, (
                    f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid NPC count range\n"
                    f"{colored('Required:', 'cyan')} When min_npc_count > 0, max_npc_count must be >= min_npc_count\n"
                    f"{colored('Provided:', 'yellow')} min_npc_count={min_count}, max_npc_count={max_count}"
                )

        # Validate min_sticky_count >= 0
        if 'min_sticky_count' in data:
            min_sticky = data['min_sticky_count']
            assert min_sticky >= 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid min_sticky_count value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {min_sticky}"
            )

        # Validate max_sticky_count >= 0
        if 'max_sticky_count' in data:
            max_sticky = data['max_sticky_count']
            assert max_sticky >= 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid max_sticky_count value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0\n"
                f"{colored('Provided:', 'yellow')} {max_sticky}"
            )

        # Validate min_sticky_count <= max_sticky_count when min_sticky_count > 0
        if 'min_sticky_count' in data and 'max_sticky_count' in data:
            min_sticky = data['min_sticky_count']
            max_sticky = data['max_sticky_count']
            if min_sticky > 0:
                assert max_sticky >= min_sticky, (
                    f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid sticky NPC count range\n"
                    f"{colored('Required:', 'cyan')} When min_sticky_count > 0, max_sticky_count must be >= min_sticky_count\n"
                    f"{colored('Provided:', 'yellow')} min_sticky_count={min_sticky}, max_sticky_count={max_sticky}"
                )

        # Validate sticky_jump_probability is in [0, 1]
        if 'sticky_jump_probability' in data:
            jump_prob = data['sticky_jump_probability']
            assert 0.0 <= jump_prob <= 1.0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid sticky_jump_probability value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.0, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {jump_prob}"
            )

        # If sticky_enabled is True, at least one sticky sprite source must be specified
        if 'sticky_enabled' in data and data['sticky_enabled']:
            sticky_sprite_options = ['sticky_sprite_dir', 'sticky_sprite_dirs', 'sticky_sprite_path']
            provided_sticky_options = [key for key in sticky_sprite_options if key in data and data[key] is not None and (key != 'sticky_sprite_dirs' or data[key])]
            assert len(provided_sticky_options) > 0, (
                f"{colored('NPC CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Sticky NPCs are enabled but no sprite source specified\n"
                f"{colored('Required:', 'cyan')} When sticky_enabled=true, specify at least one of {sticky_sprite_options}\n"
                f"{colored('Solution:', 'green', attrs=['bold'])} Set sticky_enabled=false or provide sprite source"
            )

    # Special validation for DistractorConfig
    if cls.__name__ == "DistractorConfig":
        # Valid shape types (includes all supported names and aliases)
        VALID_SHAPE_TYPES = set(SHAPE_TYPES)

        # Validate shape_types contains only valid shape names
        if 'shape_types' in data:
            shape_types = data['shape_types']
            # Handle both single string and list of strings
            if isinstance(shape_types, str):
                shape_list = [shape_types]
            else:
                shape_list = shape_types

            invalid_shapes = [shape for shape in shape_list if shape not in VALID_SHAPE_TYPES]
            if invalid_shapes:
                assert False, (
                    f"{colored('DISTRACTOR CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid shape names in shape_types\n"
                    f"{colored('Invalid shapes:', 'yellow')} {invalid_shapes}\n"
                    f"{colored('Valid shapes:', 'cyan')} {sorted(VALID_SHAPE_TYPES)}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid shape names from the list above"
                )

        # Validate shape_colors contains only valid color names
        if 'shape_colors' in data:
            shape_colors = data['shape_colors']
            # Handle both single string and list of strings
            if isinstance(shape_colors, str):
                color_list = [shape_colors]
            else:
                color_list = shape_colors

            invalid_colors = [color for color in color_list if color not in SHAPE_COLORS]
            if invalid_colors:
                assert False, (
                    f"{colored('DISTRACTOR CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color names in shape_colors\n"
                    f"{colored('Invalid colors:', 'yellow')} {invalid_colors}\n"
                    f"{colored('Valid colors:', 'cyan')} {list(SHAPE_COLORS.keys())}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid color names from the list above"
                )

    # Special validation for FilterConfig
    if cls.__name__ == "FilterConfig":
        # Photometric adjustments
        if 'brightness' in data:
            brightness = data['brightness']
            assert -1.0 <= brightness <= 1.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid brightness value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [-1.0, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {brightness}"
            )

        if 'contrast' in data:
            contrast = data['contrast']
            assert contrast > 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid contrast value\n"
                f"{colored('Required:', 'cyan')} Value must be > 0.0\n"
                f"{colored('Provided:', 'yellow')} {contrast}"
            )

        if 'gamma' in data:
            gamma = data['gamma']
            assert 0.5 <= gamma <= 2.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid gamma value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.5, 2.0]\n"
                f"{colored('Provided:', 'yellow')} {gamma}"
            )

        if 'saturation' in data:
            saturation = data['saturation']
            assert 0.0 <= saturation <= 2.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid saturation value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.0, 2.0]\n"
                f"{colored('Provided:', 'yellow')} {saturation}"
            )

        if 'hue_shift' in data:
            hue_shift = data['hue_shift']
            assert -180.0 <= hue_shift <= 180.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid hue_shift value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [-180.0, 180.0]\n"
                f"{colored('Provided:', 'yellow')} {hue_shift}"
            )

        if 'color_temp' in data:
            color_temp = data['color_temp']
            assert -1.0 <= color_temp <= 1.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color_temp value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [-1.0, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {color_temp}"
            )

        # Stochastic effects
        if 'color_jitter_std' in data:
            color_jitter_std = data['color_jitter_std']
            assert color_jitter_std >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color_jitter_std value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {color_jitter_std}"
            )

        if 'gaussian_noise_std' in data:
            gaussian_noise_std = data['gaussian_noise_std']
            assert gaussian_noise_std >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid gaussian_noise_std value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {gaussian_noise_std}"
            )

        if 'poisson_noise_scale' in data:
            poisson_noise_scale = data['poisson_noise_scale']
            assert 0.0 <= poisson_noise_scale <= 1.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid poisson_noise_scale value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.0, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {poisson_noise_scale}"
            )

        # Spatial/detail transforms
        if 'blur_sigma' in data:
            blur_sigma = data['blur_sigma']
            assert blur_sigma >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid blur_sigma value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {blur_sigma}"
            )

        if 'sharpen_amount' in data:
            sharpen_amount = data['sharpen_amount']
            assert sharpen_amount >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid sharpen_amount value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {sharpen_amount}"
            )

        if 'pixelate_factor' in data:
            pixelate_factor = data['pixelate_factor']
            assert isinstance(pixelate_factor, int) and pixelate_factor >= 1, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid pixelate_factor value\n"
                f"{colored('Required:', 'cyan')} Value must be integer >= 1\n"
                f"{colored('Provided:', 'yellow')} {pixelate_factor}"
            )

        # Global shading/lighting
        if 'vignette_strength' in data:
            vignette_strength = data['vignette_strength']
            assert vignette_strength >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid vignette_strength value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {vignette_strength}"
            )

        if 'radial_light_strength' in data:
            radial_light_strength = data['radial_light_strength']
            assert radial_light_strength >= 0.0, (
                f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid radial_light_strength value\n"
                f"{colored('Required:', 'cyan')} Value must be >= 0.0\n"
                f"{colored('Provided:', 'yellow')} {radial_light_strength}"
            )

        # JPEG quality
        if 'jpeg_quality' in data:
            jpeg_quality = data['jpeg_quality']
            if jpeg_quality is not None:
                assert 0.0 <= jpeg_quality <= 1.0, (
                    f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid jpeg_quality value\n"
                    f"{colored('Required:', 'cyan')} Value must be None or in range [0.0, 1.0]\n"
                    f"{colored('Provided:', 'yellow')} {jpeg_quality}"
                )

        # Pop filter list validation
        if 'pop_filter_list' in data:
            pop_filter_list = data['pop_filter_list']
            valid_filters = ["vintage", "retro", "cyberpunk", "horror", "noir", "none"]
            if isinstance(pop_filter_list, list):
                invalid_filters = [f for f in pop_filter_list if f not in valid_filters]
                if invalid_filters:
                    assert False, (
                        f"{colored('FILTER CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                        f"{colored('Problem:', 'red', attrs=['bold'])} Invalid pop_filter_list entries\n"
                        f"{colored('Invalid filters:', 'yellow')} {invalid_filters}\n"
                        f"{colored('Valid filters:', 'cyan')} {valid_filters}\n"
                        f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid filter names from the list above"
                    )

    # Special validation for EffectConfig
    if cls.__name__ == "EffectConfig":
        # Point light effect validation
        if 'point_light_enabled' in data:
            point_light_enabled = data['point_light_enabled']
            assert isinstance(point_light_enabled, bool), (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_enabled type\n"
                f"{colored('Required:', 'cyan')} Value must be boolean\n"
                f"{colored('Provided:', 'yellow')} {point_light_enabled} (type: {type(point_light_enabled).__name__})"
            )

        if 'point_light_intensity' in data:
            point_light_intensity = data['point_light_intensity']
            assert isinstance(point_light_intensity, (int, float)) and 0.1 <= point_light_intensity <= 5.0, (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_intensity value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.1, 5.0]\n"
                f"{colored('Provided:', 'yellow')} {point_light_intensity}"
            )

        if 'point_light_radius' in data:
            point_light_radius = data['point_light_radius']
            assert isinstance(point_light_radius, (int, float)) and 0.01 <= point_light_radius <= 1.0, (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_radius value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [0.01, 1.0]\n"
                f"{colored('Provided:', 'yellow')} {point_light_radius}"
            )

        if 'point_light_falloff' in data:
            point_light_falloff = data['point_light_falloff']
            assert isinstance(point_light_falloff, (int, float)) and 1.0 <= point_light_falloff <= 4.0, (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_falloff value\n"
                f"{colored('Required:', 'cyan')} Value must be in range [1.0, 4.0]\n"
                f"{colored('Provided:', 'yellow')} {point_light_falloff}"
            )

        if 'point_light_count' in data:
            point_light_count = data['point_light_count']
            assert isinstance(point_light_count, int) and 1 <= point_light_count <= 5, (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_count value\n"
                f"{colored('Required:', 'cyan')} Value must be integer in range [1, 5]\n"
                f"{colored('Provided:', 'yellow')} {point_light_count}"
            )

        if 'point_light_color_names' in data:
            point_light_color_names = data['point_light_color_names']
            assert isinstance(point_light_color_names, list), (
                f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid point_light_color_names format\n"
                f"{colored('Required:', 'cyan')} Value must be list of strings\n"
                f"{colored('Provided:', 'yellow')} {point_light_color_names}"
            )
            # Import LIGHT_COLORS from effects module for validation
            from ..systems.generation.effects import LIGHT_COLORS
            invalid_colors = [name for name in point_light_color_names if name not in LIGHT_COLORS]
            if invalid_colors:
                assert False, (
                    f"{colored('EFFECT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color names in point_light_color_names\n"
                    f"{colored('Invalid colors:', 'yellow')} {invalid_colors}\n"
                    f"{colored('Valid colors:', 'cyan')} {list(LIGHT_COLORS.keys())}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid color names from the list above"
                )

    # Special validation for LayoutConfig
    if cls.__name__ == "LayoutConfig":
        # Validate step height range
        if 'min_step_height' in data and 'max_step_height' in data:
            min_step_height = data['min_step_height']
            max_step_height = data['max_step_height']
            assert isinstance(min_step_height, int) and isinstance(max_step_height, int), (
                f"{colored('LAYOUT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid step height types\n"
                f"{colored('Required:', 'cyan')} Both min_step_height and max_step_height must be integers\n"
                f"{colored('Provided:', 'yellow')} min_step_height={min_step_height}, max_step_height={max_step_height}"
            )
            assert max_step_height >= min_step_height, (
                f"{colored('LAYOUT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                f"{colored('Problem:', 'red', attrs=['bold'])} Invalid step height range\n"
                f"{colored('Required:', 'cyan')} max_step_height must be >= min_step_height\n"
                f"{colored('Provided:', 'yellow')} min_step_height={min_step_height}, max_step_height={max_step_height}"
            )

        # Validate layout colors
        if 'layout_colors' in data:
            layout_colors = data['layout_colors']
            # Import COLOR_PALETTE for validation
            from ..gen_axis.image_bg import COLOR_PALETTE

            # Handle both single string and list of strings
            if isinstance(layout_colors, str):
                colors_to_validate = [layout_colors]
            elif isinstance(layout_colors, list):
                colors_to_validate = layout_colors
            else:
                assert False, (
                    f"{colored('LAYOUT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid layout_colors format\n"
                    f"{colored('Required:', 'cyan')} Value must be string or list of strings\n"
                    f"{colored('Provided:', 'yellow')} {layout_colors} (type: {type(layout_colors).__name__})"
                )

            invalid_colors = [color for color in colors_to_validate if color not in COLOR_PALETTE]
            if invalid_colors:
                assert False, (
                    f"{colored('LAYOUT CONFIG ERROR', 'white', 'on_red', attrs=['bold'])}\n"
                    f"{colored('Problem:', 'red', attrs=['bold'])} Invalid color names in layout_colors\n"
                    f"{colored('Invalid colors:', 'yellow')} {invalid_colors}\n"
                    f"{colored('Available colors:', 'cyan')} {sorted(COLOR_PALETTE.keys())}\n"
                    f"{colored('Solution:', 'green', attrs=['bold'])} Use only valid color names from the list above"
                )

    # Use get_type_hints to resolve string forward references
    try:
        type_hints = get_type_hints(cls)
    except Exception:
        # Fallback if resolving fails (e.g. strict forward refs not in scope)
        type_hints = {f.name: f.type for f in fields(cls)}

    kwargs = {}

    for key, value in data.items():
        if key in type_hints:
            field_type = type_hints[key]

            # Handle optional types (naive implementation, assumes Union[Type, None])
            if hasattr(field_type, "__origin__"):
                # Unwrap Optional/Union
                args = field_type.__args__
                # Find the non-None type
                real_type = next((a for a in args if a is not type(None)), None)
                if real_type and is_dataclass(real_type) and isinstance(value, dict):
                    kwargs[key] = from_dict(real_type, value)
                    continue

            if is_dataclass(field_type) and isinstance(value, dict):
                kwargs[key] = from_dict(field_type, value)
            else:
                # If value matches field type or if field type is not dataclass
                kwargs[key] = value

    return cls(**kwargs)

def load_config_from_yaml(config_path: str, base_config: EnvConfig = None) -> EnvConfig:
    """
    Load configuration from YAML file, optionally merging with a base config.
    
    Args:
        config_path: Path to YAML config file
        base_config: Optional base EnvConfig to override (if None, uses defaults)
        
    Returns:
        EnvConfig: Loaded configuration
    """
    # Start with default config converted to dict
    if base_config is None:
        base_config = EnvConfig()
    
    # We need a way to convert dataclass -> dict to merge
    # But since EnvConfig uses default values, we can just instantiate a new one with overrides
    # However, for nested structure, merging dicts is easier.
    
    # Simple approach: Load YAML -> Dict. Instantiate EnvConfig with it.
    # But existing EnvConfig has defaults.
    
    path = Path(config_path)
    assert path.exists(), (
        f"{colored('FILE ERROR', 'white', 'on_red', attrs=['bold'])}\n"
        f"{colored('Problem:', 'red', attrs=['bold'])} Configuration file not found\n"
        f"{colored('Path:', 'cyan')} {config_path}\n"
        f"{colored('Solution:', 'green', attrs=['bold'])} Check if the file exists and path is correct"
    )
        
    with open(path, "r") as f:
        yaml_config = yaml.safe_load(f) or {}

    # If we want to support partial updates to a full default config:
    # 1. We rely on dataclass defaults.
    # 2. But for nested dataclasses (like layout, background), if we pass a dict, 
    #    we need to make sure we don't wipe out their internal defaults if we fully replace the object.
    
    # Better approach: 
    # Use the recursive from_dict loader which inspects the class structure.
    # This allows passing a sparse dict and letting the dataclass constructor handle defaults for missing fields.
    
    return from_dict(EnvConfig, yaml_config)
