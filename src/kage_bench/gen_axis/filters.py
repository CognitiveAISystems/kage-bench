"""
Image filters and effects for JAX platformer observations.

All filters are JIT-compatible and GPU-friendly.
Filters can be applied individually or chained together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import jax.scipy as jsp


# ============================================================================
# Filter Configuration
# ============================================================================

@dataclass
class FilterConfig:
    """Configuration for image filters and effects."""
    
    # Photometric filters
    brightness: float = 0.0  # [-1, 1], 0 = no change
    contrast: float = 1.0  # [0, 2], 1 = no change
    gamma: float = 1.0  # [0.5, 2.0], 1 = no change
    saturation: float = 1.0  # [0, 2], 1 = no change
    hue_shift: float = 0.0  # [-180, 180] degrees
    
    # Color temperature (kelvin-style)
    color_temp: float = 0.0  # [-1, 1], 0 = neutral, <0 = cooler, >0 = warmer
    
    # Color jitter (random color mixing)
    color_jitter_std: float = 0.0  # [0, 0.3] for subtle, 0 = disabled
    
    # Noise
    gaussian_noise_std: float = 0.0  # [0, 50], 0 = no noise
    poisson_noise_scale: float = 0.0  # [0, 1], 0 = no noise
    
    # Blur and sharpening
    blur_sigma: float = 0.0  # [0, 5], 0 = no blur
    sharpen_amount: float = 0.0  # [0, 2], 0 = no sharpen
    
    # Pixelation
    pixelate_factor: int = 1  # [1, 8], 1 = no pixelation
    
    # JPEG-like compression artifacts
    jpeg_quality: Optional[float] = None  # [0, 1], None = no compression, 1 = max quality
    
    # Vignette / radial light
    vignette_strength: float = 0.0  # [0, 1], 0 = no vignette
    radial_light_strength: float = 0.0  # [0, 1], 0 = no light
    radial_light_color: tuple[int, int, int] = (255, 255, 200)  # Light color
    
    # Pop filters (preset effects)
    # Can be single string or list of strings for random selection
    pop_filter: Optional[str] = None  # "vintage", "horror", "retro", "cyberpunk", "noir"
    pop_filter_list: list[str] = field(default_factory=list)  # List for random selection
    
    # Random seed for stochastic effects (noise, jitter)
    seed: Optional[int] = None


# ============================================================================
# Core Filter Functions (all JIT-compatible)
# ============================================================================

def apply_brightness(img: jnp.ndarray, brightness: float) -> jnp.ndarray:
    """
    Adjust brightness: shift all pixel values.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        brightness: [-1, 1], negative = darker, positive = brighter
    
    Returns:
        img: (H, W, 3) float32 clipped to [0, 255]
    """
    shift = brightness * 100.0
    return jnp.clip(img + shift, 0.0, 255.0)


def apply_contrast(img: jnp.ndarray, contrast: float) -> jnp.ndarray:
    """
    Adjust contrast: scale around midpoint (128).
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        contrast: [0, 2], 1 = no change
    
    Returns:
        img: (H, W, 3) float32 clipped to [0, 255]
    """
    midpoint = 128.0
    return jnp.clip((img - midpoint) * contrast + midpoint, 0.0, 255.0)


def apply_gamma(img: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Apply gamma correction (power-law transform).
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        gamma: [0.5, 2.0], <1 = brighter, >1 = darker
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    normalized = img / 255.0
    corrected = jnp.power(normalized, gamma)
    return corrected * 255.0


def rgb_to_hsv(rgb: jnp.ndarray) -> jnp.ndarray:
    """
    Convert RGB to HSV.
    
    Args:
        rgb: (H, W, 3) float32 in [0, 1]
    
    Returns:
        hsv: (H, W, 3) float32, H in [0, 360], S/V in [0, 1]
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = jnp.maximum(jnp.maximum(r, g), b)
    min_c = jnp.minimum(jnp.minimum(r, g), b)
    delta = max_c - min_c
    
    # Hue
    h = jnp.zeros_like(max_c)
    mask_r = (max_c == r) & (delta > 0)
    mask_g = (max_c == g) & (delta > 0)
    mask_b = (max_c == b) & (delta > 0)
    
    h = jnp.where(mask_r, 60.0 * (((g - b) / delta) % 6.0), h)
    h = jnp.where(mask_g, 60.0 * (((b - r) / delta) + 2.0), h)
    h = jnp.where(mask_b, 60.0 * (((r - g) / delta) + 4.0), h)
    
    # Saturation
    s = jnp.where(max_c > 0, delta / max_c, 0.0)
    
    # Value
    v = max_c
    
    return jnp.stack([h, s, v], axis=-1)


def hsv_to_rgb(hsv: jnp.ndarray) -> jnp.ndarray:
    """
    Convert HSV to RGB.
    
    Args:
        hsv: (H, W, 3) float32, H in [0, 360], S/V in [0, 1]
    
    Returns:
        rgb: (H, W, 3) float32 in [0, 1]
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h = h / 60.0
    c = v * s
    x = c * (1.0 - jnp.abs((h % 2.0) - 1.0))
    m = v - c
    
    i = jnp.floor(h).astype(jnp.int32) % 6
    
    # Create RGB components for each of 6 hue sectors
    zeros = jnp.zeros_like(c)
    
    r = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                    [c, x, zeros, zeros, x, c], default=zeros)
    g = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                    [x, c, c, x, zeros, zeros], default=zeros)
    b = jnp.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                    [zeros, zeros, x, c, c, x], default=zeros)
    
    return jnp.stack([r + m, g + m, b + m], axis=-1)


def apply_saturation(img: jnp.ndarray, saturation: float) -> jnp.ndarray:
    """
    Adjust color saturation via HSV.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        saturation: [0, 2], 0 = grayscale, 1 = no change, 2 = hyper-saturated
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    rgb_norm = img / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    hsv = hsv.at[..., 1].multiply(saturation)
    hsv = hsv.at[..., 1].set(jnp.clip(hsv[..., 1], 0.0, 1.0))
    rgb_norm = hsv_to_rgb(hsv)
    return jnp.clip(rgb_norm * 255.0, 0.0, 255.0)


def apply_hue_shift(img: jnp.ndarray, hue_shift: float) -> jnp.ndarray:
    """
    Shift hue in HSV space.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        hue_shift: [-180, 180] degrees
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    rgb_norm = img / 255.0
    hsv = rgb_to_hsv(rgb_norm)
    hsv = hsv.at[..., 0].add(hue_shift)
    hsv = hsv.at[..., 0].set(hsv[..., 0] % 360.0)
    rgb_norm = hsv_to_rgb(hsv)
    return jnp.clip(rgb_norm * 255.0, 0.0, 255.0)


def apply_color_temperature(img: jnp.ndarray, temp: float) -> jnp.ndarray:
    """
    Apply color temperature adjustment (warm/cool).
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        temp: [-1, 1], negative = cooler (blue), positive = warmer (orange)
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    # Simple implementation: boost red/orange for warm, blue for cool
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    
    # Warm: boost R, reduce B
    warm_scale = jnp.maximum(0.0, temp)
    r = r + warm_scale * 30.0
    b = b - warm_scale * 15.0
    
    # Cool: boost B, reduce R
    cool_scale = jnp.maximum(0.0, -temp)
    b = b + cool_scale * 30.0
    r = r - cool_scale * 15.0
    
    return jnp.stack([
        jnp.clip(r, 0.0, 255.0),
        jnp.clip(g, 0.0, 255.0),
        jnp.clip(b, 0.0, 255.0)
    ], axis=-1)


def apply_color_jitter(img: jnp.ndarray, key: jax.Array, std: float) -> jnp.ndarray:
    """
    Apply random color channel mixing via 3x3 matrix.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        key: JAX PRNG key
        std: [0, 0.3], standard deviation for jitter
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    # Identity + small random perturbation
    identity = jnp.eye(3, dtype=jnp.float32)
    noise = jax.random.normal(key, shape=(3, 3)) * std
    color_matrix = identity + noise
    
    # Normalize to preserve brightness approximately
    color_matrix = color_matrix / jnp.sum(jnp.abs(color_matrix), axis=1, keepdims=True)
    
    # Apply: (H, W, 3) @ (3, 3).T -> (H, W, 3)
    img_transformed = jnp.dot(img, color_matrix.T)
    return jnp.clip(img_transformed, 0.0, 255.0)


def apply_gaussian_noise(img: jnp.ndarray, key: jax.Array, std: float) -> jnp.ndarray:
    """
    Add Gaussian noise.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        key: JAX PRNG key
        std: [0, 50], noise standard deviation
    
    Returns:
        img: (H, W, 3) float32 clipped to [0, 255]
    """
    noise = jax.random.normal(key, shape=img.shape) * std
    return jnp.clip(img + noise, 0.0, 255.0)


def apply_poisson_noise(img: jnp.ndarray, key: jax.Array, scale: float) -> jnp.ndarray:
    """
    Add Poisson (shot) noise.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        key: JAX PRNG key
        scale: [0, 1], noise scale
    
    Returns:
        img: (H, W, 3) float32 clipped to [0, 255]
    """
    if scale == 0.0:
        return img
    
    # Scale image to Poisson-appropriate range
    lam = img * scale
    noisy = jax.random.poisson(key, lam=lam, shape=img.shape).astype(jnp.float32)
    noisy = noisy / scale
    return jnp.clip(noisy, 0.0, 255.0)


def gaussian_kernel(size: int, sigma: float) -> jnp.ndarray:
    """Create 2D Gaussian kernel."""
    x = jnp.arange(size) - (size - 1) / 2.0
    gauss_1d = jnp.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / jnp.sum(gauss_1d)
    kernel = gauss_1d[:, None] * gauss_1d[None, :]
    return kernel


def apply_blur(img: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Apply fast box blur approximation for Gaussian blur.
    
    Uses jax.lax.reduce_window (pure XLA, no cuDNN) for maximum efficiency
    with large batched workloads. 2 passes of 5x5 box blur approximate Gaussian.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        sigma: [0, 5], blur strength
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if sigma == 0.0:
        return img
    
    # Use fixed 3x3 box blur (1 pass for speed, approximates light blur)
    # Reduced from 5x5 with 2 passes for better performance with large batches
    window_size = 3
    radius = window_size // 2
    
    # Box blur using reduce_window (pure XLA, no cuDNN)
    def box_blur_channel(channel):
        # Add batch dim for reduce_window: (1, H, W, 1)
        channel_4d = channel[None, :, :, None]
        
        # Pad to maintain output size
        padded = jnp.pad(
            channel_4d,
            ((0, 0), (radius, radius), (radius, radius), (0, 0)),
            mode='edge'
        )
        
        # Apply reduce_window for box blur (mean pooling)
        blurred_4d = jax.lax.reduce_window(
            padded,
            init_value=0.0,
            computation=jax.lax.add,
            window_dimensions=(1, window_size, window_size, 1),
            window_strides=(1, 1, 1, 1),
            padding='VALID'
        )
        blurred_4d = blurred_4d / (window_size * window_size)
        
        return blurred_4d[0, :, :, 0]
    
    # Apply 1 pass of 3x3 box blur for speed (reduced from 2 passes of 5x5)
    blurred_channels = []
    for i in range(3):
        channel = img[..., i]
        # Single pass for performance
        channel = box_blur_channel(channel)
        blurred_channels.append(channel)
    
    blurred_img = jnp.stack(blurred_channels, axis=-1)
    
    # Blend between original and blurred based on sigma
    # Adjusted for lighter blur (3x3, 1 pass)
    blend_factor = jnp.clip(sigma / 2.5, 0.0, 1.0)
    return img * (1.0 - blend_factor) + blurred_img * blend_factor


def apply_sharpen(img: jnp.ndarray, amount: float) -> jnp.ndarray:
    """
    Apply unsharp mask sharpening.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        amount: [0, 2], sharpening strength
    
    Returns:
        img: (H, W, 3) float32 clipped to [0, 255]
    """
    if amount == 0.0:
        return img
    
    # Create blurred version
    blurred = apply_blur(img, sigma=1.0)
    # Unsharp mask: img + amount * (img - blurred)
    sharpened = img + amount * (img - blurred)
    return jnp.clip(sharpened, 0.0, 255.0)


def apply_pixelate(img: jnp.ndarray, factor: int) -> jnp.ndarray:
    """
    Pixelate image by downsampling then upsampling.

    Args:
        img: (H, W, 3) float32 in [0, 255]
        factor: [1, 8], pixelation factor (1 = no effect)

    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if factor <= 1:
        return img

    H, W = img.shape[:2]

    # Downsample
    small_H, small_W = H // factor, W // factor
    downsampled = jax.image.resize(img, (small_H, small_W, 3), method='nearest')

    # Upsample back
    upsampled = jax.image.resize(downsampled, (H, W, 3), method='nearest')
    return upsampled


def apply_jpeg_compression(img: jnp.ndarray, quality: float) -> jnp.ndarray:
    """
    Simulate JPEG compression artifacts.

    Args:
        img: (H, W, 3) float32 in [0, 255]
        quality: [0, 1], 0 = heavy compression, 1 = minimal compression

    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if quality >= 1.0:
        return img

    # Convert to YCbCr color space (approximation for JPEG)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # YCbCr conversion (simplified)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    # Apply quantization based on quality
    # Lower quality = more aggressive quantization
    quant_factor = 1.0 + (1.0 - quality) * 50.0

    # Quantize chroma channels more aggressively (typical JPEG behavior)
    cb_quant = jnp.round(cb / quant_factor) * quant_factor
    cr_quant = jnp.round(cr / quant_factor) * quant_factor

    # Luma quantization (less aggressive)
    luma_quant_factor = quant_factor * 0.5
    y_quant = jnp.round(y / luma_quant_factor) * luma_quant_factor

    # Convert back to RGB
    r_new = y_quant + 1.402 * (cr_quant - 128)
    g_new = y_quant - 0.34414 * (cb_quant - 128) - 0.71414 * (cr_quant - 128)
    b_new = y_quant + 1.772 * (cb_quant - 128)

    return jnp.clip(jnp.stack([r_new, g_new, b_new], axis=-1), 0.0, 255.0)


def apply_vignette(img: jnp.ndarray, strength: float) -> jnp.ndarray:
    """
    Apply vignette effect (darkening at edges).
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        strength: [0, 1], vignette strength
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if strength == 0.0:
        return img
    
    H, W = img.shape[:2]
    y, x = jnp.ogrid[:H, :W]
    
    # Distance from center
    center_y, center_x = H / 2.0, W / 2.0
    max_dist = jnp.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)
    dist = jnp.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) / max_dist
    
    # Vignette mask (1 at center, fades to 0 at edges)
    vignette_mask = 1.0 - strength * dist ** 2
    vignette_mask = jnp.clip(vignette_mask, 0.0, 1.0)
    
    return img * vignette_mask[..., None]


def apply_radial_light(
    img: jnp.ndarray,
    strength: float,
    color: tuple[int, int, int]
) -> jnp.ndarray:
    """
    Apply radial light source effect (bright center, fading outward).
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        strength: [0, 1], light strength
        color: (R, G, B) tuple for light color
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if strength == 0.0:
        return img
    
    H, W = img.shape[:2]
    y, x = jnp.ogrid[:H, :W]
    
    # Distance from center
    center_y, center_x = H / 2.0, W / 2.0
    max_dist = jnp.sqrt((H / 2.0) ** 2 + (W / 2.0) ** 2)
    dist = jnp.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) / max_dist
    
    # Light mask (1 at center, fades to 0 at edges)
    light_mask = 1.0 - dist ** 1.5
    light_mask = jnp.clip(light_mask, 0.0, 1.0) * strength
    
    # Create light overlay
    light_color = jnp.array(color, dtype=jnp.float32)
    light_overlay = light_mask[..., None] * light_color[None, None, :]
    
    # Additive blending
    return jnp.clip(img + light_overlay, 0.0, 255.0)


def apply_pop_filter_vintage(img: jnp.ndarray) -> jnp.ndarray:
    """Vintage/sepia filter."""
    # Desaturate slightly
    img = apply_saturation(img, 0.7)
    # Warm color cast
    img = apply_color_temperature(img, 0.3)
    # Slight vignette
    img = apply_vignette(img, 0.3)
    # Boost contrast slightly
    img = apply_contrast(img, 1.1)
    return img


def apply_pop_filter_horror(img: jnp.ndarray) -> jnp.ndarray:
    """Horror/creepy filter."""
    # Desaturate heavily
    img = apply_saturation(img, 0.3)
    # Cool color cast
    img = apply_color_temperature(img, -0.4)
    # Darken
    img = apply_brightness(img, -0.2)
    # Heavy vignette
    img = apply_vignette(img, 0.6)
    # High contrast
    img = apply_contrast(img, 1.4)
    return img


def apply_pop_filter_retro(img: jnp.ndarray) -> jnp.ndarray:
    """Retro/80s filter."""
    # Boost saturation
    img = apply_saturation(img, 1.5)
    # Magenta/cyan cast (80s aesthetic)
    img = apply_hue_shift(img, 15.0)
    # Brighten slightly
    img = apply_brightness(img, 0.1)
    # Lower contrast
    img = apply_contrast(img, 0.9)
    return img


def apply_pop_filter_cyberpunk(img: jnp.ndarray) -> jnp.ndarray:
    """Cyberpunk/neon filter."""
    # High saturation
    img = apply_saturation(img, 1.8)
    # Cool color temperature
    img = apply_color_temperature(img, -0.2)
    # High contrast
    img = apply_contrast(img, 1.3)
    # Slight sharpen
    img = apply_sharpen(img, 0.5)
    return img


def apply_pop_filter_noir(img: jnp.ndarray) -> jnp.ndarray:
    """Film noir filter (black and white, high contrast)."""
    # Grayscale
    img = apply_saturation(img, 0.0)
    # Very high contrast
    img = apply_contrast(img, 1.6)
    # Vignette
    img = apply_vignette(img, 0.4)
    return img


def apply_pop_filter_by_name(img: jnp.ndarray, filter_name: str) -> jnp.ndarray:
    """
    Apply pop filter by name.
    
    Args:
        img: (H, W, 3) float32 in [0, 255]
        filter_name: Name of filter preset
    
    Returns:
        img: (H, W, 3) float32 in [0, 255]
    """
    if filter_name == "vintage":
        return apply_pop_filter_vintage(img)
    elif filter_name == "horror":
        return apply_pop_filter_horror(img)
    elif filter_name == "retro":
        return apply_pop_filter_retro(img)
    elif filter_name == "cyberpunk":
        return apply_pop_filter_cyberpunk(img)
    elif filter_name == "noir":
        return apply_pop_filter_noir(img)
    else:
        return img


def select_filter_by_index(idx: jax.Array, filter_names: list[str], img: jnp.ndarray) -> jnp.ndarray:
    """
    Select and apply filter by index (JAX-compatible).
    
    Args:
        idx: Integer index into filter_names (JAX array)
        filter_names: List of filter preset names
        img: (H, W, 3) float32 in [0, 255]
    
    Returns:
        filtered_img: (H, W, 3) float32 in [0, 255]
    """
    if not filter_names:
        return img
    if len(filter_names) == 1:
        return apply_pop_filter_by_name(img, filter_names[0])
    
    # Create branch functions for jax.lax.switch
    def make_branch(filter_name):
        return lambda: apply_pop_filter_by_name(img, filter_name)
    
    branches = [make_branch(f) for f in filter_names]
    
    # Use jax.lax.switch to select based on index (JIT-compatible)
    return jax.lax.switch(idx, branches)


# ============================================================================
# Main Filter Application
# ============================================================================

def apply_filters(
    img: jnp.ndarray,
    config: FilterConfig,
    key: Optional[jax.Array] = None,
    selected_filter: Optional[str] = None
) -> jnp.ndarray:
    """
    Apply configured filters to image.
    
    Args:
        img: (H, W, 3) uint8 observation
        config: FilterConfig with all filter parameters
        key: JAX PRNG key for stochastic effects (optional)
        selected_filter: Pre-selected filter name (overrides config.pop_filter)
    
    Returns:
        filtered_img: (H, W, 3) uint8
    """
    # Convert to float for processing
    img_f = img.astype(jnp.float32)
    
    # Generate key if needed for stochastic effects
    if key is None and config.seed is not None:
        key = jax.random.PRNGKey(config.seed)
    
    # Determine which filter to apply
    filter_to_apply = selected_filter if selected_filter is not None else config.pop_filter
    
    # Apply pop filter first if specified (overrides other settings partially)
    if filter_to_apply is not None:
        img_f = apply_pop_filter_by_name(img_f, filter_to_apply)
    
    # Photometric adjustments
    if config.brightness != 0.0:
        img_f = apply_brightness(img_f, config.brightness)
    
    if config.contrast != 1.0:
        img_f = apply_contrast(img_f, config.contrast)
    
    if config.gamma != 1.0:
        img_f = apply_gamma(img_f, config.gamma)
    
    if config.saturation != 1.0:
        img_f = apply_saturation(img_f, config.saturation)
    
    if config.hue_shift != 0.0:
        img_f = apply_hue_shift(img_f, config.hue_shift)
    
    if config.color_temp != 0.0:
        img_f = apply_color_temperature(img_f, config.color_temp)
    
    # Color jitter (stochastic)
    if config.color_jitter_std > 0.0 and key is not None:
        key, subkey = jax.random.split(key)
        img_f = apply_color_jitter(img_f, subkey, config.color_jitter_std)
    
    # Blur/sharpen
    if config.blur_sigma > 0.0:
        img_f = apply_blur(img_f, config.blur_sigma)
    
    if config.sharpen_amount > 0.0:
        img_f = apply_sharpen(img_f, config.sharpen_amount)
    
    # Pixelation
    if config.pixelate_factor > 1:
        img_f = apply_pixelate(img_f, config.pixelate_factor)

    # JPEG compression artifacts
    if config.jpeg_quality is not None and config.jpeg_quality < 1.0:
        img_f = apply_jpeg_compression(img_f, config.jpeg_quality)

    # Noise (stochastic)
    if config.gaussian_noise_std > 0.0 and key is not None:
        key, subkey = jax.random.split(key)
        img_f = apply_gaussian_noise(img_f, subkey, config.gaussian_noise_std)
    
    if config.poisson_noise_scale > 0.0 and key is not None:
        key, subkey = jax.random.split(key)
        img_f = apply_poisson_noise(img_f, subkey, config.poisson_noise_scale)
    
    # Vignette and radial light
    if config.vignette_strength > 0.0:
        img_f = apply_vignette(img_f, config.vignette_strength)
    
    if config.radial_light_strength > 0.0:
        img_f = apply_radial_light(img_f, config.radial_light_strength, config.radial_light_color)
    
    # Convert back to uint8
    return jnp.clip(img_f, 0.0, 255.0).astype(jnp.uint8)

