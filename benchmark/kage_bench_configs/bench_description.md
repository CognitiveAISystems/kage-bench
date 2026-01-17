# 1. Background
1. config-1
    train:  background.mode=black
    eval:   background.mode=noise

2. config-2
    train:  background.mode=black
    eval:   background.mode=color
            background.color_names: ["purple"]

3. config-3
    train:  background.mode=black
    eval:   background.mode=color
            background.color_names: ["purple", "lime", "indigo"]

4. config-4
    train:  background.mode=color
            background.color_names: ["red", "green", "blue"]
    eval:   background.mode=color
            background.color_names: ["purple", "lime", "indigo"]

5. config-5
    train:  background.mode=black
    eval:   background.mode=image
            background.image_dir: "src/kage_bench/assets/backgrounds"

6. config-6
    train:  background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
    eval:   background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-128.jpeg"

7. config-7
    train:  background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
                - "src/kage_bench/assets/backgrounds/bg-2.jpeg"
                - "src/kage_bench/assets/backgrounds/bg-3.jpeg"
    eval:   background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-128.jpeg"

8. config-8
    train:  background.mode: "black"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   background.mode: "color"
            background.color_names: ["purple"]
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"

# 2. Agent appearance
1. config-1
    train:  character.shape_types=["circle"]
            character.shape_colors=["teal"]
    eval:   character.shape_types=["line"]
            character.shape_colors=["teal"]

2. config-2
    train:  character.shape_types=["circle"]
            character.shape_colors=["teal"]
    eval:   character.shape_types=["circle"]
            character.shape_colors=["pink"]

3. config-3
    train:  character.shape_types=["circle"]
            character.shape_colors=["teal"]
    eval:   character.shape_types=["line"]
            character.shape_colors=["pink"]

4. config-4
    train:  character.shape_types=["circle"]
            character.shape_colors=["teal"]
    eval:   character.use_sprites: true
            character.sprite_paths:
             - "src/kage_bench/assets/sprites/skeleton"
            character.enable_animation: true

5. config-5
    train:  character.use_sprites: true
            character.sprite_paths:
             - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
             - "src/kage_bench/assets/sprites/clown"


# 3. Background & Agent
1. config-1
    train:  background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-128.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"

2. config-2
    train:  background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
                - "src/kage_bench/assets/backgrounds/bg-2.jpeg"
                - "src/kage_bench/assets/backgrounds/bg-3.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   background.mode=image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-128.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"

# 4. Distractors
1. config-1
    train:  character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            npc.enabled: true
            npc.sprite_paths: 
                - "src/kage_bench/assets/sprites/skeleton"

2. config-2
    train:  character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            npc.enabled: true
            npc.sprite_dir: "src/kage_bench/assets/sprites"

3. config-3
    train:  character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            npc.sticky_enabled: true
            npc.sticky_sprite_dirs:
                - "src/kage_bench/assets/sprites/skeleton"

4. config-4
    train:  character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            npc.sticky_enabled: true
            npc.sticky_sprite_dir: "src/kage_bench/assets/sprites"

5. config-5
    train:  distactors.enabled: false
    eval:   distactors.enabled: true
            distractors.count: 7
            distractors.shape_types: ["circle"]
            distractors.shape_types: ["teal]
            distractors.min_size: 16
            distractors.max_size: 16

6. config-6
    train:  character.use_shape: true
            character.shape_types: ["circle"]
            character.shape_colors: ["teal"]
    eval:   character.use_shape: true
            character.shape_types: ["circle"]
            character.shape_colors: ["indigo"]
            distractors.enabled: true
            distractors.count: 1
            distractors.shape_types: ["circle"]
            distractors.shape_colors["teal]
            distractors.min_size: 16
            distractors.max_size: 16

# 5. Filters
1. config-1
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.brightness: 1

2. config-2
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.contrast: 128

3. config-3
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.saturation: 0.0

4. config-4
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.hue_shift: 180.0

5. config-5
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.color_jitter_std: 2.0

6. config-6
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.gaussian_noise_std: 100.0

7. config-7
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.pixelate_factor: 3.0

8. config-8
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.vignette_strength: 10.0

9. config-9
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            filters.radial_light_strength: 1.0

# 6. Effects
1. config-1
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            effects.point_light_enabled: true
            effects.point_light_intensity: 0.5

2. config-2
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            effects.point_light_enabled: true
            effects.point_light_falloff: 4.0

3. config-3
    train:  background.mode: image
            background.image_paths:
                - "src/kage_bench/assets/backgrounds/bg-1.jpeg"
            character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
    eval:   character.use_sprites: true
            character.sprite_paths:
                - "src/kage_bench/assets/sprites/skeleton"
            effects.point_light_enabled: true
            effects.point_light_count: 4

# 7. Layout
1. config-1
    train:  layout.layout_colors: ["cyan"]
    eval:   layout.layout_colors: ["red"]
