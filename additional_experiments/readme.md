# 1. sticky_npcs

```bash
bash additional_experiments/bench_multi_conf_ppo_jax.sh \
  additional_experiments/sticky_npcs/config_5_train.yaml 1 10 \
  additional_experiments/sticky_npcs/config_5_val_0.yaml \
  additional_experiments/sticky_npcs/config_5_val_1.yaml \
  additional_experiments/sticky_npcs/config_5_val_2.yaml \
  additional_experiments/sticky_npcs/config_5_val_3.yaml \
  additional_experiments/sticky_npcs/config_5_val_5.yaml \
  additional_experiments/sticky_npcs/config_5_val_7.yaml \
  additional_experiments/sticky_npcs/config_5_val_9.yaml \
  additional_experiments/sticky_npcs/config_5_val_11.yaml
```

# 2. radial_light
```bash
bash additional_experiments/bench_multi_conf_ppo_jax.sh \
  additional_experiments/radial_light/config_9_train.yaml 6 10 \
  additional_experiments/radial_light/config_9_val_0.0.yaml \
  additional_experiments/radial_light/config_9_val_0.25.yaml \
  additional_experiments/radial_light/config_9_val_0.5.yaml \
  additional_experiments/radial_light/config_9_val_0.75.yaml \
  additional_experiments/radial_light/config_9_val_1.0.yaml
```

# 3. num_of_train_bg_colors
```bash
bash additional_experiments/bench_multi_conf_ppo_jax.sh \
  additional_experiments/num_of_train_bg_colors/config_2_train.yaml 1 10 \
  additional_experiments/num_of_train_bg_colors/config_2_val_black.yaml \
  additional_experiments/num_of_train_bg_colors/config_2_val_black_white.yaml \
  additional_experiments/num_of_train_bg_colors/config_2_val_black_white_red.yaml \
  additional_experiments/num_of_train_bg_colors/config_2_val_black_white_red_green.yaml \
  additional_experiments/num_of_train_bg_colors/config_2_val_black_white_red_green_blue.yaml
```

