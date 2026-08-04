# Statistical Summary

Baseline condition: baseline_clean
Permutation iterations: 2000

## Condition Overview

| condition_id | family | severity | n | success | within_tol | exact_match | fallback | mean_hour_err | mean_minute_err | mean_second_err |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_clean | baseline | none | 40 | 1.000 | 0.025 | 0.025 | 1.000 | 2.375 | 11.175 | 15.300 |
| combined_stress | combined | high | 40 | 1.000 | 0.025 | 0.000 | 1.000 | 2.500 | 9.275 | 13.200 |
| distractor_line | artifact | medium | 40 | 1.000 | 0.250 | 0.175 | 1.000 | 2.100 | 7.125 | 11.000 |
| gaussian_blur_5 | blur | medium | 40 | 1.000 | 0.050 | 0.000 | 1.000 | 2.925 | 7.400 | 13.900 |
| mirror_tracked | orientation | low | 40 | 1.000 | 0.100 | 0.000 | 1.000 | 2.350 | 9.400 | 14.850 |
| mirror_untracked | orientation | high | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 3.125 | 15.200 | 16.125 |
| noise_std_8 | noise | medium | 40 | 1.000 | 0.300 | 0.175 | 1.000 | 1.750 | 9.650 | 10.700 |
| occlusion_patch | occlusion | medium | 40 | 0.950 | 0.025 | 0.000 | 0.950 | 2.026 | 9.553 | 13.026 |
| rotate_16deg | rotation | medium | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 2.775 | 8.725 | 15.575 |
| rotate_8deg | rotation | low | 40 | 1.000 | 0.000 | 0.000 | 1.000 | 3.250 | 9.525 | 16.300 |

## Baseline vs Condition (within_tolerance)

| condition_id | delta_mean (baseline - condition) | 95% CI | permutation p-value |
|---|---:|---|---:|
| combined_stress | 0.0000 | [-0.0750, 0.0750] | 1.0000 |
| distractor_line | -0.2250 | [-0.3750, -0.0750] | 0.0070 |
| gaussian_blur_5 | -0.0250 | [-0.1000, 0.0500] | 1.0000 |
| mirror_tracked | -0.0750 | [-0.1750, 0.0250] | 0.3673 |
| mirror_untracked | 0.0250 | [0.0000, 0.0750] | 1.0000 |
| noise_std_8 | -0.2750 | [-0.4250, -0.1250] | 0.0015 |
| occlusion_patch | 0.0000 | [-0.0750, 0.0750] | 1.0000 |
| rotate_16deg | 0.0250 | [0.0000, 0.0750] | 1.0000 |
| rotate_8deg | 0.0250 | [0.0000, 0.0750] | 1.0000 |
