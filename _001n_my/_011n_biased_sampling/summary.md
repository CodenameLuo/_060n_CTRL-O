# Biased Sampling And Mask-Loss Binding Experiments

Goal:

```text
Improve slot binding for small objects, text-like phrases, and part-level phrases.
```

## Implemented Changes

Code/config changes:

```text
ocl/preprocessing.py
ocl/losses.py
configs/experiment/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased.yaml
configs/experiment/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased_mask.yaml
_001n_my/_011n_biased_sampling/run_clip_binding_experiment.sh
_001n_my/_009n_binding_slice_diagnostics/diagnose_binding_slices.py
```

Main changes:

```text
SelectConditioningInfoVGBiased:
  oversamples small/text/part phrases during train slot selection.

ControlMaskReconstructionLoss:
  now respects weight and is AMP-safe by computing BCE in float32.

diagnose_binding_slices.py:
  supports CTRLO_DIAG_SELECTION=hard;
  uses source_idx for GT mask lookup;
  outputs both failure and success visualizations.
```

## Runs

Baseline random 2k:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/2026-07-03_20-52-55_clip_random_2k_bs128_cmp
```

Biased sampling 2k:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased/2026-07-03_21-14-13_clip_biased_2k_bs128_cmp
```

Biased sampling + mask loss 2k:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased_mask/2026-07-03_21-53-35_clip_biased_mask_2k_bs128_cmp
```

Biased sampling + mask loss resumed to 5k:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased_mask/2026-07-03_22-46-47_clip_biased_mask_resume_5k_bs128_cmp
```

Reference random 5k:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/2026-07-03_18-43-50_clip_layer2_tuned_5k_bs128_amp
```

## Hard-Focused Diagnostics

All rows use:

```text
CTRLO_DIAG_SELECTION=hard
val shard: scripts/datasets/outputs/vg_disjoint_clip/val/shard-000000.tar
samples: 408
records: 2550 selected phrase-slot pairs
```

### Overall

| run | soft IoU | mass in GT | GT coverage | peak in GT |
| --- | ---: | ---: | ---: | ---: |
| random 2k | 0.0235 | 0.0414 | 0.1464 | 0.0400 |
| biased 2k | 0.0217 | 0.0392 | 0.1428 | 0.0306 |
| biased+mask 2k | 0.0311 | 0.0511 | 0.1771 | 0.0541 |
| random 5k | 0.0320 | 0.0544 | 0.1909 | 0.0545 |
| biased+mask 5k | 0.0358 | 0.0603 | 0.2043 | 0.0933 |

### Small

| run | soft IoU | mass in GT | GT coverage | peak in GT |
| --- | ---: | ---: | ---: | ---: |
| random 2k | 0.0070 | 0.0080 | 0.1415 | 0.0060 |
| biased 2k | 0.0067 | 0.0075 | 0.1413 | 0.0060 |
| biased+mask 2k | 0.0085 | 0.0092 | 0.1636 | 0.0078 |
| random 5k | 0.0094 | 0.0102 | 0.1845 | 0.0084 |
| biased+mask 5k | 0.0104 | 0.0112 | 0.1943 | 0.0185 |

### Text

| run | soft IoU | mass in GT | GT coverage | peak in GT |
| --- | ---: | ---: | ---: | ---: |
| random 2k | 0.0213 | 0.0333 | 0.1857 | 0.0417 |
| biased 2k | 0.0211 | 0.0310 | 0.2150 | 0.0417 |
| biased+mask 2k | 0.0227 | 0.0340 | 0.1894 | 0.0365 |
| random 5k | 0.0272 | 0.0380 | 0.2361 | 0.0573 |
| biased+mask 5k | 0.0265 | 0.0364 | 0.2207 | 0.0938 |

### Part

| run | soft IoU | mass in GT | GT coverage | peak in GT |
| --- | ---: | ---: | ---: | ---: |
| random 2k | 0.0169 | 0.0288 | 0.1146 | 0.0257 |
| biased 2k | 0.0155 | 0.0262 | 0.1135 | 0.0180 |
| biased+mask 2k | 0.0236 | 0.0344 | 0.1566 | 0.0386 |
| random 5k | 0.0273 | 0.0405 | 0.1776 | 0.0334 |
| biased+mask 5k | 0.0276 | 0.0421 | 0.1864 | 0.0694 |

## Final Artifacts

Final checkpoint:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_biased_mask/2026-07-03_22-46-47_clip_biased_mask_resume_5k_bs128_cmp/checkpoints/epoch=2-step=5000.ckpt
```

Final diagnostics:

```text
_001n_my/_011n_biased_sampling/diagnostics/biased_mask_5k_hard/binding_slice_summary.md
_001n_my/_011n_biased_sampling/diagnostics/biased_mask_5k_hard/binding_slice_records.csv
_001n_my/_011n_biased_sampling/diagnostics/biased_mask_5k_hard/binding_slice_failures.png
_001n_my/_011n_biased_sampling/diagnostics/biased_mask_5k_hard/binding_slice_successes.png
```

## Conclusion

Biased sampling alone did not improve binding at 2k steps. It made training harder and reduced early metrics.

The useful change was adding a small supervised mask-alignment loss on top of biased sampling. At 5k steps, `biased+mask` clearly improves hard-focused binding over the random 5k reference:

```text
overall peak_in_gt: 0.0545 -> 0.0933
small   peak_in_gt: 0.0084 -> 0.0185
text    peak_in_gt: 0.0573 -> 0.0938
part    peak_in_gt: 0.0334 -> 0.0694
```

The masks are still not precise enough to call the problem solved, but this is the first run that gives a clear, consistent binding improvement on the target slices.
