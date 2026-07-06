# CLIP VG 336 Smoke

Run directory:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_336_smoke/2026-07-03_20-28-06_clip_336_smoke_200_bs32
```

Command:

```bash
MAX_STEPS=200 RUN_NAME=clip_336_smoke_200_bs32 bash _001n_my/_010n_clip_336_smoke/run_clip_336_smoke.sh
```

Configuration:

```text
image_size=336
batch_size_per_gpu=32
precision=16-mixed
train_shards=vg_disjoint_clip/train/shard-{000000..000009}.tar
train_size=3950
limit_val_batches=0
max_steps=200
```

Checkpoint:

```text
outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_336_smoke/2026-07-03_20-28-06_clip_336_smoke_200_bs32/checkpoints/epoch=1-step=200.ckpt
```

Scalar sanity:

| metric | first logged | last logged |
| --- | ---: | ---: |
| train/loss_total | 3.21815 @ step 9 | 2.97102 @ step 199 |
| train/mse | 1.05165 @ step 9 | 0.816014 @ step 199 |
| train/contrastive_loss_lang | 1.08192 @ step 9 | 1.07743 @ step 199 |
| train/contrastive_loss_point | 1.08459 @ step 9 | 1.07757 @ step 199 |
| train/acc_avg | 0.140476 @ step 9 | 0.147727 @ step 199 |

Boundary:

```text
The local VG CLIP shards store 224x224 images. This smoke only verifies that
the 336x336 model path, resized masks, losses, logging, and checkpointing work.
It is not evidence that true high-resolution fine-detail binding has improved.
```
