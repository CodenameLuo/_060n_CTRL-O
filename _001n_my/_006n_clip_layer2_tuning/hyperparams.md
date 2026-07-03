# CTRL-O CLIP layer-2 tuning

Date: 2026-07-03

## Goal

Continue the small CLIP replacement route on the local Visual Genome CLIP shards while keeping the run stable on a single RTX 3090.

## Local data

- Dataset repo: `adidolkar123/vg_clip`
- Local train shards: `scripts/datasets/outputs/vg_disjoint_clip/train/shard-000000.tar` to `shard-000009.tar`
- Local val shard: `scripts/datasets/outputs/vg_disjoint_clip/val/shard-000000.tar`
- One train shard contains about 395 samples.
- The 10 train shards contain about 3950 samples.
- The single val shard contains about 408 samples.
- The CLIP text embedding shape is `(10, 512)`.

## Probe results

Short probes used 20 optimization steps, no checkpoint, no logger.

| Setting | Result |
| --- | --- |
| `batch_size=8`, fp32, `num_workers=0` | OK, about 11.4 s |
| `batch_size=16`, fp32, `num_workers=0` | OK, about 12.9 s |
| `batch_size=32`, fp32, `num_workers=0` | OK, about 16.6 s |
| `batch_size=64`, fp32, `num_workers=0` | OK, about 24.1 s |
| `batch_size=96`, fp32, `num_workers=0` | OK, about 30.1 s |
| `batch_size=128`, fp32, `num_workers=0` | OOM |
| `batch_size=96`, fp32, `num_workers=2` | OK, about 27.3 s |
| `batch_size=96`, fp32, `num_workers=4` | OK, about 26.8 s |
| `batch_size=96`, fp32, `num_workers=8` | OK, about 27.6 s |
| `batch_size=128`, `16-mixed`, `num_workers=4` | OK, about 24.9 s |
| `batch_size=128`, `bf16-mixed`, `num_workers=4` | Failed: GRU cell has no BF16 CUDA implementation |
| `batch_size=192`, `16-mixed`, `num_workers=4` | OK, about 30.4 s |
| `batch_size=256`, `16-mixed`, `num_workers=4` | OOM |

## Chosen layer-2 defaults

- `experiment.batch_size_per_gpu=128`
- `trainer.precision=16-mixed`
- `dataset.num_workers=4`
- `trainer.accumulate_grad_batches=1`
- `dataset.train_size=3950`
- `dataset.val_size=408`
- `trainer.max_steps=5000`
- `trainer.val_check_interval=500`
- `trainer.limit_val_batches=4`
- `trainer.log_every_n_steps=50`
- `training_vis_frequency=1000`
- `experiment.checkpoint_every_n_steps=1000`

This uses the repository learning-rate rule unchanged:

```text
total_lr = 0.0004 * sqrt(batch_size / 64)
mapping_lr = 0.1 * total_lr
```

For `batch_size=128`, `total_lr` is about `5.66e-4`, and `mapping_lr` is about `5.66e-5`.

`batch_size=192` with `16-mixed` also passed the short probe and has better throughput, but it changes both the learning-rate scale and the number of batch-contrastive negatives relative to the author's CLIP config. Use it as a speed mode after the `batch_size=128` trend is understood.

## Run

```bash
bash _001n_my/_006n_clip_layer2_tuning/run_clip_layer2_tuned.sh
```

Override examples:

```bash
MAX_STEPS=1000 RUN_NAME=clip_layer2_tuned_1k bash _001n_my/_006n_clip_layer2_tuning/run_clip_layer2_tuned.sh
BATCH_SIZE=192 MAX_STEPS=5000 RUN_NAME=clip_layer2_bs192_amp bash _001n_my/_006n_clip_layer2_tuning/run_clip_layer2_tuned.sh
```

## Verification

The default script was verified with:

```bash
MAX_STEPS=50 \
VAL_CHECK_INTERVAL=50 \
LIMIT_VAL_BATCHES=1 \
LOG_EVERY_N_STEPS=10 \
CHECKPOINT_EVERY_N_STEPS=50 \
RUN_NAME=clip_layer2_tuned_verify_50_bs128 \
bash _001n_my/_006n_clip_layer2_tuning/run_clip_layer2_tuned.sh
```

Result:

- Output directory: `outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/2026-07-03_16-05-59_clip_layer2_tuned_verify_50_bs128`
- Checkpoint: `checkpoints/epoch=1-step=50.ckpt`
- Event scalars: 21 tags
- No NaN or Inf in checked loss/metric scalars

Checked scalar snapshot:

| Metric | First | Last |
| --- | --- | --- |
| `train/loss_total` | step 9: 3.7906 | step 49: 3.7625 |
| `train/mse` | step 9: 1.0769 | step 49: 1.0443 |
| `train/contrastive_loss_lang` | step 9: 1.3567 | step 49: 1.3576 |
| `train/contrastive_loss_point` | step 9: 1.3571 | step 49: 1.3606 |
| `train/acc_avg` | step 9: 0.1415 | step 49: 0.1411 |
| `val/loss_total` | step 49: 3.7585 | step 49: 3.7585 |
| `val/mse` | step 49: 1.0602 | step 49: 1.0602 |
| `val/binding_hits` | step 49: 0.2107 | step 49: 0.2107 |
| `val/acc_avg` | step 49: 0.1437 | step 49: 0.1437 |

## 5000-step trend run

Command:

```bash
bash _001n_my/_006n_clip_layer2_tuning/run_clip_layer2_tuned.sh
```

Output:

- Run directory: `outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip/2026-07-03_18-43-50_clip_layer2_tuned_5k_bs128_amp`
- Final checkpoint: `checkpoints/epoch=1-step=5000.ckpt`
- Trend plot: `_001n_my/_007n_clip_layer2_5k_trend/clip_layer2_5k_scalars.png`
- Scalar CSV: `_001n_my/_007n_clip_layer2_5k_trend/clip_layer2_5k_scalars.csv`
- Scalar summary: `_001n_my/_007n_clip_layer2_5k_trend/clip_layer2_5k_summary.txt`

Main scalar changes:

| Metric | First | Last |
| --- | --- | --- |
| `train/loss_total` | step 49: 3.7625 | step 4999: 1.9865 |
| `train/mse` | step 49: 1.0443 | step 4999: 0.4810 |
| `train/contrastive_loss_lang` | step 49: 1.3576 | step 4999: 0.4070 |
| `train/contrastive_loss_point` | step 49: 1.3606 | step 4999: 1.0985 |
| `train/acc_avg` | step 49: 0.1415 | step 4999: 0.8724 |
| `val/loss_total` | step 499: 3.3924 | step 4999: 2.4340 |
| `val/mse` | step 499: 0.7817 | step 4999: 0.6169 |
| `val/binding_hits` | step 499: 0.2051 | step 4999: 0.1863 |
| `val/acc_avg` | step 499: 0.2020 | step 4999: 0.9375 |
| `val/instance_mbo` | step 499: 0.0764 | step 4999: 0.1390 |
| `val/gt_matched_instance_mbo` | step 499: 0.0596 | step 4999: 0.1254 |
| `val/instance_ari` | step 499: 0.0208 | step 4999: 0.3262 |

No NaN or Inf was found in the checked scalar tags.
