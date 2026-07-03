#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /etc/profile.d/zz_autodl_tmpdir.sh

export DATASET_PREFIX="${DATASET_PREFIX:-$REPO_ROOT/scripts/datasets/outputs}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_009n_CTRL_O/bin/python}"

if [[ -z "${TRAIN_SHARDS:-}" ]]; then
  TRAIN_SHARDS="$DATASET_PREFIX/vg_disjoint_clip/train/shard-{000000..000009}.tar"
fi
if [[ -z "${VAL_SHARDS:-}" ]]; then
  VAL_SHARDS="$DATASET_PREFIX/vg_disjoint_clip/val/shard-000000.tar"
fi

BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRECISION="${PRECISION:-16-mixed}"
MAX_STEPS="${MAX_STEPS:-5000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-500}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-4}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
TRAINING_VIS_FREQUENCY="${TRAINING_VIS_FREQUENCY:-1000}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-1000}"
RUN_NAME="${RUN_NAME:-clip_layer2_tuned_5k_bs128_amp}"

"$PYTHON_BIN" -m ocl.cli.train \
  +experiment=projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip \
  "dataset.train_shards='$TRAIN_SHARDS'" \
  "dataset.val_shards='$VAL_SHARDS'" \
  dataset.train_size=3950 \
  dataset.val_size=408 \
  experiment.batch_size_per_gpu="$BATCH_SIZE" \
  dataset.num_workers="$NUM_WORKERS" \
  trainer.precision="$PRECISION" \
  trainer.max_steps="$MAX_STEPS" \
  trainer.val_check_interval="$VAL_CHECK_INTERVAL" \
  trainer.limit_val_batches="$LIMIT_VAL_BATCHES" \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps="$LOG_EVERY_N_STEPS" \
  trainer.enable_progress_bar=false \
  trainer.accumulate_grad_batches=1 \
  training_vis_frequency="$TRAINING_VIS_FREQUENCY" \
  experiment.checkpoint_every_n_steps="$CHECKPOINT_EVERY_N_STEPS" \
  seed=0 \
  +experiment.name="$RUN_NAME"
