#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source /etc/profile.d/zz_autodl_tmpdir.sh

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/_000n_00000000/_003n_backend/_001n_z001/_001n_miniconda/_002n_conda_list/_009n_CTRL_O/bin/python}"
export DATASET_PREFIX="${DATASET_PREFIX:-$REPO_ROOT/scripts/datasets/outputs}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_STEPS="${MAX_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RUN_NAME="${RUN_NAME:-clip_336_smoke_${MAX_STEPS}_bs${BATCH_SIZE}}"

"$PYTHON_BIN" -m ocl.cli.train \
  +experiment=projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep_clip_336_smoke \
  "dataset.train_shards='$DATASET_PREFIX/vg_disjoint_clip/train/shard-{000000..000009}.tar'" \
  "dataset.val_shards='$DATASET_PREFIX/vg_disjoint_clip/val/shard-000000.tar'" \
  dataset.train_size=3950 \
  dataset.val_size=408 \
  experiment.batch_size_per_gpu="$BATCH_SIZE" \
  dataset.num_workers="$NUM_WORKERS" \
  trainer.precision=16-mixed \
  trainer.max_steps="$MAX_STEPS" \
  trainer.limit_val_batches=0 \
  trainer.num_sanity_val_steps=0 \
  trainer.log_every_n_steps=10 \
  trainer.enable_progress_bar=false \
  experiment.checkpoint_every_n_steps="$MAX_STEPS" \
  training_vis_frequency=100000 \
  seed=0 \
  +experiment.name="$RUN_NAME"
