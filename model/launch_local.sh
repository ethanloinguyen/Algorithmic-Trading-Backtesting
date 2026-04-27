#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_local.sh — Single-node multi-GPU launch (no SLURM)
#
# For a single VM with multiple GPUs (e.g. GCP a2-highgpu-4g, AWS p3.8xlarge).
# Uses torchrun's --standalone mode which handles rendezvous internally.
#
# Usage:
#   chmod +x launch_local.sh
#   ./launch_local.sh
#
# Override GPU count:
#   N_GPUS=2 ./launch_local.sh
# ──────────────────────────────────────────────────────────────────────────────

N_GPUS=${N_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
SAVE_DIR=${SAVE_DIR:-"checkpoints/deltalag"}
RESUME=${RESUME:-""}   # set to checkpoint path to resume: RESUME=checkpoints/deltalag/deltalag_epoch0080.pt

echo "Launching DeltaLag v10 on ${N_GPUS} GPU(s)"
mkdir -p "$SAVE_DIR" logs

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "Resuming from: $RESUME"
fi

torchrun \
    --standalone \
    --nproc_per_node="$N_GPUS" \
    train_distributed.py \
        --data_start    2000-01-01 \
        --train_end     2018-02-28 \
        --val_end       2021-07-31 \
        --test_end      2024-12-31 \
        --n_stocks      2000 \
        --hidden_n      128 \
        --k_leaders     5 \
        --window_l      40 \
        --l_max         10 \
        --epochs        150 \
        --lr            5e-6 \
        --tail_focus_alpha  0.5 \
        --tail_binary_alpha 0.4 \
        --diversity_alpha   0.05 \
        --lag_entropy_alpha 0.1 \
        --grad_accum_steps  5 \
        --checkpoint_every  10 \
        --save_dir      "$SAVE_DIR" \
        --backend       nccl \
        $RESUME_ARG \
    2>&1 | tee "logs/deltalag_$(date +%Y%m%d_%H%M%S).log"
