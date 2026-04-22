#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# launch_slurm.sh — DeltaLag v10 SLURM job script
#
# Usage:
#   sbatch launch_slurm.sh
#
# Tested on:
#   GCP Batch / GKE (a2-highgpu-4g, 4× A100 40GB per node)
#   AWS ParallelCluster (p3.8xlarge, 4× V100 per node)
#   On-premise SLURM with NCCL-capable InfiniBand interconnect
#
# Tune #SBATCH directives below for your cluster's node types and queue names.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=deltalag_v10
#SBATCH --nodes=2                        # number of nodes
#SBATCH --ntasks-per-node=4              # one task per GPU (4 GPUs per node = 8 total)
#SBATCH --gres=gpu:4                     # GPUs per node
#SBATCH --cpus-per-task=4               # CPU threads per GPU task
#SBATCH --mem=128G                       # RAM per node
#SBATCH --time=48:00:00                  # wall-clock limit
#SBATCH --partition=gpu                  # queue name — adjust for your cluster
#SBATCH --output=logs/deltalag_%j.out
#SBATCH --error=logs/deltalag_%j.err

# ── Environment setup ─────────────────────────────────────────────────────────
module load cuda/12.1          # adjust to your cluster's CUDA module
module load python/3.11

source /path/to/your/venv/bin/activate   # activate your Python venv

mkdir -p logs checkpoints/deltalag

# ── Network configuration for NCCL ───────────────────────────────────────────
# These environment variables ensure NCCL uses the fast InfiniBand/RoCE
# interconnect between nodes rather than the slower Ethernet interface.
# Adjust NCCL_SOCKET_IFNAME to match your cluster's network interface name
# (check with: ip link show | grep -E "^[0-9]+: (ib|eth|ens|enp)")
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0          # InfiniBand: ib0 / Ethernet: eth0, ens3, etc.
export NCCL_IB_DISABLE=0               # set to 1 to force TCP if InfiniBand is broken
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false    # suppress HuggingFace tokenizer warning

# ── Node communication setup ──────────────────────────────────────────────────
# SLURM sets SLURM_NODELIST automatically; we parse out the master node IP.
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent hosts "$MASTER_NODE" | awk '{ print $1 }')
MASTER_PORT=29500

echo "============================================================"
echo " Job ID     : $SLURM_JOB_ID"
echo " Nodes      : $SLURM_JOB_NODELIST"
echo " Master     : $MASTER_ADDR:$MASTER_PORT"
echo " Tasks      : $SLURM_NTASKS total ($SLURM_NTASKS_PER_NODE per node)"
echo "============================================================"

# ── Launch with srun + torchrun ───────────────────────────────────────────────
# srun allocates one process per --ntasks slot; torchrun inside each srun
# process discovers its LOCAL_RANK automatically from the environment.
srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
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
        --save_dir      checkpoints/deltalag \
        --backend       nccl
