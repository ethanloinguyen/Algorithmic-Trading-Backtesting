# DeltaLag v10 — Distributed Training

Parallelised version of the v10 Colab notebook.
All model logic, loss functions, and hyperparameters are identical to the notebook.

## Files

| File | Purpose |
|---|---|
| `train_distributed.py` | Main training script (PyTorch DDP) |
| `launch_local.sh` | Single-node multi-GPU launch (no SLURM) |
| `launch_slurm.sh` | Multi-node SLURM cluster launch |
| `requirements.txt` | Pinned dependencies |

---

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Single node, all available GPUs
```bash
chmod +x launch_local.sh
./launch_local.sh
```

### 3. Single node, specific GPU count
```bash
N_GPUS=2 ./launch_local.sh
```

### 4. Resume from checkpoint
```bash
RESUME=checkpoints/deltalag/deltalag_epoch0080.pt ./launch_local.sh
```

### 5. Multi-node SLURM cluster
```bash
# Edit #SBATCH directives in launch_slurm.sh for your cluster's node type and queue
sbatch launch_slurm.sh
```

### 6. Manual multi-node (no SLURM)
```bash
# Node 0 (master) — replace 10.0.0.1 with this machine's IP
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 \
         --master_addr=10.0.0.1 \
         --master_port=29500 \
         train_distributed.py

# Node 1 — same command, node_rank=1
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 \
         --master_addr=10.0.0.1 \
         --master_port=29500 \
         train_distributed.py
```

---

## How parallelism works

### Data sharding
Each training sample is one full trading day (S stocks × 40 timesteps).
`DistributedSampler` divides the list of training days across ranks:
- 4,500 training days ÷ 8 ranks = ~562 days per rank per epoch
- Each rank processes its own shard; gradients are averaged via all-reduce
- Validation days are also sharded; metrics are averaged across ranks before logging

### Within-day computation
The cross-sectional ranking loss requires the full day's cross-section —
it cannot be split further. Each rank processes one complete trading day at a time.
This is the fundamental unit of work and cannot be parallelised within a single rank.

### Communication pattern
```
Rank 0 (GPU 0)  ──┐
Rank 1 (GPU 1)  ──┤── all-reduce gradients ──► all ranks synchronised
Rank 2 (GPU 2)  ──┤   (every GRAD_ACCUM_STEPS steps)
Rank 3 (GPU 3)  ──┘
```

`model.no_sync()` suppresses all-reduce on intermediate accumulation steps,
reducing communication overhead by `GRAD_ACCUM_STEPS`× (default 5×).

### Checkpoint I/O
Only rank 0 writes checkpoints and logs to disk.
All ranks participate in training; rank 0 also handles all console output.

### Scaler broadcasting
`RobustScaler` is fit on rank 0 using the full training feature matrix,
then broadcast to all ranks via pickle over the process group.
This ensures every rank normalises features identically.

---

## Expected speedup

| Configuration | Epoch time (approx) | Speedup vs 1× T4 |
|---|---|---|
| 1× T4 (Colab notebook) | ~142s | 1× |
| 4× A100 40GB, 1 node | ~18–22s | ~6–8× |
| 8× A100 40GB, 2 nodes | ~10–14s | ~10–14× |
| 16× A100 80GB, 4 nodes | ~6–8s | ~18–24× |

Speedup is sub-linear because:
- Data broadcasting at startup is sequential (one-time cost)
- All-reduce communication overhead grows with world size
- The per-day forward/backward pass is fixed cost regardless of world size

---

## Output files

All written by rank 0 to `--save_dir` (default: `checkpoints/deltalag/`):

| File | Contents |
|---|---|
| `deltalag_epoch{N:04d}.pt` | Periodic checkpoint every 10 epochs |
| `deltalag_best.pt` | Best validation IC weights |
| `training_log.csv` | Per-epoch metrics (all columns from the notebook) |
| `daily_pnl.csv` | Test-set daily P&L series |

---

## Troubleshooting

**NCCL timeout / hang at startup**
- Check firewall rules: all nodes must be able to reach `MASTER_ADDR:MASTER_PORT` (default 29500)
- Try `--backend gloo` for CPU-only debugging or when InfiniBand is unavailable

**`NCCL error: unhandled system error`**
- Usually a CUDA device visibility issue. Ensure `CUDA_VISIBLE_DEVICES` is not
  accidentally restricting GPU access. `torchrun` sets `LOCAL_RANK` automatically.

**`RuntimeError: size mismatch`**
- Checkpoint was saved with a different architecture (different `hidden_n`, `k_leaders`, etc.)
- The resume cell rebuilds the model from `ckpt["hparams"]` automatically to handle this

**`broadcast_object` hangs**
- All ranks must reach the broadcast call. If rank 0 crashes during download,
  other ranks will wait indefinitely. Check rank 0 logs first.
