"""
DeltaLag — Distributed Training Script
=============================================

Launch commands
---------------
Single node, 4 GPUs:
    torchrun --standalone --nproc_per_node=4 train_distributed.py

Multi-node (2 nodes, 4 GPUs each = 8 total):
    # On node 0 (master):
    torchrun --nnodes=2 --nproc_per_node=4 \
             --node_rank=0 \
             --master_addr=<NODE_0_IP> \
             --master_port=29500 \
             train_distributed.py

    # On node 1:
    torchrun --nnodes=2 --nproc_per_node=4 \
             --node_rank=1 \
             --master_addr=<NODE_0_IP> \
             --master_port=29500 \
             train_distributed.py

GCP / AWS cluster with SLURM:
    sbatch launch_slurm.sh   (see launch_slurm.sh)

Key design decisions
--------------------
1.  DATA SHARDING BY DAY
    Each training sample is one full trading day (all S stocks × L timesteps).
    DDP shards the list of training days across ranks via DistributedSampler,
    so rank 0 processes days [0, W, 2W, ...], rank 1 processes days [1, W+1, ...],
    etc.  Within each day, all stocks are processed together — the cross-sectional
    ranking loss requires the full day's cross-section and cannot be split further.

2.  GRADIENT SYNCHRONISATION
    DDP all-reduces gradients automatically after each backward() call.
    GRAD_ACCUM_STEPS reduces the all-reduce frequency — synchronisation only
    happens every GRAD_ACCUM_STEPS steps, matching the effective batch size of
    the single-GPU notebook.

3.  CHECKPOINT I/O
    Only rank 0 writes checkpoints and logs to disk to avoid write conflicts.
    All ranks participate in training; rank 0 also handles all console output.

4.  SCALER BROADCASTING
    RobustScaler is fit on rank 0 using the full training feature matrix,
    then broadcast to all other ranks via pickle over the process group.
    This avoids each rank independently fitting a slightly different scaler
    on its local data shard.

5.  METRIC AGGREGATION
    IC, AR, SR, tail accuracy, and pair accuracy are computed locally on each
    rank's validation shard, then averaged across ranks via all_reduce before
    logging. This matches the notebook's per-day evaluation methodology.

Dependencies
------------
    pip install torch yfinance pandas scikit-learn tqdm scipy matplotlib
"""

import argparse
import io
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F_
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DeltaLag distributed training — v10")

    # Data
    p.add_argument("--data_start",  default="2000-01-01")
    p.add_argument("--train_end",   default="2018-02-28")
    p.add_argument("--val_end",     default="2021-07-31")
    p.add_argument("--test_end",    default="2024-12-31")
    p.add_argument("--n_stocks",    type=int, default=2000)

    # Model architecture
    p.add_argument("--window_l",    type=int,   default=40)
    p.add_argument("--l_max",       type=int,   default=10)
    p.add_argument("--k_leaders",   type=int,   default=5)
    p.add_argument("--hidden_n",    type=int,   default=128)
    p.add_argument("--mlp_dropout", type=float, default=0.1)

    # Loss weights
    p.add_argument("--tail_focus_alpha",  type=float, default=0.5)
    p.add_argument("--tail_binary_alpha", type=float, default=0.4)
    p.add_argument("--diversity_alpha",   type=float, default=0.05)
    p.add_argument("--lag_entropy_alpha", type=float, default=0.1)

    # Training
    p.add_argument("--epochs",            type=int,   default=150)
    p.add_argument("--lr",                type=float, default=5e-6)
    p.add_argument("--weight_decay",      type=float, default=0.0)
    p.add_argument("--grad_accum_steps",  type=int,   default=5)
    p.add_argument("--lr_patience",       type=int,   default=8)
    p.add_argument("--lr_factor",         type=float, default=0.5)
    p.add_argument("--lr_min",            type=float, default=5e-8)
    p.add_argument("--ic_smooth_window",  type=int,   default=10)
    p.add_argument("--patience",          type=int,   default=20)
    p.add_argument("--long_short_pct",    type=float, default=0.10)

    # I/O
    p.add_argument("--save_dir",          default="/checkpoints/deltalag")
    p.add_argument("--resume",            default=None, help="Path to .pt checkpoint to resume from")
    p.add_argument("--checkpoint_every",  type=int, default=10)
    p.add_argument("--lag_diag_freq",     type=int, default=1)

    # Distributed (set automatically by torchrun; override only if needed)
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"],
                   help="nccl for GPU clusters, gloo for CPU-only or debugging")

    return p.parse_args()


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_distributed(backend: str):
    """
    Initialise the process group from environment variables set by torchrun.
    torchrun sets LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    """
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    dist.destroy_process_group()


def is_main() -> bool:
    """True only on rank 0."""
    return dist.get_rank() == 0


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / dist.get_world_size()


def broadcast_object(obj, src: int = 0):
    """
    Broadcast an arbitrary Python object from rank `src` to all ranks.
    Uses pickle serialisation over a ByteTensor buffer.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return obj

    if dist.get_rank() == src:
        buf = pickle.dumps(obj)
        size_tensor = torch.tensor([len(buf)], dtype=torch.long)
    else:
        size_tensor = torch.tensor([0], dtype=torch.long)

    dist.broadcast(size_tensor, src=src)
    buf_size = size_tensor.item()

    if dist.get_rank() == src:
        buf_tensor = torch.ByteTensor(list(buf))
    else:
        buf_tensor = torch.ByteTensor(buf_size)

    dist.broadcast(buf_tensor, src=src)

    if dist.get_rank() != src:
        obj = pickle.loads(bytes(buf_tensor.tolist()))

    return obj


# ── Data download & feature engineering ──────────────────────────────────────

def fetch_iwm_constituents():
    import io as _io
    import urllib.request
    url = (
        "https://www.ishares.com/us/products/239710/"
        "ishares-russell-2000-etf/1467271812596.ajax"
        "?fileType=csv&fileName=IWM_holdings&dataType=fund"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.ishares.com/",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        lines = raw.splitlines()
        header_idx = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
        csv_body = "\n".join(lines[header_idx:])
        df = pd.read_csv(_io.StringIO(csv_body))
        df = df[df["Asset Class"].str.upper().str.contains("EQUITY", na=False)]
        tickers = (
            df["Ticker"].str.strip().replace("-", pd.NA).dropna().tolist()
        )
        tickers = [t for t in tickers if t.isalpha() and 1 <= len(t) <= 5]
        return tickers
    except Exception as e:
        if is_main():
            print(f"  iShares fetch failed ({e}); using fallback seed list")
        return None


def download_data(tickers, start, end):
    import yfinance as yf
    if is_main():
        print(f"Downloading {len(tickers)} tickers ({start} → {end})...")
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=is_main(), threads=True,
    )
    data = {}
    for ticker in tickers:
        try:
            df = pd.DataFrame({
                "Open":   raw["Open"][ticker],
                "High":   raw["High"][ticker],
                "Low":    raw["Low"][ticker],
                "Close":  raw["Close"][ticker],
                "Volume": raw["Volume"][ticker],
            }).dropna()
            if len(df) > 60:
                data[ticker] = df
        except Exception:
            pass
    if is_main():
        print(f"  Loaded {len(data)} tickers ({len(tickers)-len(data)} skipped)")
    return data


def build_features(df):
    feat = pd.DataFrame(index=df.index)
    feat["open_ratio"] = df["Open"]  / df["Close"]
    feat["high_ratio"] = df["High"]  / df["Close"]
    feat["low_ratio"]  = df["Low"]   / df["Close"]
    feat["daily_ret"]  = df["Close"].pct_change()
    feat["log_vol"]    = np.log1p(df["Volume"])
    feat["turnover"]   = df["Volume"] / df["Volume"].rolling(20).mean().clip(lower=1)
    return feat.replace([np.inf, -np.inf], np.nan).dropna()


def build_feature_panel(data):
    features_raw, features, returns = {}, {}, {}
    for ticker, df in data.items():
        feat = build_features(df)
        ret  = df["Close"].pct_change().shift(-1)
        valid_idx = feat.index.intersection(ret.dropna().index)
        features_raw[ticker] = feat.loc[valid_idx]
        features[ticker]     = feat.loc[valid_idx]
        returns[ticker]      = ret.loc[valid_idx]
    all_dates = sorted(set.union(*[set(v.index) for v in features.values()]))
    return features_raw, features, returns, all_dates


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeltaLagDataset(Dataset):
    """
    One sample = one TRADING DAY (all S stocks × L timesteps).
    DistributedSampler shards the list of days across ranks.
    Each rank processes a disjoint subset of trading days per epoch.
    """
    def __init__(self, features_raw, features, returns, dates, tickers,
                 window_l, scaler=None, fit_scaler=False):
        self.features_raw = features_raw
        self.features     = features
        self.returns      = returns
        self.tickers      = tickers
        self.L = window_l
        self.S = len(tickers)
        self.F = 6

        if fit_scaler:
            all_feat = np.concatenate(
                [features[t].values for t in tickers if t in features], axis=0
            )
            self.scaler = RobustScaler().fit(all_feat)
        else:
            self.scaler = scaler

        self.scaled = {}
        for t in tickers:
            if t in features:
                self.scaled[t] = self.scaler.transform(
                    features[t].values).astype(np.float32)
            else:
                self.scaled[t] = np.zeros((1, self.F), dtype=np.float32)

        self.raw = {}
        for t in tickers:
            if t in features_raw:
                self.raw[t] = features_raw[t].values.astype(np.float32)
            else:
                self.raw[t] = np.zeros((1, self.F), dtype=np.float32)

        self.date_pos = {}
        for t in tickers:
            if t in features:
                self.date_pos[t] = {d: i for i, d in enumerate(features[t].index)}

        self.dates = []
        for date in dates:
            available = [
                si for si, t in enumerate(tickers)
                if t in features
                and date in self.date_pos[t]
                and self.date_pos[t][date] >= self.L
                and date in returns[t].index
                and not np.isnan(returns[t].loc[date])
            ]
            if len(available) >= max(10, int(self.S * 0.1)):
                self.dates.append((date, available))

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date, available_sis = self.dates[idx]
        X_scaled = np.zeros((self.S, self.L, self.F), dtype=np.float32)
        X_raw    = np.zeros((self.S, self.L, self.F), dtype=np.float32)

        for si, t in enumerate(self.tickers):
            if t not in self.features:
                continue
            if date not in self.date_pos[t]:
                continue
            pos = self.date_pos[t][date]
            if pos < self.L:
                continue
            X_scaled[si] = self.scaled[t][pos - self.L + 1 : pos + 1]
            X_raw[si]    = self.raw[t][pos - self.L + 1 : pos + 1]

        ys     = np.array([self.returns[self.tickers[si]].loc[date]
                           for si in available_sis], dtype=np.float32)
        t_idxs = np.array(available_sis, dtype=np.int64)
        date_ns = int(date.value)

        return (
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(X_raw,    dtype=torch.float32),
            torch.tensor(ys,       dtype=torch.float32),
            torch.tensor(t_idxs,   dtype=torch.long),
            date_ns,
        )


def collate_fn(batch):
    return batch


# ── Model ─────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    def __init__(self, F, N):
        super().__init__()
        self.lstm = nn.LSTM(F, N, batch_first=True)
        self.norm = nn.LayerNorm(N)

    def forward(self, x):
        B, S, L, F = x.shape
        out, _ = self.lstm(x.view(B * S, L, F))
        return self.norm(out).view(B, S, L, -1)


class DeltaLag(nn.Module):
    """DeltaLag v10 — cosine attention + learnable lag bias + two-stream MLP."""

    def __init__(self, S, F=6, N=128, L=40, l_max=10, k=5, dropout=0.1):
        super().__init__()
        self.S, self.F, self.N = S, F, N
        self.L, self.l_max, self.k = L, l_max, k

        self.encoder  = TemporalEncoder(F, N)
        self.W_Q      = nn.Linear(N, N, bias=False)
        self.W_K      = nn.Linear(N, N, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(math.log(N ** 0.25)))
        self.lag_bias = nn.Parameter(torch.zeros(l_max))

        mlp_in = 2 * F
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self.last_leaders   = None
        self.last_lags      = None
        self.last_attn_flat = None

    def forward(self, X_scaled, X_raw, target_idx):
        N_tgt = target_idx.shape[0]
        H = self.encoder(X_scaled).squeeze(0)
        X = X_raw.squeeze(0)

        q      = F_.normalize(self.W_Q(H[target_idx, -1, :]), dim=-1)
        H_keys = F_.normalize(self.W_K(H[:, self.L - self.l_max:, :]), dim=-1)

        temp = self.log_temp.exp().clamp(0.1, self.N ** 0.5)
        attn = (q.view(N_tgt, 1, 1, self.N) * H_keys.unsqueeze(0)).sum(-1) / temp
        attn = attn + self.lag_bias.view(1, 1, self.l_max)

        self_mask = torch.zeros(N_tgt, self.S,
                                device=X_scaled.device, dtype=torch.bool)
        self_mask.scatter_(1, target_idx.unsqueeze(1), True)
        attn = attn.masked_fill(self_mask.unsqueeze(-1), float('-inf'))

        attn_flat = attn.view(N_tgt, self.S * self.l_max)
        self.last_attn_flat = attn_flat.detach()

        topk_vals, topk_flat = torch.topk(attn_flat, self.k, dim=1)
        leader_idx = topk_flat // self.l_max
        lag_j      = topk_flat  % self.l_max
        lag_pos    = (self.L - 1 - (self.l_max - lag_j)).clamp(0, self.L - 1)
        tau        = self.l_max - lag_j

        self.last_leaders = leader_idx.detach().cpu()
        self.last_lags    = tau.detach().cpu()

        X_exp  = X.unsqueeze(0).expand(N_tgt, -1, -1, -1)
        li     = leader_idx.view(N_tgt, self.k, 1, 1).expand(N_tgt, self.k, self.L, self.F)
        x_lead = X_exp.gather(1, li)
        lp     = lag_pos.view(N_tgt, self.k, 1, 1).expand(N_tgt, self.k, 1, self.F)
        z      = x_lead.gather(2, lp).squeeze(2)

        weights = F_.softmax(topk_vals, dim=1)
        z_agg   = (weights.unsqueeze(-1) * z).sum(1)
        z_top1  = z[:, 0, :]

        return self.mlp(torch.cat([z_agg, z_top1], dim=-1)).squeeze(-1)


# ── Loss functions ────────────────────────────────────────────────────────────

def monotonic_ranking_loss(preds, targets):
    dp = preds.unsqueeze(1)   - preds.unsqueeze(0)
    dt = targets.unsqueeze(1) - targets.unsqueeze(0)
    return torch.log1p(torch.exp(-torch.tanh(dp) * torch.tanh(dt))).sum()


def tail_focused_loss(preds, targets, pct=0.10):
    n = len(preds)
    k = max(1, int(n * pct))
    order = torch.argsort(targets)
    pt  = preds[order[-k:]]
    pb  = preds[order[:k]]
    tt  = targets[order[-k:]]
    tb  = targets[order[:k]]
    dp  = pt.unsqueeze(1) - pb.unsqueeze(0)
    dt  = tt.unsqueeze(1) - tb.unsqueeze(0)
    return torch.log1p(torch.exp(-torch.tanh(dp) * torch.tanh(dt))).sum()


def binary_tail_loss(preds, targets, pct=0.25):
    n = len(preds)
    k = max(1, int(n * pct))
    order      = torch.argsort(targets)
    tail_idx   = torch.cat([order[:k], order[-k:]])
    tail_preds = preds[tail_idx]
    tail_labels = torch.cat([
        torch.zeros(k, device=preds.device),
        torch.ones(k,  device=preds.device),
    ])
    return F_.binary_cross_entropy_with_logits(tail_preds, tail_labels)


def diversity_loss(model):
    if model.last_attn_flat is None:
        return torch.tensor(0.0)
    N_tgt = model.last_attn_flat.shape[0]
    S     = model.S
    a     = model.last_attn_flat.view(N_tgt, S, model.l_max)
    a     = a.nan_to_num(nan=0.0, posinf=0.0, neginf=-1e4)
    pop   = torch.softmax(a.max(dim=2).values, dim=1).sum(dim=0)
    pop   = pop / pop.sum()
    ent   = -(pop * (pop + 1e-8).log()).sum()
    return -ent / math.log(S)


def lag_entropy_loss(model):
    if model.last_attn_flat is None:
        return torch.tensor(0.0)
    N_tgt = model.last_attn_flat.shape[0]
    a     = model.last_attn_flat.view(N_tgt, model.S, model.l_max)
    a     = a.nan_to_num(nan=0.0, posinf=0.0, neginf=-1e4)
    marg  = torch.softmax(a, dim=-1).mean(dim=[0, 1])
    return -(marg * (marg + 1e-8).log()).sum()


# ── Metric helpers ────────────────────────────────────────────────────────────

def tail_accuracy(preds, targets, pct=0.10):
    if isinstance(preds, np.ndarray):
        preds   = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    n = len(preds)
    k = max(1, int(n * pct))
    order = torch.argsort(targets)
    return (preds[order[-k:]].unsqueeze(1) > preds[order[:k]].unsqueeze(0)).float().mean().item()


def pair_accuracy(preds, targets):
    if isinstance(preds, np.ndarray):
        preds   = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    dp   = preds.unsqueeze(1)   - preds.unsqueeze(0)
    dt   = targets.unsqueeze(1) - targets.unsqueeze(0)
    mask = dt != 0
    n    = mask.sum().item()
    return ((dp * dt > 0) & mask).sum().item() / n if n else 0.5


def compute_portfolio_metrics(results, long_short_pct):
    daily_pnl, ic_vals = {}, []
    for date, preds, targets in results:
        n = len(preds)
        k = max(1, int(n * long_short_pct))
        order = np.argsort(preds)
        daily_pnl[date] = (targets[order[-k:]].mean() - targets[order[:k]].mean()) / 2.0
        if len(preds) > 1:
            rho, _ = spearmanr(preds, targets)
            if not np.isnan(rho):
                ic_vals.append(rho)

    daily_pnl = pd.Series(daily_pnl).sort_index()
    ic  = float(np.mean(ic_vals)) if ic_vals else 0.0
    ann = daily_pnl.mean() * 252
    vol = daily_pnl.std()  * np.sqrt(252)
    sr  = float(ann / vol) if vol > 1e-8 else 0.0
    return ic, float(ann), sr, daily_pnl


def lag_entropy_metric(lag_tensor, l_max):
    counts = torch.zeros(l_max)
    for v in lag_tensor.flatten():
        idx = int(v.item()) - 1
        if 0 <= idx < l_max:
            counts[idx] += 1
    probs = counts / counts.sum().clamp(min=1)
    return (-(probs * (probs + 1e-8).log()).sum().item(),
            counts.numpy())


def leader_stability(leader_sets, window=10):
    if len(leader_sets) < 2:
        return 0.0
    pairs = list(zip(leader_sets[:-1], leader_sets[1:]))[-window:]
    jacs  = [len(a & b) / len(a | b) for a, b in pairs if len(a | b) > 0]
    return float(np.mean(jacs)) if jacs else 0.0


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, args, device):
    """
    Distributed training epoch.

    DDP handles gradient synchronisation automatically. The model is wrapped
    in DDP before this function is called, so loss.backward() triggers an
    all-reduce across all ranks. We only need to call optimizer.step() once
    per GRAD_ACCUM_STEPS steps — the same as the single-GPU notebook.

    Note: to_sync / no_sync context manager is used to suppress intermediate
    all-reduces during gradient accumulation, matching the single-GPU cadence
    and reducing communication overhead by GRAD_ACCUM_STEPS×.
    """
    model.train()
    total_loss = total_tail = total_pair = n = 0
    optimizer.zero_grad()

    # Access the underlying module for buffer reads (last_attn_flat etc.)
    raw_model = model.module if hasattr(model, "module") else model

    for step, batch in enumerate(tqdm(loader, desc=f"train r{dist.get_rank()}",
                                       leave=False, disable=not is_main())):
        X_scaled, X_raw, y, t_idx, _ = batch[0]
        X_scaled = X_scaled.unsqueeze(0).to(device)
        X_raw    = X_raw.unsqueeze(0).to(device)
        y        = y.to(device)
        t_idx    = t_idx.to(device)

        # Suppress all-reduce on accumulation steps; sync on the last step
        is_last = (step + 1) % args.grad_accum_steps == 0 \
                  or (step + 1) == len(loader)
        ctx = model.no_sync() if not is_last else torch.no_grad().__class__()
        # Use no_sync to skip all-reduce on intermediate accumulation steps
        if not is_last and hasattr(model, "no_sync"):
            ctx = model.no_sync()
        else:
            import contextlib
            ctx = contextlib.nullcontext()

        with ctx:
            preds = model(X_scaled, X_raw, t_idx)
            loss = (
                monotonic_ranking_loss(preds, y)
                + args.tail_focus_alpha  * tail_focused_loss(preds, y)
                + args.tail_binary_alpha * binary_tail_loss(preds, y)
                + args.diversity_alpha   * diversity_loss(raw_model)
                + args.lag_entropy_alpha * lag_entropy_loss(raw_model)
            )
            (loss / args.grad_accum_steps).backward()

        # Detach main_loss scalar for logging (not the combined loss)
        with torch.no_grad():
            main_loss_val = monotonic_ranking_loss(preds.detach(), y).item()

        total_loss += main_loss_val
        total_tail += tail_accuracy(preds.detach().cpu(), y.cpu())
        total_pair += pair_accuracy(preds.detach().cpu(), y.cpu())
        n += 1

        if is_last:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    # Average metrics across ranks so rank 0 logs the global mean
    metrics = torch.tensor(
        [total_loss / max(n, 1),
         total_tail / max(n, 1),
         total_pair / max(n, 1)],
        device=device
    )
    all_reduce_mean(metrics)
    return metrics[0].item(), metrics[1].item(), metrics[2].item()


@torch.no_grad()
def evaluate(model, loader, args, device, collect_diag=False):
    """
    Distributed evaluation.
    Each rank evaluates its own shard of validation days, then metrics
    are averaged across ranks via all_reduce.
    """
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
    results, tail_acc, pair_acc_list = [], [], []
    all_lags, all_leader_sets = [], []

    for batch in tqdm(loader, desc="eval", leave=False, disable=not is_main()):
        X_scaled, X_raw, y, t_idx, date_ns = batch[0]
        X_scaled = X_scaled.unsqueeze(0).to(device)
        X_raw    = X_raw.unsqueeze(0).to(device)
        preds_t  = model(X_scaled, X_raw, t_idx.to(device))
        preds    = preds_t.cpu().numpy()
        y_np     = y.numpy()
        results.append((pd.Timestamp(date_ns), preds, y_np))
        tail_acc.append(tail_accuracy(preds_t.cpu(), y))
        pair_acc_list.append(pair_accuracy(preds_t.cpu(), y))

        if collect_diag and raw_model.last_lags is not None:
            all_lags.append(raw_model.last_lags.clone())
            all_leader_sets.append(set(raw_model.last_leaders.flatten().tolist()))

    ic, ar, sr, daily_pnl = compute_portfolio_metrics(results, args.long_short_pct)
    avg_tail = float(np.mean(tail_acc))
    avg_pair = float(np.mean(pair_acc_list))

    # All-reduce scalar metrics across ranks
    m = torch.tensor([ic, ar, sr, avg_tail, avg_pair], device=device)
    all_reduce_mean(m)
    ic, ar, sr, avg_tail, avg_pair = m.tolist()

    diag = {}
    if collect_diag and all_lags:
        lag_stack = torch.cat(all_lags, dim=0)
        ent, counts = lag_entropy_metric(lag_stack, args.l_max)
        uniq = float(np.mean([len(s) for s in all_leader_sets]))
        stab = leader_stability(all_leader_sets)
        diag = {
            "lag_entropy":      ent,
            "lag_counts":       counts,
            "n_unique_leaders": uniq,
            "leader_pct":       uniq / max(1, loader.dataset.S) * 100,
            "leader_stability": stab,
        }

    return ic, ar, sr, daily_pnl, avg_tail, avg_pair, diag


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    rank    = setup_distributed(args.backend)
    device  = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    is_rank0 = is_main()

    if is_rank0:
        print(f"\nDeltaLag v10 — Distributed Training")
        print(f"  World size : {dist.get_world_size()}")
        print(f"  Backend    : {args.backend}")
        print(f"  Device     : {device}")
        print()

    # ── Data: only rank 0 downloads, then broadcast ───────────────────────────
    if is_rank0:
        tickers = fetch_iwm_constituents() or []
        seen, TICKERS = set(), []
        for t in tickers:
            t = t.strip()
            if t and t not in seen and t.isalpha() and 1 <= len(t) <= 5:
                seen.add(t)
                TICKERS.append(t)
        TICKERS = TICKERS[:args.n_stocks]
        data = download_data(TICKERS, args.data_start, args.test_end)
        ACTIVE_TICKERS = [t for t in TICKERS if t in data]
        features_raw, features, returns, all_dates = build_feature_panel(data)
        print(f"Active universe: {len(ACTIVE_TICKERS)} tickers")
    else:
        ACTIVE_TICKERS = None
        features_raw = features = returns = all_dates = None

    # Broadcast data structures to all ranks
    ACTIVE_TICKERS = broadcast_object(ACTIVE_TICKERS)
    features_raw   = broadcast_object(features_raw)
    features       = broadcast_object(features)
    returns        = broadcast_object(returns)
    all_dates      = broadcast_object(all_dates)

    # ── Date splits ───────────────────────────────────────────────────────────
    train_dates = [d for d in all_dates if str(d.date()) <= args.train_end]
    val_dates   = [d for d in all_dates
                   if args.train_end < str(d.date()) <= args.val_end]
    test_dates  = [d for d in all_dates
                   if args.val_end   < str(d.date()) <= args.test_end]

    if is_rank0:
        print(f"Train: {len(train_dates)} days | "
              f"Val: {len(val_dates)} days | "
              f"Test: {len(test_dates)} days")

    # ── Datasets — rank 0 fits scaler, broadcasts to all ranks ───────────────
    if is_rank0:
        train_ds = DeltaLagDataset(
            features_raw, features, returns, train_dates,
            ACTIVE_TICKERS, args.window_l, fit_scaler=True
        )
        scaler = train_ds.scaler
    else:
        scaler   = None
        train_ds = None

    scaler = broadcast_object(scaler)

    if train_ds is None:
        train_ds = DeltaLagDataset(
            features_raw, features, returns, train_dates,
            ACTIVE_TICKERS, args.window_l, scaler=scaler
        )

    val_ds = DeltaLagDataset(
        features_raw, features, returns, val_dates,
        ACTIVE_TICKERS, args.window_l, scaler=scaler
    )
    test_ds = DeltaLagDataset(
        features_raw, features, returns, test_dates,
        ACTIVE_TICKERS, args.window_l, scaler=scaler
    )

    # ── Distributed samplers — shard days across ranks ────────────────────────
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)
    test_sampler  = DistributedSampler(test_ds,  shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=2,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, sampler=val_sampler,
                              collate_fn=collate_fn, num_workers=2,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1, sampler=test_sampler,
                              collate_fn=collate_fn, num_workers=2,
                              pin_memory=True)

    # ── Model + DDP ───────────────────────────────────────────────────────────
    S = len(ACTIVE_TICKERS)
    raw_model = DeltaLag(
        S=S, N=args.hidden_n, L=args.window_l, l_max=args.l_max,
        k=args.k_leaders, dropout=args.mlp_dropout
    ).to(device)

    model = DDP(raw_model, device_ids=[rank] if torch.cuda.is_available() else None,
                find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor,
        patience=args.lr_patience, min_lr=args.lr_min,
    )

    n_params = sum(p.numel() for p in model.parameters())
    if is_rank0:
        print(f"Model parameters: {n_params:,}")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    log_rows    = []
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model_state"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError:
            pass
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            pass
        log_rows    = ckpt.get("log", [])
        start_epoch = ckpt["epoch"] + 1
        if is_rank0:
            print(f"Resumed from epoch {ckpt['epoch']} → continuing from {start_epoch}")

    # ── Training loop ─────────────────────────────────────────────────────────
    save_dir = Path(args.save_dir)
    if is_rank0:
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val_ic       = -np.inf
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        # Tell sampler which epoch we're in so shuffling is different each epoch
        train_sampler.set_epoch(epoch)

        t0 = time.time()
        train_loss, train_tail, train_pair = train_epoch(
            model, train_loader, optimizer, args, device
        )

        run_diag = (epoch % args.lag_diag_freq == 0)
        val_ic, val_ar, val_sr, _, val_tail, val_pair, val_diag = evaluate(
            model, val_loader, args, device, collect_diag=run_diag
        )

        scheduler.step(val_ic)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_secs = time.time() - t0
        attn_temp  = model.module.log_temp.exp().item()

        if is_rank0:
            row = dict(
                epoch=epoch,
                train_loss=train_loss, train_tail=train_tail, train_pair=train_pair,
                val_IC=val_ic, val_AR=val_ar, val_SR=val_sr,
                val_tail=val_tail, val_pair=val_pair,
                lr=current_lr, epoch_secs=epoch_secs, attn_temp=attn_temp,
                lag_entropy=val_diag.get("lag_entropy", float("nan")),
                n_unique_leaders=val_diag.get("n_unique_leaders", float("nan")),
                leader_stability=val_diag.get("leader_stability", float("nan")),
            )
            log_rows.append(row)

            diag_str = ""
            if run_diag and val_diag:
                diag_str = (
                    f"  | lag_H={val_diag['lag_entropy']:.2f}"
                    f"  uniq={val_diag['n_unique_leaders']:.0f}"
                    f"({val_diag['leader_pct']:.1f}%)"
                    f"  stab={val_diag['leader_stability']:.3f}"
                    f"  temp={attn_temp:.2f}"
                )

            print(
                f"Epoch {epoch:3d}/{args.epochs}  "
                f"loss={train_loss:,.1f}  "
                f"tail={train_tail:.4f}  pair={train_pair:.4f}  "
                f"| val IC={val_ic:+.4f}  AR={val_ar:+.3f}  SR={val_sr:+.2f}  "
                f"tail={val_tail:.4f}  "
                f"| lr={current_lr:.2e}  ({epoch_secs:.0f}s)"
                + diag_str
            )

            # Checkpoint
            if epoch % args.checkpoint_every == 0:
                ckpt_path = save_dir / f"deltalag_epoch{epoch:04d}.pt"
                torch.save({
                    "epoch":           epoch,
                    "model_state":     model.module.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler":          scaler,
                    "tickers":         ACTIVE_TICKERS,
                    "hparams": {
                        "S": S, "F": 6, "N": args.hidden_n,
                        "L": args.window_l, "l_max": args.l_max,
                        "k": args.k_leaders,
                    },
                    "log": log_rows,
                }, ckpt_path)
                print(f"  ── checkpoint saved → {ckpt_path.name}")

            # Early stopping (track on rank 0)
            if epoch >= args.ic_smooth_window:
                smoothed = np.mean([r["val_IC"]
                                    for r in log_rows[-args.ic_smooth_window:]])
                if smoothed > best_val_ic:
                    best_val_ic       = smoothed
                    best_weights      = {k: v.clone()
                                         for k, v in model.module.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        # Broadcast early-stop signal from rank 0 to all ranks
        stop_signal = torch.tensor(
            [1 if is_rank0 and epochs_no_improve >= args.patience else 0]
        )
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            if is_rank0:
                print(f"\nEarly stopping at epoch {epoch}.")
            break

    # ── Final test evaluation ─────────────────────────────────────────────────
    if best_weights is not None:
        model.module.load_state_dict(best_weights)

    test_ic, test_ar, test_sr, daily_pnl, test_tail, test_pair, test_diag = evaluate(
        model, test_loader, args, device, collect_diag=True
    )

    if is_rank0:
        print("\n── Test set results (v10) ─────────────────────────────")
        print(f"  Information Coefficient : {test_ic:+.4f}  (paper SP500: +0.0261)")
        print(f"  Annualised Return       : {test_ar:+.2%}  (paper SP500: +24.7%)")
        print(f"  Sharpe Ratio            : {test_sr:+.2f}   (paper SP500: +2.12)")
        print()
        print(f"  Tail accuracy : {test_tail:.4f}  (random=0.50, paper≈0.85)")
        print(f"  Pair accuracy : {test_pair:.4f}  (random=0.50)")

        if best_weights:
            torch.save({
                "model_state":  best_weights,
                "scaler":       scaler,
                "tickers":      ACTIVE_TICKERS,
                "hparams":      {"S": S, "F": 6, "N": args.hidden_n,
                                 "L": args.window_l, "l_max": args.l_max,
                                 "k": args.k_leaders},
                "test_metrics": {"IC": test_ic, "AR": test_ar, "SR": test_sr},
            }, save_dir / "deltalag_best.pt")
            print(f"\nBest model saved → {save_dir / 'deltalag_best.pt'}")

        pd.DataFrame(log_rows).to_csv(save_dir / "training_log.csv", index=False)
        daily_pnl.to_csv(save_dir / "daily_pnl.csv", header=["daily_return"])

    cleanup_distributed()


if __name__ == "__main__":
    main()