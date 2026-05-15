"""
Entry point for offline SAC ionospheric RL training.

Usage examples
──────────────
  # Preprocess cnvmap files → numpy arrays (run once)
  python launch.py --preprocess --cnvmap_dir /data3/rst/extracted_data --data_dir /data2/rl_data --grid_size 120 --max_files 4000

  # Basic run from pre-saved data directory
  python launch.py --data_dir /data2/rl_data --batch_size 16

  # With W&B logging
  WANDB_API_KEY=xxx python launch.py --data_dir /data/... --wandb

  # SLURM (generates sbatch script, does not submit)
  python launch.py --data_dir /data/... --slurm

Hyperparameters map 1:1 to SACOfflineAgent and RLDataModule kwargs.
"""

import argparse
import datetime
import os
import sys
import time
import numpy as np

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor,
)
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

# Add parent directory to path so we can import the rl_forecast package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rl_forecast.agent import SACOfflineAgent
from rl_forecast.datamodule import RLDataModule

WANDB_PROJECT = "SuperDARN-RL"
WANDB_ENTITY  = "st7ma784"

torch.set_float32_matmul_precision('medium')

# Import preprocessing utilities from baseline
_ptl_path = os.path.join(os.path.dirname(__file__), '..', 'weatherlearn', 'PTL')
if _ptl_path not in sys.path:
    sys.path.insert(0, _ptl_path)
try:
    from run_baseline import preprocess_to_disk, record_to_grid
except ImportError:
    preprocess_to_disk = None
    record_to_grid = None


def preprocess_if_needed(cnvmap_dir, data_dir, grid_size, max_files=None, 
                         force=False, min_mlat=50.0, max_mlat=90.0):
    """
    Preprocess cnvmap files to numpy arrays if needed.
    
    Checks if preprocessed data exists; if not or if force=True, runs preprocessing.
    """
    os.makedirs(data_dir, exist_ok=True)
    shape_file = os.path.join(data_dir, "shape.txt")
    
    # Check if preprocessing already done
    if os.path.exists(shape_file) and not force:
        npy_files = [f for f in os.listdir(data_dir) if f.startswith("dataA_")]
        print(f"Cache found: {len(npy_files)} chunks in {data_dir}")
        return data_dir
    
    # Run preprocessing
    if preprocess_to_disk is None:
        raise RuntimeError(
            f"Could not import preprocess_to_disk from {_ptl_path}. "
            "Ensure the weatherlearn submodule is checked out and pydarnio is installed."
        )
    
    print(f"Preprocessing {cnvmap_dir} → {data_dir}")
    preprocess_to_disk(cnvmap_dir, data_dir, grid_size, max_files=max_files,
                       min_mlat=min_mlat, max_mlat=max_mlat)
    return data_dir


def train(args):
    os.environ['PYTHONUNBUFFERED'] = '1'

    # Build the DDP strategy object so PL actually uses the chosen backend.
    # Passing a string like 'ddp_find_unused_parameters_true' always picks NCCL;
    # constructing DDPStrategy explicitly is the only reliable way to select gloo.
    n_devices = torch.cuda.device_count() if args.devices == -1 else args.devices
    multi_gpu = n_devices > 1

    if multi_gpu:
        strategy = DDPStrategy(
            process_group_backend=args.backend,
            find_unused_parameters=True,
        )
        os.environ['NCCL_DEBUG'] = 'WARN'
    else:
        strategy = 'auto'

    print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
    print(f"[DEBUG] CUDA device count: {torch.cuda.device_count()}")
    print(f"[DEBUG] Requested devices: {args.devices}")
    print(f"[DEBUG] Strategy: DDPStrategy(backend={args.backend}, find_unused=True)" if multi_gpu else "[DEBUG] Strategy: auto (single GPU)")
    print(f"[DEBUG] Backend: {args.backend}")
    
    pl.seed_everything(args.seed, workers=True)

    datamodule = RLDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        gamma=args.gamma,
        num_input_frames=args.num_input_frames,
        temporal_agg_frames=args.temporal_agg_frames,
        val_split=args.val_split,
    )

    model = SACOfflineAgent(
        grid_size=args.grid_size,
        in_channels=6,
        latent_dim=args.latent_dim,
        action_latent_dim=args.action_latent_dim,
        base_channels=args.base_channels,
        n_steps=args.n_steps,
        gamma=args.gamma,
        tau=args.tau,
        alpha_init=args.alpha_init,
        cql_alpha=args.cql_alpha,
        bc_weight=args.bc_weight,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        actor_update_freq=args.actor_update_freq,
        warmup_steps=args.warmup_steps,
        compile_networks=args.compile,
        normalise_rewards=args.normalise_rewards,
    )

    run_name = "rl-{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logtool  = None

    if args.wandb:
        try:
            import wandb
            from pytorch_lightning.loggers import WandbLogger
            logtool = WandbLogger(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=run_name,
                save_dir=args.log_dir,
            )
        except ImportError:
            print("wandb not installed — running without logging")

    ckpt_dir = os.path.join(args.log_dir, run_name)
    callbacks = [
        TQDMProgressBar(refresh_rate=20),
        EarlyStopping(
            monitor="val/rmse", mode="min",
            patience=args.patience, check_finite=True,
        ),
        ModelCheckpoint(
            monitor="val/rmse", mode="min",
            dirpath=ckpt_dir,
            filename="rl-{epoch:03d}-{val/rmse:.4f}",
            save_top_k=2,
            save_last=True,
        ),
    ]
    if logtool is not None:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=strategy,
        logger=logtool,
        callbacks=callbacks,
        precision=args.precision,
        gradient_clip_val=None,   # manual clipping inside training_step
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
    )

    trainer.fit(model, datamodule)

    if logtool is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline SAC for ionospheric forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Preprocessing
    g = p.add_argument_group("Preprocessing")
    g.add_argument("--preprocess",       action="store_true",
                   help="Force re-preprocess cnvmap files even if cache exists")
    g.add_argument("--cnvmap_dir",       type=str, default=None,
                   help="Raw cnvmap data directory (required if --preprocess is used)")
    g.add_argument("--min_mlat",         type=float, default=50.0,
                   help="Minimum magnetic latitude for polar grid (degrees)")
    g.add_argument("--max_mlat",         type=float, default=90.0,
                   help="Maximum magnetic latitude for polar grid (degrees)")
    
    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",           type=str,   required=True,
                   help="Pre-saved dataset directory (contains dataA_*.npy, dataB_*.npy)")
    g.add_argument("--batch_size",         type=int,   default=16)
    g.add_argument("--num_input_frames",   type=int,   default=1)
    g.add_argument("--temporal_agg_frames",type=int,   default=1)
    g.add_argument("--val_split",          type=float, default=0.1)
    g.add_argument("--grid_size",          type=int,   default=300)
    g.add_argument("--max_files",          type=int,   default=None,
                   help="Max files to preprocess (None = all)")


    # Model
    g = p.add_argument_group("Model")
    g.add_argument("--latent_dim",         type=int,   default=256)
    g.add_argument("--action_latent_dim",  type=int,   default=128)
    g.add_argument("--base_channels",      type=int,   default=64)

    # RL hyperparameters
    g = p.add_argument_group("RL")
    g.add_argument("--n_steps",            type=int,   default=3,
                   help="N-step return horizon; must match datamodule")
    g.add_argument("--gamma",              type=float, default=0.99)
    g.add_argument("--tau",                type=float, default=0.005)
    g.add_argument("--alpha_init",         type=float, default=0.2)
    g.add_argument("--cql_alpha",          type=float, default=1.0)
    g.add_argument("--bc_weight",          type=float, default=0.5)
    g.add_argument("--actor_update_freq",  type=int,   default=2)

    # Optimiser
    g = p.add_argument_group("Optimiser")
    g.add_argument("--actor_lr",           type=float, default=3e-4)
    g.add_argument("--critic_lr",          type=float, default=3e-4)
    g.add_argument("--alpha_lr",           type=float, default=3e-4)
    g.add_argument("--warmup_steps",       type=int,   default=1000)
    g.add_argument("--compile",            action="store_true",
                   help="torch.compile the encoder and critic (GPU, PyTorch 2+)")
    g.add_argument("--normalise_rewards",  action="store_true", default=True,
                   help="Enable Welford reward normalization (recommended, default=True)")

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--max_epochs",         type=int,   default=200)
    g.add_argument("--patience",           type=int,   default=20)
    g.add_argument("--accelerator",        type=str,   default="gpu",
                   help="'gpu' for GPU, 'cpu' for CPU, 'auto' for auto-detect")
    g.add_argument("--devices",            type=int,   default=-1,
                   help="-1 = all GPUs, 1 = single GPU, or comma-separated list of GPU IDs")
    g.add_argument("--strategy",           type=str,   default="auto",
                   help="Distributed strategy: 'auto', 'ddp', 'ddp_find_unused_parameters_true'")
    g.add_argument("--backend",            type=str,   default="gloo",
                   help="DDP backend: 'gloo' (stable, works without NVLink), 'nccl' (faster but requires NVML)")
    g.add_argument("--precision",          type=str,   default="16-mixed")
    g.add_argument("--seed",               type=int,   default=42)
    g.add_argument("--fast_dev_run",       action="store_true")
    g.add_argument("--log_every_n_steps",  type=int,   default=20)

    # Logging / output
    g = p.add_argument_group("Logging")
    g.add_argument("--log_dir",            type=str,
                   default=os.path.join(os.getenv("global_scratch", "/data"), "rl_logs"))
    g.add_argument("--wandb",              action="store_true")
    g.add_argument("--slurm",              action="store_true",
                   help="Print an sbatch script and exit (does not submit)")

    return p


def _slurm_script(args) -> str:
    lines = [
        "#!/bin/bash",
        "#SBATCH --time=24:00:00",
        "#SBATCH --job-name=superdarn-rl",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --mem=96G",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --mail-type=END,FAIL",
        "#SBATCH --mail-user=st7ma784@gmail.com",
        "",
        "source /etc/profile",
        "module add opence",
        "conda activate $CONDADIR",
        "",
    ]
    script_path = os.path.realpath(sys.argv[0])
    arg_str = " ".join(
        f"--{k} {v}" for k, v in vars(args).items()
        if k not in ("slurm",) and v is not None and v is not False
    )
    lines.append(f"srun python {script_path} {arg_str}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.slurm:
        print(_slurm_script(args))
        sys.exit(0)

    # Preprocess if requested
    if args.preprocess:
        if not args.cnvmap_dir:
            parser.error("--cnvmap_dir is required when using --preprocess")
        if not os.path.exists(args.cnvmap_dir):
            parser.error(f"--cnvmap_dir does not exist: {args.cnvmap_dir}")
        preprocess_if_needed(args.cnvmap_dir, args.data_dir, args.grid_size, 
                             max_files=args.max_files, force=True,
                             min_mlat=args.min_mlat, max_mlat=args.max_mlat)
        print(f"Preprocessing complete. Preprocessed data saved to {args.data_dir}")
        sys.exit(0)

    train(args)
