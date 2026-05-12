"""
Entry point for offline SAC ionospheric RL training.

Usage examples
──────────────
  # Basic run from pre-saved data directory
  python launch.py --data_dir /data/convmap_data/data/abc123 --batch_size 16

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

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    TQDMProgressBar, EarlyStopping, ModelCheckpoint, LearningRateMonitor,
)

from agent import SACOfflineAgent
from datamodule import RLDataModule

WANDB_PROJECT = "SuperDARN-RL"
WANDB_ENTITY  = "st7ma784"

torch.set_float32_matmul_precision('medium')


def train(args):
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
    # Data
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",           type=str,   required=True,
                   help="Pre-saved dataset directory (contains dataA_*.npy, dataB_*.npy)")
    g.add_argument("--batch_size",         type=int,   default=16)
    g.add_argument("--num_input_frames",   type=int,   default=1)
    g.add_argument("--temporal_agg_frames",type=int,   default=1)
    g.add_argument("--val_split",          type=float, default=0.1)
    g.add_argument("--grid_size",          type=int,   default=300)

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
    g.add_argument("--normalise_rewards",  action="store_true", default=True)

    # Training
    g = p.add_argument_group("Training")
    g.add_argument("--max_epochs",         type=int,   default=200)
    g.add_argument("--patience",           type=int,   default=20)
    g.add_argument("--accelerator",        type=str,   default="auto")
    g.add_argument("--devices",            type=str,   default="auto")
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

    train(args)
