"""
Multi-step supervised rollout agent for ionospheric convection forecasting.

Replaces offline SAC with direct autoregressive supervision:

    For k = 0 … K-1:
        ŝ_{t+k+1} = ŝ_{t+k} + Decoder(Encoder(ŝ_{t+k}), Predictor(z_{t+k}))
        loss_k     = weighted_channel_loss(ŝ_{t+k+1}, s_{t+k+1})

    total_loss = (1/K) Σ_k w_k · loss_k

Two mechanisms address the train/inference distribution gap:

  Scheduled sampling  — with probability p_ss (annealed 0 → p_ss_max over
                        training) use the model's own ŝ as the next input
                        rather than the ground-truth s.  This exposes the
                        model to its own prediction errors during training.

  Curriculum on K     — start with K=init_rollout_steps (default 1), increase
                        by 1 every curriculum_step_epochs epochs up to
                        max_rollout_steps.  The model masters short horizons
                        before being asked to handle long ones.

Benchmark (SAC + one-sided CQL, ~30 epochs):
    val/skill_pers   (1-step, vs persistence)  ≈ 7.0 %   (peak epoch 27)
    val/skill_1step  (same metric, cleaner def) ≈ 5.7 %
    val/rmse                                   ≈ 0.658
"""

import math
import copy
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .networks import ConvEncoder, DeterministicPredictor, GridDecoder
from .reward   import compute_reward, persistence_reward

# Channel metadata — shared with agent.py
_CHANNEL_NAMES = (
    'obs_vel_north', 'obs_vel_east',
    'model_vel_north', 'model_vel_east',
    'soft_occ', 'boundary_dist',
)
_OBS_CHANNELS = {0, 1}
_CMAPS        = ('RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'Greys_r', 'PuOr')

# Per-channel loss weights matching reward.py
_CHANNEL_WEIGHTS = torch.tensor([1.5, 1.5, 1.0, 1.0, 0.3, 1.2], dtype=torch.float32)


# ── Time conditioning helper (identical to agent.py) ─────────────────────────

def _advance_time(time_vec: torch.Tensor, dt_hours: float = 2.0 / 60.0) -> torch.Tensor:
    two_pi = 2.0 * math.pi
    sin_ut = float(time_vec[0, 0])
    cos_ut = float(time_vec[0, 1])
    angle  = math.atan2(sin_ut, cos_ut) + two_pi * dt_hours / 24.0
    return torch.tensor(
        [[math.sin(angle), math.cos(angle), float(time_vec[0, 2]), float(time_vec[0, 3])]],
        dtype=time_vec.dtype, device=time_vec.device,
    )


# ── Main agent ────────────────────────────────────────────────────────────────

class MultiStepForecastAgent(pl.LightningModule):
    """
    Supervised multi-step rollout forecaster.

    Batch format from MultiStepDataModule (MultiStepDataset):
        states    (B, max_rollout_steps+1, 6, H, W)   frames t … t+K_max
        time_vecs (B, max_rollout_steps+1, 4)          time features per frame

    Constructor args
    ─────────────────
    grid_size              spatial H=W of the polar grid
    in_channels            channels per frame (default 6)
    latent_dim             ConvEncoder output dimension
    action_latent_dim      predictor / decoder action dimension
    base_channels          encoder base channel count
    max_rollout_steps      maximum rollout length K (must match datamodule)
    init_rollout_steps     starting K for curriculum (1 = one-step supervised)
    curriculum_step_epochs increase K by 1 every N epochs
    p_ss_max               maximum scheduled-sampling probability
    p_ss_anneal_epochs     epochs over which p_ss is linearly annealed 0→p_ss_max
    step_weight_scheme     'uniform' | 'geometric'
    gamma                  discount for geometric step weights
    tv_weight              total-variation penalty weight on obs channels
    lr                     Adam learning rate (single optimiser)
    warmup_steps           linear LR warmup steps
    """

    automatic_optimization = False

    def __init__(
        self,
        grid_size:             int   = 300,
        in_channels:           int   = 6,
        latent_dim:            int   = 256,
        action_latent_dim:     int   = 128,
        base_channels:         int   = 64,
        max_rollout_steps:     int   = 12,
        init_rollout_steps:    int   = 1,
        curriculum_step_epochs: int  = 5,
        p_ss_max:              float = 0.5,
        p_ss_anneal_epochs:    int   = 20,
        step_weight_scheme:    str   = 'uniform',
        gamma:                 float = 0.99,
        tv_weight:             float = 0.01,
        lr:                    float = 3e-4,
        warmup_steps:          int   = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.max_rollout_steps      = max_rollout_steps
        self.init_rollout_steps     = init_rollout_steps
        self.curriculum_step_epochs = curriculum_step_epochs
        self.p_ss_max               = p_ss_max
        self.p_ss_anneal_epochs     = p_ss_anneal_epochs
        self.step_weight_scheme     = step_weight_scheme
        self.gamma                  = gamma
        self.tv_weight              = tv_weight
        self.warmup_steps           = warmup_steps

        # Runtime curriculum state
        self._current_k   = init_rollout_steps
        self._current_p_ss = 0.0

        # Networks
        self.encoder   = ConvEncoder(in_channels, latent_dim, base_channels, time_dim=4)
        self.predictor = DeterministicPredictor(latent_dim, action_latent_dim)
        self.decoder   = GridDecoder(
            self.encoder.feat_channels, action_latent_dim, in_channels, base_channels
        )

        self.register_buffer('_channel_weights', _CHANNEL_WEIGHTS)

    # ── Step weights ──────────────────────────────────────────────────────────

    def _step_weights(self, K: int) -> list:
        if self.step_weight_scheme == 'geometric':
            ws = [self.gamma ** k for k in range(K)]
            s  = sum(ws)
            return [w / s for w in ws]
        # uniform
        return [1.0 / K] * K

    # ── Per-step loss ─────────────────────────────────────────────────────────

    def _step_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Channel-weighted Smooth-L1 loss (beta=0.1) over the full spatial grid
        for all channels.

        Obs channels (0-1) are NOT masked to covered cells: outside radar
        coverage they are defined as zero in both s_t and s_{t+1}, so the
        model SHOULD learn to predict delta ≈ 0 there.  Masking removes that
        gradient signal and allows arbitrary wrong deltas in unobserved
        regions, which catastrophically hurts the full-grid skill metric.

        TV smoothness penalty is still applied only inside covered cells so it
        doesn't penalise the (correctly near-zero) unobserved region.
        """
        occ_mask = (target[:, 4:5] > 0.05).float()          # (B, 1, H, W)
        w        = self._channel_weights.view(1, -1, 1, 1)

        raw  = F.smooth_l1_loss(pred, target, reduction='none', beta=0.1)
        loss = (raw * w).mean()                              # full grid, all channels

        # TV smoothness on predicted obs velocity in covered cells only
        pred_obs = pred[:, :2] * occ_mask
        tv_h     = (pred_obs[:, :, 1:, :] - pred_obs[:, :, :-1, :]).pow(2).mean()
        tv_w     = (pred_obs[:, :, :, 1:] - pred_obs[:, :, :, :-1]).pow(2).mean()

        return loss + self.tv_weight * (tv_h + tv_w)

    # ── Single autoregressive forward ─────────────────────────────────────────

    def _rollout_step(self, s: torch.Tensor,
                      time_vec: torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
        """One encoder→predictor→decoder step. Returns (s_next, delta)."""
        z, feats  = self.encoder(s, time_vec)
        a_lat     = self.predictor(z)
        delta     = self.decoder(feats, a_lat, s.shape[-2:])
        return s + delta, delta

    # ── Curriculum / scheduled-sampling update ────────────────────────────────

    def on_train_epoch_start(self):
        epoch = self.current_epoch

        # Curriculum: increase K every N epochs
        new_k = min(
            self.init_rollout_steps + epoch // self.curriculum_step_epochs,
            self.max_rollout_steps,
        )
        if new_k != self._current_k:
            print(f"\n[curriculum] K {self._current_k} → {new_k}  "
                  f"(epoch {epoch})")
            self._current_k = new_k

        # Scheduled sampling: linear anneal 0 → p_ss_max
        self._current_p_ss = min(
            self.p_ss_max * epoch / max(1, self.p_ss_anneal_epochs),
            self.p_ss_max,
        )

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        states, time_vecs = batch       # (B, K+1, C, H, W), (B, K+1, 4)
        B, _, C, H, W = states.shape
        K    = self._current_k
        p_ss = self._current_p_ss
        ws   = self._step_weights(K)

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        s_curr      = states[:, 0]
        total_loss  = torch.tensor(0.0, device=self.device, dtype=states.dtype)
        step_losses = []

        for k in range(K):
            s_next, _delta = self._rollout_step(s_curr, time_vecs[:, k])
            loss_k = self._step_loss(s_next, states[:, k + 1])
            step_losses.append(loss_k.detach())
            total_loss = total_loss + ws[k] * loss_k

            # Scheduled sampling: use own prediction or ground truth as next input
            if k < K - 1:
                if torch.rand(1, device=self.device).item() < p_ss:
                    s_curr = s_next.detach()
                else:
                    s_curr = states[:, k + 1]

        self.manual_backward(total_loss)
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()
        self.lr_schedulers().step()

        self.log('train/loss',                total_loss, prog_bar=True, on_step=True)
        self.log('train/rollout_k',           float(K),                  on_step=True)
        self.log('train/p_scheduled_sampling', p_ss,                     on_step=True)
        self.log('train/lr', self.lr_schedulers().get_last_lr()[0],      on_step=True)
        for k, l in enumerate(step_losses):
            self.log(f'train/loss_step_{k + 1}', l, on_step=True)

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        states, time_vecs = batch       # (B, K_max+1, C, H, W), (B, K_max+1, 4)
        B, _, C, H, W = states.shape
        s0 = states[:, 0]              # reference state for persistence baseline

        with torch.no_grad():
            s_curr = s0

            for k in range(self.max_rollout_steps):
                s_next, delta = self._rollout_step(s_curr, time_vecs[:, k])
                s_true        = states[:, k + 1]

                mse      = F.mse_loss(s_next, s_true)
                pers_mse = F.mse_loss(s0,     s_true)
                skill    = 1.0 - mse / pers_mse.clamp(min=1e-8)

                self.log(f'val/rmse_step_{k + 1}',  mse.sqrt(), sync_dist=True)
                self.log(f'val/skill_step_{k + 1}', skill,      sync_dist=True)

                # Step-1 metrics kept for backward compat + progress bar
                if k == 0:
                    self.log('val/rmse',       mse.sqrt(), prog_bar=True, sync_dist=True)
                    self.log('val/skill_pers', skill,      prog_bar=True, sync_dist=True)
                    self.log('val/skill_1step', skill,     prog_bar=True, sync_dist=True)

                    if not self.trainer.sanity_checking:
                        r_policy = compute_reward(s0, delta, s_true).mean()
                        r_pers   = persistence_reward(s0, s_true).mean()
                        self.log('val/r_policy', r_policy,           sync_dist=True)
                        self.log('val/r_pers',   r_pers,             sync_dist=True)
                        self.log('val/r_delta',  r_policy - r_pers,  sync_dist=True)

                    # Per-channel breakdown at step 1
                    if not self.trainer.sanity_checking:
                        occ_mask = (s_true[:, 4:5] > 0.05).float()
                        for ch, name in enumerate(_CHANNEL_NAMES):
                            y_ch    = s_true[:, ch]
                            pred_ch = s_next[:, ch]
                            pers_ch = s0[:, ch]

                            if ch in _OBS_CHANNELS:
                                mask  = occ_mask[:, 0]
                                denom = mask.sum().clamp(min=1.0)
                                se_p  = ((pred_ch - y_ch).pow(2) * mask).sum() / denom
                                se_b  = ((pers_ch - y_ch).pow(2) * mask).sum() / denom
                                bias  = ((pred_ch - y_ch)          * mask).sum() / denom
                            else:
                                se_p = (pred_ch - y_ch).pow(2).mean()
                                se_b = (pers_ch - y_ch).pow(2).mean()
                                bias = (pred_ch - y_ch).mean()

                            self.log(f'val/rmse_{name}',      se_p.sqrt(), sync_dist=True)
                            self.log(f'val/pers_rmse_{name}', se_b.sqrt(), sync_dist=True)
                            self.log(f'val/skill_{name}',     1.0 - se_p / se_b.clamp(min=1e-8),
                                     sync_dist=True)
                            self.log(f'val/bias_{name}',      bias,        sync_dist=True)

                # Final-step metrics (max horizon)
                if k == self.max_rollout_steps - 1:
                    self.log('val/rmse_final',  mse.sqrt(), prog_bar=True, sync_dist=True)
                    self.log('val/skill_final', skill,      prog_bar=True, sync_dist=True)

                # Cache seed for visualisation
                if batch_idx == 0 and self.trainer.is_global_zero and k == 0:
                    self._val_vis_seed = (
                        s0[0:1].detach(),
                        time_vecs[0:1, 0].detach(),
                        states[0, self.max_rollout_steps].detach().cpu(),
                    )

                s_curr = s_next     # always use own predictions in validation

    # ── Epoch-end summary ─────────────────────────────────────────────────────

    def on_validation_epoch_end(self):
        log_root = self.trainer.log_dir
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        m = self.trainer.callback_metrics

        def _f(key: str) -> str:
            v = m.get(key)
            return '  n/a  ' if v is None else f'{float(v):7.4f}'

        K = self._current_k
        header = (
            f"\n{'─'*72}\n"
            f"Multi-step forecast epoch {self.current_epoch:>3d}  "
            f"(K={K}/{self.max_rollout_steps}, SS={self._current_p_ss:.2f})\n"
            f"  1-step  RMSE {_f('val/rmse')}  Skill {_f('val/skill_pers')}   "
            f"Final-step RMSE {_f('val/rmse_final')}  Skill {_f('val/skill_final')}\n"
            f"  Reward policy {_f('val/r_policy')}  Pers {_f('val/r_pers')}  "
            f"Δreward {_f('val/r_delta')}\n"
        )
        col  = f"  {'Channel':<20} {'RMSE':>9} {'PersRMSE':>9} {'Skill':>7} {'Bias':>9}"
        rows = [col, "  " + "-"*58]
        for name in _CHANNEL_NAMES:
            note = " [masked]" if name in ('obs_vel_north', 'obs_vel_east') else ""
            rows.append(
                f"  {name:<20}"
                f" {_f(f'val/rmse_{name}'):>9}"
                f" {_f(f'val/pers_rmse_{name}'):>9}"
                f" {_f(f'val/skill_{name}'):>7}"
                f" {_f(f'val/bias_{name}'):>9}"
                f"{note}"
            )
        # Per-step skill summary row
        step_row = "  K-step skills:"
        for k in range(1, self.max_rollout_steps + 1):
            v = m.get(f'val/skill_step_{k}')
            step_row += f"  k{k}={float(v):.3f}" if v is not None else f"  k{k}=n/a"
        rows.append(step_row)
        rows.append("─"*72)
        print(header + "\n".join(rows))

        # Autoregressive rollout visualisation (rank 0 only)
        if hasattr(self, '_val_vis_seed'):
            s_seed, time_seed, s_tn_cpu = self._val_vis_seed
            with torch.no_grad():
                rollout   = [s_seed[0].cpu()]
                s_roll    = s_seed
                time_roll = time_seed
                for _ in range(self.max_rollout_steps):
                    z_r, feats_r = self.encoder(s_roll, time_roll)
                    a_r          = self.predictor(z_r)
                    d_r          = self.decoder(feats_r, a_r, s_roll.shape[-2:])
                    s_roll       = s_roll + d_r
                    time_roll    = _advance_time(time_roll)
                    rollout.append(s_roll[0].cpu())
            self._log_forecast_figure(rollout, s_tn_cpu, log_root=log_root)

    # ── Visualisation (adapted from agent.py) ────────────────────────────────

    @staticmethod
    def _polar_cap_axes_setup(ax, H, W, min_mlat=50.0):
        import numpy as np
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        R      = min(H, W) / 2.0
        ys, xs = np.ogrid[:H, :W]
        outside = ((xs - cx)**2 + (ys - cy)**2) > R**2
        theta   = np.linspace(0, 2*math.pi, 360)
        for mlat in range(int(min_mlat)+10, 90, 10):
            frac = (90.0 - mlat) / (90.0 - min_mlat)
            r_px = frac * R
            ax.plot(cx + r_px*np.sin(theta), cy - r_px*np.cos(theta),
                    color='white', lw=0.5, alpha=0.5, zorder=3)
            ax.text(cx, cy - r_px + 2, f'{mlat}°',
                    fontsize=5, color='white', ha='center', va='bottom',
                    alpha=0.7, zorder=4)
        ax.plot(cx + R*np.sin(theta), cy - R*np.cos(theta), 'k-', lw=1.2, zorder=4)
        for label, deg in [('00', 0), ('06', 90), ('12', 180), ('18', 270)]:
            rad = math.radians(deg)
            dx, dy = math.sin(rad), -math.cos(rad)
            ax.plot([cx, cx+R*dx], [cy, cy+R*dy],
                    color='white', lw=0.5, alpha=0.5, zorder=3)
            ax.text(cx+(R+6)*dx, cy+(R+6)*dy, label,
                    fontsize=6, color='k', ha='center', va='center', zorder=4)
        ax.set_xlim(-2, W+2); ax.set_ylim(H+2, -2)
        ax.set_xticks([]); ax.set_yticks([])
        return outside

    def _log_forecast_figure(self, rollout, s_tn, min_mlat=50.0, log_root=None):
        import os
        rollout_np = [t.numpy().copy() for t in rollout]
        stn_np     = s_tn.numpy().copy()
        epoch      = self.current_epoch
        n_steps    = self.max_rollout_steps
        global_step = self.global_step
        if log_root is None:
            log_root = self.trainer.default_root_dir

        def _worker():
            try:
                import matplotlib, numpy as np
                if matplotlib.get_backend().lower() != 'agg':
                    matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                self._render_forecast_figures(
                    rollout_np, stn_np, min_mlat,
                    plt, np, os, log_root, epoch, n_steps, global_step,
                )
            except Exception as exc:
                print(f"  [vis] figure generation failed ({exc!r}) — skipping")
            finally:
                try:
                    import matplotlib.pyplot as plt; plt.close('all')
                except Exception: pass

        threading.Thread(target=_worker, daemon=True).start()

    def _render_forecast_figures(self, rollout, s_tn, min_mlat, plt, np, os,
                                 log_root, epoch, n_steps, global_step):
        s    = rollout[0]
        pred = rollout[-1]
        H, W = s.shape[-2:]
        occ_np = s[4]

        step1 = max(1, n_steps // 3)
        step2 = max(step1+1, 2*n_steps//3)
        step1 = min(step1, len(rollout)-1)
        step2 = min(step2, len(rollout)-1)

        col_labels = ['Current  s_t', f't+{step1}', f't+{step2}',
                      f'Predicted  t+{n_steps}', f'Actual  t+{n_steps}',
                      'Error  pred − actual']
        fig1, axes1 = plt.subplots(len(_CHANNEL_NAMES), 6,
                                   figsize=(18, len(_CHANNEL_NAMES)*3.0),
                                   squeeze=False, facecolor='#1a1a2e')
        fig1.suptitle(f"Polar-cap forecast — Epoch {epoch}  (K={n_steps})",
                      fontsize=12, fontweight='bold', color='white', y=1.005)
        for col, label in enumerate(col_labels):
            axes1[0, col].set_title(label, fontsize=8, fontweight='bold', color='white')

        for ch, (name, cmap) in enumerate(zip(_CHANNEL_NAMES, _CMAPS)):
            s_np    = s[ch].copy()
            mid1_np = rollout[step1][ch].copy()
            mid2_np = rollout[step2][ch].copy()
            pred_np = pred[ch].copy()
            stn_np  = s_tn[ch].copy()
            err_np  = pred_np - stn_np

            outside = self._polar_cap_axes_setup(axes1[ch, 0], H, W, min_mlat)
            for arr in (s_np, mid1_np, mid2_np, pred_np, stn_np, err_np):
                arr[outside] = np.nan

            if cmap != 'Greys_r':
                valid = np.concatenate([s_np[~outside], pred_np[~outside], stn_np[~outside]])
                vmax  = max(float(np.nanpercentile(np.abs(valid), 98)), 1e-6)
                vmin  = -vmax
            else:
                vmin, vmax = 0.0, 1.0
            err_vmax = max(float(np.nanpercentile(np.abs(err_np[~outside]), 98)), 1e-6)

            panels = [
                (s_np,    cmap,     vmin,      vmax),
                (mid1_np, cmap,     vmin,      vmax),
                (mid2_np, cmap,     vmin,      vmax),
                (pred_np, cmap,     vmin,      vmax),
                (stn_np,  cmap,     vmin,      vmax),
                (err_np,  'RdBu_r', -err_vmax, err_vmax),
            ]
            for col, (data, cm, lo, hi) in enumerate(panels):
                ax = axes1[ch, col]
                ax.set_facecolor('#1a1a2e')
                im = ax.imshow(data, cmap=cm, vmin=lo, vmax=hi,
                               origin='upper', interpolation='nearest',
                               extent=[0, W, H, 0])
                ax.contour(occ_np, levels=[0.05], colors='yellow',
                           linewidths=0.7, alpha=0.6, zorder=5)
                if col > 0:
                    self._polar_cap_axes_setup(ax, H, W, min_mlat)
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6, colors='white')
                cb.outline.set_edgecolor('white')
            axes1[ch, 0].set_ylabel(name, fontsize=8, rotation=0,
                                    ha='right', va='center', labelpad=65, color='white')

        fig1.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()

        # E×B quiver figure
        stride = max(1, min(H, W) // 25)
        ys_q   = np.arange(stride//2, H, stride)
        xs_q   = np.arange(stride//2, W, stride)
        Xq, Yq = np.meshgrid(xs_q, ys_q)
        cy, cx  = (H-1)/2.0, (W-1)/2.0
        R       = min(H, W)/2.0
        in_cap  = ((Xq-cx)**2 + (Yq-cy)**2) <= R**2

        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8), squeeze=False,
                                   facecolor='#1a1a2e')
        fig2.suptitle(f"E×B drift vectors — Epoch {epoch}  (K={n_steps})",
                      fontsize=11, fontweight='bold', color='white')
        for col, lbl in enumerate(['Actual', 'Predicted', 'Error (pred − actual)']):
            axes2[0, col].set_title(lbl, fontsize=9, color='white', fontweight='bold')

        for row, (n_idx, e_idx, pair_label) in enumerate(
                [(0, 1, 'Observed E×B'), (2, 3, 'Model E×B')]):
            vn_act  = s_tn[n_idx]; ve_act  = s_tn[e_idx]
            vn_pred = pred[n_idx]; ve_pred = pred[e_idx]
            covered = np.sqrt(vn_act**2 + ve_act**2)[occ_np > 0.05]
            speed_max = float(np.nanpercentile(covered, 98)) if len(covered) > 0 else 1.0
            speed_max = speed_max if np.isfinite(speed_max) and speed_max > 0 else 1.0

            for col, (vn, ve) in enumerate(
                    [(vn_act, ve_act), (vn_pred, ve_pred),
                     (vn_pred-vn_act, ve_pred-ve_act)]):
                ax = axes2[row, col]
                ax.set_facecolor('#1a1a2e')
                outside = self._polar_cap_axes_setup(ax, H, W, min_mlat)
                speed   = np.sqrt(vn**2 + ve**2); speed[outside] = np.nan
                vlim    = speed_max if col < 2 else speed_max*0.5
                im = ax.imshow(speed, cmap='plasma', vmin=0, vmax=vlim,
                               origin='upper', interpolation='nearest',
                               extent=[0, W, H, 0], alpha=0.6)
                U =  ve[ys_q[:, None], xs_q[None, :]].copy()
                V = -vn[ys_q[:, None], xs_q[None, :]].copy()
                U[~in_cap] = np.nan; V[~in_cap] = np.nan
                ax.quiver(Xq, Yq, U, V, scale=speed_max*stride*1.5,
                          color='white', alpha=0.85, width=0.003,
                          headwidth=4, headlength=4, zorder=6)
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6, colors='white')
                cb.outline.set_edgecolor('white')
            axes2[row, 0].set_ylabel(pair_label, fontsize=8, color='white',
                                     rotation=90, va='center')

        fig2.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()

        # Save / W&B dispatch
        save_dir = os.path.join(log_root, 'val_vis')
        os.makedirs(save_dir, exist_ok=True)
        logged = False
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({'val/polar_cap_channels': wandb.Image(fig1),
                           'val/velocity_vectors':   wandb.Image(fig2)},
                          step=global_step)
                logged = True
        except Exception: pass
        if not logged:
            p1 = os.path.join(save_dir, f'epoch_{epoch:03d}_channels.png')
            p2 = os.path.join(save_dir, f'epoch_{epoch:03d}_vectors.png')
            fig1.savefig(p1, dpi=120, bbox_inches='tight', facecolor=fig1.get_facecolor())
            fig2.savefig(p2, dpi=120, bbox_inches='tight', facecolor=fig2.get_facecolor())
            print(f"  [vis] {p1}\n  [vis] {p2}")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x_last: torch.Tensor,
                time_vec: 'torch.Tensor | None' = None,
                n_steps: int = 1) -> 'list[torch.Tensor]':
        """
        Autoregressively predict n_steps frames ahead.
        Returns a list [ŝ_{t+1}, …, ŝ_{t+n_steps}].
        """
        squeeze = x_last.ndim == 3
        if squeeze:
            x_last = x_last.unsqueeze(0)
        if time_vec is not None and time_vec.ndim == 1:
            time_vec = time_vec.unsqueeze(0)

        preds   = []
        s_curr  = x_last
        t_curr  = time_vec
        for _ in range(n_steps):
            s_next, _ = self._rollout_step(s_curr, t_curr)
            preds.append(s_next.squeeze(0) if squeeze else s_next)
            s_curr = s_next
            if t_curr is not None:
                t_curr = _advance_time(t_curr)
        return preds

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),
                                lr=self.hparams.lr, weight_decay=1e-4)
        warmup = self.hparams.warmup_steps
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.05, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, 200_000 - warmup), eta_min=self.hparams.lr * 0.01)
        sch = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup_sched, cosine_sched], milestones=[warmup])
        return [opt], [sch]
