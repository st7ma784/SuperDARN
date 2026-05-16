"""
Offline SAC agent with n-step returns and Conservative Q-Learning (CQL).

Key training optimisations over the naive implementation
─────────────────────────────────────────────────────────
1. Single encoder pass per step — s and s_next are concatenated and encoded
   together (one conv2d call instead of two), then split. Halves the dominant
   bottleneck.

2. N-step discounted returns — the datamodule pre-computes R_n = Σ γ^k r_{t+k}
   and supplies the bootstrap state s_{t+n}.  The critic target becomes
       Q_target = R_n + γ^n * (min_Q_tgt(s_{t+n}, a') - α * log π(a'|s_{t+n}))
   This gives much lower variance targets for a slow-varying physical system.

3. torch.compile — optional; wraps encoder + critic for graph compilation.
   Activated via `compile_networks=True` in the constructor.

4. LR scheduling — linear warmup over `warmup_steps` then cosine annealing
   for the remainder.  Stepped manually once per training batch.

5. Running reward normalisation — a Welford online mean/variance tracker
   normalises the batch reward before it enters the critic target, preventing
   Q-value scale drift when the reward function changes over training.

6. CQL conservative penalty — keeps Q-values lower for policy actions than
   for data actions, preventing out-of-distribution exploitation.

7. Behavioural cloning anchor — actor loss includes a SmoothL1 term between
   the decoded delta and the observed delta, keeping predictions physically
   plausible early in training.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .networks import ConvEncoder, ActionEncoder, LatentActor, GridDecoder, DoubleQCritic
from .reward import compute_reward, persistence_reward

# Ordered names for the 6-channel state tensor (matches DatasetFromPresaved)
#   0  obs_vel_north   — SH-fitted northward E×B drift in radar-covered cells (m/s)
#   1  obs_vel_east    — SH-fitted eastward  E×B drift in radar-covered cells (m/s)
#   2  model_vel_north — Weimer/TS96 background model northward drift (m/s)
#   3  model_vel_east  — Weimer/TS96 background model eastward  drift (m/s)
#   4  soft_occ        — soft radar-coverage fraction [0, 1]
#   5  boundary_dist   — signed mlat offset from Heppner-Maynard boundary (deg)
_CHANNEL_NAMES = (
    'obs_vel_north', 'obs_vel_east',
    'model_vel_north', 'model_vel_east',
    'soft_occ', 'boundary_dist',
)
# Channels 0-1 are only meaningful inside radar-covered cells; evaluate masked.
_OBS_CHANNELS = {0, 1}
# Colormaps per channel: velocity→diverging, occupancy→sequential, boundary→diverging
_CMAPS = ('RdBu_r', 'RdBu_r', 'RdBu_r', 'RdBu_r', 'Greys_r', 'PuOr')


# ── Online Welford reward normaliser ─────────────────────────────────────────

class RunningMeanStd:
    """Welford online estimator; normalises tensors to zero mean unit variance."""

    def __init__(self, eps: float = 1e-4):
        self.mean  = 0.0
        self.var   = 1.0
        self.count = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        batch_mean  = x.float().mean().item()
        batch_var   = x.float().var().item() if x.numel() > 1 else 0.0
        batch_count = x.numel()
        delta       = batch_mean - self.mean
        total       = self.count + batch_count
        self.mean  += delta * batch_count / total
        m_a         = self.var   * self.count
        m_b         = batch_var  * batch_count
        self.var    = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count  = total

    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var ** 0.5 + 1e-8)


# ── Main agent ────────────────────────────────────────────────────────────────

class SACOfflineAgent(pl.LightningModule):
    """
    Offline SAC + CQL for ionospheric convection forecasting.

    Batch format from RLDataModule (NStepRLTransitionDataset):
        s_t     (B, 6, H, W)   current frame
        a_t     (B, 6, H, W)   observed delta (behaviour action)
        R_n     (B,)            n-step discounted return
        s_tn    (B, 6, H, W)   bootstrap frame at t+n
        done    (B, 1)          always 0 for offline data

    Constructor args
    ─────────────────
    grid_size          spatial H=W of the polar grid
    in_channels        channels per frame (default 6)
    latent_dim         ConvEncoder output dimension
    action_latent_dim  actor / critic action dimension
    base_channels      encoder base channel count
    n_steps            must match datamodule n_steps (used for γ^n bootstrap)
    gamma              RL discount factor
    tau                soft target-network update rate
    alpha_init         initial SAC temperature
    target_entropy_scale  target_entropy = -scale * action_latent_dim
    cql_alpha          weight of the CQL conservative term
    bc_weight          weight of the BC decoder anchor in actor loss
    actor_lr           Adam lr for actor + decoder
    critic_lr          Adam lr for encoder + action_enc + critics
    alpha_lr           Adam lr for log_alpha
    actor_update_freq  update actor every N critic steps
    warmup_steps       linear LR warmup length (steps)
    compile_networks   whether to torch.compile encoder + critic
    normalise_rewards  whether to apply running Welford normalisation
    """

    automatic_optimization = False  # three separate optimisers

    def __init__(
        self,
        grid_size:            int   = 300,
        in_channels:          int   = 6,
        latent_dim:           int   = 256,
        action_latent_dim:    int   = 128,
        base_channels:        int   = 64,
        n_steps:              int   = 3,
        gamma:                float = 0.99,
        tau:                  float = 0.005,
        alpha_init:           float = 0.2,
        target_entropy_scale: float = 0.98,
        cql_alpha:            float = 1.0,
        bc_weight:            float = 0.5,
        actor_lr:             float = 3e-4,
        critic_lr:            float = 3e-4,
        alpha_lr:             float = 3e-4,
        actor_update_freq:    int   = 2,
        warmup_steps:         int   = 1000,
        compile_networks:     bool  = False,
        normalise_rewards:    bool  = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.grid_size         = grid_size
        self.n_steps           = n_steps
        self.gamma             = gamma
        self.gamma_n           = gamma ** n_steps    # γ^n for bootstrap
        self.tau               = tau
        self.cql_alpha         = cql_alpha
        self.bc_weight         = bc_weight
        self.actor_update_freq = actor_update_freq
        self.warmup_steps      = warmup_steps
        self.normalise_rewards = normalise_rewards

        # ── Networks ──────────────────────────────────────────────────────
        self.encoder       = ConvEncoder(in_channels, latent_dim, base_channels)
        self.action_enc    = ActionEncoder(in_channels, action_latent_dim)
        self.actor         = LatentActor(latent_dim, action_latent_dim)
        self.decoder       = GridDecoder(
            self.encoder.feat_channels, action_latent_dim, in_channels, base_channels
        )
        self.critic        = DoubleQCritic(latent_dim, action_latent_dim)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        if compile_networks:
            self.encoder = torch.compile(self.encoder)
            self.critic  = torch.compile(self.critic)
            # critic_target compiled separately so it stays in sync
            self.critic_target = torch.compile(copy.deepcopy(
                self.critic._orig_mod if hasattr(self.critic, '_orig_mod') else self.critic
            ))
            for p in self.critic_target.parameters():
                p.requires_grad_(False)

        # ── Entropy temperature ────────────────────────────────────────────
        self.log_alpha      = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.target_entropy = float(-target_entropy_scale * action_latent_dim)

        # ── Reward normaliser ──────────────────────────────────────────────
        self._reward_rms    = RunningMeanStd()

        # ── Step counter for actor update frequency ────────────────────────
        self._n_critic_steps = 0

    # ── Utility ───────────────────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.tau
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def _unpack(self, batch):
        s, a_data, r_n, s_tn, done = batch
        return s, a_data, r_n.float(), s_tn, done.float()

    # ── Optimised encoder: single forward pass for both s and s_next ──────────

    def _encode_pair(self, s: torch.Tensor, s_next: torch.Tensor):
        """
        Encodes s and s_next in a single conv pass by batching them together.
        Returns (z_s, feats_s, z_next) — feats_next is not needed downstream.
        """
        both   = torch.cat([s, s_next], dim=0)       # (2B, C, H, W)
        z_both, feats_both = self.encoder(both)
        B      = s.shape[0]
        z_s,    z_next    = z_both[:B],    z_both[B:]
        feats_s            = feats_both[:B]
        return z_s, feats_s, z_next

    # ── Critic step ───────────────────────────────────────────────────────────

    def _critic_loss(
        self,
        s:      torch.Tensor,
        a_data: torch.Tensor,
        r_n:    torch.Tensor,
        s_tn:   torch.Tensor,
        done:   torch.Tensor,
    ):
        # Log raw rewards before normalization
        with torch.no_grad():
            self.log('train/raw_reward_mean', r_n.mean(), on_step=True, on_epoch=False)
            self.log('train/raw_reward_std',  r_n.std(),  on_step=True, on_epoch=False)
            self.log('train/raw_reward_min',  r_n.min(),  on_step=True, on_epoch=False)
            self.log('train/raw_reward_max',  r_n.max(),  on_step=True, on_epoch=False)
        
        if self.normalise_rewards:
            self._reward_rms.update(r_n)
            r_n_normalized = self._reward_rms.normalise(r_n)
            with torch.no_grad():
                self.log('train/norm_reward_mean', r_n_normalized.mean(), on_step=True, on_epoch=False)
                self.log('train/norm_reward_std',  r_n_normalized.std(),  on_step=True, on_epoch=False)
            r_n = r_n_normalized
        else:
            with torch.no_grad():
                self.log('train/norm_reward_mean', torch.tensor(0.0), on_step=True, on_epoch=False)
                self.log('train/norm_reward_std',  torch.tensor(0.0),  on_step=True, on_epoch=False)

        # Single encoder pass for s and s_{t+n}
        z_s, feats, z_next = self._encode_pair(s, s_tn)

        a_data_lat = self.action_enc(a_data)

        with torch.no_grad():
            a_next, lp_next, _ = self.actor.sample(z_next)
            # N-step bootstrap: Q_tgt = R_n + γ^n * (Q_tgt(s_{t+n}, a') - α log π)
            q_next  = self.critic_target.q_min(z_next, a_next)
            q_target = (
                r_n.unsqueeze(1)
                + self.gamma_n * (1.0 - done) * (q_next - self.alpha * lp_next)
            )

        q1, q2   = self.critic(z_s, a_data_lat)
        
        # Use Huber loss instead of MSE to prevent Q-value divergence
        # Huber is quadratic for small errors (stable) but linear for large errors (robust)
        td_loss  = F.huber_loss(q1, q_target, delta=1.0) + F.huber_loss(q2, q_target, delta=1.0)
        
        # Log Q-value statistics to diagnose if clipping is the issue
        with torch.no_grad():
            self.log('train/q1_mean', q1.mean(), on_step=True, on_epoch=False)
            self.log('train/q1_std',  q1.std(),  on_step=True, on_epoch=False)
            self.log('train/q2_mean', q2.mean(), on_step=True, on_epoch=False)
            self.log('train/q2_std',  q2.std(),  on_step=True, on_epoch=False)
            self.log('train/q_target_mean', q_target.mean(), on_step=True, on_epoch=False)
            self.log('train/q_target_std',  q_target.std(),  on_step=True, on_epoch=False)
            # Check if Q-values are at the bounds
            self.log('train/q1_max', q1.max(), on_step=True, on_epoch=False)
            self.log('train/q1_min', q1.min(), on_step=True, on_epoch=False)

        # CQL: only apply after some warmup when Q-estimates are more stable
        # TEMPORARILY DISABLED: CQL is causing loss explosion; focus on TD learning first
        cql_penalty = torch.tensor(0.0, device=q1.device, dtype=q1.dtype)
        # step = self.global_step
        # if step >= self.warmup_steps // 2:
        #     with torch.no_grad():
        #         a_pi_cql, _, _ = self.actor.sample(z_s)
        #     q1_pi, q2_pi = self.critic(z_s, a_pi_cql)
        #     q1_pi = torch.clamp(q1_pi, min=-100.0, max=10.0)
        #     q2_pi = torch.clamp(q2_pi, min=-100.0, max=10.0)
        #     cql_penalty  = ((q1_pi - q1.detach()) + (q2_pi - q2.detach())).mean() * 0.5

        loss = td_loss + self.cql_alpha * cql_penalty
        return loss, td_loss.detach(), cql_penalty.detach(), z_s, feats

    # ── Actor step ────────────────────────────────────────────────────────────

    def _actor_loss(
        self,
        s:              torch.Tensor,
        a_data:         torch.Tensor,
        z_s_detached:   torch.Tensor,
        feats_detached: torch.Tensor,
    ):
        a_pi, log_pi, _ = self.actor.sample(z_s_detached)
        q_pi   = self.critic.q_min(z_s_detached, a_pi)
        # TEMPORARILY DISABLED: Clip Q-values to prevent explosion in actor loss
        # q_pi   = torch.clamp(q_pi, min=-100.0, max=10.0)
        
        # Clip alpha to prevent it from growing unbounded
        alpha_clipped = torch.clamp(self.alpha, min=1e-6, max=1.0)
        
        rl_loss = (alpha_clipped.detach() * log_pi - q_pi).mean()

        target_size = (self.grid_size, self.grid_size)
        delta_hat   = self.decoder(feats_detached, a_pi, target_size)
        bc_loss     = F.smooth_l1_loss(delta_hat, a_data, beta=0.1)

        loss = rl_loss + self.bc_weight * bc_loss
        return loss, log_pi.detach(), rl_loss.detach(), bc_loss.detach()

    # ── Alpha step ────────────────────────────────────────────────────────────

    def _alpha_loss(self, log_pi_detached: torch.Tensor) -> torch.Tensor:
        return -(self.log_alpha * (log_pi_detached + self.target_entropy)).mean()

    # ── Training step ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        opt_critic, opt_actor, opt_alpha = self.optimizers()
        sch_critic, sch_actor, sch_alpha = self.lr_schedulers()
        s, a_data, r_n, s_tn, done       = self._unpack(batch)

        # 1. Critic update (encoder included)
        opt_critic.zero_grad(set_to_none=True)
        c_loss, td_loss, cql_pen, z_s, feats = self._critic_loss(s, a_data, r_n, s_tn, done)
        self.manual_backward(c_loss)
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.action_enc.parameters()) +
            list(self.critic.parameters()),
            max_norm=1.0,
        )
        opt_critic.step()
        sch_critic.step()
        self._soft_update_targets()

        self.log('train/critic_loss', c_loss,   prog_bar=True, on_step=True)
        self.log('train/td_loss',     td_loss,  on_step=True)
        self.log('train/cql',         cql_pen,  on_step=True)
        self.log('train/lr_critic',   sch_critic.get_last_lr()[0], on_step=True)
        
        # Log diagnostics
        with torch.no_grad():
            self.log('train/reward_mean', r_n.mean(), on_step=True)
            self.log('train/reward_std',  r_n.std(),  on_step=True)

        # 2. Actor + alpha update (every actor_update_freq critic steps)
        self._n_critic_steps += 1
        if self._n_critic_steps % self.actor_update_freq == 0:
            opt_actor.zero_grad(set_to_none=True)
            a_loss, log_pi, rl_loss, bc_loss = self._actor_loss(
                s, a_data, z_s.detach(), feats.detach()
            )
            self.manual_backward(a_loss)
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0,
            )
            opt_actor.step()
            sch_actor.step()

            opt_alpha.zero_grad(set_to_none=True)
            alph_loss = self._alpha_loss(log_pi)
            self.manual_backward(alph_loss)
            opt_alpha.step()
            sch_alpha.step()

            self.log('train/actor_loss', a_loss,   prog_bar=True, on_step=True)
            self.log('train/rl_loss',    rl_loss,  on_step=True)
            self.log('train/bc_loss',    bc_loss,  on_step=True)
            self.log('train/alpha_loss', alph_loss,on_step=True)
            self.log('train/alpha',      self.alpha, on_step=True)
            self.log('train/lr_actor',   sch_actor.get_last_lr()[0], on_step=True)
            self.log('train/log_pi_mean', log_pi.mean(), on_step=True)

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        s, a_data, r_n, s_tn, done = self._unpack(batch)

        with torch.no_grad():
            z_s, feats = self.encoder(s)
            _, _, a_det = self.actor.sample(z_s)
            delta_hat   = self.decoder(feats, a_det, (self.grid_size, self.grid_size))
            y_hat       = s + delta_hat

            # Use s_tn as the forecast target.  For n_steps=1 this is the true
            # next frame; for n_steps>1 it is the state at t+n (persistence and
            # policy predictions are both one-step, so the skill comparison is
            # still well-defined relative to the same reference).
            mse      = F.mse_loss(y_hat, s_tn)
            pers_mse = F.mse_loss(s, s_tn)
            skill    = 1.0 - mse / pers_mse.clamp(min=1e-8)

            r_policy = compute_reward(s, delta_hat, s_tn).mean()
            r_pers   = persistence_reward(s, s_tn).mean()

            # ── Per-channel breakdown ──────────────────────────────────────
            # soft_occ (ch 4) tells us which cells are radar-constrained.
            # Obs-velocity channels are only evaluated inside covered cells.
            occ_mask = (s_tn[:, 4:5] > 0.05).float()   # (B, 1, H, W)

            # Cache one sample (CPU): autoregressive n-step rollout for vis.
            if batch_idx == 0 and self.trainer.is_global_zero:
                rollout = [s[0].detach().cpu()]
                s_roll = s[0:1]
                for _ in range(self.n_steps):
                    z_r, feats_r = self.encoder(s_roll)
                    _, _, a_r    = self.actor.sample(z_r)
                    d_r          = self.decoder(feats_r, a_r, (self.grid_size, self.grid_size))
                    s_roll       = (s_roll + d_r)
                    rollout.append(s_roll[0].detach().cpu())
                self._val_vis = (rollout, s_tn[0].detach().cpu())

            # Skip per-channel metrics during the sanity check — the extra
            # sync_dist all_reduces deadlock against PL's sanity-check metric
            # finalization path in DDP.
            if not self.trainer.sanity_checking:
                for ch, name in enumerate(_CHANNEL_NAMES):
                    y_ch    = s_tn[:, ch]    # (B, H, W)
                    pred_ch = y_hat[:, ch]
                    pers_ch = s[:, ch]

                    if ch in _OBS_CHANNELS:
                        mask  = occ_mask[:, 0]                       # (B, H, W)
                        denom = mask.sum().clamp(min=1.0)
                        se_p  = ((pred_ch - y_ch).pow(2) * mask).sum() / denom
                        se_b  = ((pers_ch - y_ch).pow(2) * mask).sum() / denom
                        bias  = ((pred_ch - y_ch) * mask).sum() / denom
                    else:
                        se_p = (pred_ch - y_ch).pow(2).mean()
                        se_b = (pers_ch - y_ch).pow(2).mean()
                        bias = (pred_ch - y_ch).mean()

                    skill_ch = 1.0 - se_p / se_b.clamp(min=1e-8)

                    self.log(f'val/rmse_{name}',      se_p.sqrt(),  sync_dist=True)
                    self.log(f'val/pers_rmse_{name}', se_b.sqrt(),  sync_dist=True)
                    self.log(f'val/skill_{name}',     skill_ch,     sync_dist=True)
                    self.log(f'val/bias_{name}',      bias,         sync_dist=True)

        self.log('val/rmse',         mse.sqrt(),         prog_bar=True, sync_dist=True)
        self.log('val/skill_pers',   skill,              prog_bar=True, sync_dist=True)
        self.log('val/r_policy',     r_policy,           sync_dist=True)
        self.log('val/r_pers',       r_pers,             sync_dist=True)
        self.log('val/r_delta',      r_policy - r_pers,  sync_dist=True)

    def on_validation_epoch_end(self):
        """Print a human-readable per-channel forecast summary to stdout (rank 0 only)."""
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        m = self.trainer.callback_metrics

        def _f(key: str) -> str:
            v = m.get(key)
            if v is None:
                return '  n/a  '
            return f'{float(v):7.4f}'

        header = (
            f"\n{'─'*72}\n"
            f"Validation epoch {self.current_epoch:>3d}  "
            f"(n_steps={self.n_steps}, γ={self.gamma})\n"
            f"  Overall  RMSE  {_f('val/rmse')}   Skill vs pers {_f('val/skill_pers')}\n"
            f"  Reward policy  {_f('val/r_policy')}   Reward pers   {_f('val/r_pers')}   "
            f"Δreward {_f('val/r_delta')}\n"
        )
        col = f"  {'Channel':<20} {'Pred RMSE':>9} {'Pers RMSE':>9} {'Skill':>7} {'Bias':>9}"
        rows = [col, "  " + "-" * 58]
        for name in _CHANNEL_NAMES:
            note = " [masked]" if name in ('obs_vel_north', 'obs_vel_east') else ""
            row = (
                f"  {name:<20}"
                f" {_f(f'val/rmse_{name}'):>9}"
                f" {_f(f'val/pers_rmse_{name}'):>9}"
                f" {_f(f'val/skill_{name}'):>7}"
                f" {_f(f'val/bias_{name}'):>9}"
                f"{note}"
            )
            rows.append(row)
        rows.append("─" * 72)
        print(header + "\n".join(rows))

        # Skip figure during sanity check — it would block rank-0 while other
        # DDP ranks move on, causing a deadlock.
        if (
            self.trainer.is_global_zero
            and hasattr(self, '_val_vis')
            and not self.trainer.sanity_checking
        ):
            rollout_vis, s_tn_vis = self._val_vis
            self._log_forecast_figure(rollout_vis, s_tn_vis)

    # ── Spatial forecast figures ──────────────────────────────────────────────

    @staticmethod
    def _polar_cap_axes_setup(ax, H, W, min_mlat=50.0):
        """
        Overlay a zenithal-equidistant polar cap grid on an imshow axis.

        Draws:
          - circular polar-cap boundary (at min_mlat)
          - concentric magnetic-latitude rings every 10°
          - MLT spokes at 0h/6h/12h/18h with labels
            (midnight at top, dawn 06h at right, standard SuperDARN convention)

        Returns a boolean array (H, W) that is True outside the polar cap so
        callers can NaN-mask their data before plotting.
        """
        import numpy as np
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        R = min(H, W) / 2.0          # pixels from centre to min_mlat

        # Outside-cap mask
        ys, xs = np.ogrid[:H, :W]
        outside = ((xs - cx) ** 2 + (ys - cy) ** 2) > R ** 2

        theta = np.linspace(0, 2 * np.pi, 360)

        # Latitude rings — every 10° from min_mlat inward
        for mlat in range(int(min_mlat) + 10, 90, 10):
            frac = (90.0 - mlat) / (90.0 - min_mlat)   # 0 at pole, 1 at boundary
            r_px = frac * R
            ax.plot(cx + r_px * np.sin(theta), cy - r_px * np.cos(theta),
                    color='white', lw=0.5, alpha=0.5, zorder=3)
            ax.text(cx, cy - r_px + 2, f'{mlat}°',
                    fontsize=5, color='white', ha='center', va='bottom',
                    alpha=0.7, zorder=4)

        # Polar-cap boundary circle
        ax.plot(cx + R * np.sin(theta), cy - R * np.cos(theta),
                'k-', lw=1.2, zorder=4)

        # MLT spokes and labels (midnight up = −y in image coords)
        mlt_labels = [('00', 0), ('06', 90), ('12', 180), ('18', 270)]
        for label, deg in mlt_labels:
            rad = math.radians(deg)
            dx, dy = math.sin(rad), -math.cos(rad)
            ax.plot([cx, cx + R * dx], [cy, cy + R * dy],
                    color='white', lw=0.5, alpha=0.5, zorder=3)
            ax.text(cx + (R + 6) * dx, cy + (R + 6) * dy, label,
                    fontsize=6, color='k', ha='center', va='center', zorder=4)

        ax.set_xlim(-2, W + 2)
        ax.set_ylim(H + 2, -2)
        ax.set_xticks([])
        ax.set_yticks([])
        return outside

    def _log_forecast_figure(
        self,
        rollout:  list,            # [s_t, ŝ_{t+1}, ..., ŝ_{t+n}] each (6, H, W) CPU
        s_tn:     'torch.Tensor',  # (6, H, W) actual state at t+n
        min_mlat: float = 50.0,
    ):
        """
        Produce two figures and dispatch to W&B or {log_dir}/val_vis/.

        Figure 1 — channel grid (6 rows × 6 cols):
            s_t | ŝ_{t+n//3} | ŝ_{t+2*n//3} | ŝ_{t+n}(pred) | s_{t+n}(actual) | Error
        Figure 2 — E×B drift quiver (obs + model, pred vs actual).
        """
        import threading, os

        rollout_np = [t.numpy().copy() for t in rollout]
        stn_np     = s_tn.numpy().copy()
        epoch      = self.current_epoch
        n_steps    = self.n_steps

        log_root = (
            getattr(self.trainer, 'log_dir', None)
            or self.trainer.default_root_dir
        )
        global_step = self.global_step

        def _worker():
            try:
                import matplotlib
                if matplotlib.get_backend().lower() != 'agg':
                    matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np
                self._render_forecast_figures(
                    rollout_np, stn_np, min_mlat,
                    plt, np, os, log_root, epoch, n_steps, global_step,
                )
            except Exception as exc:
                print(f"  [vis] figure generation failed ({exc!r}) — skipping")
            finally:
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except Exception:
                    pass

        # Daemon thread — never blocks training or DDP barriers
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _render_forecast_figures(
        self, rollout, s_tn, min_mlat, plt, np, os,
        log_root, epoch, n_steps, global_step,
    ):
        """Runs on a daemon thread — all required state passed explicitly.

        rollout: list of (6, H, W) numpy arrays [s_t, ŝ_{t+1}, ..., ŝ_{t+n}]
        s_tn:    (6, H, W) actual ground truth at t+n
        """
        s    = rollout[0]    # s_t
        pred = rollout[-1]   # ŝ_{t+n}
        H, W = s.shape[-2:]
        occ_np = s[4]

        # Two intermediate frames spaced across the rollout
        step1 = max(1, n_steps // 3)
        step2 = max(step1 + 1, 2 * n_steps // 3)
        step1 = min(step1, len(rollout) - 1)
        step2 = min(step2, len(rollout) - 1)

        # ── Figure 1: per-channel polar-cap grid (6 cols) ────────────────────
        # Col: s_t | ŝ_{step1} | ŝ_{step2} | ŝ_{t+n}(pred) | s_{t+n}(actual) | Error
        col_labels = [
            'Current  s_t',
            f't+{step1}',
            f't+{step2}',
            f'Predicted  t+{n_steps}',
            f'Actual  t+{n_steps}',
            'Error  pred − actual',
        ]
        ncols = 6
        n_ch  = len(_CHANNEL_NAMES)
        fig1, axes1 = plt.subplots(
            n_ch, ncols,
            figsize=(ncols * 3.0, n_ch * 3.0),
            squeeze=False,
            facecolor='#1a1a2e',
        )
        fig1.suptitle(
            f"Polar-cap forecast — Epoch {epoch}  (n_steps={n_steps})",
            fontsize=12, fontweight='bold', color='white', y=1.005,
        )
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
                valid = np.concatenate([
                    s_np[~outside], pred_np[~outside], stn_np[~outside]
                ])
                vmax = max(float(np.nanpercentile(np.abs(valid), 98)), 1e-6)
                vmin = -vmax
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
                                    ha='right', va='center', labelpad=65,
                                    color='white')

        fig1.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()

        # ── Figure 2: E×B drift velocity vectors ─────────────────────────────
        stride = max(1, min(H, W) // 25)
        ys_q   = np.arange(stride // 2, H, stride)
        xs_q   = np.arange(stride // 2, W, stride)
        Xq, Yq = np.meshgrid(xs_q, ys_q)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        R      = min(H, W) / 2.0
        in_cap = ((Xq - cx) ** 2 + (Yq - cy) ** 2) <= R ** 2

        vel_pairs = [
            (0, 1, 'Observed  E×B drift'),
            (2, 3, 'Model  E×B drift'),
        ]
        fig2, axes2 = plt.subplots(
            2, 3,
            figsize=(3 * 4.0, 2 * 4.0),
            squeeze=False,
            facecolor='#1a1a2e',
        )
        fig2.suptitle(
            f"E×B drift vectors — Epoch {epoch}  (n_steps={n_steps})",
            fontsize=11, fontweight='bold', color='white',
        )
        for col, lbl in enumerate(['Actual', 'Predicted', 'Error (pred − actual)']):
            axes2[0, col].set_title(lbl, fontsize=9, color='white', fontweight='bold')

        for row, (n_ch_idx, e_ch_idx, pair_label) in enumerate(vel_pairs):
            vn_act  = s_tn[n_ch_idx]
            ve_act  = s_tn[e_ch_idx]
            vn_pred = pred[n_ch_idx]
            ve_pred = pred[e_ch_idx]
            vn_err  = vn_pred - vn_act
            ve_err  = ve_pred - ve_act

            covered = np.sqrt(vn_act**2 + ve_act**2)[occ_np > 0.05]
            speed_max = float(np.nanpercentile(covered, 98)) if len(covered) > 0 else 1.0
            speed_max = speed_max if np.isfinite(speed_max) and speed_max > 0 else 1.0

            triples = [(vn_act, ve_act), (vn_pred, ve_pred), (vn_err, ve_err)]
            for col, (vn, ve) in enumerate(triples):
                ax = axes2[row, col]
                ax.set_facecolor('#1a1a2e')
                outside = self._polar_cap_axes_setup(ax, H, W, min_mlat)

                speed = np.sqrt(vn**2 + ve**2)
                speed[outside] = np.nan
                vlim = speed_max if col < 2 else speed_max * 0.5
                im = ax.imshow(speed, cmap='plasma', vmin=0, vmax=vlim,
                               origin='upper', interpolation='nearest',
                               extent=[0, W, H, 0], alpha=0.6)

                U =  ve[ys_q[:, None], xs_q[None, :]]
                V = -vn[ys_q[:, None], xs_q[None, :]]
                U[~in_cap] = np.nan
                V[~in_cap] = np.nan
                ax.quiver(Xq, Yq, U, V,
                          scale=speed_max * stride * 1.5,
                          color='white', alpha=0.85,
                          width=0.003, headwidth=4, headlength=4,
                          zorder=6)

                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6, colors='white')
                cb.set_label('speed (norm.)', fontsize=6, color='white')
                cb.outline.set_edgecolor('white')

            axes2[row, 0].set_ylabel(pair_label, fontsize=8,
                                     color='white', rotation=90, va='center')

        fig2.patch.set_facecolor('#1a1a2e')
        plt.tight_layout()

        # ── Dispatch: W&B → disk ─────────────────────────────────────────────
        save_dir = os.path.join(log_root, 'val_vis')
        os.makedirs(save_dir, exist_ok=True)

        logged = False
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'val/polar_cap_channels': wandb.Image(fig1),
                    'val/velocity_vectors':   wandb.Image(fig2),
                }, step=global_step)
                logged = True
        except Exception:
            pass

        if not logged:
            p1 = os.path.join(save_dir, f'epoch_{epoch:03d}_channels.png')
            p2 = os.path.join(save_dir, f'epoch_{epoch:03d}_vectors.png')
            fig1.savefig(p1, dpi=120, bbox_inches='tight', facecolor=fig1.get_facecolor())
            fig2.savefig(p2, dpi=120, bbox_inches='tight', facecolor=fig2.get_facecolor())
            print(f"  [vis] {p1}")
            print(f"  [vis] {p2}")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x_last: torch.Tensor) -> torch.Tensor:
        squeeze = x_last.ndim == 3
        if squeeze:
            x_last = x_last.unsqueeze(0)
        z_s, feats  = self.encoder(x_last)
        _, _, a_det = self.actor.sample(z_s)
        delta_hat   = self.decoder(feats, a_det, x_last.shape[-2:])
        y_hat       = x_last + delta_hat
        return y_hat.squeeze(0) if squeeze else y_hat

    # ── Optimisers + schedulers ───────────────────────────────────────────────

    def configure_optimizers(self):
        warmup = self.hparams.warmup_steps

        opt_critic = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.action_enc.parameters()) +
            list(self.critic.parameters()),
            lr=self.hparams.critic_lr, weight_decay=1e-4,
        )
        opt_actor = torch.optim.AdamW(
            list(self.actor.parameters()) +
            list(self.decoder.parameters()),
            lr=self.hparams.actor_lr, weight_decay=1e-4,
        )
        opt_alpha = torch.optim.Adam(
            [self.log_alpha], lr=self.hparams.alpha_lr,
        )

        def _schedule(optimizer, base_lr):
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.05, end_factor=1.0, total_iters=warmup,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, 200_000 - warmup), eta_min=base_lr * 0.01,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_sched, cosine_sched], milestones=[warmup],
            )

        sch_critic = _schedule(opt_critic, self.hparams.critic_lr)
        sch_actor  = _schedule(opt_actor,  self.hparams.actor_lr)
        # Alpha gets constant LR (no warmup/decay) for stability
        sch_alpha  = torch.optim.lr_scheduler.ConstantLR(opt_alpha, factor=1.0)

        return (
            [opt_critic, opt_actor, opt_alpha],
            [sch_critic, sch_actor, sch_alpha],
        )
