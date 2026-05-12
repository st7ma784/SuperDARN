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
        if self.normalise_rewards:
            self._reward_rms.update(r_n)
            r_n = self._reward_rms.normalise(r_n)

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
        td_loss  = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # CQL: raise data Q, lower policy Q
        with torch.no_grad():
            a_pi_cql, _, _ = self.actor.sample(z_s)
        q1_pi, q2_pi = self.critic(z_s, a_pi_cql)
        cql_penalty  = ((q1_pi - q1.detach()) + (q2_pi - q2.detach())).mean() * 0.5

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
        rl_loss = (self.alpha.detach() * log_pi - q_pi).mean()

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
        sch_critic, sch_actor            = self.lr_schedulers()
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

            self.log('train/actor_loss', a_loss,   prog_bar=True, on_step=True)
            self.log('train/rl_loss',    rl_loss,  on_step=True)
            self.log('train/bc_loss',    bc_loss,  on_step=True)
            self.log('train/alpha_loss', alph_loss,on_step=True)
            self.log('train/alpha',      self.alpha, on_step=True)
            self.log('train/lr_actor',   sch_actor.get_last_lr()[0], on_step=True)

    # ── Validation step ───────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        s, a_data, r_n, s_tn, done = self._unpack(batch)

        with torch.no_grad():
            z_s, feats = self.encoder(s)
            _, _, a_det = self.actor.sample(z_s)
            delta_hat   = self.decoder(feats, a_det, (self.grid_size, self.grid_size))
            y_hat       = s + delta_hat

            # Use s_tn as a proxy for the immediate next state when n_steps=1;
            # for n>1 we still compute RMSE against s_tn (the trajectory endpoint)
            # as an approximate measure of multi-step accuracy.
            mse      = F.mse_loss(y_hat, s_tn)
            pers_mse = F.mse_loss(s, s_tn)
            skill    = 1.0 - mse / pers_mse.clamp(min=1e-8)

            r_policy = compute_reward(s, delta_hat, s_tn).mean()
            r_pers   = persistence_reward(s, s_tn).mean()

        self.log('val/rmse',         mse.sqrt(),         prog_bar=True, sync_dist=True)
        self.log('val/skill_pers',   skill,              prog_bar=True, sync_dist=True)
        self.log('val/r_policy',     r_policy,           sync_dist=True)
        self.log('val/r_pers',       r_pers,             sync_dist=True)
        self.log('val/r_delta',      r_policy - r_pers,  sync_dist=True)

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

        return (
            [opt_critic, opt_actor, opt_alpha],
            [sch_critic, sch_actor],
        )
