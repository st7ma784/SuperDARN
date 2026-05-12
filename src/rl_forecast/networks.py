"""
Neural network components for offline SAC ionospheric RL.

ConvEncoder    — shared state encoder, (B, C, H, W) → latent + spatial features
ActionEncoder  — encodes a grid-space delta to a latent vector for the critic
LatentActor    — Gaussian policy in latent action space (reparameterised + tanh)
GridDecoder    — decodes (spatial features, latent action) → delta grid
DoubleQCritic  — twin Q-networks, (state_latent, action_latent) → (Q1, Q2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared building blocks ───────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── State encoder ────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Maps a (B, in_channels, H, W) state to:
      - z  (B, latent_dim)           — state latent used by actor & critic
      - feats (B, feat_ch, H/8, W/8) — spatial feature map used by decoder
    """

    def __init__(self, in_channels: int = 6, latent_dim: int = 256,
                 base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 7, stride=2, padding=3, bias=False),  # H/2
            nn.GroupNorm(8, c),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(ResBlock(c), ResBlock(c))
        self.down1  = nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False)  # H/4
        self.stage2 = nn.Sequential(ResBlock(c * 2), ResBlock(c * 2))
        self.down2  = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False)  # H/8
        self.stage3 = nn.Sequential(ResBlock(c * 4), ResBlock(c * 4))

        self.feat_channels = c * 4
        self.pool = nn.AdaptiveAvgPool2d(4)              # → (B, c*4, 4, 4)
        self.proj = nn.Linear(c * 4 * 16, latent_dim)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        feats = self.stage3(x)                           # (B, feat_ch, H/8, W/8)
        z = self.proj(self.pool(feats).flatten(1))       # (B, latent_dim)
        return z, feats


# ── Action encoder (data actions → latent for critic) ────────────────────────

class ActionEncoder(nn.Module):
    """
    Encodes a full-grid delta (B, 6, H, W) into a latent vector (B, action_latent_dim).
    Used to represent data-set actions in the critic's input space.
    Lightweight: AdaptivePool + MLP, no transposed convolutions needed.
    """

    def __init__(self, in_channels: int = 6, action_latent_dim: int = 128,
                 pool_size: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        flat_dim  = in_channels * pool_size * pool_size
        self.mlp  = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_latent_dim),
        )

    def forward(self, a_grid: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pool(a_grid).flatten(1))


# ── Gaussian policy in latent action space ───────────────────────────────────

class LatentActor(nn.Module):
    """
    State-latent → Gaussian policy over latent actions.
    Uses reparameterisation with tanh squashing so actions lie in (-1, 1)^D.
    """
    LOG_SIG_MAX =  2.0
    LOG_SIG_MIN = -5.0

    def __init__(self, state_latent_dim: int = 256, action_latent_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()
        self.action_latent_dim = action_latent_dim
        self.net = nn.Sequential(
            nn.Linear(state_latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_head       = nn.Linear(hidden_dim, action_latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, action_latent_dim)

    def _forward(self, z_s: torch.Tensor):
        h         = self.net(z_s)
        mu        = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        return mu, log_sigma

    def sample(self, z_s: torch.Tensor):
        """
        Returns:
            a:        (B, action_latent_dim)  tanh-squashed sampled action
            log_pi:   (B, 1)                  log-probability with squash correction
            a_det:    (B, action_latent_dim)  deterministic (mean) action
        """
        mu, log_sigma = self._forward(z_s)
        sigma = log_sigma.exp()
        eps   = torch.randn_like(mu)
        x_t   = mu + sigma * eps
        a     = torch.tanh(x_t)
        # log_pi = Σ[ log N(eps;0,1) - log(sigma) - log(1 - tanh²(x_t) + ε) ]
        log_pi = (
            -0.5 * (eps.pow(2) + math.log(2.0 * math.pi))
            - log_sigma
            - torch.log(1.0 - a.pow(2) + 1e-6)
        ).sum(dim=-1, keepdim=True)
        return a, log_pi, torch.tanh(mu)


# ── Grid decoder ─────────────────────────────────────────────────────────────

class GridDecoder(nn.Module):
    """
    Decodes (spatial_features, latent_action) → (B, 6, H, W) delta grid.

    spatial_features comes from ConvEncoder.  latent_action is the actor sample.
    The action is spatially broadcast and concatenated with the features before
    upsampling, so it modulates the decode at every spatial location.
    """

    def __init__(self, feat_channels: int = 256, action_latent_dim: int = 128,
                 out_channels: int = 6, base_channels: int = 64):
        super().__init__()
        c = base_channels
        self.action_proj = nn.Linear(action_latent_dim, feat_channels)
        in_ch = feat_channels * 2   # cat(feats, broadcast action)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, c * 2, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            ResBlock(c * 2),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
            ResBlock(c),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.GELU(),
        )
        self.head = nn.Conv2d(c, out_channels, 1)
        # Zero-init: delta = 0 (persistence) at start of training
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats: torch.Tensor, a_latent: torch.Tensor,
                target_size: tuple) -> torch.Tensor:
        """
        Args:
            feats:       (B, feat_ch, Hf, Wf)
            a_latent:    (B, action_latent_dim)
            target_size: (H, W) final output spatial size
        Returns:
            delta:       (B, 6, H, W)
        """
        a_sp = self.action_proj(a_latent).unsqueeze(-1).unsqueeze(-1)
        a_sp = a_sp.expand(-1, -1, feats.shape[2], feats.shape[3])
        x = torch.cat([feats, a_sp], dim=1)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.head(x)
        if x.shape[-2:] != torch.Size(target_size):
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


# ── Twin Q-critic ────────────────────────────────────────────────────────────

class DoubleQCritic(nn.Module):
    """
    Two independent Q-networks for pessimistic value estimation (SAC / TD3).
    Both operate on (state_latent ‖ action_latent).
    """

    def __init__(self, state_latent_dim: int = 256, action_latent_dim: int = 128,
                 hidden_dim: int = 512):
        super().__init__()
        in_dim = state_latent_dim + action_latent_dim

        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        self.q1 = _mlp()
        self.q2 = _mlp()

    def forward(self, z_s: torch.Tensor,
                a_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([z_s, a_latent], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q_min(self, z_s: torch.Tensor, a_latent: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(z_s, a_latent)
        return torch.min(q1, q2)
