import torch

# Per-channel importance: obs velocity > boundary > model > coverage
CHANNEL_WEIGHTS = torch.tensor([1.5, 1.5, 1.0, 1.0, 0.3, 1.2], dtype=torch.float32)


def compute_reward(
    x_last: torch.Tensor,
    delta_pred: torch.Tensor,
    y_true: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shaped reward for a single-step ionospheric delta prediction.

    The reward is negative weighted MSE (obs channels masked to radar cells),
    plus physics-motivated penalties for spatial roughness and boundary drift.

    Args:
        x_last:      (B, 6, H, W) most-recent normalised state frame
        delta_pred:  (B, 6, H, W) predicted delta (action in RL terms)
        y_true:      (B, 6, H, W) ground-truth next frame
        channel_weights: optional (6,) override

    Returns:
        reward: (B,) float tensor, detached from autograd graph
    """
    if channel_weights is None:
        channel_weights = CHANNEL_WEIGHTS.to(x_last.device)
    w = channel_weights.view(1, -1, 1, 1)

    y_pred = x_last + delta_pred
    err = (y_pred - y_true).pow(2) * w                    # (B, 6, H, W)

    # Observed channels (0-1) supervised only inside radar-covered cells
    occ_mask = (y_true[:, 4:5] > 0.05).float()            # (B, 1, H, W)
    obs_w = occ_mask.expand_as(err[:, :2])                 # (B, 2, H, W)
    full_w = torch.ones_like(err[:, 2:])                   # (B, 4, H, W)

    obs_err  = (err[:, :2] * obs_w).sum(dim=(1, 2, 3))
    obs_denom = obs_w.sum(dim=(1, 2, 3)).clamp(min=1.0)
    full_err  = (err[:, 2:] * full_w).sum(dim=(1, 2, 3))
    full_denom = full_w.sum(dim=(1, 2, 3)).clamp(min=1.0)

    mse_per_sample = obs_err / obs_denom + full_err / full_denom   # (B,)

    # Total variation of predicted obs velocity in covered cells — penalise noise
    pred_obs = y_pred[:, :2] * occ_mask
    tv_h = (pred_obs[:, :, 1:, :] - pred_obs[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
    tv_w = (pred_obs[:, :, :, 1:] - pred_obs[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))

    # Boundary distance accuracy
    bnd_err = (y_pred[:, 5] - y_true[:, 5]).abs().mean(dim=(-2, -1))

    reward = -mse_per_sample - 0.01 * (tv_h + tv_w) - 0.05 * bnd_err
    return reward.detach()


def persistence_reward(
    x_last: torch.Tensor,
    y_true: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reward of the trivial persistence forecast (predict no change)."""
    zeros = torch.zeros_like(x_last)
    return compute_reward(x_last, zeros, y_true, channel_weights)
