RL Forecast — Offline SAC for Ionospheric Convection
=====================================================

This document describes the reinforcement-learning forecast module
(``src/rl_forecast/``).  It trains an offline Soft Actor-Critic (SAC) agent
to predict the next frame of the polar ionospheric convection field from
SuperDARN radar data.

.. contents:: Contents
   :local:
   :depth: 2


Scientific data: the six-channel state
---------------------------------------

Every observation fed to the agent is a polar-grid tensor of shape
``(6, H, W)``.  The grid is a zenithal equidistant projection of the northern
(or southern) polar cap, covering magnetic latitudes from ``min_mlat``
(default 50°) to the pole, at a resolution of ``grid_size × grid_size`` pixels
(default 300 × 300).

.. list-table::
   :header-rows: 1
   :widths: 6 24 60

   * - Index
     - Name
     - Description
   * - 0
     - ``obs_vel_north``
     - Northward component of the SH-fitted E×B convection drift (m/s) in
       radar-covered cells.  Equal to the model northward velocity where
       ``soft_occ > 0.05``; **zero elsewhere**.  Channels 0–1 together give
       the observed 2-D plasma velocity vector only where radar measurements
       constrain the spherical-harmonic (SH) fit.
   * - 1
     - ``obs_vel_east``
     - Eastward component of the SH-fitted E×B drift (m/s) in radar-covered
       cells.  Zero outside covered regions.
   * - 2
     - ``model_vel_north``
     - Northward E×B drift from the background statistical convection model
       (Weimer / TS96), available across the **full polar cap** regardless of
       radar coverage.  Provides a physics-based prior in regions where no
       SuperDARN data exist.
   * - 3
     - ``model_vel_east``
     - Eastward drift from the background model (m/s), full polar cap.
   * - 4
     - ``soft_occ``
     - Soft radar-coverage fraction in ``[0, 1]``.  Values above 0.05
       indicate radar-constrained cells; cells with lower values are
       model-only.  Used as a spatial mask when evaluating the observed
       velocity channels.
   * - 5
     - ``boundary_dist``
     - Signed magnetic-latitude distance from the Heppner-Maynard convection
       boundary (degrees).  Positive values are poleward of (inside) the
       convection zone; negative values are equatorward.  Derived from the
       ``boundary.mlat`` / ``boundary.mlon`` arrays in each cnvmap record and
       interpolated onto the grid.

The tensor is **normalised** per-channel using statistics estimated on the
training split (Welford online mean / variance).  All network inputs and
outputs therefore live in approximately zero-mean unit-variance space; the
reward function likewise operates on normalised values.

.. note::

   Obs-velocity channels (0–1) are spatially sparse: only the radar-footprint
   cells are non-zero.  Evaluation metrics for these channels are therefore
   computed **only inside covered cells** (``soft_occ > 0.05``) to avoid
   diluting the RMSE with trivially-zero regions.


RL transition tuple
-------------------

The datamodule (``RLDataModule`` / ``NStepRLTransitionDataset``) converts
consecutive frame pairs from the pre-saved dataset into offline RL
transitions::

    (s_t, a_t, R_n, s_{t+n}, done)

where

* ``s_t``   — the normalised 6-channel state at time *t*
* ``a_t``   — the *observed* delta ``y_t − s_t`` (what the atmosphere
  actually did; used as the data-set action for the critic and as a
  behavioural-cloning target for the actor)
* ``R_n``   — n-step discounted return
  ``Σ_{k=0}^{n-1} γ^k · r(s_{t+k}, a_{t+k})``,
  where the per-step reward is computed relative to a persistence baseline
  (see :ref:`reward`)
* ``s_{t+n}`` — bootstrap state *n* steps later, used for the critic target
* ``done``  — always 0 for offline data (no episode boundaries)

Only temporally contiguous indices (within a single mmap'd file chunk) are
exposed, ensuring that bootstrap states are physically meaningful.

.. _reward:

Reward signal
-------------

The shaped reward has three components:

1. **Weighted MSE** — negative channel-weighted squared error between the
   predicted next frame and the ground-truth, with per-channel weights
   ``[1.5, 1.5, 1.0, 1.0, 0.3, 1.2]`` (obs velocity weighted highest;
   soft-occ lowest).  Obs-channel errors are masked to radar-covered cells.

2. **Spatial total-variation penalty** — penalises high-frequency noise in
   the predicted obs-velocity field inside covered cells (weight 0.01).

3. **Boundary accuracy penalty** — penalises drift of the predicted
   Heppner-Maynard boundary latitude from the true value (weight 0.05).

During **offline data collection** the reward is computed relative to the
persistence baseline (predict no change) so that the agent receives a
meaningful signal even though the data actions were all executed in the real
atmosphere::

    r_relative = r(obs_action) − r(zero_delta)

This avoids the degenerate situation where every observed trajectory gets
the same absolute reward regardless of how informative it was.


Soft Actor-Critic (SAC) — synopsis
------------------------------------

SAC [Haarnoja et al., 2018] is a maximum-entropy off-policy actor-critic
algorithm.  It augments the standard RL objective with an entropy bonus,
encouraging the policy to remain as stochastic as possible while maximising
cumulative reward::

    J(π) = Σ_t  E[ r(s_t, a_t) + α · H(π(·|s_t)) ]

where ``α`` (the *temperature*) is a Lagrange multiplier that is adapted
automatically to hit a target entropy ``H_target``.  The key properties of
SAC that make it suitable for this problem are:

* **Sample efficiency** — off-policy learning from a replay buffer (here an
  offline dataset) allows full reuse of every transition.
* **Stability** — the twin-critic (double-Q) trick prevents overestimation
  of Q-values; the entropy bonus prevents policy collapse to a deterministic
  mode prematurely.
* **Continuous actions** — the reparameterised Gaussian with tanh squashing
  gives closed-form, low-variance policy gradients over continuous latent
  action spaces.

The three interleaved update rules per step are:

**Critic** — minimise Bellman residual with entropy-regularised targets::

    Q_target = r + γ · [min_i Q_tgt_i(s', a') − α · log π(a'|s')]
    L_critic  = Σ_i  Huber(Q_i(s, a),  Q_target)

**Actor** — maximise soft Q-value while satisfying the entropy constraint::

    L_actor = E_a~π[ α · log π(a|s) − min_i Q_i(s, a) ]

**Temperature** — dual gradient descent on the entropy constraint::

    L_α = −log_α · (log π(a|s) + H_target)

This implementation uses ``manual_optimization`` (PyTorch Lightning) with
three separate Adam optimisers, one per component.


Critic: data seen and training
--------------------------------

The critic estimates the *soft Q-value* ``Q(s, a)`` — the expected
discounted return from state *s* when action *a* is taken, assuming the
policy ``π`` thereafter.

**Input**

The critic sees two encoded vectors concatenated together:

* **State latent** ``z_s`` (dim 256) — produced by the
  ``ConvEncoder`` (shared with the actor).  This encodes the full 6-channel
  ``(6, H, W)`` polar-grid state.

* **Action latent** (dim 128) — either

  * *Data action*: the observed delta ``a_t = y_t − s_t`` passed through
    ``ActionEncoder`` (lightweight pool + MLP) to produce a latent vector.
    Used for the TD-error loss.

  * *Policy action*: a latent sample from ``LatentActor`` (tanh-squashed
    Gaussian), decoded back to grid space by ``GridDecoder`` for the BC
    anchor; the latent vector itself is used for actor and CQL losses.

The concatenated ``[z_s ‖ a_latent]`` vector (dim 384) is fed to two
independent MLP heads (``DoubleQCritic``) to give ``Q1`` and ``Q2``.  The
pessimistic estimate ``min(Q1, Q2)`` is used wherever a scalar value
estimate is needed (actor update, bootstrap target), preventing
overestimation.

**Training objective**

With n-step returns and the CQL conservative penalty disabled (see
``agent.py`` comments), the critic loss is::

    Q_target = R_n + γ^n · (1 − done) · [min_i Q_tgt_i(s_{t+n}, a') − α · log π(a'|s_{t+n})]
    L_critic  = Huber(Q1(s_t, a_t), Q_target) + Huber(Q2(s_t, a_t), Q_target)

The Huber loss (δ=1.0) replaces MSE to limit the gradient magnitude for
large TD errors early in training.  Target networks ``Q_tgt`` are updated by
Polyak averaging at rate ``τ = 0.005`` after every critic step.

Rewards entering the target are normalised by a Welford running
mean/variance tracker to prevent Q-value scale drift as the reward
distribution shifts over training.

The critic's optimiser also owns the ``ConvEncoder`` and ``ActionEncoder``
parameters, because the encoder is the performance bottleneck and it is most
efficiently trained by the dense per-pixel signal available through the
critic loss.


Actor: data seen and training
-------------------------------

The actor produces the policy ``π(a | s)`` — a distribution over latent
actions given the current state.

**Input**

The actor sees only the **state latent** ``z_s`` (dim 256) from the shared
``ConvEncoder``.  It never sees the raw grid directly; all spatial
information is compressed into ``z_s`` by the encoder.

**Architecture**

``LatentActor`` is a two-hidden-layer MLP (512 units, GELU activations) that
outputs a mean ``μ`` and log-standard-deviation ``log σ`` over a
128-dimensional latent action space.  Sampling uses the reparameterisation
trick with tanh squashing::

    ε ~ N(0, I)
    x_t = μ + σ · ε
    a   = tanh(x_t)       ∈ (−1, 1)^128

The log-probability includes the tanh Jacobian correction::

    log π(a|s) = Σ_d [ −½(ε_d² + log 2π) − log σ_d − log(1 − a_d² + ε) ]

The deterministic action ``a_det = tanh(μ)`` (no noise) is used at inference
time and during validation.

**Decoding to grid space**

The latent action ``a`` is decoded back to a ``(6, H, W)`` delta grid by
``GridDecoder``, which spatially broadcasts the latent vector, concatenates
it with the encoder's spatial feature map (``feats``, shape
``(feat_ch, H/8, W/8)``), and upsamples via three transposed-convolution
stages with residual blocks.  The decoder head is zero-initialised so that
the agent starts from persistence (delta = 0) at the beginning of training.

**Training objective**

The actor loss has two terms::

    L_actor = L_RL + bc_weight · L_BC

*RL term* — standard SAC policy gradient::

    L_RL = E_a~π[ α · log π(a|s) − min_i Q_i(z_s, a) ]

The actor gradient is computed with the encoder and critic frozen (their
gradients are detached).  The SAC temperature ``α`` is clipped to
``[1e-6, 1.0]`` to prevent it from growing unbounded early in training.

*Behavioural cloning anchor* — SmoothL1 loss between the decoded policy
delta and the observed dataset delta::

    L_BC = SmoothL1(GridDecoder(feats, a_det), a_data,  β=0.1)

This keeps the decoded prediction close to the distribution of observed
atmospheric transitions, preventing the actor from drifting into
physically implausible regions of the grid-space action distribution
before the Q-function is well-calibrated.

The actor is updated every ``actor_update_freq = 2`` critic steps.

**Temperature adaptation**

``log α`` is a scalar ``nn.Parameter`` updated by gradient descent on::

    L_α = −log_α · (log π + H_target)

where ``H_target = −0.98 · action_latent_dim``.  This automatically adjusts
the trade-off between exploration (high entropy) and exploitation (low
entropy) over the course of training.


Validation metrics
------------------

Each validation epoch reports both aggregate and per-channel metrics, printed
to stdout as a formatted table and logged to W&B (if enabled).

Aggregate metrics:

* ``val/rmse`` — RMS error between ``y_hat = s + delta_hat`` and ``s_tn``
* ``val/skill_pers`` — skill score vs persistence: ``1 − RMSE_policy / RMSE_pers``
* ``val/r_policy``, ``val/r_pers`` — mean shaped rewards for policy and persistence
* ``val/r_delta`` — reward improvement of policy over persistence

Per-channel metrics (suffix ``_{channel_name}``):

* ``val/rmse_{ch}`` — RMSE for that channel (obs channels masked to covered cells)
* ``val/pers_rmse_{ch}`` — persistence RMSE (same mask)
* ``val/skill_{ch}`` — skill score for that channel
* ``val/bias_{ch}`` — mean signed error (positive = overprediction)

.. note::

   All values are in **normalised** units.  To convert to physical units,
   multiply by the per-channel standard deviation used during preprocessing
   (stored in ``shape.txt`` alongside the ``dataA_*.npy`` / ``dataB_*.npy``
   files).  For velocity channels this is typically on the order of 200–600 m/s.


References
----------

* Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
  *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
  with a Stochastic Actor.*
  ICML 2018. https://arxiv.org/abs/1801.01290

* Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020).
  *Conservative Q-Learning for Offline Reinforcement Learning.*
  NeurIPS 2020. https://arxiv.org/abs/2006.04779

* Weimer, D. R. (2005). *Improved ionospheric electrodynamic models and
  application to calculating Joule heating rates.*
  Journal of Geophysical Research, 110(A5).

* Heppner, J. P., & Maynard, N. C. (1987). *Empirical high-latitude electric
  field models.*  Journal of Geophysical Research, 92(A5), 4467–4489.
