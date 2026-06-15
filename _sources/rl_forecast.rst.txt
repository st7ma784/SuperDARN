RL/ML Forecast — Ionospheric Convection
========================================

This document describes the forecast module (``src/rl_forecast/``).
Two approaches are available, selected via ``--mode``:

* **multi_step** *(default, recommended)* — supervised multi-step rollout with
  curriculum on horizon length K and scheduled sampling.
* **sac** *(legacy)* — offline Soft Actor-Critic + Conservative Q-Learning.

See :ref:`approach_history` for benchmark comparisons and the rationale for
the switch.

.. contents:: Contents
   :local:
   :depth: 2

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.10+ with the ``open-ce`` conda environment activated
* ``pydarnio`` (for reading raw cnvmap files)
* ``pytorch-lightning``, ``torch``, ``wandb`` (all in ``open-ce``)
* The ``weatherlearn`` submodule checked out:

  .. code-block:: bash

     git submodule update --init --recursive

Step 1 — Preprocess raw cnvmap data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw SuperDARN cnvmap files must be converted to memory-mapped numpy arrays
before training.  Run this once per dataset; results are cached and re-used
automatically on subsequent runs.

.. code-block:: bash

   cd src/rl_forecast
   python launch.py \
       --preprocess \
       --cnvmap_dir /data3/rst/extracted_data \
       --data_dir   /data2/rl_data \
       --grid_size  120 \
       --max_files  4000

This writes ``dataA_*.npy``, ``dataB_*.npy``, and ``shape.txt`` into
``--data_dir``.  ``--grid_size`` must match the pixel resolution of your
polar-cap projection (the data at ``/data2/rl_data`` uses **120 × 120**).

.. note::

   Use ``--max_files`` to limit the number of cnvmap files processed (useful
   for quick tests).  Omit it to process the full dataset.

Step 2 — Extract timestamps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model uses UT-of-day and day-of-year sinusoidal features to capture
ionospheric diurnal variation.  Timestamps must be extracted separately and
saved chunk-aligned with the numpy arrays:

.. code-block:: bash

   cd src/weatherlearn/PTL
   python -c "
   from run_baseline import extract_timestamps_to_disk
   extract_timestamps_to_disk(
       cnvmap_dir='/data3/rst/extracted_data',
       out_dir='/data2/rl_data',
       max_files=4000,
   )
   "

This writes ``timestamps_*.npy`` files (shape ``(N, 2)``: ``[ut_hours, doy]``)
into ``--data_dir`` in the same chunk order as the ``dataA_*`` files.
If no timestamp files are found at training time, time conditioning is
silently disabled (all time features set to zero).

Step 3 — Train the model
~~~~~~~~~~~~~~~~~~~~~~~~~

**Multi-step supervised** (recommended, default):

.. code-block:: bash

   cd src/rl_forecast
   python launch.py \
       --data_dir            /data2/rl_data \
       --grid_size           120 \
       --max_rollout_steps   12 \
       --devices             -1 \
       --wandb

**Legacy SAC** (for comparison / ablation):

.. code-block:: bash

   python launch.py \
       --mode sac \
       --data_dir  /data2/rl_data \
       --grid_size 120 \
       --n_steps   12 \
       --devices   -1 \
       --wandb

Key flags:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--mode``
     - ``multi_step``
     - ``multi_step`` (supervised rollout) or ``sac`` (offline SAC+CQL).
   * - ``--data_dir``
     - *(required)*
     - Directory containing the preprocessed ``dataA_*.npy`` files.
   * - ``--grid_size``
     - 300
     - Must match the grid used during preprocessing.  Use ``120`` for the
       current dataset.
   * - ``--max_rollout_steps``
     - 12
     - Maximum autoregressive horizon K (2-min steps; 12 = 24 min ahead).
   * - ``--init_rollout_steps``
     - 1
     - Curriculum starting point.  The model is first trained on 1-step
       prediction, then K grows every ``--curriculum_step_epochs`` epochs.
   * - ``--curriculum_step_epochs``
     - 5
     - Increase K by 1 every N epochs.
   * - ``--p_ss_max``
     - 0.5
     - Maximum scheduled-sampling probability (linearly annealed from 0).
   * - ``--p_ss_anneal_epochs``
     - 20
     - Epochs to reach ``p_ss_max``.
   * - ``--step_weight_scheme``
     - ``uniform``
     - Per-step loss weighting: ``uniform`` (equal) or ``geometric``
       (earlier steps weighted more, discounted by ``--gamma``).
   * - ``--tv_weight``
     - 0.01
     - Total-variation smoothness penalty on predicted obs-velocity channels.
   * - ``--lr``
     - 3e-4
     - Adam learning rate (single optimiser).
   * - ``--devices``
     - -1
     - Number of GPUs: ``-1`` = all, ``1`` = single GPU.
   * - ``--batch_size``
     - 16
     - Samples per gradient step per GPU.
   * - ``--max_epochs``
     - 200
     - Training will stop early via ``EarlyStopping`` on ``val/rmse``.
   * - ``--patience``
     - 20
     - Early stopping patience (epochs without improvement).
   * - ``--wandb``
     - False
     - Enable Weights & Biases logging.
   * - ``--backend``
     - gloo
     - DDP backend.  ``gloo`` is stable on all hardware; ``nccl`` is faster
       but requires NVML/NVLink.

Step 4 — Monitor training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Log output is written to stdout.  After every validation epoch a table is
printed.  For ``--mode multi_step``:

.. code-block:: text

   ────────────────────────────────────────────────────────────────────────
   Multi-step forecast epoch   3  (K=1/12, SS=0.15)
     1-step  RMSE  0.6600  Skill  0.0820   Final-step RMSE  0.7100  Skill  0.0310
     Reward policy  -5.23  Pers  -5.31  Δreward  0.08
     Channel              RMSE  PersRMSE   Skill      Bias
     ----------------------------------------------------------
     obs_vel_north       1.821     1.998  0.0887    0.021 [masked]
     ...
     K-step skills:  k1=0.082  k2=0.065  k3=0.051  ...  k12=0.031
   ────────────────────────────────────────────────────────────────────────

If ``--wandb`` is set, all metrics (including a 6-column forecast figure per
epoch) are synced to the ``SuperDARN-RL`` W&B project.

Step 5 — SLURM submission
~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate (not submit) an sbatch script for HPC clusters:

.. code-block:: bash

   python launch.py --data_dir /data2/rl_data --grid_size 120 \
       --n_steps 12 --wandb --slurm > job.sh
   # review job.sh, then:
   sbatch job.sh

Checkpoints
~~~~~~~~~~~

Model checkpoints are saved to ``{log_dir}/rl-<timestamp>/`` (default log
root: ``$global_scratch/rl_logs``).  The best and last checkpoints are kept.
To resume training from a checkpoint:

.. code-block:: bash

   python launch.py --data_dir /data2/rl_data --grid_size 120 \
       --resume_from /data/rl_logs/rl-<timestamp>/last.ckpt


Scientific data: the six-channel state
---------------------------------------

Every observation fed to the agent is a polar-grid tensor of shape
``(6, H, W)``.  The grid is a zenithal equidistant projection of the northern
(or southern) polar cap, covering magnetic latitudes from ``min_mlat``
(default 50°) to the pole, at a resolution of ``grid_size × grid_size`` pixels.

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
training split (z-score: zero mean, unit variance).  Channel 4 (``soft_occ``,
naturally in ``[0, 1]``) is kept unscaled so that the reward function's
occupancy threshold (``> 0.05``) remains valid after normalisation.

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

    (s_t, a_t, R_n, s_{t+n}, done, time_t, time_tn)

where

* ``s_t``      — normalised 6-channel state at time *t*  ``(6, H, W)``
* ``a_t``      — observed delta ``y_t − s_t``, used as the data-set action
  ``(6, H, W)``
* ``R_n``      — n-step discounted return
  ``Σ_{k=0}^{n-1} γ^k · r(s_{t+k}, a_{t+k})``,
  where the per-step reward is computed relative to a persistence baseline
  (see :ref:`reward`)
* ``s_{t+n}``  — bootstrap state *n* steps later  ``(6, H, W)``
* ``done``     — always 0 for offline data (no episode boundaries)
* ``time_t``   — sinusoidal time features at *t*: ``[sin_UT, cos_UT, sin_DOY, cos_DOY]`` ``(4,)``
* ``time_tn``  — same features advanced by ``n × 2 minutes``  ``(4,)``

Only temporally contiguous indices (within a single mmap'd file chunk) are
exposed, ensuring that bootstrap states are physically meaningful.

If no timestamp files are present in ``data_dir``, ``time_t`` and
``time_tn`` are zero vectors and time conditioning is effectively disabled.


Time conditioning
-----------------

Ionospheric convection depends strongly on universal time (UT) even when
data are expressed in magnetic local time (MLT) coordinates.  The reason is
the ~11° offset between Earth's geographic and magnetic poles: as Earth
rotates, the orientation of the solar-wind forcing relative to the
magnetosphere changes on a 24-hour cycle, modulating convection even within
the co-rotating MLT frame.

To capture this, the encoder receives a 4-element sinusoidal time vector:

.. math::

   \mathbf{t} = \bigl[\sin(2\pi\,\mathrm{UT}/24),\;\cos(2\pi\,\mathrm{UT}/24),\;
                       \sin(2\pi\,\mathrm{DOY}/365.25),\;\cos(2\pi\,\mathrm{DOY}/365.25)\bigr]

This is projected to the encoder latent dimension and added to the state
latent *before* the actor and critic see it::

    z_s = ConvEncoder(s_t) + W_time · t

The projection keeps ``in_channels = 6`` unchanged (the time features are
*not* broadcast as additional spatial channels), so the action space, reward
computation, and channel normalisation are all unaffected.

During autoregressive rollout at validation time, the time vector is advanced
by 2 minutes per step using circular arithmetic on the UT angle, keeping DOY
fixed within a single rollout.


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
  ``(6, H, W)`` polar-grid state, optionally conditioned on the time vector.

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

With n-step returns and the CQL conservative penalty, the critic loss is::

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

**Aggregate metrics** (both modes):

* ``val/rmse`` — 1-step RMSE: ``‖ŝ_{t+1} − s_{t+1}‖``
* ``val/skill_pers`` / ``val/skill_1step`` — 1-step skill vs persistence
* ``val/rmse_final`` — RMSE at the maximum rollout step K_max
* ``val/skill_final`` — skill vs persistence at K_max
* ``val/r_policy``, ``val/r_pers``, ``val/r_delta`` — shaped reward metrics

**Per-step metrics** (``multi_step`` mode, critical for curriculum tuning):

* ``val/rmse_step_{k}`` — RMSE at rollout step k (k = 1 … K_max)
* ``val/skill_step_{k}`` — skill vs persistence at step k

  Watch these to judge whether the curriculum is progressing well: skill at
  step k should be non-trivially positive before increasing K past k.

**Per-channel metrics** (suffix ``_{channel_name}``, evaluated at step 1):

* ``val/rmse_{ch}`` — RMSE for that channel (obs channels masked to covered cells)
* ``val/pers_rmse_{ch}`` — persistence RMSE (same mask)
* ``val/skill_{ch}`` — skill score for that channel
* ``val/bias_{ch}`` — mean signed error (positive = overprediction)

**Training diagnostics** (``multi_step`` mode):

* ``train/loss`` — total weighted rollout loss
* ``train/loss_step_{k}`` — per-step contribution to training loss
* ``train/rollout_k`` — current curriculum K
* ``train/p_scheduled_sampling`` — current scheduled sampling probability

A **forecast figure** is generated at the end of each validation epoch and
logged to W&B (and saved to ``{log_dir}/val_vis/``).  It shows six columns
for each of the six data channels:

  ``s_t`` | ``ŝ_{t+K/3}`` | ``ŝ_{t+2K/3}`` | ``ŝ_{t+K}`` (pred) | ``s_{t+K}`` (actual) | Error

.. note::

   All values are in **normalised** units.  To convert to physical units,
   multiply by the per-channel standard deviation used during preprocessing
   (stored in ``shape.txt`` alongside the ``dataA_*.npy`` / ``dataB_*.npy``
   files).  For velocity channels this is typically on the order of 200–600 m/s.


.. _approach_history:

Approach history and benchmarks
---------------------------------

Phase 1 — Offline SAC + CQL (``--mode sac``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial approach trained an offline Soft Actor-Critic agent with
Conservative Q-Learning (CQL) to learn a latent policy over grid-space
prediction deltas.

**Architecture**: ConvEncoder → LatentActor (stochastic Gaussian, 128-dim
latent action) → GridDecoder → delta grid.  A separate ActionEncoder and
DoubleQCritic estimated state-action values.

**Key findings**:

* The model reached **~7% persistence skill** (``val/skill_pers ≈ 0.070``) at
  its peak (epoch 27, ~30 training epochs, 4×GPU).
* Without CQL, Q-values overestimated without bound (actor loss drifted
  −5 → −16.4 over training), confirming the classic offline RL distribution
  problem.
* Two-sided CQL was too aggressive: it brought actor loss to ~0 in a single
  epoch then overshot, making critic loss negative and causing a loss
  explosion (actor loss > 200) that destroyed the learned policy.
* Switching to **one-sided (ReLU) CQL** stabilised training by stopping the
  penalty gradient once Q(s,π) ≤ Q(s,a_data) (no overshoot possible).
* After stabilisation, skill oscillated in the 5–7% band without clear
  improvement, suggesting a fundamental ceiling imposed by the 1-step
  predictor / distribution-shift gap between training and autoregressive
  inference.

**Benchmark** (best checkpoint, ``rl-20260518-171634/rl-epoch=020-val/``):

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Metric
     - Value
     - Notes
   * - ``val/skill_pers`` (peak)
     - 0.070
     - Epoch 27; oscillates 5–7 % thereafter
   * - ``val/skill_1step``
     - 0.057
     - Cleaner 1-step metric (vs true next frame)
   * - ``val/rmse``
     - 0.658
     - Normalised units
   * - Actor loss range
     - −5 → −16
     - Growing without CQL; ~0 with ReLU CQL
   * - Training time to plateau
     - ~30 epochs
     - 4 × GPU, ~70 min/epoch

Phase 2 — Multi-step supervised rollout (``--mode multi_step``, current)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Replaced the actor-critic with direct autoregressive supervision:

**Why the switch**: The SAC reward is negative weighted MSE — essentially
a supervised loss in disguise.  The actor-critic wrapper added CQL tuning
complexity (two-sided vs one-sided, alpha magnitude, explosion risk) for
marginal benefit.  The core problem — the model never sees its own predictions
during training — was better addressed by scheduled sampling and a curriculum
on rollout length.

**Architecture**: ConvEncoder → DeterministicPredictor (MLP, no sampling) →
GridDecoder → delta.  Single Adam optimiser.  No actor-critic, no temperature,
no CQL.

**Key design choices**:

* **Curriculum on K** — starts at K=1 (clean 1-step supervision), increases
  K every ``curriculum_step_epochs`` (default 5) epochs.  The model masters
  short horizons before being challenged with longer ones.
* **Scheduled sampling** — with linearly-annealed probability ``p_ss``, the
  model's own prediction is fed back as the next input rather than ground
  truth, bridging the train/inference distribution gap.
* **Channel-weighted Smooth-L1 loss** — consistent weights with the SAC reward
  function (obs velocity × 1.5, boundary × 1.2, soft-occ × 0.3); obs channels
  masked to radar-covered cells.
* **TV regularisation** — small total-variation penalty (weight 0.01) on
  predicted obs-velocity field inside covered cells, suppressing grid noise.

**Expected improvements over SAC**:

* Skill at every rollout step K is directly optimised (not just step 1).
* Scheduled sampling closes the exposure-bias gap without CQL complexity.
* Single optimiser with cosine LR schedule — no three-way gradient conflict.

**Hyperparameters to tune first**:

1. ``--curriculum_step_epochs`` — if ``val/skill_step_k`` is still low when K
   increases to k, increase this to give more time at each horizon.
2. ``--p_ss_max`` — 0.5 is conservative; raise to 0.7–0.8 if skill plateaus
   again (more aggressive scheduled sampling reduces compounding error).
3. ``--p_ss_anneal_epochs`` — anneal too fast and early training is noisy;
   too slow and the distribution gap persists.
4. ``--step_weight_scheme=geometric`` — experiment if later steps are lagging;
   uniform weights all steps equally.

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
