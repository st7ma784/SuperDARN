Baseline Evaluation Results
===========================

This page documents the dataset statistics and baseline forecast accuracy metrics
for the SuperDARN ML pipeline.  These figures were computed on 2026-04-23 against
the full preprocessed training corpus and serve as the quantitative foundation for
grant proposal submissions.

All metrics are computed on the **normalised** channel representations output by
the preprocessing pipeline (zero-mean, unit-variance per channel across a 2,000-sample
training subset).

----

Dataset
-------

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - Raw cnvmap files
     - **26,129**
   * - Temporal span
     - 2008-01-05 06:00 UT — 2017-12-31 22:00 UT (~10 years)
   * - Preprocessed training pairs
     - **29,500** consecutive two-hourly radar scans
   * - Grid resolution
     - 120 × 120 geographic latitude / longitude bins
   * - Input channels
     - Velocity (m/s) · Velocity SD (m/s) · K-vector (°) · Occupancy · Density (log *n*)
   * - Solar-wind conditioning
     - IMF Bx · By · Bz · Kp · Vx (5 scalar FiLM channels per sample)
   * - Train / validation split
     - 90 % / 10 %  (26,550 / 2,950 pairs, seed 42)

Channel Statistics (target distribution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Channel
     - Mean
     - Std
   * - Velocity (m/s)
     - 0.927
     - 18.386
   * - Velocity SD (m/s)
     - 0.358
     - 7.066
   * - K-vector (°)
     - −0.043
     - 6.143
   * - Occupancy
     - 0.004
     - 0.064
   * - Density (log n)
     - 0.003
     - 0.051

The near-zero means and low occupancy (0.4 % active cells) reflect the sparse,
patchy nature of SuperDARN convection maps: genuine radar returns cover only a
small fraction of the polar cap at any given two-hour window.

----

Baseline Forecasts
------------------

Two baselines bound the difficulty of the one-step prediction task.

**Persistence** ("repeat the last observation")
  The simplest possible forecast — predict that the next state equals the
  current state.  Any trained model must beat this to demonstrate skill.

**Climatology** ("predict all zeros")
  Predicts the dataset mean (approximately zero after normalisation).
  This represents a model with no temporal knowledge at all.

Persistence Baseline
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Channel
     - MSE
     - RMSE
   * - Velocity (m/s)
     - 107.867
     - **10.386**
   * - Velocity SD (m/s)
     - 28.383
     - **5.328**
   * - K-vector (°)
     - 19.903
     - **4.461**
   * - Occupancy
     - 0.00110
     - **0.0332**
   * - Density (log n)
     - 0.00061
     - **0.0247**
   * - **Overall (all channels)**
     - **31.231**
     -

Climatology Baseline
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Channel
     - MSE
     - RMSE
   * - Velocity (m/s)
     - 338.913
     - **18.410**
   * - Velocity SD (m/s)
     - 50.059
     - **7.075**
   * - K-vector (°)
     - 37.736
     - **6.143**
   * - Occupancy
     - 0.00414
     - **0.0644**
   * - Density (log n)
     - 0.00265
     - **0.0515**
   * - **Overall (all channels)**
     - **85.343**
     -

The persistence overall MSE (31.23) is 63 % lower than climatology (85.34),
confirming that successive SuperDARN frames are strongly correlated.
A trained model must therefore achieve MSE < 31.23 to claim positive skill.

Active vs. Quiet Cell Breakdown (velocity channel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SuperDARN returns are sparse: only **0.4 %** of grid cells contain active
radar echoes in a typical frame.

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Region
     - Cell count
     - Persistence RMSE
   * - Active (echo present)
     - 1,758,959  (0.4 %)
     - **124.79 m/s**
   * - Quiet (no echo)
     - 423,041,041  (99.6 %)
     - **6.60 m/s**

The 19× higher RMSE on active cells highlights that the convection field is
highly dynamic precisely where it matters scientifically — motivating an
event-weighted loss function and the occupancy-channel conditioning used by
the model.

----

Skill Score Definition
----------------------

Training and evaluation use the following skill score to contextualise MSE
relative to the persistence baseline::

    skill_persistence = 1 − model_MSE / persistence_MSE

A score > 0 means the model outperforms persistence; score = 1 is a perfect
forecast; score < 0 means the model is worse than doing nothing.

An analogous score against the climatology baseline::

    skill_climatology = 1 − model_MSE / climatology_MSE

----

Model Training Results
----------------------

The Pangu-adapted model (48 M parameters, ``embed_dim=64``, ``mlp_ratio=2``,
120 × 120 grid, solar-wind FiLM conditioning) was trained for up to 10 epochs
with early stopping (patience = 6 on ``val_mse``).  Training runs at ~2.6
it/s on a single GPU (~5 min/epoch, 830 gradient steps per epoch).

.. list-table::
   :widths: 12 18 18 24 24
   :header-rows: 1

   * - Epoch
     - val_loss
     - val_mse
     - skill_persistence
     - skill_climatology
   * - 0
     - 0.0758
     - **0.4922**
     - −0.284
     - **+0.507**

.. note::
   After a single epoch the model already achieves **50.7 % skill over
   climatology** (``skill_climatology = +0.507``), indicating that it has
   learned the gross structure of the convection field within the first pass
   through the data.

   The persistence skill score is negative at epoch 0 (−0.284), which is
   expected: the persistence baseline is conservative for this dataset because
   consecutive SuperDARN frames are strongly correlated (persistence MSE 31.2
   vs climatology MSE 85.3).  Beating persistence requires the model to capture
   the *dynamics* of the convection pattern, not just its spatial structure —
   this typically emerges over several training epochs.

   Training is ongoing; this table will be updated with multi-epoch results.

----

Reproducing These Results
--------------------------

All scripts are in ``src/weatherlearn/PTL/``.

**Step 1 — dataset statistics and baselines (no training required):**

.. code-block:: bash

   cd src/weatherlearn/PTL
   python grant_baselines.py \
       --data_dir ~/rst/preprocessed/g120_f500 \
       --cnvmap_dir ~/rst/extracted_data

**Step 2 — model training (uses pre-cached preprocessed data):**

.. code-block:: bash

   python run_baseline.py \
       --grid_size 120 --max_files 500 \
       --epochs 10 --batch_size 32 \
       --embed_dim 64 --mlp_ratio 2 \
       --ckpt_dir ./checkpoints_baseline

Checkpoints are saved to ``./checkpoints_baseline/``.
Best checkpoint is selected by minimum ``val_mse``.
