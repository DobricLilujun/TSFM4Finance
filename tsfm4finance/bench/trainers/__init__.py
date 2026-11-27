"""Per-model fine-tuning trainers (spec §4.1 adaptation ladder).

Why one trainer per family instead of one generic loop: the families disagree on
everything that matters for training - input normalisation (TimesFM raw units,
Chronos instance-norm, Toto causal scaling, Moirai patched windows), loss heads
(9-quantile pinball vs sampled paths vs AR token cross-entropy), and which
parameters should move at all.  A single abstraction would have to fake those
differences; here each family's loop is explicit and independently testable.

Common contract (enforced by ``tsfm4finance/tests/test_trainers.py``):

* windows are sampled **from the train split only** (``span`` never touches
  validation/test); the checkpoint metadata records ``train_end`` so the
  validation-span control can retrain and verify it;
* budgets: ``fs16`` = 16 windows / 40 steps, ``fs64`` = 64 windows / 120 steps,
  ``full`` = every train window for ``--epochs`` epochs (default 3);
* full-parameter updates by default (``--heads-only`` exists for ablation, and the
  metadata says which one ran, so a head-only run can never be reported as full);
* checkpoint: ``reports/ft_ckpts/<model>__<series>__<budget>.pt`` plus a JSON
  sidecar with steps, loss curve, trainable-parameter count, profile;
* ``--smoke`` runs two optimiser steps on two windows: cheap proof that the loop
  is wired (gradients flow, loss decreases) without claiming a real fine-tune.

Usage (the heavy runs are meant for the GPU server; see docs/05_fine_tuning.md):

  python -m tsfm4finance.bench.trainers --model timesfm-2.5-200m --series sp500_daily --budget full
  python -m tsfm4finance.bench.trainers --all-tier-A --series sp500_daily --budget fs64
  python -m tsfm4finance.bench.trainers --model chronos-2 --series btc_long --budget full --smoke
"""
