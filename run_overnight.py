"""
run_overnight.py — Unattended overnight run for both likelihoods.

Runs the full pipeline (Optuna search → final retraining → test evaluation)
sequentially for GaussianLMCLikelihood then ChainedGammaLikelihood.

Outputs
-------
  outputs_gaussian/   ← Gaussian baseline (faster, finishes first)
  outputs_gamma/      ← Chained-Gamma model (principled, slower)

Optuna strategy
---------------
A more explorative configuration is used compared to the default run.py:
  - More trials (N_OPTUNA_TRIALS_OVERNIGHT) to cover the search space better.
  - More epochs per trial so the ELBO signal is reliable before pruning.
  - HyperbandPruner: aggressive early-stopping for clearly bad trials while
    protecting promising ones through successive halving brackets.
    This lets us run many more trials in the same wall-clock time.
  - CmaEsSampler: covariance-matrix adaptation — much better than TPE for
    continuous hyperparameter spaces (M, Q are treated as continuous here).

Usage
-----
    nohup python run_overnight.py > overnight.log 2>&1 &
    # or
    python run_overnight.py | tee overnight.log
"""

import optuna
from pathlib import Path

from src.models.likelihoods import ChainedGammaLikelihood, GaussianLMCLikelihood
from run import main

# ── Overnight configuration ───────────────────────────────────────────────────
N_OPTUNA_TRIALS_OVERNIGHT = 40    # more trials for better coverage
N_EPOCHS_OPTUNA_OVERNIGHT = 500  # longer per trial — reliable ELBO signal
N_EPOCHS_FINAL_OVERNIGHT  = 8000  # longer final training overnight

# HyperbandPruner: brackets successive halving so a trial that is in the
# bottom half at epoch 50 is pruned before epoch 100, etc.
# min_resource=30   → never prune in the first 30 epochs (GP warm-up)
# max_resource=N_EPOCHS_OPTUNA_OVERNIGHT → full budget for bracket calculation
# reduction_factor=3 → each bracket is 3× longer than the previous
OVERNIGHT_PRUNER = optuna.pruners.HyperbandPruner(
    min_resource=30,
    max_resource=N_EPOCHS_OPTUNA_OVERNIGHT,
    reduction_factor=3,
)

# RandomSampler: pure random search — no model-based assumptions, uniform
# coverage of the search space. Good when the number of trials is large
# enough (>=40) and the space is not too smooth for Bayesian methods to help.
OVERNIGHT_SAMPLER = optuna.samplers.RandomSampler(seed=42)
# OVERNIGHT_SAMPLER = optuna.samplers.TPESampler()


def run_for_likelihood(likelihood_cls, out_dir_name: str) -> None:
    """Run the full pipeline for one likelihood class."""
    print("\n" + "=" * 72)
    print(f"  Starting run: {likelihood_cls.__name__}")
    print(f"  Output dir  : {out_dir_name}/")
    print("=" * 72 + "\n")

    main(
        likelihood=likelihood_cls,
        n_optuna_trials=N_OPTUNA_TRIALS_OVERNIGHT,
        n_epochs_optuna=N_EPOCHS_OPTUNA_OVERNIGHT,
        n_epochs_final=N_EPOCHS_FINAL_OVERNIGHT,
        out_dir=Path(out_dir_name),
        pruner=optuna.pruners.NopPruner(),   # no pruning — every trial runs to completion
        sampler=OVERNIGHT_SAMPLER,
    )


if __name__ == "__main__":
    # Gaussian first — faster, good sanity-check while gamma runs later
    # run_for_likelihood(GaussianLMCLikelihood, "outputs_gaussian")

    # Gamma second — principled model for positive skewed reservoir volumes
    run_for_likelihood(ChainedGammaLikelihood, "outputs_gamma")

    print("\n" + "=" * 72)
    print("  Overnight run complete.")
    print("  Results:")
    print("    outputs_gaussian/results/metrics_H1.txt")
    print("    outputs_gamma/results/metrics_H1.txt")
    print("=" * 72)
