"""
run_overnight.py — Unattended overnight run for both likelihoods.

Runs training + test evaluation sequentially for GaussianLMCLikelihood and
ChainedGammaLikelihood across multiple horizons, reusing already tuned
Optuna best hyperparameters (no Optuna search in this script).

Outputs
-------
  outputs_gaussian/   ← Gaussian baseline (faster, finishes first)
  outputs_gamma/      ← Chained-Gamma model (principled, slower)

Train-only strategy
-------------------
  - No Optuna is executed.
  - For each likelihood, best params are read from:
      outputs_<likelihood>/results/best_params_H1.txt
  - The same M/Q/T are reused for all requested horizons.

Usage
-----
    nohup python run_overnight.py > overnight.log 2>&1 &
    # or
    python run_overnight.py | tee overnight.log
"""

from pathlib import Path

from src.models.likelihoods import ChainedGammaLikelihood, GaussianLMCLikelihood
from run import main

# ── Overnight configuration ───────────────────────────────────────────────────
N_EPOCHS_FINAL_OVERNIGHT  = 100  # longer final training overnight
HORIZONS = [2, 3, 4, 5, 6, 7, 14, 21, 30]


def run_for_likelihood_and_horizons(likelihood_cls, out_dir_name: str) -> None:
    """Run train+eval for one likelihood across all configured horizons."""
    best_params_path = Path(out_dir_name) / "results" / "best_params_H1.txt"
    if not best_params_path.exists():
        raise FileNotFoundError(
            f"Best-params file not found: {best_params_path}. "
            "Run Optuna once for H=1 first."
        )

    print("\n" + "=" * 72)
    print(f"  Starting train-only sweep: {likelihood_cls.__name__}")
    print(f"  Output dir             : {out_dir_name}/")
    print(f"  Best params source     : {best_params_path}")
    print(f"  Horizons               : {HORIZONS}")
    print("=" * 72 + "\n")

    for h in HORIZONS:
        print("\n" + "-" * 72)
        print(f"  [{likelihood_cls.__name__}] Running H={h}")
        print("-" * 72)
        main(
            likelihood=likelihood_cls,
            n_epochs_final=N_EPOCHS_FINAL_OVERNIGHT,
            out_dir=Path(out_dir_name),
            horizon=h,
            best_params_file=best_params_path,
        )


if __name__ == "__main__":
    # Gaussian first — faster, good sanity-check while gamma runs later
    run_for_likelihood_and_horizons(GaussianLMCLikelihood, "outputs_gaussian")

    # Gamma second — principled model for positive skewed reservoir volumes
    run_for_likelihood_and_horizons(ChainedGammaLikelihood, "outputs_gamma")

    print("\n" + "=" * 72)
    print("  Overnight run complete.")
    print("  Results are available under:")
    print("    outputs_gaussian/results/metrics_H*.txt")
    print("    outputs_gamma/results/metrics_H*.txt")
    print("=" * 72)
