"""
run.py — Main pipeline for the ChdGamma hydrology forecasting model.

Stages
------
1. Load and preprocess reservoir volume data from Excel.
2. Run Optuna study to tune M (inducing points), Q (latent GPs), T (window).
   - Study is persisted to SQLite; re-running resumes from the last checkpoint.
   - Objective: NLPD on a temporal val split carved from the training set.
   - The test set is NEVER accessed during this stage.
3. Retrain with best hyper-parameters on the FULL training set.
4. Evaluate on the held-out test set (NLPD + predictive samples).
5. Save model, scaler, results.

Configuration
-------------
Edit the constants below to change the forecast horizon (H),
number of Optuna trials, epochs, learning rates, etc.
"""

import joblib
from pathlib import Path

import torch
import optuna

from src.data.preprocessing import load_and_preprocess
from src.data.build_dataset import build_dataset
from src.models.lmc_gp import LMCChdGP
from src.models.likelihoods import ChainedGammaLikelihood, GaussianLMCLikelihood
from src.training.trainer import train_model, predict, nlpd_metric
from src.tuning.objective import make_objective
from src.utils.seed import set_seed

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts/servers
import matplotlib.pyplot as plt


def _load_best_params_file(path: Path) -> dict:
    """Load best params saved as key=value lines."""
    params = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            params[k.strip()] = v.strip()
    return params

# ── Configuration ──────────────────────────────────────────────────────────────
EXCEL_PATH          = "Ejercicio hidrología v2.xlsx"
SHEET_INDEX         = 1           # "Datos generales" (0-based)

H                   = 1           # forecast horizon — change here for multi-horizon runs

# ── Likelihood toggle ─────────────────────────────────────────────────────────
# Set to ChainedGammaLikelihood (principled for positive skewed data) or
# GaussianLMCLikelihood (faster, symmetric — useful as a sanity-check baseline).
# Both use the same LMC model (num_tasks=2*D): first D outputs → distribution
# location, last D outputs → scale/rate via softplus.
LIKELIHOOD_CHOICES  = {
    "gaussian": GaussianLMCLikelihood,
    "gamma": ChainedGammaLikelihood,
}
LIKELIHOOD_NAME     = "gaussian"
LIKELIHOOD          = LIKELIHOOD_CHOICES[LIKELIHOOD_NAME]
# ─────────────────────────────────────────────────────────────────────────────

N_OPTUNA_TRIALS     = 15         # Optuna trials
N_EPOCHS_OPTUNA     = 1000         # epochs per trial (fewer than final training)
N_EPOCHS_FINAL      = 5000         # epochs for final model (best params, full train set)

BATCH_SIZE          = 500#3554
LR_ADAM             = 0.01
LR_NGD              = 0.1
SEED                = np.random.randint(0, 1_000_000)

OUT_DIR             = Path("outputs")
# OPTUNA_DB is built dynamically inside main() so it encodes LIKELIHOOD.__name__
# ───────────────────────────────────────────────────────────────────────────────

# ── Default Optuna pruner (used when run directly via __main__) ───────────────
DEFAULT_PRUNER = optuna.pruners.NopPruner()   # no pruning by default
# To enable pruning, replace with e.g.:
# DEFAULT_PRUNER = optuna.pruners.MedianPruner(
#     n_startup_trials=5,
#     n_warmup_steps=20,
#     interval_steps=5,
# )

# ── Default Optuna sampler (used when run directly via __main__) ──────────────
DEFAULT_SAMPLER = None  # None → Optuna uses TPE (default)


def save_forecast_plots(
    Y_test, mean, median, p025, p975,
    reservoir_names, scaler, out_dir, H,
):
    """
    Save one PNG per reservoir showing test observations vs GP predictive
    distribution (mean, median, and 0.025–0.975 credible-interval shading).
    All tensors are moved to CPU and denormalised to original m³ units.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_min = scaler["x_min"]  # (1, D)  numpy
    x_max = scaler["x_max"]  # (1, D)  numpy
    scale = x_max - x_min    # (1, D)

    def denorm(t):
        arr = t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        return arr * scale + x_min   # broadcast over N dimension (N, D)

    Y_np    = denorm(Y_test)
    mean_np = denorm(mean)
    med_np  = denorm(median)
    lo_np   = denorm(p025)
    hi_np   = denorm(p975)

    N = Y_np.shape[0]
    steps = np.arange(N)

    for d, name in enumerate(reservoir_names):
        fig, ax = plt.subplots(figsize=(14, 4))

        ax.fill_between(
            steps, lo_np[:, d], hi_np[:, d],
            alpha=0.30, color="steelblue", label="95 % CI (0.025–0.975)"
        )
        ax.plot(steps, Y_np[:, d],    color="black",      lw=1.2, label="Observed")
        ax.plot(steps, mean_np[:, d], color="tomato",     lw=1.2, label="Predictive mean")
        ax.plot(steps, med_np[:, d],  color="darkorange", lw=1.0,
                linestyle="--", label="Predictive median")

        ax.set_title(f"{name}  —  H = {H} step-ahead forecast (test set)")
        ax.set_xlabel("Test time step")
        ax.set_ylabel("Volume (m\u00b3)")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        safe_name = name.replace(" ", "_").replace("/", "-")
        out_path = fig_dir / f"forecast_H{H}_{safe_name}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"    [{d+1}/{len(reservoir_names)}] {safe_name}")

    print(f"  Forecast plots saved to '{fig_dir}/'")


def main(
    *,
    likelihood=None,
    n_optuna_trials=None,
    n_epochs_optuna=None,
    n_epochs_final=None,
    out_dir=None,
    pruner=None,
    sampler=None,
    seed=None,
    horizon=None,
    best_params=None,
    best_params_file=None,
):
    """
    Run the full pipeline.

    All parameters default to the module-level constants so that calling
    ``main()`` with no arguments reproduces the original behaviour.
    External scripts (e.g. run_overnight.py) can override any of these
    to run multiple configurations without duplicating code.

    Parameters
    ----------
    likelihood      : likelihood class (ChainedGammaLikelihood or GaussianLMCLikelihood)
    n_optuna_trials : number of Optuna trials
    n_epochs_optuna : training epochs per Optuna trial
    n_epochs_final  : training epochs for final model
    out_dir         : Path — root output directory (e.g. Path("outputs_gamma"))
    pruner          : optuna.pruners.BasePruner instance
    sampler         : optuna.samplers.BaseSampler instance (None → TPE)
    seed            : integer random seed (None → random)
    horizon         : forecast horizon override (None → module constant H)
    best_params     : dict with fixed params {"M", "Q", "T"} to skip Optuna
    best_params_file: Path to a key=value file containing M/Q/T (and optional
                      val_nlpd) to skip Optuna
    """
    # Apply defaults from module-level constants
    if likelihood is None:
        likelihood = LIKELIHOOD
    if n_optuna_trials is None:
        n_optuna_trials = N_OPTUNA_TRIALS
    if n_epochs_optuna is None:
        n_epochs_optuna = N_EPOCHS_OPTUNA
    if n_epochs_final is None:
        n_epochs_final = N_EPOCHS_FINAL
    if out_dir is None:
        out_dir = OUT_DIR
    if pruner is None:
        pruner = DEFAULT_PRUNER
    if sampler is None:
        sampler = DEFAULT_SAMPLER
    if seed is None:
        seed = SEED

    if horizon is None:
        horizon = H

    if best_params is None and best_params_file is not None:
        best_params = _load_best_params_file(Path(best_params_file))

    out_dir = Path(out_dir)

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Create output directories
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)
    (out_dir / "optuna").mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess ─────────────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    V_train_norm, V_test_norm, scaler, reservoir_names = load_and_preprocess(
        EXCEL_PATH, sheet=SHEET_INDEX
    )
    D = V_train_norm.shape[1]
    print(f"  Reservoirs (D)  : {D}")
    print(f"  Training rows   : {len(V_train_norm)}")
    print(f"  Test rows       : {len(V_test_norm)}")
    print(f"  Reservoir's names: {reservoir_names}")

    # Move to device after preprocessing (scaler stays on CPU)
    V_train_norm = V_train_norm.to(device)
    V_test_norm  = V_test_norm.to(device)

    # ── 2. Optuna study ──────────────────────────────────────────────────
    val_nlpd_optuna = None
    if best_params is None:
        print(f"\n[2/5] Optuna search — {n_optuna_trials} trials, H={horizon}, likelihood={likelihood.__name__}...")

        objective = make_objective(
            V_train_norm=V_train_norm,
            D=D,
            H=horizon,
            n_epochs_per_trial=n_epochs_optuna,
            batch_size=BATCH_SIZE,
            lr_adam=LR_ADAM,
            lr_ngd=LR_NGD,
            likelihood_cls=likelihood,
            device=device,
        )

        study_name = f"chdgamma_{likelihood.__name__}_H{horizon}"
        OPTUNA_DB  = str(out_dir / "optuna" / f"{study_name}.db")
        print(f"  Study DB : {OPTUNA_DB}")

        # Persist study to SQLite — re-running this script resumes the study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            storage=f"sqlite:///{OPTUNA_DB}",
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )
        study.optimize(
            objective,
            n_trials=n_optuna_trials,
            show_progress_bar=True,
            catch=(ValueError, RuntimeError),   # trial-level errors → TrialFailed, not study crash
        )

        best = study.best_params
        val_nlpd_optuna = float(study.best_value)
        print(f"\n  Best params     : {best}")
        print(f"  Best NLPD (val) : {val_nlpd_optuna:.4f}")

        # Persist best params as a text file
        results_file = out_dir / "results" / f"best_params_H{horizon}.txt"
        with open(results_file, "w", encoding="utf-8") as f:
            f.write(f"H={horizon}\n")
            for k, v in best.items():
                f.write(f"{k}={v}\n")
            f.write(f"val_nlpd={val_nlpd_optuna:.6f}\n")
    else:
        print(f"\n[2/5] Optuna skipped — using fixed params, H={horizon}, likelihood={likelihood.__name__}...")
        best = dict(best_params)
        if "val_nlpd" in best:
            try:
                val_nlpd_optuna = float(best["val_nlpd"])
            except (TypeError, ValueError):
                val_nlpd_optuna = None

        for key in ("M", "Q", "T"):
            if key not in best:
                raise ValueError(f"Missing required fixed param '{key}'")

    M_best = int(best["M"])
    Q_best = int(best["Q"])
    T_best = int(best["T"])

    # ── 3. Final retraining on the FULL training set ─────────────────────
    print(f"\n[3/5] Retraining — M={M_best}, Q={Q_best}, T={T_best}, H={horizon}...")

    X_train, Y_train = build_dataset(V_train_norm, T_best, horizon)
    Y_train = Y_train.clamp(min=1e-6)
    print(f"  Training samples: {len(X_train)}")

    from sklearn.cluster import MiniBatchKMeans
    _km = MiniBatchKMeans(n_clusters=M_best, n_init=5, random_state=seed)
    _km.fit(X_train.cpu().numpy())
    inducing_points = (
        torch.tensor(_km.cluster_centers_, dtype=X_train.dtype, device=device)
        .unsqueeze(0).expand(Q_best, -1, -1).contiguous()   # (Q, M, input_dim)
    )

    likelihood_cls = likelihood   # keep reference to class before instantiation

    model = LMCChdGP(
        num_tasks=2 * D,
        num_latents=Q_best,
        inducing_points=inducing_points,
    ).to(device)

    likelihood_instance = likelihood_cls(
        num_tasks=D,
        num_latents=2 * D,
    ).to(device)

    train_model(
        model,
        likelihood_instance,
        X_train,
        Y_train,
        num_epochs=n_epochs_final,
        batch_size=BATCH_SIZE,
        lr_adam=LR_ADAM,
        lr_ngd=LR_NGD,
    )

    # Save model and scaler
    torch.save(model.state_dict(),               out_dir / "models" / f"model_H{horizon}.pt")
    torch.save(likelihood_instance.state_dict(), out_dir / "models" / f"likelihood_H{horizon}.pt")
    joblib.dump(scaler,                          out_dir / "models" / f"scaler_H{horizon}.pkl")
    print("  Checkpoints saved.")

    # ── 4. Test evaluation — ONLY here, never before ─────────────────────
    print("\n[4/5] Test evaluation...")

    # Prepend the last T_best observations from training as input context
    # so the very first test sample has a valid input window.
    V_test_ctx = torch.cat([V_train_norm[-T_best:], V_test_norm], dim=0)
    X_test, Y_test = build_dataset(V_test_ctx, T_best, horizon)
    Y_test = Y_test.clamp(min=1e-6)
    print(f"  Test samples    : {len(X_test)}")

    nlpd_test = nlpd_metric(model, likelihood_instance, X_test, Y_test)
    print(f"  NLPD (test)     : {nlpd_test:.4f}")

    mean, median, p025, p975 = predict(model, likelihood_instance, X_test)

    # Save test metrics
    with open(out_dir / "results" / f"metrics_H{horizon}.txt", "w", encoding="utf-8") as f:
        f.write(f"Model      : ChdGP + {likelihood_cls.__name__}\n")
        f.write(f"H={horizon}  M={M_best}  Q={Q_best}  T={T_best}\n")
        f.write(f"NLPD_test      : {nlpd_test:.6f}\n")
        if val_nlpd_optuna is not None:
            f.write(f"NLPD_val_optuna: {val_nlpd_optuna:.6f}\n")

    # ── 5. Forecast plots ────────────────────────────────────────────────
    print("\n[5/5] Saving per-reservoir forecast plots...")
    save_forecast_plots(
        Y_test, mean, median, p025, p975,
        reservoir_names, scaler, out_dir, horizon,
    )

    print(f"\nAll outputs saved to '{out_dir}/'")
    return model, likelihood_instance, nlpd_test, mean, median, p025, p975


if __name__ == "__main__":
    main()
