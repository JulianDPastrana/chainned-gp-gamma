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

# ── Configuration ──────────────────────────────────────────────────────────────
EXCEL_PATH          = "Ejercicio hidrología v2.xlsx"
SHEET_INDEX         = 1           # "Datos generales" (0-based)

H                   = 1           # forecast horizon — change here for multi-horizon runs

# ── Likelihood toggle ─────────────────────────────────────────────────────────
# Set to ChainedGammaLikelihood (principled for positive skewed data) or
# GaussianLMCLikelihood (faster, symmetric — useful as a sanity-check baseline).
# Both use the same LMC model (num_tasks=2*D): first D outputs → distribution
# location, last D outputs → scale/rate via softplus.
# LIKELIHOOD          = ChainedGammaLikelihood   # ← swap to GaussianLMCLikelihood to compare
LIKELIHOOD          = GaussianLMCLikelihood   # ← swap to ChainedGammaLikelihood for the principled choice
# ─────────────────────────────────────────────────────────────────────────────

N_OPTUNA_TRIALS     = 15         # Optuna trials
N_EPOCHS_OPTUNA     = 1000         # epochs per trial (fewer than final training)
N_EPOCHS_FINAL      = 5000         # epochs for final model (best params, full train set)

BATCH_SIZE          = 3554
LR_ADAM             = 0.01
LR_NGD              = 0.1
SEED                = np.random.randint(0, 1_000_000)

OUT_DIR             = Path("outputs")
# OPTUNA_DB is built dynamically inside main() so it encodes LIKELIHOOD.__name__
# ───────────────────────────────────────────────────────────────────────────────


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


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Create output directories
    (OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "optuna").mkdir(parents=True, exist_ok=True)

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
    print(f"\n[2/5] Optuna search — {N_OPTUNA_TRIALS} trials, H={H}, likelihood={LIKELIHOOD.__name__}...")

    objective = make_objective(
        V_train_norm=V_train_norm,
        D=D,
        H=H,
        n_epochs_per_trial=N_EPOCHS_OPTUNA,
        batch_size=BATCH_SIZE,
        lr_adam=LR_ADAM,
        lr_ngd=LR_NGD,
        likelihood_cls=LIKELIHOOD,
        device=device,
    )

    study_name = f"chdgamma_{LIKELIHOOD.__name__}_H{H}"
    OPTUNA_DB  = str(OUT_DIR / "optuna" / f"{study_name}.db")
    print(f"  Study DB : {OPTUNA_DB}")

    # Persist study to SQLite — re-running this script resumes the study
    # MedianPruner: skip first 5 trials (need baseline) and first 20 epochs
    # of each trial (GP needs warm-up before ELBO becomes a reliable signal).
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=f"sqlite:///{OPTUNA_DB}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=5,     # check every 5 epochs, not every epoch
        ),
    )
    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
        catch=(ValueError, RuntimeError),   # trial-level errors → TrialFailed, not study crash
    )

    best = study.best_params
    print(f"\n  Best params     : {best}")
    print(f"  Best NLPD (val) : {study.best_value:.4f}")

    # Persist best params as a text file
    results_file = OUT_DIR / "results" / f"best_params_H{H}.txt"
    with open(results_file, "w") as f:
        f.write(f"H={H}\n")
        for k, v in best.items():
            f.write(f"{k}={v}\n")
        f.write(f"val_nlpd={study.best_value:.6f}\n")

    M_best = best["M"]
    Q_best = best["Q"]
    T_best = best["T"]

    # ── 3. Final retraining on the FULL training set ─────────────────────
    print(f"\n[3/5] Retraining — M={M_best}, Q={Q_best}, T={T_best}, H={H}...")

    X_train, Y_train = build_dataset(V_train_norm, T_best, H)
    Y_train = Y_train.clamp(min=1e-6)
    print(f"  Training samples: {len(X_train)}")

    # k-means inducing points.
    # Shape (Q, M, input_dim): each latent GP starts from the same centres but
    # learns its own locations — without the Q leading dim all latents share one
    # compromise set and expressivity is limited.
    from sklearn.cluster import MiniBatchKMeans
    _km = MiniBatchKMeans(n_clusters=M_best, n_init=5, random_state=SEED)
    _km.fit(X_train.cpu().numpy())
    inducing_points = (
        torch.tensor(_km.cluster_centers_, dtype=X_train.dtype, device=device)
        .unsqueeze(0).expand(Q_best, -1, -1).contiguous()   # (Q, M, input_dim)
    )

    model = LMCChdGP(
        num_tasks=2 * D,
        num_latents=Q_best,
        inducing_points=inducing_points,
    ).to(device)

    likelihood = LIKELIHOOD(
        num_tasks=D,
        num_latents=2 * D,
    ).to(device)

    train_model(
        model,
        likelihood,
        X_train,
        Y_train,
        num_epochs=N_EPOCHS_FINAL,
        batch_size=BATCH_SIZE,
        lr_adam=LR_ADAM,
        lr_ngd=LR_NGD,
    )

    # Save model and scaler
    torch.save(model.state_dict(),      OUT_DIR / "models" / f"model_H{H}.pt")
    torch.save(likelihood.state_dict(), OUT_DIR / "models" / f"likelihood_H{H}.pt")
    joblib.dump(scaler,                 OUT_DIR / "models" / f"scaler_H{H}.pkl")
    print("  Checkpoints saved.")

    # ── 4. Test evaluation — ONLY here, never before ─────────────────────
    print("\n[4/5] Test evaluation...")

    # Prepend the last T_best observations from training as input context
    # so the very first test sample has a valid input window.
    V_test_ctx = torch.cat([V_train_norm[-T_best:], V_test_norm], dim=0)
    X_test, Y_test = build_dataset(V_test_ctx, T_best, H)
    Y_test = Y_test.clamp(min=1e-6)
    print(f"  Test samples    : {len(X_test)}")

    nlpd_test = nlpd_metric(model, likelihood, X_test, Y_test)
    print(f"  NLPD (test)     : {nlpd_test:.4f}")

    mean, median, p025, p975 = predict(model, likelihood, X_test)

    # Save test metrics
    with open(OUT_DIR / "results" / f"metrics_H{H}.txt", "w") as f:
        f.write(f"Model      : ChdGP + {LIKELIHOOD.__name__}\n")
        f.write(f"H={H}  M={M_best}  Q={Q_best}  T={T_best}\n")
        f.write(f"NLPD_test      : {nlpd_test:.6f}\n")
        f.write(f"NLPD_val_optuna: {study.best_value:.6f}\n")

    # ── 5. Forecast plots ────────────────────────────────────────────────
    print("\n[5/5] Saving per-reservoir forecast plots...")
    save_forecast_plots(
        Y_test, mean, median, p025, p975,
        reservoir_names, scaler, OUT_DIR, H,
    )

    print(f"\nAll outputs saved to '{OUT_DIR}/'")
    return model, likelihood, nlpd_test, mean, median, p025, p975


if __name__ == "__main__":
    main()
