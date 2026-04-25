"""
objective.py — Optuna objective factory for the ChdGamma model.

Tuned hyper-parameters
----------------------
  M : inducing points       (32 – 256, step 32)
      Controls the quality of the sparse GP approximation; larger M
      gives a better fit but is O(M³) more expensive per step.

  Q : latent GPs            (5 – 40, step 5)
      Sets the rank of the LMC coregionalisation matrix.

  T : model order / window  (1 – 7)
      How many past time steps are used as input features.

Validation strategy
-------------------
A temporal (causal) hold-out is carved from the training tensor:
  - Optuna-train : first  (1 − VAL_FRACTION) rows of V_train_norm
  - Optuna-val   : last   VAL_FRACTION        rows of V_train_norm

This guarantees no information leakage.  For sliding-window samples, the
target of sample n is V[n+H-1].  With a random shuffle a later training
sample's input window can directly contain a val target — label leakage.
The temporal boundary ensures every training target index is strictly less
than every validation target index:
  max train target = n_tr_rows − 1
  min val   target = n_tr_rows          (no overlap)

Val inputs ARE allowed to look back into training time — that is just
providing input context, not leaking future targets.

The val split is NEVER exposed during final model retraining in run.py.
NLPD on the val set is the minimised Optuna objective.
"""

import torch
import optuna
from gpytorch.utils.errors import NotPSDError

from src.data.build_dataset import build_dataset
from src.models.lmc_gp import LMCChdGP
from src.models.likelihoods import ChainedGammaLikelihood, GaussianLMCLikelihood
from src.training.trainer import train_model, nlpd_metric

# Last VAL_FRACTION of the training tensor rows is held out for Optuna val.
# Causal ordering is preserved: val targets come strictly after all train targets.
VAL_FRACTION = 0.2


def make_objective(
    V_train_norm: torch.Tensor,
    D: int,
    *,
    H: int = 1,
    n_epochs_per_trial: int = 80,
    batch_size: int = 256,
    lr_adam: float = 0.01,
    lr_ngd: float = 0.1,
    likelihood_cls=ChainedGammaLikelihood,
    device: torch.device = None,
):
    """
    Returns an Optuna-compatible objective callable.

    Parameters
    ----------
    V_train_norm      : (N_train, D) normalised training tensor (on CPU or GPU)
    D                 : number of reservoirs
    H                 : forecast horizon (fixed per study; change in run.py)
    n_epochs_per_trial: training epochs per Optuna trial
    likelihood_cls    : ChainedGammaLikelihood (default) or GaussianLMCLikelihood
    device            : torch device (defaults to CUDA if available)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Temporal val split — row boundary computed once outside trial loop ─
    # All samples whose target index < n_tr_rows go to Optuna-train;
    # the rest go to Optuna-val.  The boundary guarantees:
    #   max(train target row) = n_tr_rows - 1
    #   min(val  target row)  = n_tr_rows
    # so no val target can ever appear as an input feature in a train sample.
    N        = len(V_train_norm)
    n_tr_rows = N - int(N * VAL_FRACTION)  # first n_tr_rows rows → optuna-train

    # Datasets are built per trial because T varies per trial.

    def objective(trial: optuna.Trial) -> float:
        M = trial.suggest_int("M", 1, 300)
        Q = trial.suggest_int("Q", 1, 70)
        T = trial.suggest_categorical("T", [1, 2, 3, 7, 14, 21, 30])

        # ── Build train/val sample pairs for this T ────────────────────────
        X_all, Y_all = build_dataset(V_train_norm, T, H)
        # Temporal split: sample k has target V_train_norm[T + k + H - 1].
        # Targets are in Optuna-train when T + k + H - 1 < n_tr_rows,
        # i.e. k < n_tr_rows - T - H + 1  →  n_tr_samples train samples.
        n_tr_samples = n_tr_rows - T - H + 1
        if n_tr_samples <= 0 or len(X_all) - n_tr_samples <= 0:
            raise optuna.exceptions.TrialPruned()

        X_tr,  Y_tr  = X_all[:n_tr_samples],  Y_all[:n_tr_samples]
        X_val, Y_val = X_all[n_tr_samples:],  Y_all[n_tr_samples:]

        Y_tr  = Y_tr.clamp(min=1e-6).to(device)
        Y_val = Y_val.clamp(min=1e-6).to(device)
        X_tr  = X_tr.to(device)
        X_val = X_val.to(device)

        # ── Inducing points: k-means centres in input space ───────────────────
        # Random subsets often cluster inducing points in dense regions.
        # k-means spreads them more evenly, giving better GP approximation
        # for the same M — especially important for structured time-series.
        n_inducing = min(M, len(X_tr))
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=n_inducing, n_init=3, random_state=0)
        km.fit(X_tr.cpu().numpy())
        # Shape (Q, M, input_dim): each latent GP starts from the same k-means
        # centres but learns its own locations during training.  Without the Q
        # leading dimension, all latents share one set of locations and gradient
        # updates are a compromise across Q — limiting expressivity.
        inducing_points = torch.tensor(
            km.cluster_centers_, dtype=X_tr.dtype, device=device
        ).unsqueeze(0).expand(Q, -1, -1).contiguous()  # (Q, M, input_dim)

        # ── Build model and likelihood ─────────────────────────────────────
        model = LMCChdGP(
            num_tasks=2 * D,   # D alpha + D beta channels
            num_latents=Q,
            inducing_points=inducing_points,
        ).to(device)

        likelihood = likelihood_cls(
            num_tasks=D,
            num_latents=2 * D,
        ).to(device)

        # ── Train on optuna-train portion ──────────────────────────────────
        # epoch_callback reports the negative ELBO to the pruner each epoch.
        # Using ELBO (free, already computed) avoids expensive val NLPD probes.
        # n_warmup_steps in MedianPruner prevents pruning until epoch >= warmup.
        def _report_and_prune(epoch: int, avg_neg_elbo: float):
            trial.report(avg_neg_elbo, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            train_model(
                model,
                likelihood,
                X_tr,
                Y_tr,
                num_epochs=n_epochs_per_trial,
                batch_size=batch_size,
                lr_adam=lr_adam,
                lr_ngd=lr_ngd,
                epoch_callback=_report_and_prune,
            )
        except (NotPSDError, ValueError, RuntimeError):
            # Numerical failure (NaN, non-PSD) for this (M, Q, T) combination.
            # Prune gracefully so Optuna records it as pruned and moves on.
            raise optuna.TrialPruned()

        # ── Evaluate NLPD on temporal val set (Optuna minimises this) ──────
        # Reached only if the trial was not pruned
        nlpd = nlpd_metric(model, likelihood, X_val, Y_val)

        # Clean up GPU memory between trials
        del model, likelihood
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return nlpd

    return objective
