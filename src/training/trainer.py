"""
trainer.py — Core training and evaluation primitives for the ChdGamma model.

Public API
----------
  train_model()  : mini-batch ELBO with hybrid NGD + Adam optimisation
  predict()      : Monte Carlo predictive samples → mean / CI
  nlpd_metric()  : Negative Log Predictive Density  (lower is better)

The Optuna objective (make_objective) lives in src/tuning/objective.py
and imports these primitives.
"""

import numpy as np
import torch
import gpytorch as gpy


# ─── 1. TRAINING ──────────────────────────────────────────────────────────────

def train_model(
    model: gpy.models.ApproximateGP,
    likelihood: gpy.likelihoods.Likelihood,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    *,
    num_epochs: int = 200,
    batch_size: int = 256,
    lr_adam: float = 0.01,
    lr_ngd: float = 0.01,
    epoch_callback=None,
) -> list:
    """
    Mini-batch variational training with hybrid NGD + Adam optimisation.

    Parameter groups
    ----------------
    NGD  → variational parameters  {mean vector m_q, covariance S_q,
                                    inducing locations Z_q}
           Natural gradients are the correct update rule for distributions
           in the exponential family; they account for the curvature of the
           KL divergence and converge faster than ordinary SGD on q.

    Adam → kernel hyper-parameters (lengthscale, outputscale) and
           likelihood parameters (softplus shape/rate of Gamma)

    Update order: NGD first, then Adam.  Both optimisers see the same
    backward pass, so there is no extra forward call.

    epoch_callback : optional callable(epoch: int, avg_neg_elbo: float)
        Called at the end of every epoch.  May raise optuna.TrialPruned
        to abort training early.  Keeps trainer.py decoupled from Optuna.
    """
    device = X_train.device
    N = X_train.size(0)

    # Natural Gradient for the variational distribution q(u)
    ngd_optimizer = gpy.optim.NGD(
        model.variational_parameters(),
        num_data=N,
        lr=lr_ngd,
    )

    # Adam for everything else (kernel + likelihood hyperparameters)
    adam_optimizer = torch.optim.Adam(
        [
            {"params": model.hyperparameters()},
            {"params": likelihood.parameters()},
        ],
        lr=lr_adam,
    )

    # ELBO = E_q[log p(y|f)] − KL[q(f) ‖ p(f)]
    mll = gpy.mlls.VariationalELBO(likelihood, model, num_data=N)

    model.train()
    likelihood.train()

    loss_history = []

    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            x_b, y_b = X_train[idx], Y_train[idx]

            ngd_optimizer.zero_grad()
            adam_optimizer.zero_grad()

            # Slight Cholesky jitter for numerical stability with LMC kernels.
            # num_likelihood_samples=64: GPyTorch default is 10, which is
            # sufficient for Gaussian likelihoods (analytic E_q[log p]) but
            # causes 2× extra gradient noise for non-Gaussian likelihoods like
            # ChainedGamma where expected_log_prob uses Monte Carlo sampling.
            # 64 samples reduces that noise by √(64/10) ≈ 2.5× at modest cost.
            with gpy.settings.cholesky_jitter(1e-3), \
                 gpy.settings.num_likelihood_samples(64):
                output = model(x_b)
                loss = -mll(output, y_b)   # minimise negative ELBO

            # Skip batch if loss is non-finite (NaN or ±inf).
            # inf occurs when a tail f-sample gives an extreme Gamma parameter
            # (beta → ∞ → log_prob = -∞) before the likelihood clamp takes effect.
            # NaN occurs with ill-conditioned Cholesky at initialisation.
            # In both cases propagating gradients corrupts the model permanently.
            if not torch.isfinite(loss):
                ngd_optimizer.zero_grad()
                adam_optimizer.zero_grad()
                continue

            loss.backward()

            # Clip variational gradients BEFORE the NGD step.
            # When the Fisher matrix is ill-conditioned (early training, large Q),
            # the natural gradient F^{-1} g can explode even if g is small.
            # Clamping g first prevents the natural parameter from going to NaN.
            # torch.nn.utils.clip_grad_norm_(
            #     list(model.variational_parameters()), max_norm=5.0
            # )
            # # Clip kernel/hyperparameter gradients (Adam side)
            # torch.nn.utils.clip_grad_norm_(
            #     list(model.hyperparameters()), max_norm=1.0
            # )

            ngd_optimizer.step()   # update variational params (q)
            adam_optimizer.step()  # update kernel / likelihood hyperparams

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / n_batches
        loss_history.append(avg)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{num_epochs} | −ELBO: {avg:.4f}")

        # Notify the caller (e.g. Optuna pruner) after each epoch
        if epoch_callback is not None:
            epoch_callback(epoch, avg)

    return loss_history


# ─── 2. PREDICTION ────────────────────────────────────────────────────────────

def predict(
    model: gpy.models.ApproximateGP,
    likelihood: gpy.likelihoods.Likelihood,
    X_test: torch.Tensor,
    *,
    n_samples: int = 1024,
):
    """
    Monte Carlo predictive distribution over test points.

    Draws S samples from q(f), passes each through the Gamma likelihood
    (softplus link), then returns summary statistics over output samples.

    Returns
    -------
    mean, median, p025, p975   each of shape (N, D)
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpy.settings.fast_pred_var(), \
         gpy.settings.num_likelihood_samples(64):
        f_dist = model(X_test)
        # f_samples: (S, N, 2D) — clamp prevents softplus overflow
        f_samples = f_dist.rsample(torch.Size([n_samples]))# .clamp(-8.0, 8.0)

        # Delegate to likelihood.forward() — applies softplus link and builds
        # Independent(Gamma(alpha, beta), 1) consistently with training
        y_dist    = likelihood.forward(f_samples)   # batch (S,N), event (D,)
        y_samples = y_dist.rsample()                # (S, N, D)

    mean   = y_samples.mean(0)
    median = y_samples.median(0).values
    p025   = y_samples.quantile(0.025, dim=0)
    p975   = y_samples.quantile(0.975, dim=0)

    return mean, median, p025, p975


# ─── 3. NLPD ──────────────────────────────────────────────────────────────────

def nlpd_metric(
    model: gpy.models.ApproximateGP,
    likelihood: gpy.likelihoods.Likelihood,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    *,
    n_samples: int = 512,
) -> float:
    """
    Negative Log Predictive Density  (lower is better; Optuna minimises).

    NLPD = −(1/N) Σ_n  log p(y_n* | y_train)

    The marginal log-likelihood is intractable, so we use an importance-
    sampling estimate with S samples from q(f):

        log p(y_n*|y) ≈ logsumexp_s [log p(y_n*|f_s)] − log S

    Averaging over D reservoirs before the logsumexp keeps the estimate
    in a numerically stable range regardless of D.
    """
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpy.settings.fast_pred_var():
        f_dist = model(X_test)
        f_samples = f_dist.rsample(torch.Size([n_samples])).clamp(-8.0, 8.0)  # (S, N, 2D)

        # Use likelihood.forward() — same softplus+Gamma as training;
        # Independent(Gamma, 1) sums log_prob over D reservoirs (joint NLPD)
        y_dist = likelihood.forward(f_samples)   # batch (S, N), event (D,)

        # log p(y* | f_s): joint over D reservoirs  →  (S, N)
        log_p = y_dist.log_prob(
            Y_test.unsqueeze(0).expand(n_samples, -1, -1)
        )

        # IS estimate: log(1/S Σ_s p(y*|f_s)) = logsumexp_s − log S
        log_pred = torch.logsumexp(log_p, dim=0) - np.log(n_samples)
        nlpd = -log_pred.mean().item()

    return nlpd


