import  torch
import torch.distributions as dist
import gpytorch as gpy


class ChainedGammaLikelihood(gpy.likelihoods.Likelihood):
    def __init__(self, num_tasks, num_latents):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_latents = num_latents

    def forward(self, function_samples):
        assert function_samples.size(-1) == self.num_latents, (
            "Input size mismatch: expected twice the number of tasks."
        )
        f_alpha = function_samples[..., :self.num_tasks]
        f_beta  = function_samples[..., self.num_tasks:]

        # alpha: softplus(f) + 1  guarantees α ≥ 1 smoothly everywhere.
        #   α ≥ 1  →  Gamma is always unimodal (mode = (α-1)/β ≥ 0).
        #   α < 1  →  anti-modal spike near y=0; numerically unstable log_prob.
        #   The +1 shift is a zero-gradient-kill alternative to .clamp(min=1):
        #   gradients flow to zero as f→-∞ but are never exactly zero.
        #   Upper bound 11: softplus(f)+1 ≤ 11  ⇒  f ≤ softplus_inv(10) ≈ 10.
        alphas = (torch.nn.functional.softplus(f_alpha) + 1.0).clamp(max=11.0)

        # beta:  softplus(f) maps ℝ→(0,∞); clamp to [1e-4, 10].
        #   Lower 1e-4: prevents β→0 (rate near zero → E[y] diverges).
        #   Upper 10  : prevents β·y overflow (y∈[1e-6,1] → β·y ≤ 10).
        #   The clamp only activates outside normal operating range, so
        #   gradients are preserved through the entire training distribution.
        betas  = torch.nn.functional.softplus(f_beta).clamp(min=1e-4, max=10.0)

        return dist.Independent(dist.Gamma(alphas, betas), 1)


class GaussianLMCLikelihood(gpy.likelihoods.Likelihood):
    """
    Heteroscedastic Gaussian likelihood mirroring the chained-GP structure.

    The LMC model produces 2×D outputs.  This class interprets them as:
      f[..., :D]  →  μ  (predicted mean,  unconstrained)
      f[..., D:]  →  σ  (predicted std,   softplus → (0,∞))

    Returning Independent(Normal(μ, σ), 1) makes it a drop-in swap for
    ChainedGammaLikelihood: same num_tasks / num_latents interface, same
    expected_log_prob signature, same forward shape.

    When to prefer this over Gamma
    --------------------------------
    Use Gaussian for a quick sanity check or when volume residuals appear
    symmetric and approximately normal after min-max scaling.  Gamma is
    the principled choice for strictly-positive skewed data (reservoir
    volumes), but Gaussian is faster to optimise and easier to debug.
    """
    def __init__(self, num_tasks, num_latents):
        super().__init__()
        self.num_tasks   = num_tasks    # D
        self.num_latents = num_latents  # 2D

    def forward(self, function_samples):
        assert function_samples.size(-1) == self.num_latents, (
            "Input size mismatch: expected twice the number of tasks."
        )
        f_mu    = function_samples[..., :self.num_tasks]
        f_sigma = function_samples[..., self.num_tasks:]

        # μ: unconstrained — the LMC mean can be anything in ℝ.
        mu = f_mu

        # σ: softplus ensures σ > 0 everywhere with smooth gradients.
        #   Clamp to [1e-4, 5]: lower keeps Normal log_prob finite;
        #   upper prevents the distribution from becoming so wide that
        #   the ELBO is trivially satisfied by a flat predictive.
        sigma = torch.nn.functional.softplus(f_sigma).clamp(min=1e-4, max=5.0)

        return dist.Independent(dist.Normal(mu, sigma), 1)

    @property
    def noise_sigma(self):
        """Convenience property: returns the current mean predictive σ (for logging)."""
        return None  # parametric — no fixed noise parameter
