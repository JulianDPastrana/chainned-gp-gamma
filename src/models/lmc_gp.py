import  gpytorch as gpy
import torch

class LMCChdGP(gpy.models.ApproximateGP):
    def __init__(self, num_tasks, num_latents, inducing_points):
        # TrilNaturalVariationalDistribution parameterises q(u) via a lower-
        # triangular factor of S⁻¹, so the covariance is PSD by construction.
        variational_distribution = gpy.variational.TrilNaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = gpy.variational.LMCVariationalStrategy(
            base_variational_strategy=gpy.variational.VariationalStrategy(
                model=self,
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )

        super().__init__(variational_strategy)

        # ConstantMean: one learned offset per latent GP.
        # Reservoir volumes are normalised to [1e-6, 1] but their mean is not
        # zero, so a constant mean term is warranted.
        self.mean_module = gpy.means.ConstantMean(batch_shape=torch.Size([num_latents]))

        # Matérn 5/2 with ARD — one lengthscale per input feature (T*D dims).
        #
        # Why Matérn 5/2 instead of RBF:
        #   RBF is infinitely differentiable, implying the latent function is
        #   perfectly smooth everywhere.  Reservoir volumes are driven by
        #   seasonal transitions, storm inflows, and dry spells — physical
        #   processes that produce "kinks" (non-smooth behaviour).  Matérn 5/2
        #   (twice differentiable) is better calibrated for such systems and is
        #   the standard kernel in physical/environmental GP literature.
        #   At long lengthscales, Matérn 5/2 approximates linear behaviour, so
        #   there is no need for an explicit LinearKernel addend.
        #
        # Why no LinearKernel addend (compared to the previous version):
        #   LinearKernel(ard_num_dims=D) stores (Q, 1, D) per-feature variances
        #   — nearly as many parameters as the Matérn lengthscales — and
        #   duplicates what a long-lengthscale Matérn already provides.  The sum
        #   ScaleKernel(Matérn) + LinearKernel creates a bimodal optimisation
        #   landscape and roughly doubles kernel parameters with no benefit.
        input_dim = inducing_points.size(-1)
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(
                # nu=2.5,
                ard_num_dims=input_dim,
                batch_shape=torch.Size([num_latents]),
            ),
            batch_shape=torch.Size([num_latents]),
        )

        # Initialise every ARD lengthscale to sqrt(input_dim) instead of the
        # GPyTorch default of 1.0.  Justification:
        #   For inputs uniformly in [0,1]^D, E[||x-x'||^2] = D/6, so the
        #   expected Matérn argument is sqrt(5)*sqrt(D/6)/l.  At l=1 this
        #   equals sqrt(5D/6); for D=161 (T=7) that is ≈11.6, giving
        #   k≈5e-4 — a near-zero kernel everywhere.  At l=sqrt(D) the
        #   argument becomes sqrt(5/6)≈0.91, giving k≈0.88 — proper
        #   correlation.  Without this fix the GP prior has no structure
        #   and training starts in a collapsed, uninformative state.
        import math
        self.covar_module.base_kernel.lengthscale = math.sqrt(input_dim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)