import torch


def get_arm_with_largest_upper_bound(lr_fit, x_arms, key_args):
    # quantiles are invariant to monotonic transformations
    # (that is, we can map the quantiles through the transformation?),
    # so for W ~ N(m, S) and Z = X W ~ N(X m, X S X'), we have
    # logistic(Z) and Z will give the same arm as the max. quantile.
    # note: t runs from zero
    t = key_args["t"]
    bc = key_args["bc"]
    qa = x_arms.new_full((), 1.0 - 1.0 / ((t + 1.0) * bc))

    w_mean = lr_fit["w_mean"]
    w_cov = lr_fit["w_chol"] @ lr_fit["w_chol"].t()

    # marginal quantiles
    z_mean = x_arms @ w_mean
    z_std = torch.sqrt(torch.diag(x_arms @ w_cov @ x_arms.t()))

    qj = torch.distributions.Normal(z_mean, z_std).icdf(qa)
    assert not torch.any(torch.isnan(qj))

    # handle ties with small amount of added noise
    j = torch.argmax(qj + torch.randn(qj.size(0), dtype=qj.dtype) * 1e-10)

    return j, None


def get_arm_with_thompson_sampling(lr_fit, x_arms, key_args):
    # NOTE: samples from the latent Gaussian approximate dist
    #       (no need to transform to reward probs, since the
    #        transformation is monotonic and max. is invariant to such)
    N, M = x_arms.size()

    w_mean = lr_fit["w_mean"]
    w_cov = lr_fit["w_chol"] @ lr_fit["w_chol"].t()

    # arm distribution
    z_mean = x_arms @ w_mean
    z_cov = x_arms @ w_cov @ x_arms.t() + torch.diag(1e-6 * x_arms.new_ones(N))
    z_chol = torch.cholesky(z_cov, upper=False)

    z = z_chol @ torch.randn(z_mean.numel(), dtype=z_mean.dtype) + z_mean
    # z = torch.distributions \
    #          .MultivariateNormal(z_mean, covariance_matrix=z_cov).sample()

    qj = torch.argmax(z)

    return qj, (z_mean, z_chol)


def get_arm_with_largest_mean(lr_fit, x_arms, key_args):
    w_mean = lr_fit["w_mean"]
    z_mean = x_arms @ w_mean

    j = torch.argmax(z_mean)

    return j, None
