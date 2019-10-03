import torch
import numpy as np


def estimate_gaussian_thompson_sampling_probabilities_rbmc(
    m, L, n_samples, renormalize=True
):
    """Rao-Blackwellized (conditional) Monte Carlo estimates of each variable
       being the maximum value in a multivariate normal distribution."""
    # m is mean and L is lower triangular Cholesky factor of covariance matrix
    M = m.numel()
    dtype = m.dtype

    # samples
    x = L @ torch.randn(M, n_samples, dtype=dtype) + m.unsqueeze(1)
    x_max, inds = torch.max(x, dim=0)
    x[inds, range(n_samples)] = -np.inf
    x_2ndmax, inds_2nd = torch.max(x, dim=0)
    x[inds, range(n_samples)] = x_max
    x_max = x_max.expand_as(x).clone()
    x_max[inds, range(n_samples)] = x_2ndmax

    # conditional distributions
    # Lambda = torch.potri(L, upper=False).contiguous()
    # eta = torch.potrs(m, L, upper=False)
    Lambda = torch.cholesky_inverse(L, upper=False).contiguous()
    eta = torch.cholesky_solve(m.view(-1, 1), L, upper=False)

    # note: this is just Mx1 since variance does not depend on the conditioning
    s2_cond = 1.0 / Lambda.diag()

    # zero out diagonal to compute leave-one-variable-out sample quantities
    Lambda.view(-1)[0 : (M * M) : (M + 1)] = 0.0
    B = Lambda @ x

    # note: this is M x n_samples
    m_cond = s2_cond.unsqueeze(1) * (eta - B)

    z = (x_max - m_cond) / (2.0 * s2_cond).sqrt().unsqueeze(1)

    probs = 0.5 * (1.0 - torch.erf(z)).mean(dim=1)

    # this doesn't calculate coherent multivariate estimates, so they do not
    # sum to 1
    if renormalize:
        probs = probs / probs.sum()

    return probs


def estimate_gaussian_thompson_sampling_probabilities_mc(m, L, n_samples):
    # m is mean and L is lower triangular Cholesky factor of covariance matrix
    M = m.numel()
    dtype = m.dtype

    # samples
    x = L @ torch.randn(M, n_samples, dtype=dtype) + m.unsqueeze(1)
    inds = torch.argmax(x, dim=0)

    probs = torch.zeros_like(m)
    for i in range(n_samples):
        probs[inds[i]] += 1.0

    probs = probs / n_samples
    return probs
