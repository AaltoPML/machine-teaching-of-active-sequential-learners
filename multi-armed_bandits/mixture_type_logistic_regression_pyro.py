import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.contrib
import pyro.distributions as dist


def logsumexp(val1, val2):
    # NOTE: newer versions of pytorch should have a better logsumexp function
    #       inbuilt
    return torch.log1p(torch.exp(val2 - val1)) + val1


class MixtureObsDistribution(dist.Distribution):
    """Mixture of direct action observation and lookahead observation.

       This is not implemented for general use.
    """

    def __init__(self, alpha, logits, logits_a0_vs_a1):
        if not logits.size() == logits_a0_vs_a1.size():
            raise ValueError("probs and probs_a0_vs_a1 should be of same size.")

        self.N = logits.numel()
        self.alpha = alpha
        self.logits = logits
        self.logits_a0_vs_a1 = logits_a0_vs_a1

    def sample(self, sample_shape=torch.Size()):
        res = self.probs.new_zeros(self.N)
        probs = torch.sigmoid(self.logits)
        probs_a0_vs_a1 = torch.sigmoid(self.logits_a0)
        with torch.no_grad():
            for n in range(self.N):
                z = torch.bernoulli(self.alpha)
                if z == 1:
                    res[n] = torch.bernoulli(probs[n])
                else:
                    res[n] = torch.bernoulli(probs_a0_vs_a1[n])
            return res

    def log_prob(self, value):
        log_alpha = torch.log(self.alpha)
        log_1malpha = torch.log1p(-self.alpha)

        A = log_alpha - torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits, value, reduce=False
        )

        B = log_1malpha - torch.nn.functional.binary_cross_entropy_with_logits(
            self.logits_a0_vs_a1, value, reduce=False
        )

        # logsumexp(A, B)
        log_p = torch.log1p(torch.exp(B - A)) + A
        return log_p.sum()


def logistic_regression_mixture_obs_model_mla(
    x, y, x_arms, P, M, beta, tau, N_lookahead
):
    # assumes x and y has been augmented with N_lookahead samples for the
    # lookahead which is always a direct observation

    # beta is preference observation "inverse temperature"
    if isinstance(tau, tuple):
        tau_ = pyro.sample("tau", dist.Gamma(tau[0], tau[1]))
    else:
        tau_ = tau
    if isinstance(beta, tuple):
        beta_ = pyro.sample("beta", dist.Gamma(beta[0], beta[1]))
    else:
        beta_ = beta

    w_ = pyro.sample(
        "w", dist.Normal(torch.zeros(M, dtype=torch.double), tau_).independent(1)
    )
    a_ = pyro.sample(
        "alpha",
        dist.Beta(
            torch.tensor(1.0, dtype=torch.double), torch.tensor(1.0, dtype=torch.double)
        ),
    )
    logits = x @ w_

    N = y.numel()
    N_not_la = N - N_lookahead
    if N_not_la > 0:
        # multistep lookahead observations
        with torch.no_grad():
            x_a0 = x_arms.new_zeros(N_not_la, M)
            x_a1 = x_arms.new_zeros(N_not_la, M)

            inds = list(range(P[0].size()[0]))
            n_branching = len(inds) // 2
            inds_0 = inds[0:n_branching]
            inds_1 = inds[n_branching : len(inds)]

            logits_tmp = x_arms @ w_

            for i in range(len(P)):
                i0 = torch.argmax(P[i][inds_0,] @ logits_tmp)
                i1 = torch.argmax(P[i][inds_1,] @ logits_tmp)

                x_a0[i,] = P[i][inds_0[i0],] @ x_arms
                x_a1[i,] = P[i][inds_1[i1],] @ x_arms

        logits_a0_vs_a1 = (x_a1 - x_a0) @ (beta_ * w_)

        pyro.sample(
            "y",
            MixtureObsDistribution(a_, logits[0:N_not_la], logits_a0_vs_a1),
            obs=y[0:N_not_la],
        )
    if N_lookahead > 0:
        pyro.sample(
            "y_lookahead", dist.Bernoulli(logits=logits[N_not_la:N]), obs=y[N_not_la:N]
        )

    return w_, tau_, beta_


def logistic_regression_mixture_obs_model(x, y, x_paired, M, beta, tau, N_lookahead):
    # assumes x and y has been augmented with N_lookahead samples for the
    # lookahead which is always a direct observation

    # beta is preference observation "inverse temperature"
    if isinstance(tau, tuple):
        tau_ = pyro.sample("tau", dist.Gamma(tau[0], tau[1]))
    else:
        tau_ = tau
    if isinstance(beta, tuple):
        beta_ = pyro.sample("beta", dist.Gamma(beta[0], beta[1]))
    else:
        beta_ = beta

    N = y.numel()
    w_ = pyro.sample(
        "w", dist.Normal(torch.zeros(M, dtype=torch.double), tau_).independent(1)
    )
    a_ = pyro.sample(
        "alpha",
        dist.Beta(
            torch.tensor(1.0, dtype=torch.double), torch.tensor(1.0, dtype=torch.double)
        ),
    )
    logits = x @ w_
    N_not_la = N - N_lookahead
    if N_not_la > 0:
        logits_a0_vs_a1 = beta_ * (x_paired @ w_)
        pyro.sample(
            "y",
            MixtureObsDistribution(a_, logits[0:N_not_la], logits_a0_vs_a1),
            obs=y[0:N_not_la],
        )
    if N_lookahead > 0:
        pyro.sample(
            "y_lookahead", dist.Bernoulli(logits=logits[N_not_la:N]), obs=y[N_not_la:N]
        )

    return w_, tau_, beta_


def fit_logistic_regression_mixture_obs_lap(
    data, inits=None, learning_rate=0.01, max_iter=1000, lookahead=False
):
    assert not isinstance(data["tau_prior"], tuple)
    assert not isinstance(data["beta_prior"], tuple)
    model = logistic_regression_mixture_obs_model
    if lookahead:
        N_lookahead = 1
    else:
        N_lookahead = 0
    optim = pyro.optim.SGD({"lr": learning_rate})
    delta_guide = pyro.contrib.autoguide.AutoLaplaceApproximation(model)
    svi = pyro.infer.SVI(
        model, delta_guide, optim, loss=pyro.infer.Trace_ELBO(num_particles=1)
    )
    w_dim = data["M"]
    latent_dim = w_dim + 1

    pyro.clear_param_store()
    initvec = torch.zeros(latent_dim, dtype=torch.double)
    if inits is not None:
        if ("w_mean" in inits) and (inits["w_mean"] is not None):
            initvec[0:w_dim] = inits["w_mean"]
        if ("alpha_latent" in inits) or (inits["alpha_latent"] is not None):
            initvec[w_dim] = inits["alpha_latent"]
    pyro.param("auto_loc", initvec)

    for i in range(max_iter):
        loss = svi.step(
            data["x"],
            data["y"],
            data["xpaired"],
            data["M"],
            data["beta_prior"],
            data["tau_prior"],
            N_lookahead,
        )
        # print(i, loss, pyro.param('auto_loc'))
    guide = delta_guide.laplace_approximation(
        data["x"],
        data["y"],
        data["xpaired"],
        data["M"],
        data["beta_prior"],
        data["tau_prior"],
        N_lookahead,
    )

    # note: cholesky of leading submatrix is the corresponding submatrix of the full
    #       matrix cholesky
    res = {
        "w_mean": pyro.param("auto_loc")[0:w_dim].detach().clone(),
        "w_chol": pyro.param("auto_scale_tril")[0:w_dim, 0:w_dim].detach().clone(),
        "alpha_latent": pyro.param("auto_loc")[w_dim].detach().clone(),
    }
    assert not torch.any(torch.isnan(res["w_chol"]))
    assert not torch.any(torch.isnan(res["w_mean"]))
    assert not torch.isnan(res["alpha_latent"])
    return res


def fit_logistic_regression_mixture_obs_mla_lap(
    data, inits=None, learning_rate=0.01, max_iter=1000, N_lookahead=0
):
    assert not isinstance(data["tau_prior"], tuple)
    assert not isinstance(data["beta_prior"], tuple)
    model = logistic_regression_mixture_obs_model_mla
    optim = pyro.optim.SGD({"lr": learning_rate})
    delta_guide = pyro.contrib.autoguide.AutoLaplaceApproximation(model)
    svi = pyro.infer.SVI(
        model, delta_guide, optim, loss=pyro.infer.Trace_ELBO(num_particles=1)
    )
    w_dim = data["M"]
    latent_dim = w_dim + 1

    pyro.clear_param_store()
    initvec = torch.zeros(latent_dim, dtype=torch.double)
    if inits is not None:
        if ("w_mean" in inits) and (inits["w_mean"] is not None):
            initvec[0:w_dim] = inits["w_mean"]
        if ("alpha_latent" in inits) or (inits["alpha_latent"] is not None):
            initvec[w_dim] = inits["alpha_latent"]
    pyro.param("auto_loc", initvec)

    for i in range(max_iter):
        loss = svi.step(
            data["x"],
            data["y"],
            data["x_arms"],
            data["P"],
            data["M"],
            data["beta_prior"],
            data["tau_prior"],
            N_lookahead,
        )
        # print(i, loss)
    guide = delta_guide.laplace_approximation(
        data["x"],
        data["y"],
        data["x_arms"],
        data["P"],
        data["M"],
        data["beta_prior"],
        data["tau_prior"],
        N_lookahead,
    )

    # note: cholesky of leading submatrix is the corresponding submatrix of the full
    #       matrix cholesky
    res = {
        "w_mean": pyro.param("auto_loc")[0:w_dim].detach().clone(),
        "w_chol": pyro.param("auto_scale_tril")[0:w_dim, 0:w_dim].detach().clone(),
        "alpha_latent": pyro.param("auto_loc")[w_dim].detach().clone(),
    }
    assert not torch.any(torch.isnan(res["w_chol"]))
    assert not torch.any(torch.isnan(res["w_mean"]))
    assert not torch.isnan(res["alpha_latent"])
    return res


fit_logistic_regression_mixture_obs = fit_logistic_regression_mixture_obs_lap
fit_logistic_regression_mixture_obs_mla = fit_logistic_regression_mixture_obs_mla_lap
