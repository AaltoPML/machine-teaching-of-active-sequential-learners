import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import pyro.contrib


def logistic_regression_model(x, y, x_a0, x_a1, y_a0_vs_a1, M, beta, tau):
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
    if y.size()[0] > 0:
        # direct observations
        probs = x @ w_
        pyro.sample("y", dist.Bernoulli(logits=probs), obs=y)
    if y_a0_vs_a1.size()[0] > 0:
        # pairwise preference observations
        prob_a0_vs_a1 = (x_a1 - x_a0) @ (beta_ * w_)

        pyro.sample("y_a0_vs_a1", dist.Bernoulli(logits=prob_a0_vs_a1), obs=y_a0_vs_a1)

    return w_, tau_, beta_


def fit_logistic_regression_lap(data, inits=None, learning_rate=0.01, max_iter=100):
    assert not isinstance(data["tau_prior"], tuple)
    assert not isinstance(data["beta_prior"], tuple)
    optim = pyro.optim.SGD({"lr": learning_rate})
    delta_guide = pyro.contrib.autoguide.AutoLaplaceApproximation(
        logistic_regression_model
    )
    svi = pyro.infer.SVI(
        logistic_regression_model,
        delta_guide,
        optim,
        loss=pyro.infer.Trace_ELBO(num_particles=1),
    )
    latent_dim = data["M"]
    pyro.clear_param_store()
    if inits is None or ("w_mean" not in inits) or (inits["w_mean"] is None):
        pyro.param("auto_loc", torch.zeros(latent_dim, dtype=torch.double))
    else:
        pyro.param("auto_loc", inits["w_mean"])

    for i in range(max_iter):
        loss = svi.step(
            data["x"],
            data["y"],
            data["x_a0"],
            data["x_a1"],
            data["y_a0_vs_a1"],
            data["M"],
            data["beta_prior"],
            data["tau_prior"],
        )
        # print(i, loss)
    guide = delta_guide.laplace_approximation(
        data["x"],
        data["y"],
        data["x_a0"],
        data["x_a1"],
        data["y_a0_vs_a1"],
        data["M"],
        data["beta_prior"],
        data["tau_prior"],
    )
    # print(pyro.get_param_store().get_all_param_names())
    # print(pyro.param("auto_loc"))
    # print(guide)

    res = {
        "w_mean": pyro.param("auto_loc").detach().clone(),
        "w_chol": pyro.param("auto_scale_tril").detach().clone(),
    }
    assert not torch.any(torch.isnan(res["w_chol"]))
    assert not torch.any(torch.isnan(res["w_mean"]))
    return res


fit_logistic_regression = fit_logistic_regression_lap
