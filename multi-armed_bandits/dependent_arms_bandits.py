import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from pylab import rcParams
import ai_models
import logistic_regression_pyro as lrp


def logistic_regression_bandit(
    task_params,
    user_model,
    ai_model,
    ai_acq_fun,
    ai_logreg_fun,
    target=None,
    starting_state=None,
    debug=False,
    verbose=False,
):
    """The Bernoulli bandit solution with logistic regression for arm
       dependencies."""
    x_arms = task_params["x_arms"]
    reward_prob = task_params["reward_probs"]
    n_horizon = task_params["n_horizon"]
    n_arms = reward_prob.size()[0]
    bc = task_params["bc"]
    word_list = task_params["word_list"]

    rewards = x_arms.new_zeros(n_horizon)
    draws = torch.zeros(n_horizon, dtype=torch.int64)
    a0s = torch.zeros(n_horizon, dtype=torch.int64)
    a1s = torch.zeros(n_horizon, dtype=torch.int64)
    p_a0s = x_arms.new_zeros(n_horizon, n_arms)
    p_a1s = x_arms.new_zeros(n_horizon, n_arms)
    P = []
    direct_feedbacks = torch.zeros(n_horizon, dtype=torch.int64)
    user_actions = x_arms.new_zeros(n_horizon)

    # data will be used in two places: user model and ai's update of log.reg.
    # note: need to be careful to keep data in correct state in the these
    #       and to not modify the "global" state inadvertently (dict fields and
    #       torch tensors are by reference, not copies.
    #       Currently, ai_model recreates the data-dict in each call to it.
    data = ai_model(
        task_params,
        draws[0:0],
        user_actions[0:0],
        direct_feedbacks[0:0],
        a0s[0:0],
        a1s[0:0],
        p_a0s[0:0,],
        p_a1s[0:0,],
        P,
    )
    naive_data = ai_models.no_lookahead(
        task_params,
        draws[0:0],
        user_actions[0:0],
        direct_feedbacks[0:0],
        a0s[0:0],
        a1s[0:0],
        p_a0s[0:0,],
        p_a1s[0:0,],
        P,
    )

    w_mean_series = torch.zeros(size=(n_horizon, data["M"]))
    alpha_a = x_arms.new_zeros(n_horizon)
    alpha_b = x_arms.new_zeros(n_horizon)
    beta_a = x_arms.new_zeros(n_horizon)
    beta_b = x_arms.new_zeros(n_horizon)
    tau_a = x_arms.new_zeros(n_horizon)
    tau_b = x_arms.new_zeros(n_horizon)
    w_covar_series = torch.zeros(size=(n_horizon, data["M"], data["M"]))
    if debug is True:
        prob_a1_vs_a0_s = x_arms.new_zeros(n_horizon)
        r_a0_s = x_arms.new_zeros(n_horizon)
        r_a1_s = x_arms.new_zeros(n_horizon)

    lr_fit = None
    lr_fit_user = None
    if "plot_user_pw" in task_params and task_params["plot_user_pw"]:
        # need to initialize for visualization
        lr_fit_user = lrp.fit_logistic_regression(naive_data, lr_fit_user)

    key_args = {}
    if debug is True and verbose is True:
        x_values = np.arange(1, 10 * len(word_list) + 1, 10)
        rcParams["figure.figsize"] = 50, 10
        plt.ion()
        fig = plt.figure()

    for t in range(n_horizon):
        # choose an arm
        if t == 0:
            if starting_state is None:
                j = np.random.randint(n_arms)
            else:
                j = np.int(starting_state)
        else:
            key_args["t"] = t
            key_args["bc"] = bc
            j, _ = ai_acq_fun(lr_fit, x_arms, key_args)

        if "plot_user_pw" in task_params and task_params["plot_user_pw"]:
            lrf = lr_fit_user
            w_mean = lrf["w_mean"].numpy()
            ndim = w_mean.shape[0]
            w_cov = lrf["w_chol"] @ lrf["w_chol"].t()
            w_95 = 1.96 * torch.diag(w_cov).sqrt().numpy()

            plt.figure("plot_user_pw")
            plt.clf()
            plt.plot(w_mean, range(ndim), "-", color="#ff0000")
            plt.plot(w_mean + w_95, range(ndim), "-", color="#ff8888")
            plt.plot(w_mean - w_95, range(ndim), "-", color="#ff8888")
            plt.plot(x_arms[j, :].numpy(), range(ndim), "b.")
            plt.plot(np.zeros(ndim), range(ndim), "k-")
            if "feature_labels" in task_params:
                plt.yticks(range(ndim), task_params["feature_labels"])
            plt.xlim(-5.0, 5.0)
            plt.show(block=False)
            plt.draw()

        # observe reward and user's action
        r = torch.bernoulli(reward_prob[j])

        user_action, meta = user_model(
            task_params, r, j, naive_data, t, lr_fit_user, debug=debug
        )
        # note: in addition to observing the action, we assume here that we get
        # some information about the user's process of coming up with the
        # action that we might not really have in a real system.

        # book-keeping
        # rewards[t] = r  # for cumulative reward
        rewards[t] = reward_prob[j]  # for expected cumulative reward
        draws[t] = j
        user_actions[t] = user_action
        a0s[t] = meta.get("a0", -1)
        a1s[t] = meta.get("a1", -1)
        p_a0s[t, :] = meta.get("p_a0", -1)
        p_a1s[t, :] = meta.get("p_a1", -1)
        p_temp = meta.get("P", ())
        P.append(meta.get("P", ()))

        direct_feedbacks[t] = meta["direct_feedback"]

        if debug is True:
            prob_a1_vs_a0_s[t] = meta.get("prob_a1_vs_a0", -1)
            r_a0_s[t] = meta.get("r_a0", -1)
            r_a1_s[t] = meta.get("r_a1", -1)

        if not isinstance(P[0], tuple):
            P_tensor = torch.stack(P)
        else:
            P_tensor = P
        # update arms
        data = ai_model(
            task_params,
            draws[0 : (t + 1)],
            user_actions[0 : (t + 1)],
            direct_feedbacks[0 : (t + 1)],
            a0s[0 : (t + 1)],
            a1s[0 : (t + 1)],
            p_a0s[0 : (t + 1),],
            p_a1s[0 : (t + 1),],
            P_tensor,
        )
        ti = time.process_time()

        lr_fit = ai_logreg_fun(data, lr_fit)
        ti = time.process_time() - ti
        naive_data = ai_models.no_lookahead(
            task_params,
            draws[0 : (t + 1)],
            user_actions[0 : (t + 1)],
            direct_feedbacks[0 : (t + 1)],
            a0s[0 : (t + 1)],
            a1s[0 : (t + 1)],
            p_a0s[0 : (t + 1),],
            p_a1s[0 : (t + 1),],
            P_tensor,
        )
        lr_fit_user = lrp.fit_logistic_regression(naive_data, lr_fit_user)

        w_mean_series[t, :] = lr_fit["w_mean"]
        # dict.get return -1 if key doesn't exist.
        alpha_a[t] = lr_fit.get("a_alpha", -1)
        alpha_b[t] = lr_fit.get("b_alpha", -1)
        beta_a[t] = lr_fit.get("a_beta", -1)
        beta_b[t] = lr_fit.get("b_beta", -1)
        tau_a[t] = lr_fit.get("a_tau", -1)
        tau_b[t] = lr_fit.get("b_tau", -1)
        w_covar = lr_fit["w_chol"] @ lr_fit["w_chol"].t()
        covar_diag = np.diag(w_covar)
        w_covar_series[t, :, :] = w_covar

        if debug is True and verbose is True:
            plt.clf()
            plt.xticks(x_values, word_list, rotation="vertical")
            bar_series = plt.bar(
                x_values,
                torch.sigmoid(x_arms @ w_mean_series[t, :].double()),
                align="center",
            )
            bar_series[target].set_color("r")
            plt.pause(0.0001)

            print("w_mean: {} \n".format(w_mean_series[t, :]))

            print("Covar Diag: {} \n".format(covar_diag))
            if alpha_a[t] != -1:
                print("Alpha a: {} \n".format(alpha_a[t]))
                print("Alpha b: {} \n".format(alpha_b[t]))
                print("Alpha mean : {}".format(alpha_a[t] / (alpha_a[t] + alpha_b[t])))

                print("Beta a: {} \n".format(beta_a[t]))
                print("Beta b: {} \n".format(beta_b[t]))
                print("Beta mean : {}".format(beta_a[t] / (beta_b[t])))

                print("Tau a: {} \n".format(tau_a[t]))
                print("Tau b: {} \n".format(tau_b[t]))
                print("Tau mean : {}".format(tau_a[t] / (tau_b[t])))

        # This is for the case where the best_arm is given to termiante the interaction
        # earlier than the horizon
        if "best_arm" in task_params:
            if j == task_params["best_arm"]:
                # terminate the study if the best arm is quieried
                break

    # note: true_best_at_end looks at the mean, not the upper conf. bound
    w_mean = lr_fit["w_mean"]
    true_best_at_end = torch.argmax(x_arms @ w_mean) == torch.argmax(reward_prob)
    return_dict = {
        "w_mean_series": w_mean_series,
        "alpha_a": alpha_a,
        "alpha_b": alpha_b,
        "beta_a": beta_a,
        "beta_b": beta_b,
        "tau_a": tau_a,
        "tau_b": tau_b,
        "w_covar_series": w_covar_series,
    }
    return_dict.update(
        {
            "prob_a1_vs_a0_s": prob_a1_vs_a0_s,
            "r_a0_s": r_a0_s,
            "r_a1_s": r_a1_s,
            "p_a0_s": p_a0s,
            "p_a1_s": p_a1s,
        }
    )

    plt.close()
    return rewards, true_best_at_end, draws, user_actions, direct_feedbacks, return_dict
