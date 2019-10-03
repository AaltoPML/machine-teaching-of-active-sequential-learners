def omnipotent_mix_obs(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    """Interprets user actions as direct rewards or preferential feedbacks
       depending on meta data."""
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    M = x_arms.size()[1]
    tau_prior = task_params["tau_prior"]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "xpaired": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    data["x"] = x_arms[draws,]
    data["xpaired"] = x_arms[a1s,] - x_arms[a0s,]
    data["y"] = user_actions

    return data


def ts_lookahead_mix_obs(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    """Interprets user actions as direct rewards or preferential feedbacks
       following a mixture model, but doesn't know which arm the user chose,
       but only the probabilities."""
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    tau_prior = task_params["tau_prior"]
    M = x_arms.size()[1]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "xpaired": x_arms.new_empty(0),
        "M": M,
        "x_a0": x_arms.new_empty(0),
        "x_a1": x_arms.new_empty(0),
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
        "y_a0_vs_a1": x_arms.new_empty(0),
    }

    data["x"] = x_arms[draws,]
    data["y"] = user_actions
    pr = direct_feedbacks == 0
    p_a0s = p_a0s[pr,]
    p_a1s = p_a1s[pr,]

    if p_a0s.size()[0] > 0:
        data["xpaired"] = (p_a1s - p_a0s) @ x_arms

    return data


def ts_lookahead_mix_obs_mla(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    """Interprets user actions as direct rewards or preferential feedbacks
       following a mixture model, but doesn't know which arm the user chose,
       but only the probabilities."""
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    tau_prior = task_params["tau_prior"]
    M = x_arms.size()[1]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "x_arms": x_arms,
        "P": P,
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    data["x"] = x_arms[draws,]
    data["y"] = user_actions

    return data
