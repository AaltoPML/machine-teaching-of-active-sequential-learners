# AI models are responsible for interpreting the history and preparing it as
# a dataset for the logistic regression update.


def omnipotent(
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
        "x_a0": x_arms.new_empty(0),
        "x_a1": x_arms.new_empty(0),
        "y_a0_vs_a1": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    di = direct_feedbacks == 1
    data["x"] = x_arms[draws[di],]
    data["y"] = user_actions[di]

    pr = direct_feedbacks == 0
    a0 = a0s[pr]
    a1 = a1s[pr]
    data["x_a0"] = x_arms[a0,]
    data["x_a1"] = x_arms[a1,]
    data["y_a0_vs_a1"] = user_actions[pr]
    return data


def ts_lookahead(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    """Interprets user actions as direct rewards or preferential feedbacks
       depending on meta data, but doesn't know which arm the user chose,
       but only the probabilities."""
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    tau_prior = task_params["tau_prior"]
    M = x_arms.size()[1]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "x_a0": x_arms.new_empty(0),
        "x_a1": x_arms.new_empty(0),
        "y_a0_vs_a1": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    di = direct_feedbacks == 1
    data["x"] = x_arms[draws[di],]
    data["y"] = user_actions[di]

    pr = direct_feedbacks == 0
    p_a0s = p_a0s[pr,]
    p_a1s = p_a1s[pr,]
    if p_a0s.size()[0] > 0:
        data["x_a0"] = p_a0s @ x_arms
        data["x_a1"] = p_a1s @ x_arms
        data["y_a0_vs_a1"] = user_actions[pr]
    return data


def ts_lookahead_mla(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    """Interprets user actions as direct rewards or preferential feedbacks
       depending on meta data, but doesn't know which arm the user chose,
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
        "y_a0_vs_a1": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    di = direct_feedbacks == 1
    data["x"] = x_arms[draws[di],]
    data["y"] = user_actions[di]

    pr = direct_feedbacks == 0
    if pr.any():
        data["y_a0_vs_a1"] = user_actions[pr]
    return data


def no_lookahead(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    "Interprets user actions as direct rewards."
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    M = x_arms.size()[1]
    tau_prior = task_params["tau_prior"]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "x_a0": x_arms.new_empty(0),
        "x_a1": x_arms.new_empty(0),
        "y_a0_vs_a1": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    data["x"] = x_arms[draws,]
    data["y"] = user_actions

    return data


def no_lookahead_mla(
    task_params, draws, user_actions, direct_feedbacks, a0s, a1s, p_a0s, p_a1s, P
):
    # This version could be combined with the no_lookahead by just adding P and
    # x_arms to the data dict there
    "Interprets user actions as direct rewards."
    x_arms = task_params["x_arms"]
    beta_prior = task_params["beta_prior"]
    M = x_arms.size()[1]
    tau_prior = task_params["tau_prior"]

    data = {
        "x": x_arms.new_empty(0),
        "y": x_arms.new_empty(0),
        "x_arms": x_arms,
        "P": P,
        "y_a0_vs_a1": x_arms.new_empty(0),
        "M": M,
        "beta_prior": beta_prior,
        "tau_prior": tau_prior,
    }

    data["x"] = x_arms[draws,]
    data["y"] = user_actions

    return data
