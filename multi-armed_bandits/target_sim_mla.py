import pickle
import sys
import time
import numpy as np
import torch
from scipy.special import expit
import ai_models
import logistic_regression_pyro as lrp
import logistic_regression_mla_pyro as lrp_mla
import user_models
from acquisition_functions import get_arm_with_thompson_sampling
from dependent_arms_bandits import logistic_regression_bandit

# input arguments:
# 0: dataset name (data loaded from "simulation_studies/X_{}.npy")
# 1: number of arms
# 2: number of replications
# 3: horizon
# 4: run id
arguments = sys.argv[1:]

x_file = "simulation_studies/X_{}.npy".format(arguments[0])

torch.manual_seed(42)
np.random.seed(42)

# Read the data, normalize, add intercept
X_ = np.load(x_file)
X_n = X_ - np.mean(X_, 0)
X_n = X_n / np.std(X_n, 0)
X_n = X_n / np.sqrt(np.sum(X_n ** 2, axis=1))[:, np.newaxis]
X_n_i = np.hstack((np.ones((X_n.shape[0], 1)), X_n))

n_customers = int(arguments[1])  # number of words
dim_customers = X_n_i.shape[1]  # number of dimensions

n_horizon = int(arguments[3])
n_arms = n_customers
m = dim_customers
n_reps = int(arguments[2])
beta = 20.0
tau = 2.0
bc = 1.0
c_w = -4.0  # intercept coefficient for ground truth
d_w = 8.0  # scaling factor for ground truth
gamma = 1.0

methods = [
    (
        "(U)4LA TS Average - (AI)MLA TS - (ACQ)TS",
        user_models.lookahead_irl_ts_avearms_multi2_naive,
        ai_models.ts_lookahead_mla,
        lrp_mla.fit_logistic_regression_mla,
        get_arm_with_thompson_sampling,
        4,
    ),
    (
        "(U)3LA TS Average - (AI)MLA TS - (ACQ)TS",
        user_models.lookahead_irl_ts_avearms_multi2_naive,
        ai_models.ts_lookahead_mla,
        lrp_mla.fit_logistic_regression_mla,
        get_arm_with_thompson_sampling,
        3,
    ),
    (
        "(U)2LA TS Average - (AI)MLA TS - (ACQ)TS",
        user_models.lookahead_irl_ts_avearms_multi2_naive,
        ai_models.ts_lookahead_mla,
        lrp_mla.fit_logistic_regression_mla,
        get_arm_with_thompson_sampling,
        2,
    ),
    (
        "(U)1LA TS Average - (AI)MLA TS - (ACQ)TS",
        user_models.lookahead_irl_ts_avearms_multi2_naive,
        ai_models.ts_lookahead_mla,
        lrp_mla.fit_logistic_regression_mla,
        get_arm_with_thompson_sampling,
        1,
    ),
]

n_methods = len(methods)

rewards = torch.zeros(n_methods, n_reps, n_horizon, dtype=torch.double)
tbae = torch.zeros(n_methods, n_reps, dtype=torch.double)
fpma = torch.zeros(n_methods, n_reps, dtype=torch.double)  # first play of best arm
system_data = {}
draws = torch.zeros(n_methods, n_reps, n_horizon, dtype=torch.long)
user_actions = torch.zeros(n_methods, n_reps, n_horizon, dtype=torch.int)
x_arms_all = torch.zeros(n_reps, n_customers, m)
reward_prob_matrix_all = torch.zeros(n_reps, n_customers)
target_words = np.zeros(n_reps)
starting_states = []

for rep in range(n_reps):
    idx = np.random.choice(X_.shape[0], n_arms, replace=False)
    X_n_i_rep = X_n_i[idx, :]
    target = torch.LongTensor(1).random_(0, X_n_i_rep.shape[0])[0]
    target_words[rep] = idx[target]
    w = d_w * X_n_i_rep[target, :]
    intercept = 0
    w[intercept] = c_w
    reward_prob_matrix = torch.from_numpy(expit(X_n_i_rep @ w))
    x_arms = torch.from_numpy(X_n_i_rep)
    x_arms_all[rep, :, :] = x_arms
    reward_prob_matrix_all[rep, :] = reward_prob_matrix
    starting_state = np.random.choice(x_arms.shape[0])
    starting_states.append(starting_state)
    print("Starting State: {}".format(starting_state))

    print("Best Arm: {} with prob {}".format(target, reward_prob_matrix[target]))
    print(np.sort(reward_prob_matrix))
    print(x_arms.shape)

    task_params = {
        "x_arms": x_arms,
        "reward_probs": reward_prob_matrix,
        "bc": bc,
        "beta_prior": beta,
        "n_horizon": n_horizon,
        "tau_prior": tau,
        "beta": beta,
        "tau": tau,
        "word_list": [],
        "w": torch.from_numpy(w),
        "gamma": gamma,
    }

    for i, (name, user_model, ai_model, logreg_fun, acq_fun, n_steps) in enumerate(
        methods
    ):
        task_params["n_steps"] = n_steps
        task_params["beta"] = beta / n_steps
        task_params["beta_prior"] = beta / n_steps
        ti = time.process_time()
        rewards[i, rep, :], tbae[i, rep], draws[i, rep, :], user_actions[
            i, rep, :
        ], df, return_dict = logistic_regression_bandit(
            task_params,
            user_model,
            ai_model,
            ai_logreg_fun=logreg_fun,
            ai_acq_fun=acq_fun,
            debug=True,
            target=target,
            verbose=False,
            starting_state=starting_state,
        )

        draws_of_best_arm = torch.nonzero(draws[i, rep, :] == target)  # indices
        if draws_of_best_arm.numel() > 0:
            fpma[i, rep] = torch.min(draws_of_best_arm) + 1  # min index
        else:
            fpma[i, rep] = 0
        ti = time.process_time() - ti
        print("Done {}, {}/{}; time: {:.1f} sec".format(name, i + 1, n_methods, ti))

        system_data.update({(i, rep): return_dict})

    print("Done replication {}/{}".format(rep + 1, n_reps))

order_method = []

with open(
    "results/experiment_mla_{}_{}_{}_{}.pickle".format(
        arguments[0], n_customers, n_horizon, arguments[4]
    ),
    "wb",
) as pickle_file:
    pickle.dump(rewards, pickle_file)
    pickle.dump(draws, pickle_file)
    pickle.dump(user_actions, pickle_file)
    pickle.dump(fpma, pickle_file)
    pickle.dump(task_params, pickle_file)
    pickle.dump(order_method, pickle_file)
    pickle.dump(methods, pickle_file)
    pickle.dump(target_words, pickle_file)
    pickle.dump(system_data, pickle_file)
    pickle.dump(starting_states, pickle_file)
    pickle.dump(x_arms_all, pickle_file)
    pickle.dump(reward_prob_matrix_all, pickle_file)
