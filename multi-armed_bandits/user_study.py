import os
import pickle
import time

import numpy as np
import torch

import ai_models
import ai_models_mix_obs
import logistic_regression_pyro as lrp
import mixture_type_logistic_regression_pyro as mlrp
import user_models
import user_models_mix_obs
from acquisition_functions import get_arm_with_largest_upper_bound
from dependent_arms_bandits import logistic_regression_bandit

# Set the seed for user study. This is to reduce the randomness in different users.
torch.manual_seed(42)

# Debug parameters: debug=True affects the return_dict.
# verbose=True means we show the current state of the system during interaction
# It's better to keep debug=True for logging everything.
# Although we may want to disable it if it's slowing things
# Use verbose=True ONLY IN PILOTS


# load the word list, data, and similarity matrix
with open("word_search_study/word_list_small", "rb") as fp:
    word_list = pickle.load(fp)


data = np.load("word_search_study/data_rbf_small.npy")

reward_prob_matrix = np.load("word_search_study/reward_prob_matrix_small.npy")
similarity = np.dot(data, data.T)
inv_mag = np.sqrt(1 / np.diag(similarity))
cos_sim = (similarity * inv_mag).T * inv_mag

n_words = data.shape[0]  # number of words
dim_words = data.shape[1]  # number of dimensions

n_horizon = 15
n_arms = n_words
m = dim_words
beta = 20.0
tau = 2.0
bc = 1.0

target_words = [i for i in range(20)]

methods = [
    (
        "(U)Look-Ahead-Average TS Naive- (AI)Look-Ahead-Average TS Mix -(ACQ)TS",
        user_models_mix_obs.lookahead_irl_ts_avearms_mix_obs_naive,
        ai_models_mix_obs.ts_lookahead_mix_obs,
        mlrp.fit_logistic_regression_mixture_obs,
        get_arm_with_largest_upper_bound,
    ),
    (
        "(U)Direct-Feedback - (AI)Vanilla - (ACQ)TS",
        user_models.no_lookahead,
        ai_models.no_lookahead,
        lrp.fit_logistic_regression,
        get_arm_with_largest_upper_bound,
    ),
]

n_methods = len(methods)
n_targets = len(target_words)

rewards = torch.zeros(n_methods, n_targets, n_horizon, dtype=torch.double)
draws = torch.zeros(n_methods, n_targets, n_horizon, dtype=torch.int)
user_actions = torch.zeros(n_methods, n_targets, n_horizon, dtype=torch.int)
# first play of best arm
fpma = torch.zeros(n_methods, n_targets, dtype=torch.double)
# user opinion about the two methods after each game with a target
user_opinions_better_method = np.zeros(n_targets)

USER_ID = input("Enter User ID:")
system_data = {}
np.random.seed(int(USER_ID))  # Set the numpy seed to USER_ID value.

# This is the first method that is used for each target word
order_method = np.zeros((n_methods, n_targets))
order_method[0, :] = np.random.choice(n_methods, n_targets, replace=True)
# The first round is a practice round and we only go through the vanilla
order_method[0, 0] = 1
order_method[1, :] = 1 - order_method[0, :]
order_method = order_method.astype(int)
# print(order_method)

# Select a random word as the starting state for each target (excluding the target word)
starting_states = np.zeros(n_targets, dtype=np.int)
for target in range(n_targets):
    valid_states = list(range(0, target_words[target])) + list(
        range(target_words[target] + 1, n_arms)
    )
    starting_states[target] = int(np.random.choice(valid_states, 1))

start_time = time.time()

os.system("cls")  # Use this in Windows
# os.system('clear') #Use this in Ubuntu and IOS

for target in range(n_targets):
    # Select the best arm
    best_arm = target_words[target]
    # Use the normalized cosine similarities in original data space to determine
    # the reward_prob
    reward_prob = torch.from_numpy(reward_prob_matrix[best_arm, :])
    reward_prob = torch.clamp(reward_prob, max=0.9999, min=0.0001)

    if target == 0:
        print("For each target word you will play with two AIs in random order.")
        print(
            "Your goal is to collaborate to get as close as possible to the target word."
        )
        print(
            "The first two rounds are practice rounds (target word =",
            word_list[best_arm],
            ")",
            ". Feel free to play around and ask questions.",
        )

    for method in range(n_methods):
        i = order_method[method, target]
        # print('METHOD THAT IS USED IS ', methods[i])

        (name, user_model, ai_model, logreg_fun, acq_fun) = methods[i]

        print(
            "\nThe target word is:",
            word_list[best_arm],
            "\nThe game ends after",
            n_horizon,
            "questions/answers.\n",
        )

        x_arms = torch.from_numpy(data).double()

        task_params = {
            "w": 0,
            "x_arms": x_arms,
            "reward_probs": reward_prob,
            "bc": bc,
            "beta_prior": beta,
            "beta": beta,
            "tau_prior": tau,
            "n_horizon": n_horizon,
            "tau": tau,
            "word_list": word_list,
        }
        task_params["user_model"] = user_model

        rewards[i, target, :], tbae, draws[i, target, :], user_actions[
            i, target, :
        ], df, return_dict = logistic_regression_bandit(
            task_params,
            user_models.interactive_user_model,
            ai_model,
            ai_logreg_fun=logreg_fun,
            ai_acq_fun=acq_fun,
            starting_state=starting_states[target],
            target=target_words[target],
            debug=True,
            verbose=False,
        )

        # system_data is a dictionary with (method_id, target_word_id) tuple as key,
        # and a dictonary as value with system's data
        system_data.update({(i, target_words[target]): return_dict})

        draws_of_best_arm = torch.nonzero(draws[i, target, :] == best_arm)  # indices
        if draws_of_best_arm.numel() > 0:
            fpma[i, target] = torch.min(draws_of_best_arm)  # min index
        else:
            fpma[i, target] = -1

        print(
            "\n",
            round(
                100 * ((n_methods * target + 1 + method) / (n_methods * n_targets)), 2
            ),
            "% of the experiment completed.",
        )

        input("Let's start the next round! \n\nPress Enter to Continue...")

        os.system("cls")  # Use this in Windows
        # os.system('clear') #Use this in Ubuntu and IOS

    if target == 0 and method == 1:
        input(
            "\nPractice rounds are over and now the study begins! \n\nPress Enter to Continue..."
        )
    os.system("cls")

# save all the important stuff for evaluation
with open("results/user_experiment_small" + str(USER_ID), "wb") as pickle_file:
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
    pickle.dump(user_opinions_better_method, pickle_file)


print("The study took", int((time.time() - start_time) / 60), "minutes")
