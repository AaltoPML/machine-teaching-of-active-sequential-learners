import pickle
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats


# load the list of words used in the study
with open("word_search_study/word_list_small", "rb") as fp:
    word_list = pickle.load(fp)

# load the data and ground truth rewards (similarity matrix).
reward_prob_matrix = np.load("word_search_study/reward_prob_matrix_small.npy")

# visualize the data and ground truth rewards
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.imshow(reward_prob_matrix, cmap="hot", interpolation="nearest")
ax.set_xticks(range(len(word_list)))
ax.set_xticklabels(word_list, rotation=90)
ax.set_yticks(range(len(word_list)))
ax.set_yticklabels(word_list)
plt.title("Data and Reward heat map")
plt.colorbar()
plt.savefig("results_user_study/Data_and_rewards.pdf", bbox_inches="tight")

# IDs of participants of the user study
user_ids = [
    "103_mix",
    "104_mix",
    "105_mix",
    "106_mix",
    "107_mix",
    "108_mix",
    "109_mix",
    "110_mix",
    "111_mix",
    "112_mix",
]
n_users = len(user_ids)

n_horizon = 15
n_arms = reward_prob_matrix.shape[0]
n_targets = 20

# We compare 3 methods:
# 0: Mixture Model (user study), 1: Naive Model (user study), 2: Random draw (baseline)
n_methods = 3
reward_all_users = np.zeros((n_methods, n_users, n_targets, n_horizon))

# load user study data and also compute random draw baseline performance
for user in range(n_users):

    USER_ID = user_ids[user]
    print("USER: ", USER_ID)

    # load all the important stuff for evaluation
    with open("results_user_study/{}".format(USER_ID), "rb") as pickle_file:
        rewards = pickle.load(pickle_file)
        draws = pickle.load(pickle_file)
        user_actions_ = pickle.load(pickle_file)
        fpma_ = pickle.load(pickle_file)
        task_params_ = pickle.load(pickle_file)
        order_method = pickle.load(pickle_file)
        methods_ = pickle.load(pickle_file)
        targets_ = pickle.load(pickle_file)
        system_data_ = pickle.load(pickle_file)
        starting_states = pickle.load(pickle_file)

    # Save the things we need for later
    for target in range(n_targets):
        temp = rewards.numpy()
        reward_all_users[0, user, target, :] = temp[0, target, :]
        reward_all_users[1, user, target, :] = temp[1, target, :]

        # Compute random draw reward
        random_draws = np.random.randint(n_arms, size=n_horizon)
        # the first draw is fixed
        random_draws[0] = starting_states[target]
        reward_all_users[2, user, target, :] = reward_prob_matrix[target, random_draws]


plt.figure()
overall_ave_Mix = reward_all_users[0, :, 1:, :]
overall_ave_Mix = np.mean(overall_ave_Mix, 0)
plt.plot(
    range(1, n_horizon + 1),
    overall_ave_Mix.cumsum(1).mean(0),
    "rs-",
    label="Human Teacher - Mixture Model",
    ms=6,
)  # only for legend
overall_ave_vanilla = reward_all_users[1, :, 1:, :]
overall_ave_vanilla = np.mean(overall_ave_vanilla, 0)
plt.plot(
    range(1, n_horizon + 1),
    overall_ave_vanilla.cumsum(1).mean(0),
    "bo-",
    label="Human Teacher - Naive Model",
    ms=6,
)  # only for legend
overall_ave_rnd = reward_all_users[2, :, 1:, :]
overall_ave_rnd = np.mean(overall_ave_rnd, 0)
plt.plot(
    range(1, n_horizon + 1),
    overall_ave_rnd.cumsum(1).mean(0),
    "k--",
    label="Random draw (baseline)",
    ms=6,
)  # only for legend

for user in range(n_users):
    r_mix = reward_all_users[0, user, 1:, :]
    ave_reward_mix = r_mix.cumsum(1).mean(0)
    plt.plot(range(1, n_horizon + 1), ave_reward_mix, "r-", alpha=0.15)

    r_vanilla = reward_all_users[1, user, 1:, :]
    ave_reward_vanilla = r_vanilla.cumsum(1).mean(0)
    plt.plot(range(1, n_horizon + 1), ave_reward_vanilla, "b-", alpha=0.15)

    r_random_draw = reward_all_users[2, user, 1:, :]
    ave_reward_random_draw = r_random_draw.cumsum(1).mean(0)
    plt.plot(range(1, n_horizon + 1), ave_reward_random_draw, "k--", alpha=0.15)

# plt.legend()
plt.xticks(np.arange(1, 16, step=2))
plt.xlabel("Number of Questions")
# plt.title('User study')
plt.ylabel("Mean Cumulative Reward")
plt.legend(frameon=False)
plt.savefig("results_user_study/user_study_cum_reward.pdf", bbox_inches="tight")


# Compute the p-value for cum_reward performance between user study conditions,
# after averaging over targets (discarding the practice target)
reward_ave_over_targets_Mix = np.mean(reward_all_users[0, :, 1:, :], 1)
reward_ave_over_targets_Mix = reward_ave_over_targets_Mix.cumsum(1)
reward_ave_over_targets_Vanilla = np.mean(reward_all_users[1, :, 1:, :], 1)
reward_ave_over_targets_Vanilla = reward_ave_over_targets_Vanilla.cumsum(1)

a, p_values = stats.ttest_rel(
    reward_ave_over_targets_Mix[:, 1:], reward_ave_over_targets_Vanilla[:, 1:]
)  # compute the p-values after second question

plt.figure()
plt.plot(range(2, n_horizon + 1), p_values, "rs--", label="p-value")
plt.plot([2, n_horizon], [0.05, 0.05], "k:", label="0.05 threshold")
plt.plot([2, n_horizon], [0.01, 0.01], "k:", label="0.01 threshold")
print("p values for average over target, cum reward between 10 users", p_values)
plt.xticks(np.arange(1, 16, step=2))
plt.title("Paired t-test between user models of online study")
plt.xlabel("Steps")
plt.ylabel("P-value")
plt.savefig("results_user_study/p_values_cum_reward_diff.pdf", bbox_inches="tight")

plt.show()
