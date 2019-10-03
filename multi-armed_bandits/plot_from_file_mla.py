import pickle
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

# File Name
arguments = sys.argv[1:]
with open("results/{}.pickle".format(arguments[0]), "rb") as pickle_file:
    rewards = pickle.load(pickle_file)
    _ = pickle.load(pickle_file)
    _ = pickle.load(pickle_file)
    _ = pickle.load(pickle_file)
    task_params = pickle.load(pickle_file)


n_horizon = task_params["n_horizon"]
n_arms = task_params["x_arms"].shape[0]
m = task_params["x_arms"].shape[1]
n_reps = rewards.shape[1]


methods = ["4", "3", "2", "1"]

colours = [0, 1, 2, 3]

claim_plots = {"C1": [0, 1, 2, 3]}
n_methods = len(methods)


cmap = plt.get_cmap("Set1").colors
method_names = []
reward_cumsums = rewards.cumsum(2)
fontP = FontProperties()
fontP.set_size("small")
for claim, mtds in claim_plots.items():
    if isinstance(mtds, tuple):
        plt.figure()
        mtd_plt1 = mtds[0]
        for mtd in mtd_plt1:
            reward_cumsum_mean = reward_cumsums[mtd, :, :].mean(0).numpy()
            reward_cumsum_std = reward_cumsums[mtd, :, :].std(0).numpy()
            plt.plot(
                range(n_horizon),
                reward_cumsum_mean,
                color=cmap[colours[mtd]],
                label=methods[mtd],
            )

            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean + reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.15,
                interpolate=True,
            )
            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean - reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.15,
                interpolate=True,
            )

        mtd_plt2 = mtds[1]
        for mtd in mtd_plt2:
            reward_cumsum_mean = reward_cumsums[mtd, :, :].mean(0).numpy()
            reward_cumsum_std = reward_cumsums[mtd, :, :].std(0).numpy()

            plt.plot(
                range(n_horizon),
                reward_cumsum_mean,
                color=cmap[colours[mtd]],
                label=methods[mtd],
            )

            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean + reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.2,
                interpolate=True,
            )
            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean - reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.2,
                interpolate=True,
            )

        plt.legend(loc="upper left", frameon=False)
        plt.xlabel("Steps")
        plt.ylabel("Expected Cumulative Reward")
        # plt.tight_layout()
        plt.savefig("results/claim_{}_{}.pdf".format(arguments[0], claim))
        plt.close()
    else:
        plt.figure()
        print("Plotting Claim:{}".format(claim))
        for mtd in mtds:
            reward_cumsum_mean = reward_cumsums[mtd, :, :].mean(0).numpy()
            reward_cumsum_std = reward_cumsums[mtd, :, :].std(0).numpy()

            plt.plot(
                range(n_horizon),
                reward_cumsum_mean,
                color=cmap[colours[mtd]],
                label=methods[mtd],
            )

            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean + reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.2,
                interpolate=True,
            )
            plt.fill_between(
                range(n_horizon),
                reward_cumsum_mean,
                reward_cumsum_mean - reward_cumsum_std * 1.96 / np.sqrt(n_reps),
                facecolor=cmap[colours[mtd]],
                alpha=0.2,
                interpolate=True,
            )
        plt.legend(loc="upper left", frameon=False)
        plt.xlabel("Steps")
        plt.ylabel("Expected Cumulative Reward")
        plt.savefig("results/claim_{}_{}.pdf".format(arguments[0], claim))
        plt.close()

for i, name in enumerate(methods):
    plt.plot(
        range(n_horizon),
        rewards[i, :, :].cumsum(1).mean(0).numpy(),
        "-",
        color=cmap[colours[i]],
        label=name,
    )
    method_names.append(name)

plt.legend()
plt.xlabel("Steps")
plt.ylabel("Expected Cumulative Reward")

plt.savefig("results/simulated_{}_{}_{}.pdf".format(arguments[0], n_arms, n_horizon))
plt.close()
