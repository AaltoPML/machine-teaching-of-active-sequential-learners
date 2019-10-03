import numpy as np
import lifelines


def compute_concordances(ground_truth_scores, w_means, x_arms):
    n_iterations = w_means.size()[0]
    predicted_scores = (x_arms @ w_means.t()).numpy()  # N_arms x iterations
    ground_truth_scores = ground_truth_scores.numpy()

    cs = np.zeros((n_iterations,))
    for i in range(n_iterations):
        cs[i] = lifelines.utils.concordance_index(
            ground_truth_scores, predicted_scores[:, i]
        )
    return cs
