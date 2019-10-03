import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression


def make_data():
    n_small = 10
    n_large = 30

    y = np.ones(4 * n_small + 2 * n_large)
    y[0 : (2 * n_small + n_large)] = 0.0

    centers = np.array(
        [[-1.0, 1.0], [-1.0, -1.0], [0.0, 0.5], [1.0, 1.0], [1.0, -1.0], [0.0, -0.5]]
    )
    sd = 0.1
    X = np.ones((y.shape[0], 3))

    ntot = 0
    for i in range(6):
        if (i + 1) % 3 == 0:
            n = n_large
        else:
            n = n_small
        X[ntot : (ntot + n), 1:3] = sd * np.random.randn(n, 2) + centers[i, :]
        ntot += n

    return X, y


def get_log_reg():
    return LogisticRegression(C=1.0, solver="liblinear", fit_intercept=False)


def maxent_query(lf, X_pool, y_pool):
    logp = lf.predict_log_proba(X_pool)
    negent = np.sum(np.exp(logp) * logp, axis=1)
    ind = np.argmin(negent)
    x = X_pool[ind, :]
    y = y_pool[ind]
    return x, y, ind


def accuracy(X_pool, y_pool, lf):
    return np.mean(lf.predict(X_pool) == y_pool)


def run_maxent(X0, y0, X_pool, y_pool, horizon, X_test, Y_test):
    X = X0.copy()
    y = y0.copy()
    X_pool = X_pool.copy()
    y_pool = y_pool.copy()
    lf = get_log_reg()
    lf.fit(X, y)
    error_per_it = np.zeros(horizon + 1)

    error_per_it[0] = accuracy(X_test, Y_test, lf)

    for i in range(horizon):
        x_new, y_new, ind_new = maxent_query(lf, X_pool, y_pool)
        X = np.append(X, x_new[np.newaxis, :], axis=0)
        y = np.append(y, y_new)

        X_pool = np.delete(X_pool, ind_new, 0)
        y_pool = np.delete(y_pool, ind_new, 0)

        # update model
        lf.fit(X, y)
        error_per_it[i + 1] = accuracy(X_test, Y_test, lf)

    return X, y, lf, error_per_it


def teacher_query_recursion(lf, X_pool, y_pool, X, y, eval_lf, inds):
    if len(inds) == 1:
        return eval_lf(lf)

    n_branching = len(inds) // 2
    inds_0 = inds[0:n_branching]
    inds_1 = inds[n_branching : len(inds)]

    # query x
    x_new, _, ind_new = maxent_query(lf, X_pool, y_pool)
    X = np.append(X, x_new[np.newaxis, :], axis=0)

    X_pool = X_pool.copy()
    X_pool = np.delete(X_pool, ind_new, 0)
    y_pool = y_pool.copy()
    y_pool = np.delete(y_pool, ind_new, 0)

    y0 = np.append(y, 0)
    lf.fit(X, y0)
    score0 = teacher_query_recursion(lf, X_pool, y_pool, X, y0, eval_lf, inds_0)

    y1 = np.append(y, 1)
    lf.fit(X, y1)
    score1 = teacher_query_recursion(lf, X_pool, y_pool, X, y1, eval_lf, inds_1)

    if score0 > score1:
        return score0
    else:
        return score1


def teacher_query(lf, X_pool, y_pool, teacher_meta, horizon):
    X = teacher_meta["X"]  # these are the data used to fit current lf
    y = teacher_meta["y"]
    eval_lf = teacher_meta["eval_lf"]

    horizon = np.min([X_pool.shape[0], horizon])
    inds = list(range(2 ** horizon))
    n_branching = len(inds) // 2
    inds_0 = inds[0:n_branching]
    inds_1 = inds[n_branching : len(inds)]

    # query x
    x_new, _, ind_new = maxent_query(lf, X_pool, y_pool)
    X = np.append(X, x_new[np.newaxis, :], axis=0)

    X_pool = X_pool.copy()
    X_pool = np.delete(X_pool, ind_new, 0)
    y_pool = y_pool.copy()
    y_pool = np.delete(y_pool, ind_new, 0)

    y0 = np.append(y, 0)
    lf.fit(X, y0)
    score0 = teacher_query_recursion(lf, X_pool, y_pool, X, y0, eval_lf, inds_0)

    y1 = np.append(y, 1)
    lf.fit(X, y1)
    score1 = teacher_query_recursion(lf, X_pool, y_pool, X, y1, eval_lf, inds_1)

    if score0 > score1:
        return x_new, 0, ind_new
    else:
        return x_new, 1, ind_new


def run_teacher(X0, y0, X_pool, y_pool, horizon, eval_lf, X_test, Y_test):
    X = X0.copy()
    y = y0.copy()
    X_pool = X_pool.copy()
    y_pool = y_pool.copy()
    error_per_it = np.zeros(horizon + 1)
    lf = get_log_reg()
    lf.fit(X, y)
    error_per_it[0] = accuracy(X_test, Y_test, lf)
    lie_ind = []

    for i in range(horizon):
        teacher_meta = {"X": X, "y": y, "eval_lf": eval_lf}
        # it would be enough to plan once to full horizon
        x_new, y_new, ind_new = teacher_query(
            lf, X_pool, y_pool, teacher_meta, horizon - i
        )
        X = np.append(X, x_new[np.newaxis, :], axis=0)
        y = np.append(y, y_new)

        # save index of switched labels for visualization
        if y_pool[ind_new] != float(y_new):
            lie_ind.append(i + 2)

        X_pool = np.delete(X_pool, ind_new, 0)
        y_pool = np.delete(y_pool, ind_new, 0)

        # update model
        lf.fit(X, y)
        error_per_it[i + 1] = accuracy(X_test, Y_test, lf)

    return X, y, lf, error_per_it, lie_ind


def run_random(X0, y0, X_pool, y_pool, horizon, X_test, Y_test):
    X = X0.copy()
    y = y0.copy()
    X_pool = X_pool.copy()
    y_pool = y_pool.copy()
    error_per_it = np.zeros(horizon + 1)
    lf = get_log_reg()
    lf.fit(X, y)
    error_per_it[0] = accuracy(X_test, Y_test, lf)

    for i in range(horizon):

        ind_new = np.random.randint(len(y_pool), size=1)
        x_new = X_pool[ind_new, :]
        y_new = y_pool[ind_new]

        X = np.append(X, x_new, axis=0)
        y = np.append(y, y_new)

        X_pool = np.delete(X_pool, ind_new, 0)
        y_pool = np.delete(y_pool, ind_new, 0)

        # update model
        lf.fit(X, y)
        error_per_it[i + 1] = accuracy(X_test, Y_test, lf)

    return X, y, lf, error_per_it


def decision_boundary_x2(x1, lf):
    # w[0] is assumed the intercept term
    w = lf.coef_[0]
    x2 = -w[1] / w[2] * x1 - w[0] / w[2]
    return x2


def visualize_data(X_pool, y_pool, X0, y0, X_me, y_me, X_te, y_te, lie_ind):
    x1_ls = np.linspace(-2, 2, 101)
    fig, ax = plt.subplots(nrows=1, ncols=3)

    # visualize the data pool and initial points
    ax[0].scatter(X_pool[y_pool == 0, 1], X_pool[y_pool == 0, 2], marker=".", c="k")
    ax[0].scatter(X_pool[y_pool == 1, 1], X_pool[y_pool == 1, 2], marker="x", c="k")

    ax[0].scatter(X0[y0 == 0, 1], X0[y0 == 0, 2], marker=".", c="b")
    ax[0].scatter(X0[y0 == 1, 1], X0[y0 == 1, 2], marker="x", c="b")

    ax[0].plot(x1_ls, decision_boundary_x2(x1_ls, lf_pool), "k-")
    ax[0].set_xlim([-1.5, 1.5])
    ax[0].set_ylim([-1.5, 1.5])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title("Full data pool and fit")
    ax[0].set_aspect(1.0)

    # visualize the active learner data without the teacher
    ax[1].scatter(X_me[y_me == 0, 1], X_me[y_me == 0, 2], marker=".", c="k")
    ax[1].scatter(X_me[y_me == 1, 1], X_me[y_me == 1, 2], marker="x", c="k")

    ax[1].scatter(X0[y0 == 0, 1], X0[y0 == 0, 2], marker=".", c="b")
    ax[1].scatter(X0[y0 == 1, 1], X0[y0 == 1, 2], marker="x", c="b")

    ax[1].plot(x1_ls, decision_boundary_x2(x1_ls, lf_me), "k-")
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set_ylim([-1.5, 1.5])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_title("Without teacher")
    ax[1].set_aspect(1.0)

    # visualize the active learner with a teacher and highlight switched labels
    ax[2].scatter(X_te[y_te == 0, 1], X_te[y_te == 0, 2], marker=".", c="k")
    ax[2].scatter(X_te[y_te == 1, 1], X_te[y_te == 1, 2], marker="x", c="k")

    ax[2].scatter(X0[y0 == 0, 1], X0[y0 == 0, 2], marker=".", c="b")
    ax[2].scatter(X0[y0 == 1, 1], X0[y0 == 1, 2], marker="x", c="b")

    for lie in lie_ind:
        if y_te[lie] == 0.0:
            ax[2].scatter(X_te[lie, 1], X_te[lie, 2], marker=".", c="r")
        else:
            ax[2].scatter(X_te[lie, 1], X_te[lie, 2], marker="x", c="r")

    ax[2].plot(x1_ls, decision_boundary_x2(x1_ls, lf_te), "k-")
    ax[2].set_xlim([-1.5, 1.5])
    ax[2].set_ylim([-1.5, 1.5])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_title("With teacher")
    ax[2].set_aspect(1.0)

    plt.tight_layout()
    plt.savefig("results/example_data.pdf", bbox_inches="tight")


if __name__ == "__main__":
    np.random.seed(945)
    horizon = 10
    n_rep = 100

    # we consider 3 methods and one GT
    accuracy_per_it_all_methods = np.zeros([horizon + 1, n_rep, 4])
    X0 = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, -0.5]])
    y0 = np.array([0.0, 1.0])

    # create test data for evaluation.
    # Note: Teacher has access to X_pool and y_pool, but not the test data.
    X_test, Y_test = make_data()

    for rep in range(n_rep):

        print(rep)

        X_pool, y_pool = make_data()

        # fit to full data
        lf_pool = get_log_reg()
        lf_pool.fit(np.append(X0, X_pool, axis=0), np.append(y0, y_pool, axis=0))
        accuracy_per_it_all_methods[:, rep, 3] = accuracy(
            X_test, Y_test, lf_pool
        )  # GT accuracy (using all training data)

        # fit random model and compute random query accuracy
        X_rnd, y_rnd, lf_rnd, error_per_it_rnd = run_random(
            X0, y0, X_pool, y_pool, horizon, X_test, Y_test
        )
        accuracy_per_it_all_methods[
            :, rep, 0
        ] = error_per_it_rnd  # random query accuracy

        # maxent
        X_me, y_me, lf_me, error_per_it_al = run_maxent(
            X0, y0, X_pool, y_pool, horizon, X_test, Y_test
        )
        accuracy_per_it_all_methods[:, rep, 1] = error_per_it_al  # no-teacher accuracy

        # Error measure for teacher
        def eval_error(lf):
            # accuracy on pool data (and not test data).
            val = accuracy(X_pool, y_pool, lf)
            return val

        X_te, y_te, lf_te, error_per_it_teacher, lie_ind = run_teacher(
            X0, y0, X_pool, y_pool, horizon, eval_error, X_test, Y_test
        )
        accuracy_per_it_all_methods[:, rep, 2] = error_per_it_teacher  # teacher error

        # visualize the full pool of data and results just once
        if rep == 0:
            visualize_data(X_pool, y_pool, X0, y0, X_me, y_me, X_te, y_te, lie_ind)

    ave_res = np.mean(accuracy_per_it_all_methods, 1)
    stds_res = np.zeros([horizon + 1, 3])
    for i in range(3):
        stds_res[:, i] = (
            np.sqrt(np.var(accuracy_per_it_all_methods[:, :, i], 1)) / n_rep
        )

    plt.figure(figsize=(4.5, 4.5))
    plt.plot(ave_res[:, 3], "k--")
    plt.plot(ave_res[:, 0:3], ".-")
    plt.legend(
        [
            "Full pool (ground truth)",
            "Random queries (baseline)",
            "Active learner without teacher",
            "Active learner with teacher",
        ],
        frameon=False,
    )
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("results/example_accuracy.pdf", bbox_inches="tight")

    plt.show()
