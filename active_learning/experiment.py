import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


def maxent_acquire(lf, X_pool, y_pool, _):
    logp = lf.predict_log_proba(X_pool)
    negent = np.sum(np.exp(logp) * logp, axis=1)
    ind = np.argmin(negent)
    x = X_pool[ind, :]
    y = y_pool[ind]
    return x, y, ind


def random_acquire(lf, X_pool, y_pool, _):
    ind = np.random.randint(X_pool.shape[0])
    x = X_pool[ind, :]
    y = y_pool[ind]
    return x, y, ind


def teacher_1step_acquire(lf, X_pool, y_pool, teacher_meta):
    # acquire x
    x_new, _, ind_new = maxent_acquire(lf, X_pool, y_pool, None)

    X = teacher_meta["X"]  # these are the data used to fit current lf
    y = teacher_meta["y"]

    # which y gives better model
    lf = get_log_reg()
    X = np.append(X, x_new[np.newaxis, :], axis=0)

    y0 = np.append(y, 0)
    lf.fit(X, y0)
    score0 = teacher_meta["eval_lf"](lf)

    y1 = np.append(y, 1)
    lf.fit(X, y1)
    score1 = teacher_meta["eval_lf"](lf)

    if score0 > score1:
        y_new = 0
    else:
        y_new = 1

    return x_new, y_new, ind_new


def run_active_learner(
    X0, y0, X_pool, y_pool, eval_lfs, eval_lf_teacher, acq_fun, horizon
):
    scores = np.zeros((horizon + 1, len(eval_lfs)))

    lf = get_log_reg()
    lf.fit(X0, y0)

    for eval_i in range(len(eval_lfs)):
        scores[0, eval_i] = eval_lfs[eval_i](lf)

    X_pool_unused = X_pool.copy()
    y_pool_unused = y_pool.copy()
    X = X0.copy()
    y = y0.copy()
    for i in range(horizon):
        # acquire new data point
        teacher_meta = {"X": X, "y": y, "eval_lf": eval_lf_teacher}
        x_new, y_new, ind_new = acq_fun(lf, X_pool_unused, y_pool_unused, teacher_meta)
        X = np.append(X, x_new[np.newaxis, :], axis=0)
        y = np.append(y, y_new)

        X_pool_unused = np.delete(X_pool_unused, ind_new, 0)
        y_pool_unused = np.delete(y_pool_unused, ind_new, 0)

        # update model
        lf.fit(X, y)

        # evaluate
        for eval_i in range(len(eval_lfs)):
            scores[i + 1, eval_i] = eval_lfs[eval_i](lf)

    return scores


def get_log_reg():
    return LogisticRegression(C=1.0, solver="liblinear", fit_intercept=False)


def subsample_keeping_label_balance(X, y, n):
    inds0 = np.where(y == 0)[0]
    inds1 = np.where(y == 1)[0]

    n0 = int(np.round(len(inds0) / y.shape[0] * n))
    n1 = int(np.round(len(inds1) / y.shape[0] * n))

    # arbitrarily make exact n
    ndiff = n - (n0 + n1)
    if ndiff == 2:
        n0 += 1
        n1 += 1
    elif ndiff == 1:
        n0 += 1
    elif ndiff == 0:
        pass
    elif ndiff == -1:
        n0 -= 1
    elif ndiff == -2:
        n0 -= 1
        n1 -= 1
    else:
        return None

    np.random.shuffle(inds0)
    inds0_in = inds0[0:n0]
    inds0_out = inds0[n0:]
    np.random.shuffle(inds1)
    inds1_in = inds1[0:n1]
    inds1_out = inds1[n1:]

    X_in = np.append(X[inds0_in, :], X[inds1_in, :], axis=0)
    y_in = np.append(y[inds0_in], y[inds1_in])

    X_out = np.append(X[inds0_out, :], X[inds1_out, :], axis=0)
    y_out = np.append(y[inds0_out], y[inds1_out])

    return X_in, y_in, X_out, y_out


if __name__ == "__main__":
    np.random.seed(8492)

    X = np.load("data/X_wine.npy")
    X = preprocessing.scale(X)
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    y = np.load("data/y_wine.npy")

    methods = [random_acquire, maxent_acquire, teacher_1step_acquire]
    method_names = {
        "random_acquire": "Random queries (baseline)",
        "maxent_acquire": "Active learner without teacher",
        "teacher_1step_acquire": "Active learner with teacher",
    }
    n_methods = len(methods)
    horizon = 100
    pool_size = 2000  # plus 2 initial points
    n_reps = 100

    titles = {"eval_pred": "Accuracy"}

    def get_title(s, nms):
        if s in nms.keys():
            return nms[s]
        else:
            return s

    scores = np.zeros((n_reps, n_methods, horizon + 1, 5))
    scores_full_tr = np.zeros((n_reps, 5))
    for r in range(n_reps):
        ind0 = [
            np.where(y == 0)[0][np.random.randint(np.sum(y == 0))],
            np.where(y == 1)[0][np.random.randint(np.sum(y == 1))],
        ]
        X0 = X[ind0, :]
        y0 = y[ind0]

        X_pool = np.delete(X, ind0, 0)
        y_pool = np.delete(y, ind0, 0)

        X_pool, y_pool, X_t, y_t = subsample_keeping_label_balance(
            X_pool, y_pool, pool_size
        )
        X_full_tr = np.append(X0, X_pool, axis=0)
        y_full_tr = np.append(y0, y_pool)

        lf_test_data = get_log_reg().fit(X_t, y_t)
        w_test_data = lf_test_data.coef_
        label_probs_test_data = lf_test_data.predict_proba(X_t)
        label_log_probs_test_data = lf_test_data.predict_log_proba(X_t)

        def eval_pred(lf):
            val = lf.score(X_t, y_t)
            return val

        def eval_lf_teacher(lf):
            val = lf.score(X_full_tr, y_full_tr)
            return val

        eval_lfs = [eval_pred]

        for m in range(n_methods):
            scores[r, m, :, :] = run_active_learner(
                X0, y0, X_pool, y_pool, eval_lfs, eval_lf_teacher, methods[m], horizon
            )

        # full pool (plus initial data) as ground truth
        lf_full_tr = get_log_reg().fit(X_full_tr, y_full_tr)
        for eval_i in range(len(eval_lfs)):
            scores_full_tr[r, eval_i] = eval_lfs[eval_i](lf_full_tr)

    mean_scores = np.mean(scores, axis=0)
    sd_scores = np.std(scores, axis=0)
    mean_scores_full_tr = np.mean(scores_full_tr, axis=0)
    sd_scores_full_tr = np.std(scores_full_tr, axis=0)

    x_locs = np.linspace(0, horizon, horizon + 1)
    for eval_i in range(len(eval_lfs)):
        fig = plt.figure(eval_i)
        p_ = plt.plot(
            [0, horizon],
            mean_scores_full_tr[eval_i] * np.ones(2),
            "k--",
            label="Full pool (ground truth)",
        )
        col_ = p_[0].get_color()
        plt.fill_between(
            [0, horizon],
            mean_scores_full_tr[eval_i],
            mean_scores_full_tr[eval_i]
            + sd_scores_full_tr[eval_i] * 1.96 / np.sqrt(n_reps),
            facecolor=col_,
            alpha=0.15,
            interpolate=True,
        )
        plt.fill_between(
            [0, horizon],
            mean_scores_full_tr[eval_i],
            mean_scores_full_tr[eval_i]
            - sd_scores_full_tr[eval_i] * 1.96 / np.sqrt(n_reps),
            facecolor=col_,
            alpha=0.15,
            interpolate=True,
        )
        for m in range(n_methods):
            p_ = plt.plot(
                x_locs,
                mean_scores[m, :, eval_i],
                label=get_title(methods[m].__name__, method_names),
            )
            col_ = p_[0].get_color()
            plt.fill_between(
                x_locs,
                mean_scores[m, :, eval_i],
                mean_scores[m, :, eval_i]
                + sd_scores[m, :, eval_i] * 1.96 / np.sqrt(n_reps),
                facecolor=col_,
                alpha=0.15,
                interpolate=True,
            )
            plt.fill_between(
                x_locs,
                mean_scores[m, :, eval_i],
                mean_scores[m, :, eval_i]
                - sd_scores[m, :, eval_i] * 1.96 / np.sqrt(n_reps),
                facecolor=col_,
                alpha=0.15,
                interpolate=True,
            )

        plt.ylabel(get_title(eval_lfs[eval_i].__name__, titles))
        plt.xlabel("Steps")
        plt.legend()
        fig.savefig(
            "results/experiment_" + eval_lfs[eval_i].__name__ + ".pdf",
            bbox_inches="tight",
        )

    plt.show()
