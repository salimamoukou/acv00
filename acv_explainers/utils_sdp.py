import numpy as np


def compute_sdp_reg(X, tX, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(N)
    elif S == []:
        return np.zeros(N)

    n_trees = len(model.trees)
    y_pred = model.predict(data)
    fX = model.predict(X)
    mean_forest = np.zeros((N, 3))

    for b in range(n_trees):
        for l in range(n_trees):
            if b == l:
                leaves_tree = model.partition_leaves_trees[b]

                for leaf_numb in range(model.leaves_nb[b]):
                    leaf_part = leaves_tree[leaf_numb]
                    leaf_id = model.leaf_idx_trees[b, leaf_numb]
                    value = model.values[b, leaf_id]

                    leaf_bool = np.prod([(X[:, s] <= leaf_part[s, 1]) * (X[:, s] >= leaf_part[s, 0]) for s in S],
                                        axis=0)

                    if np.sum(leaf_bool) == 0:
                        continue

                    lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in index],
                                 axis=0)

                    p_s = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in
                                   S], axis=0)

                    for i, x in enumerate(X):
                        if leaf_bool[i] == 0:
                            continue
                        dist = (y_pred - fX[i]) ** 2
                        up_tx = np.array(dist > tX).reshape(-1)
                        down_tx = np.array(dist <= tX).reshape(-1)

                        #             p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)

                        p_su = np.sum(p_s * up_tx)
                        p_sd = np.sum(p_s * down_tx)
                        p_ss = np.sum(p_s)

                        mean_forest[i, 0] += (np.sum(lm) * value ** 2) / (p_ss) if p_ss != 0 else 0
                        mean_forest[i, 0] -= (2 * fX[i] * np.sum(lm) * value) / (p_ss) if p_ss != 0 else 0

                        mean_forest[i, 1] += (np.sum(lm * up_tx) * value ** 2) / (
                                    p_su) if p_su != 0 else 0
                        mean_forest[i, 1] -= (2 * fX[i] * np.sum(lm * up_tx) * value) / (
                                    p_su) if p_su != 0 else 0

                        mean_forest[i, 2] += (np.sum(lm * down_tx) * value ** 2) / (
                                    p_sd) if p_sd != 0 else 0
                        mean_forest[i, 2] -= (2 * fX[i] * np.sum(lm * down_tx) * value) / (
                                    p_sd) if p_sd != 0 else 0
            else:
                for leaf_numb_b in range(model.leaves_nb[b]):
                    leaf_id_b = model.leaf_idx_trees[b, leaf_numb_b]
                    for leaf_numb_l in range(model.leaves_nb[l]):
                        leaf_id_l = model.leaf_idx_trees[l, leaf_numb_l]

                        leaf_part_b = model.partition_leaves_trees[b][leaf_numb_b]
                        value_b = model.values[b, leaf_id_b]

                        leaf_part_l = model.partition_leaves_trees[l][leaf_numb_l]
                        value_l = model.values[l, leaf_id_l]

                        leaf_bool = np.prod([(X[:, s] <= leaf_part_b[s, 1]) * (X[:, s] >= leaf_part_b[s, 0]) *
                                             (X[:, s] <= leaf_part_l[s, 1]) * (X[:, s] >= leaf_part_l[s, 0])
                                             for s in S], axis=0)

                        if np.sum(leaf_bool) == 0.:
                            continue

                        lm = np.prod([(data[:, s] <= leaf_part_b[s, 1]) * (data[:, s] >= leaf_part_b[s, 0]) *
                                      (data[:, s] <= leaf_part_l[s, 1]) * (data[:, s] >= leaf_part_l[s, 0])
                                      for s in index], axis=0)

                        p_s = np.prod([(data[:, s] <= leaf_part_b[s, 1]) * (data[:, s] >= leaf_part_b[s, 0]) *
                                       (data[:, s] <= leaf_part_l[s, 1]) * (data[:, s] >= leaf_part_l[s, 0])
                                       for s in S], axis=0)

                        for i, x in enumerate(X):
                            if leaf_bool[i] == 0:
                                continue

                            dist = (y_pred - fX[i]) ** 2
                            up_tx = np.array(dist > tX).reshape(-1)
                            down_tx = np.array(dist <= tX).reshape(-1)

                            #             p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)

                            p_su = np.sum(p_s * up_tx)
                            p_sd = np.sum(p_s * down_tx)
                            p_ss = np.sum(p_s)

                            mean_forest[i, 0] += (np.sum(lm) * value_b * value_l) / (
                                        p_ss) if p_ss != 0 else 0

                            mean_forest[i, 1] += (np.sum(lm * up_tx) * value_b * value_l) / (
                                        p_su) if p_su != 0 else 0

                            mean_forest[i, 2] += (np.sum(lm * down_tx) * value_b * value_l) / (
                                        p_sd) if p_sd != 0 else 0

    num = mean_forest[:, 1] - mean_forest[:, 0]
    den = mean_forest[:, 1] - mean_forest[:, 2]
    sdp = np.true_divide(num, den, out=np.zeros(N), where=den != 0)

    return sdp * (0 <= sdp) * (sdp <= 1) + np.ones(N) * (sdp > 1) + np.zeros(N) * (sdp < 0)


def compute_sdp_reg_cat(X, tX, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(N)
    elif S == []:
        return np.zeros(N)

    n_trees = len(model.trees)
    y_pred = model.predict(data)
    fX = model.predict(X)
    mean_forest = np.zeros((N, 3))

    for b in range(n_trees):
        for l in range(n_trees):
            if b == l:
                leaves_tree = model.partition_leaves_trees[b]
                #             leaves_tree = model.partition_leaves_trees[l]
                for leaf_numb in range(model.leaves_nb[b]):
                    leaf_part = leaves_tree[leaf_numb]
                    leaf_id = model.leaf_idx_trees[b, leaf_numb]
                    value = model.values[b, leaf_id]

                    leaf_bool = np.prod([(X[:, s] <= leaf_part[s, 1]) * (X[:, s] >= leaf_part[s, 0]) for s in S],
                                        axis=0)

                    if np.sum(leaf_bool) == 0:
                        continue

                    lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in index],
                                 axis=0)

                    for i, x in enumerate(X):
                        if leaf_bool[i] == 0:
                            continue
                        dist = (y_pred - fX[i]) ** 2
                        up_tx = np.array(dist > tX).reshape(-1)
                        down_tx = np.array(dist <= tX).reshape(-1)

                        p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)

                        p_su = np.sum(p_s * up_tx)
                        p_sd = np.sum(p_s * down_tx)
                        p_ss = np.sum(p_s)

                        mean_forest[i, 0] += (np.sum(lm * p_s) * value ** 2) / (p_ss) if p_ss != 0 else 0
                        mean_forest[i, 0] -= (2 * fX[i] * np.sum(lm * p_s) * value) / (
                                    p_ss) if p_ss != 0 else 0

                        mean_forest[i, 1] += (np.sum(lm * p_s * up_tx) * value ** 2) / (
                                    p_su) if p_su != 0 else 0
                        mean_forest[i, 1] -= (2 * fX[i] * np.sum(lm * p_s * up_tx) * value) / (
                                    p_su) if p_su != 0 else 0

                        mean_forest[i, 2] += (np.sum(lm * p_s * down_tx) * value ** 2) / (
                                    p_sd) if p_sd != 0 else 0
                        mean_forest[i, 2] -= (2 * fX[i] * np.sum(lm * p_s * down_tx) * value) / (
                                    p_sd) if p_sd != 0 else 0
            else:
                for leaf_numb_b in range(model.leaves_nb[b]):
                    leaf_id_b = model.leaf_idx_trees[b, leaf_numb_b]
                    leaf_part_b = model.partition_leaves_trees[b][leaf_numb_b]

                    for leaf_numb_l in range(model.leaves_nb[l]):
                        leaf_id_l = model.leaf_idx_trees[l, leaf_numb_l]
                        value_b = model.values[b, leaf_id_b]

                        leaf_part_l = model.partition_leaves_trees[l][leaf_numb_l]
                        value_l = model.values[l, leaf_id_l]

                        leaf_bool = np.prod([(X[:, s] <= leaf_part_b[s, 1]) * (X[:, s] >= leaf_part_b[s, 0]) *
                                             (X[:, s] <= leaf_part_l[s, 1]) * (X[:, s] >= leaf_part_l[s, 0])
                                             for s in S], axis=0)

                        if np.sum(leaf_bool) == 0.:
                            continue

                        lm = np.prod([(data[:, s] <= leaf_part_b[s, 1]) * (data[:, s] >= leaf_part_b[s, 0]) *
                                      (data[:, s] <= leaf_part_l[s, 1]) * (data[:, s] >= leaf_part_l[s, 0])
                                      for s in index], axis=0)

                        for i, x in enumerate(X):
                            if leaf_bool[i] == 0:
                                continue

                            dist = (y_pred - fX[i]) ** 2
                            up_tx = np.array(dist > tX).reshape(-1)
                            down_tx = np.array(dist <= tX).reshape(-1)

                            p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)

                            p_su = np.sum(p_s * up_tx)
                            p_sd = np.sum(p_s * down_tx)
                            p_ss = np.sum(p_s)

                            mean_forest[i, 0] += (np.sum(lm * p_s) * value_b * value_l) / (
                                        p_ss) if p_ss != 0 else 0

                            mean_forest[i, 1] += (np.sum(lm * p_s * up_tx) * value_b * value_l) / (
                                        p_su) if p_su != 0 else 0

                            mean_forest[i, 2] += (np.sum(lm * p_s * down_tx) * value_b * value_l) / (
                                        p_sd) if p_sd != 0 else 0

    num = mean_forest[:, 1] - mean_forest[:, 0]
    den = mean_forest[:, 1] - mean_forest[:, 2]
    sdp = np.true_divide(num, den, out=np.zeros(N), where=den != 0)

    return sdp * (0 <= sdp) * (sdp <= 1) + np.ones(N) * (sdp > 1) + np.zeros(N) * (sdp < 0)


def compute_sdp_clf_cat(X, tX, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(N)
    elif S == []:
        return np.zeros(N)

    n_trees = len(model.trees)
    y_pred = model.predict(data)
    fX = model.predict(X)

    if len(y_pred.shape) == 1:
        y_pred = np.array([1 - y_pred, y_pred]).T
        fX = np.array([1 - fX, fX]).T

    argmax_y_pred = np.argmax(y_pred, axis=1)
    fX = np.argmax(fX, axis=1)

    mean_forest = np.zeros((N, 3, model.values.shape[2]))

    for b in range(n_trees):
        leaves_tree = model.partition_leaves_trees[b]

        for leaf_numb in range(model.leaves_nb[b]):
            leaf_part = leaves_tree[leaf_numb]
            leaf_id = model.leaf_idx_trees[b, leaf_numb]
            value = model.values[b, leaf_id]

            leaf_bool = np.prod([(X[:, s] <= leaf_part[s, 1]) * (X[:, s] >= leaf_part[s, 0]) for s in S], axis=0)

            if np.sum(leaf_bool) == 0:
                continue

            lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in index], axis=0)

            for i, x in enumerate(X):
                if leaf_bool[i] == 0:
                    continue

                up_tx = np.array(argmax_y_pred == int(fX[i])).reshape(-1)
                down_tx = np.array(argmax_y_pred != int(fX[i])).reshape(-1)

                p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)

                p_su = np.sum(p_s * up_tx)
                p_sd = np.sum(p_s * down_tx)
                p_ss = np.sum(p_s)

                mean_forest[i, 0] += (np.sum(lm * p_s) * value) / p_ss if p_ss != 0 else 0

                mean_forest[i, 1] += (np.sum(lm * p_s * up_tx) * value) / p_su if p_su != 0 else 0

                mean_forest[i, 2] += (np.sum(lm * p_s * down_tx) * value) / p_sd if p_sd != 0 else 0

    num = mean_forest[:, 0] - mean_forest[:, 2]
    den = mean_forest[:, 1] - mean_forest[:, 2]

    sdp = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
    sdp = np.array([sdp[i][int(fX[i])] for i in range(N)])

    return sdp * (0 <= sdp) * (sdp <= 1) + np.ones(N) * (sdp > 1) + np.zeros(N) * (sdp < 0)


def compute_sdp_clf(X, tX, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(N)
    elif S == []:
        return np.zeros(N)

    n_trees = len(model.trees)
    y_pred = model.predict(data)
    fX = model.predict(X)

    if len(y_pred.shape) == 1:
        y_pred = np.array([1 - y_pred, y_pred]).T
        fX = np.array([1 - fX, fX]).T

    argmax_y_pred = np.argmax(y_pred, axis=1)
    fX = np.argmax(fX, axis=1)

    mean_forest = np.zeros((N, 3, model.values.shape[2]))

    for b in range(n_trees):
        leaves_tree = model.partition_leaves_trees[b]

        for leaf_numb in range(model.leaves_nb[b]):
            leaf_part = leaves_tree[leaf_numb]
            leaf_id = model.leaf_idx_trees[b, leaf_numb]
            value = model.values[b, leaf_id]
            leaf_bool = np.prod([(X[:, s] <= leaf_part[s, 1]) * (X[:, s] >= leaf_part[s, 0]) for s in S], axis=0)

            if np.sum(leaf_bool) == 0:
                continue

            lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in index], axis=0)
            p_s = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in
                           S], axis=0)

            for i, x in enumerate(X):
                if leaf_bool[i] == 0:
                    continue

                up_tx = np.array(argmax_y_pred == int(fX[i])).reshape(-1)
                down_tx = np.array(argmax_y_pred != int(fX[i])).reshape(-1)

                p_su = np.sum(p_s * up_tx)
                p_sd = np.sum(p_s * down_tx)
                p_ss = np.sum(p_s)

                mean_forest[i, 0] += (np.sum(lm) * value) / p_ss if p_ss != 0 else 0

                mean_forest[i, 1] += (np.sum(lm * up_tx) * value) / p_su if p_su != 0 else 0

                mean_forest[i, 2] += (np.sum(lm * down_tx) * value) / p_sd if p_sd != 0 else 0

    num = mean_forest[:, 0] - mean_forest[:, 2]
    den = mean_forest[:, 1] - mean_forest[:, 2]

    sdp = np.true_divide(num, den, out=np.zeros_like(den), where=den != 0)
    sdp = np.array([sdp[i][int(fX[i])] for i in range(N)])
    sdp = sdp * (0 <= sdp) * (sdp <= 1) + np.ones(N) * (sdp > 1) + np.zeros(N) * (sdp < 0)

    return sdp
