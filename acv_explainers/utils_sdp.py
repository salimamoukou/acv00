import numpy as np
import itertools
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

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

def single_msdp(x, S, model, rg_data):
    # fx = np.argmax(model.predict(x), axis=1) # for acvtree
    fx = model.predict(x.reshape(1, -1))
    d = x.shape[0]
    data = rg_data.copy()
    data[:, S] = x[S]

    if len(S) != d:
        # y_pred = np.argmax(model.predict(data), axis=1) # for acvtree
        y_pred = model.predict(data)
        sdp = np.mean(y_pred == fx)
        return sdp
    return 1


def msdp_mthread(X, S, model, rg_data):
    mthread = Pool()
    sdp = np.array(mthread.map(partial(single_msdp, S=S, model=model, rg_data=rg_data), X))
    mthread.close()
    mthread.join()
    return sdp

def msdp(X, S, model, rg_data):
    sdp = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        sdp[i] = single_msdp(X[i], S, model, rg_data)
    return sdp


def importance_msdp_clf_search(X, model, rg_data, C=[[]], minimal=1, pi_level=0.9, r_search_space=None, stop=True):
    N = X.shape[0]
    m = X.shape[1]

    sdp = np.zeros((N))
    sdp_global = np.zeros((m))
    len_s_star = np.zeros((N), dtype=np.int)

    R, r = [], []
    for i in range(N):
        R.append(i)

    R_buf = np.zeros((N), dtype=np.int)

    if r_search_space == None:
        search_space = [i for i in range(m)]
    else:
        search_space = r_search_space.copy()

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co), [])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2 ** 15:
            break

    power_cpp = power
    s_star = -1 * np.ones((N, X.shape[1]), dtype=np.int)
    S = np.zeros((X.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, len(power_cpp[s_0])):
            for i in range(len(power_cpp[s_0][s_1])):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = len(power_cpp[s_0][s_1])
            r = []
            N = len(R)
            for i in range(N):
                R_buf[i] = R[i]

            sdp_b = msdp(X, S[:S_size], model, rg_data)
            for i in range(N):
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.append(R[i])
                for s in range(len_s_star[R_buf[i]], X.shape[1]):  # to filter (important for coalition)
                    s_star[R_buf[i], s] = -1
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(len(r)):
            R.remove(r[i])

        if (len(R) == 0 or S_size >= X.shape[1] / 2) and stop:
            break

    return np.asarray(sdp_global) / X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star,
                                                                                          dtype=np.long), np.array(sdp)

def single_msdp_reg(x, S, model, rg_data, threshold=0.2):
    fx = model.predict(x.reshape(1, -1))
    d = x.shape[0]
    data = rg_data.copy()
    data[:, S] = x[S]

    if len(S) != d:
        y_pred = model.predict(data)
        sdp = np.mean(np.abs(y_pred - fx) <= threshold)
        return sdp
    return 1

def msdp_reg_mthread(X, S, model, rg_data, threshold=0.2):
    mthread = Pool()
    sdp = np.array(mthread.map(partial(single_msdp_reg, S=S, model=model, rg_data=rg_data, threshold=threshold), X))
    mthread.close()
    mthread.join()
    return sdp


def msdp_reg(X, S, model, rg_data, threshold=0.2):
    sdp = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        sdp[i] = single_msdp_reg(X[i], S, model, rg_data, threshold)
    return sdp


def importance_msdp_reg_search(X, model, rg_data, C=[[]], minimal=1, pi_level=0.9, threshold=0.2, r_search_space=None, stop=True):
    N = X.shape[0]
    m = X.shape[1]

    sdp = np.zeros((N))
    sdp_global = np.zeros((m))
    len_s_star = np.zeros((N), dtype=np.int)

    R, r = [], []
    for i in range(N):
        R.append(i)

    R_buf = np.zeros((N), dtype=np.int)

    if r_search_space == None:
        search_space = [i for i in range(m)]
    else:
        search_space = r_search_space.copy()

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co), [])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2 ** 15:
            break

    power_cpp = power
    s_star = -1 * np.ones((N, X.shape[1]), dtype=np.int)
    S = np.zeros((X.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, len(power_cpp[s_0])):
            for i in range(len(power_cpp[s_0][s_1])):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = len(power_cpp[s_0][s_1])
            r = []
            N = len(R)
            for i in range(N):
                R_buf[i] = R[i]

            sdp_b = msdp_reg(X, S[:S_size], model, rg_data, threshold)

            for i in range(N):
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.append(R[i])
                for s in range(len_s_star[R_buf[i]], X.shape[1]):  # to filter (important for coalition)
                    s_star[R_buf[i], s] = -1
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(len(r)):
            R.remove(r[i])

        if (len(R) == 0 or S_size >= X.shape[1] / 2) and stop:
            break

    return np.asarray(sdp_global) / X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star,
                                                                                          dtype=np.long), np.array(sdp)

