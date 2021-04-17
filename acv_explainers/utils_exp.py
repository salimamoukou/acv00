import numpy as np


def compute_exp(X, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(shape=(N, model.values.shape[2]))
    elif S == []:
        return np.zeros(shape=(N, model.values.shape[2]))

    n_trees = len(model.trees)
    mean_forest = np.zeros((N, model.values.shape[2]))

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

            p_ss = np.sum(p_s)
            for i, x in enumerate(X):
                if leaf_bool[i] == 0:
                    continue

                mean_forest[i] += (np.sum(lm) * value) / p_ss if p_ss != 0 else 0

    return mean_forest


def compute_exp_cat(X, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

    if len(S) == len(index):
        return np.ones(N)
    elif S == []:
        return np.zeros(N)

    n_trees = len(model.trees)
    mean_forest = np.zeros((N, model.values.shape[2]))

    for b in range(n_trees):
        leaves_tree = model.partition_leaves_trees[b]

        for leaf_numb in range(model.leaves_nb[b]):
            leaf_part = leaves_tree[leaf_numb]
            leaf_id = model.leaf_idx_trees[b, leaf_numb]

            leaf_bool = np.prod([(X[:, s] <= leaf_part[s, 1]) * (X[:, s] >= leaf_part[s, 0]) for s in S], axis=0)

            if np.sum(leaf_bool) == 0:
                continue

            value = model.values[b, leaf_id]
            lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in index], axis=0)

            for i, x in enumerate(X):
                if leaf_bool[i] == 0:
                    continue

                p_s = np.prod([data[:, s] == x[s] for s in S], axis=0)
                p_ss = np.sum(p_s)

                mean_forest[i] += (np.sum(lm * p_s) * value) / p_ss if p_ss != 0 else 0

    return mean_forest
