import numpy as np
from tqdm import tqdm
from acv_explainers.utils import weighted_percentile


def compute_exp(X, model, S, data):
    N = X.shape[0]
    index = list(range(X.shape[1]))

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

def compute_shaff_quantile(X, model, S, data, Y, min_node_size=5, quantile=95):
    n_trees = len(model.trees)
    weights = []
    for b in range(n_trees):
        nodes_level = [0]
        nodes_child = []
        samples = np.ones(data.shape[0]).astype(bool)
        weight = np.zeros(data.shape[0])
        for level in range(model.max_depth):
            for node in nodes_level:
                if model.trees[b].features[node] in S:
                    if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
                        nodes_child.append(model.trees[b].children_left[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
                    else:
                        nodes_child.append(model.trees[b].children_right[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
                else:
                    nodes_child.append(model.trees[b].children_left[node])
                    nodes_child.append(model.trees[b].children_right[node])
                    samples_child = samples.copy()

                if np.sum(samples_child) < min_node_size:
                    break
                # if model.trees[b].features[node] < 0:
                #     break
                else:
                    samples = samples_child.copy()
            nodes_level = nodes_child.copy()
        weight[samples] = 1/np.sum(samples)
        weights.append(weight)
    weights = np.mean(weights, axis=0)
    sorter = np.argsort(Y)
    y_quantile = weighted_percentile(Y, quantile, weights, sorter)
    return y_quantile


def compute_shaff_exp(X, model, S, data, Y, min_node_size=5):
    n_trees = len(model.trees)
    predictions = []
    for b in range(n_trees):
        nodes_level = [0]
        nodes_child = []
        samples = np.ones(data.shape[0]).astype(bool)

        for level in range(model.max_depth):
            for node in nodes_level:
                if model.trees[b].features[node] in S:
                    if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
                        nodes_child.append(model.trees[b].children_left[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
                    else:
                        nodes_child.append(model.trees[b].children_right[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
                else:
                    nodes_child.append(model.trees[b].children_left[node])
                    nodes_child.append(model.trees[b].children_right[node])
                    samples_child = samples.copy()

                if np.sum(samples_child) < min_node_size:
                    break
                # if model.trees[b].features[node] < 0:
                #     break
                else:
                    samples = samples_child.copy()
            nodes_level = nodes_child.copy()
        predictions.append(np.mean(Y[samples]))
    return np.mean(predictions)
#
#
# def compute_shaff_sdp_clf(X, y_X, model, S, data, Y, min_node_size=5):
#     n_trees = len(model.trees)
#     predictions = []
#     mean = []
#     mean_up = []
#     mean_down = []
#     for b in range(n_trees):
#         nodes_level = [0]
#         nodes_child = []
#         samples = np.ones(data.shape[0]).astype(bool)
#
#         for level in range(model.max_depth):
#             for node in nodes_level:
#                 if model.trees[b].features[node] in S:
#                     if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
#                         nodes_child.append(model.trees[b].children_left[node])
#                         samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
#                     else:
#                         nodes_child.append(model.trees[b].children_right[node])
#                         samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
#                 else:
#                     nodes_child.append(model.trees[b].children_left[node])
#                     nodes_child.append(model.trees[b].children_right[node])
#                     samples_child = samples.copy()
#
#                 if np.sum(samples_child) < min_node_size:
#                     break
#                 # if model.trees[b].features[node] < 0:
#                 #     break
#                 else:
#                     samples = samples_child.copy()
#             nodes_level = nodes_child.copy()
#         mean.append(np.mean(Y[samples]))
#         mean_up.append(np.mean(Y[samples * (Y[samples] == y_X)]))
#         mean_down.append(np.mean(Y[samples * (Y[samples] != y_X)]))
#     return (np.mean(mean) - np.mean(mean_down))/(np.mean(mean_up) - np.mean(mean_down))
        # predictions.append(np.mean(Y[samples]))
    # return np.mean(predictions)


def compute_shaff_sdp_clf(X, y_X, model, S, data, Y, min_node_size=5):
    n_trees = len(model.trees)
    weights = []
    for b in range(n_trees):
        nodes_level = [0]
        nodes_child = []
        samples = np.ones(data.shape[0]).astype(bool)
        weight = np.zeros(data.shape[0])
        for level in range(model.max_depth):
            for node in nodes_level:
                if model.trees[b].features[node] in S:
                    if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
                        nodes_child.append(model.trees[b].children_left[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
                    else:
                        nodes_child.append(model.trees[b].children_right[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
                else:
                    nodes_child.append(model.trees[b].children_left[node])
                    nodes_child.append(model.trees[b].children_right[node])
                    samples_child = samples.copy()

                if np.sum(samples_child) < min_node_size:
                    break
                # if model.trees[b].features[node] < 0:
                #     break
                else:
                    samples = samples_child.copy()
            nodes_level = nodes_child.copy()
        weight[samples] = 1/np.sum(samples)
        weights.append(weight)
    weights = np.mean(weights, axis=0)
    a = 1.*(Y == y_X)
    sdp = np.sum(weights * a)
    return sdp

#
# def compute_shaff_sdpr(X, y_X, t, model, S, data, Y, min_node_size=5):
#     n_trees = len(model.trees)
#     predictions = []
#     mean = []
#     mean_up = []
#     mean_down = []
#     for b in range(n_trees):
#         nodes_level = [0]
#         nodes_child = []
#         samples = np.ones(data.shape[0]).astype(bool)
#
#         for level in range(model.max_depth):
#             for node in nodes_level:
#                 if model.trees[b].features[node] in S:
#                     if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
#                         nodes_child.append(model.trees[b].children_left[node])
#                         samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
#                     else:
#                         nodes_child.append(model.trees[b].children_right[node])
#                         samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
#                 else:
#                     nodes_child.append(model.trees[b].children_left[node])
#                     nodes_child.append(model.trees[b].children_right[node])
#                     samples_child = samples.copy()
#
#                 if np.sum(samples_child) < min_node_size:
#                     break
#                 # if model.trees[b].features[node] < 0:
#                 #     break
#                 else:
#                     samples = samples_child.copy()
#             nodes_level = nodes_child.copy()
#
#         mean.append(np.mean(Y[samples]))
#         up = samples * (Y-y_X)**2 > t
#         print(np.sum(samples), np.sum(up))
#         # mean.append()
#         # print(np.mean(Y[samples]), np.mean(Y[up]), np.mean(Y[~up]))
#         # mean.append()
#         mean_up.append(np.mean(Y[samples * (Y - y_X)**2 > t]))
#         mean_down.append(np.mean(Y[samples * (Y - y_X)**2 <= t]))
#         print(np.mean(mean), np.mean(mean_up), np.mean(mean_down))
#
#     print((np.array(mean_up) - np.array(mean))/(np.array(mean_up) - np.array(mean_down)))
#     return (np.mean(mean_up) - np.mean(mean))/(np.mean(mean_up) - np.mean(mean_down))
        # predictions.append(np.mean(Y[samples]))
    # return np.mean(predictions)


def compute_shaff_sdp(X, y_X, model, S, data, Y, min_node_size=5, t=5):
    n_trees = len(model.trees)
    weights = []
    for b in range(n_trees):
        nodes_level = [0]
        nodes_child = []
        samples = np.ones(data.shape[0]).astype(bool)
        weight = np.zeros(data.shape[0])
        for level in range(model.max_depth):
            for node in nodes_level:
                if model.trees[b].features[node] in S:
                    if X[model.trees[b].features[node]] <= model.trees[b].thresholds[node]:
                        nodes_child.append(model.trees[b].children_left[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] <= model.trees[b].thresholds[node])
                    else:
                        nodes_child.append(model.trees[b].children_right[node])
                        samples_child = samples * (data[:, model.trees[b].features[node]] > model.trees[b].thresholds[node])
                else:
                    nodes_child.append(model.trees[b].children_left[node])
                    nodes_child.append(model.trees[b].children_right[node])
                    samples_child = samples.copy()

                if np.sum(samples_child) < min_node_size:
                    break
                # if model.trees[b].features[node] < 0:
                #     break
                else:
                    samples = samples_child.copy()
            nodes_level = nodes_child.copy()
        weight[samples] = 1/np.sum(samples)
        weights.append(weight)
    weights = np.mean(weights, axis=0)
    a = (Y - y_X) ** 2 <= t
    a = a.astype(int)
    sdp = np.sum(weights * a)
    return sdp