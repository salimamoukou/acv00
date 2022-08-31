import scipy.stats as st
import sys
import itertools
from collections import defaultdict
from operator import itemgetter
import string
import random
import scipy.stats as stats
from typing import Tuple, List
from sklearn.cluster import AffinityPropagation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
# from bigholes import HoleFinder
from tqdm import tqdm
from tqdm import tqdm
from numpy.random import rand
from numpy.random import seed
import acv_explainers
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
import pickle

import warnings

warnings.filterwarnings('ignore')


def plot_feature_importance(importance, names, model_type, xlabel='SHAP values', title=' '):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('FEATURE NAMES')


def plot_feature_importance_10(importance, names, model_type, xlabel='SHAP values', title=' '):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    seaborn_colors = sns.color_palette("tab10")
    colors = {names[i]: seaborn_colors[i] for i in range(len(names))}

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], palette=colors)
    # Add chart labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('FEATURE NAMES')


def bar_plot(values_1, values_2, values_3, labels, variables_name, title):
    x = np.arange(len(variables_name))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(dpi=250)
    rects1 = ax.bar(x - width / 2, values_1, width, label=labels[0])
    rects2 = ax.bar(x + width / 2, values_2, width, label=labels[1])
    rects3 = ax.bar(x + 1.5 * width, values_3, width, label=labels[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Shapley values')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(variables_name)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color='black')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    plt.show()


def compute_fcluster_bydistance(data, dist_type='kendall'):
    d = data.shape[1]
    if dist_type == 'kendall':
        dist_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                tau, p_value = stats.kendalltau(data[:, i], data[:, j])
                dist_matrix[i, j] = tau
    elif dist_type == 'corr':
        dist_matrix = np.cov(data.T)
    else:
        raise ValueError("This type of distance is not implemented")

    #     print(dist_matrix)
    clustering = AffinityPropagation(random_state=5)
    coal_idx = clustering.fit_predict(np.abs(dist_matrix))
    coal = []
    for c in np.unique(coal_idx):
        t = np.argwhere(c == coal_idx)
        t = list(t.reshape(-1))
        coal.append(t)
    return coal


def ecart_model(model, X, y):
    y_predict = model.predict(X)
    error = (y_predict - y) ** 2
    print('mse = {} -- max = {} -- min = {} -- q0.95 = {} -- q0.25 {}'.format(np.mean(error),
                                                                              np.max(error),
                                                                              np.min(error),
                                                                              np.quantile(error, 0.95),
                                                                              np.quantile(error, 0.5)))
    return np.mean(error), np.quantile(error, 0.95), np.quantile(error, 0.25)


def get_given_data(x, idx, N):
    val = np.array([x[idx]])
    val = np.tile(val, N)
    return val


# def gen_data_by_cat(x_in, cat_va, prob, N, S, S_bar):
#     # to do: make it real function of S, c_index, etc
#
#     mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)
#
#     rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key],
#                          self.cov[key], S_bar, S, x_in[S]) for key in cat_va
#           if np.sum(mixture_idx == key) != 0}
#
#     rg = np.concatenate([np.concatenate([rg[key], np.tile(self.dummy[key],
#                                                           (np.sum(mixture_idx == key), 1))], axis=1) for key in
#                          rg.keys()])
#
#     rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])
#
#     for val_id in S:
#         rg_data[str(val_id)] = get_given_data(val_id)
#
#     rg_data = rg_data[sorted(rg_data.columns)]
#     return rg_data


def nb_row(row, up, low, S):
    return np.prod([(row[s] <= up[s]) * (row[s] > low[s]) for s in S])


def get_partition_tree(tree, leaf_id, part):
    a = tree.children_left.copy()
    b = tree.children_right.copy()
    f = tree.feature.copy()
    t = tree.threshold.copy()
    r_w = tree.weighted_n_node_samples.copy()
    r = tree.n_node_samples.copy()
    left = np.where(tree.children_left == leaf_id)[0]
    right = np.where(tree.children_right == leaf_id)[0]

    if (len(left) == 0) * (len(right) == 0):
        return (part)

    else:
        if len(right) != 0:
            right = int(right[0])

            part[f[right]] = np.concatenate((part[f[right]], np.array([[t[right], np.inf]])))
            return get_partition_tree(tree, right, part)
        else:
            left = int(left[0])
            part[f[left]] = np.concatenate((part[f[left]], np.array([[-np.inf, t[left]]])))
            return get_partition_tree(tree, left, part)


def get_final_partition(part):
    final_partition = {}
    for i, var_part in enumerate(part):
        final_partition[i] = [np.max(var_part[:, 0]), np.min(var_part[:, 1])]
    return final_partition


def leaves_proba(tree, leaf_id, data, S):
    partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
    partition_leaves = get_partition_tree(tree, leaf_id, partition_leaves)
    partition_leaves = pd.DataFrame(get_final_partition(partition_leaves))[S]

    low = partition_leaves.iloc[0]
    up = partition_leaves.iloc[1]

    section_x = np.prod([(data[:, s] <= up[s]) * (data[:, s] >= low[s]) for s in S], axis=0)
    return np.sum(section_x)


def l1_norm(x, dim=1):
    return np.sum(np.abs(x), axis=1)


def rebuild_tree(parent_id, tree, data, y):
    tree.weighted_n_node_samples[parent_id] = len(data)
    tree.value[parent_id][0][0] = np.mean(y)

    if tree.children_right[parent_id] < 0:
        return 0
    else:
        right = tree.children_right[parent_id]
        left = tree.children_left[parent_id]

        left_cond = data[:, tree.feature[parent_id]] <= tree.threshold[parent_id]
        data_left = data[left_cond]
        y_left = y[left_cond]

        right_cond = data[:, tree.feature[parent_id]] > tree.threshold[parent_id]
        data_right = data[right_cond]
        y_right = y[right_cond]

        rebuild_tree(left, tree, data_left, y_left)
        rebuild_tree(right, tree, data_right, y_right)


def rebuild_acvtree(parent_id, tree, data, y):
    tree.node_sample_weight[parent_id] = data.shape[0]
    tree.values[parent_id] = np.mean(y, axis=0)

    if tree.children_right[parent_id] < 0:
        return 0
    else:
        right = tree.children_right[parent_id]
        left = tree.children_left[parent_id]

        left_cond = data[:, tree.features[parent_id]] <= tree.thresholds[parent_id]
        data_left = data[left_cond]
        y_left = y[left_cond]

        right_cond = data[:, tree.features[parent_id]] > tree.thresholds[parent_id]
        data_right = data[right_cond]
        y_right = y[right_cond]

        rebuild_acvtree(left, tree, data_left, y_left)
        rebuild_acvtree(right, tree, data_right, y_right)


def condMVN(mean, cov, set_bar, set, x):
    if set == []:
        return mean, cov
    else:
        mean_cond = mean[set_bar] + np.matmul(np.matmul(cov[set_bar][:, set],
                                                        np.linalg.inv(cov[set][:, set])), x - mean[set])

        cov_cond = cov[set_bar][:, set_bar] - np.matmul(
            np.matmul(cov[set_bar][:, set], np.linalg.inv(cov[set][:, set])),
            cov[set][:, set_bar])

        return mean_cond, cov_cond


def marMVN(mean, cov, set_bar, set, x):
    if set == []:
        return mean, cov
    else:
        mean_cond = mean[set_bar]
        cov_cond = cov[set_bar][:, set_bar]
        return mean_cond, cov_cond


def sampleMVN(n, mean, cov, set_bar, set, x):
    mean_cond, cov_cond = condMVN(mean, cov, set_bar, set, x)
    sample = st.multivariate_normal(mean_cond, cov_cond).rvs(n)
    return np.reshape(sample, (n, len(set_bar)))


def sampleMarMVN(n, mean, cov, set_bar, set):
    mean_cond, cov_cond = marMVN(mean, cov, set_bar, set)
    sample = st.multivariate_normal(mean_cond, cov_cond).rvs(n)
    return np.reshape(sample, (n, len(set_bar)))


def chain_l(l):
    chain = []
    if type(l) == tuple:
        for it in l:
            if type(it) != list:
                chain.append(it)
            elif type(it) == list and len(it) > 1:
                chain = chain + it
            else:
                raise ValueError('problem...')
    else:
        chain = l
    return chain


def convert_list(l):
    if type(l) == list:
        return l
    else:
        return [l]


def convert_tuple(l):
    if type(l) == tuple:
        return l
    elif type(l) == list:
        return tuple(l)
    else:
        return (l,)


def linear_regression(coefs, x):
    return np.sum(coefs * x, axis=1)


def linear_regression_0(coefs, x):
    return np.sum(coefs * x)


# utils for classifer data generation


def return_positive_semi_definite_matrix(n_dim: int) -> np.ndarray:
    """Return positive semi-definite matrix.

    Args:
        n_dim (int): size of square matrix to return
    Returns:
        p (np.array): positive semi-definite array of shape (n_dim, n_dim)
    """
    m = np.random.randn(n_dim, n_dim)
    p = np.dot(m, m.T)
    return p


def sigmoid(x: np.array) -> np.array:
    """Return sigmoid(x) for some activations x.

    Args:
        x (np.array): input activations
    Returns:
        s (np.array): sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def return_weak_features_and_targets(
        num_features: int,
        num_samples: int,
        mixing_factor: float,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Return weakly predictive features and a target variable.

    Create a multivariate Gaussian-distributed set of features and a
    response variable that is conditioned on a weighted sum of the features.

    Args:
        num_features (int): number of variables in Gaussian distribution
        num_samples (int): number of samples to take
        mixing_factor (float): squashes the weighted sum into the linear
            regime of a sigmoid.  Smaller numbers squash closer to 0.5.
    Returns:
        X (np.array): weakly predictive continuous features
            (num_samples, num_features)
        Y (np.array): targets (num_samples,)
    """

    cov = return_positive_semi_definite_matrix(num_features)
    X = np.random.multivariate_normal(mean=np.zeros(num_features), cov=cov, size=num_samples)

    weights = np.random.randn(num_features)
    y_probs = sigmoid(mixing_factor * np.dot(X, weights))
    y = np.random.binomial(1, p=y_probs)
    return X, y, cov, weights


def return_c_values(cardinality: int) -> Tuple[list, list]:
    """Return categorical values for C+ and C-.

    Create string values to be used for the categorical variable c.
    We build two sets of values C+ and C-.  All values from C+ end with
    "A" and all values from C- end with "B". The cardinality input
    determines len(c_pos) + len(c_neg).

    Args:
        cardinality (int): cardinality of c
    Returns:
        c_pos (list): categorical values from C+ sample
        c_neg (list): categorical values from C- sample
    """
    suffixes = [
        "{}{}".format(i, j)
        for i in string.ascii_lowercase
        for j in string.ascii_lowercase]
    c_pos = ["{}A".format(s) for s in suffixes][:int(cardinality / 2)]
    c_neg = ["{}B".format(s) for s in suffixes][:int(cardinality / 2)]
    return c_pos, c_neg


def return_strong_features(
        y_vals: np.array,
        cardinality: int,
        z_pivot: int = 10
) -> Tuple[np.array, np.array]:
    """Return strongly predictive features.

    Given a target variable values `y_vals`, create a categorical variable
    c and continuous variable z such that y is perfectly predictable from
    c and z, with y = 1 iff c takes a value from C+ OR z > z_pivot.

    Args:
        y_vals (np.array): targets
        cardinality (int): cardinality of the categorical variable, c
        z_pivot (float): mean of z
    Returns:
        c (np.array): strongly predictive categorical variable
        z (np.array): strongly predictive continuous variable
    """
    z = np.random.normal(loc=z_pivot, scale=5, size=2 * len(y_vals))
    z_pos, z_neg = z[z > z_pivot], z[z <= z_pivot]
    c_pos, c_neg = return_c_values(cardinality)
    c, z = list(), list()
    for y in y_vals:
        coin = np.random.binomial(1, 0.5)
        if y and coin:
            c.append(random.choice(c_pos + c_neg))
            z.append(random.choice(z_pos))
        elif y and not coin:
            c.append(random.choice(c_pos))
            z.append(random.choice(z_neg))
        else:
            c.append(random.choice(c_neg))
            z.append(random.choice(z_neg))
    return np.array(c), np.array(z)


def return_main_dataset(
        num_weak: int,
        num_samp: int,
        cardinality: int = 100,
        mixing_factor: float = 0.025,
) -> Tuple[pd.DataFrame, np.array, np.array]:
    """Generate training samples.

    Generate a dataset with features c and z that are perfectly predictive
    of y and additional features x_i that are weakly predictive of y and
    correlated with eachother.

    Args:
        num_weak (int): number of weakly predictive features x_i to create
        num_samp (int): number of sample to create
        cardinality (int): cardinality of the predictive categorical variable.
          half of these values will be correlated with y=1 and the other
          with y=0.
        mixing_factor (float): see `return_weak_features_and_targets`
    Returns:
        df (pd.DataFrame): dataframe with y, z, c, and x_i columns
    """
    X, y, cov, weights = return_weak_features_and_targets(num_weak, num_samp, mixing_factor)
    c, z = return_strong_features(y, cardinality)
    xcol_names = ['x{}'.format(i) for i in range(num_weak)]
    df = pd.DataFrame(X, columns=xcol_names)
    df['y'] = y
    df['z'] = z
    df['c'] = c
    df['c'] = df['c'].astype('category')
    df = df[['y', 'c', 'z'] + xcol_names]
    return df, cov, weights


def encode_as_onehot(df_main: pd.DataFrame) -> pd.DataFrame:
    """Replace string values for c with one-hot encoding."""
    df_onehot = pd.get_dummies(df_main, 'c')
    df_onehot['y'] = df_main['y'].copy()
    return df_onehot


def encode_as_int(df_main: pd.DataFrame) -> pd.DataFrame:
    """Replace string values for c with integer encoding."""
    ord_enc = OrdinalEncoder(dtype=np.int)
    c_encoded = ord_enc.fit_transform(df_main[['c']])
    df_catnum = df_main.copy()
    df_catnum['c'] = c_encoded
    df_catnum['c'] = df_catnum['c'].astype('category')
    return df_catnum, ord_enc


def encode_as_magic_int(df_main: pd.DataFrame) -> pd.DataFrame:
    """Replace string values for c with "magic" integer encoding.

    A magic encoding is one in which the sorted integer values keep all
    C+ values (values of c that end with "A") next to each other and all
    C- values (values of c that end with "B") next to eachother.
    """
    values = sorted(df_main['c'].unique(), key=lambda x: x[-1])
    ord_enc = OrdinalEncoder(categories=[values], dtype=np.int)
    c_encoded = ord_enc.fit_transform(df_main[['c']])
    df_catnum = df_main.copy()
    df_catnum['c'] = c_encoded
    df_catnum['c'] = df_catnum['c'].astype('category')
    return df_catnum, ord_enc


def get_feature_names(df, include_c=True):
    names = [f for f in df.columns if not f.startswith('y')]
    if not include_c:
        names = [f for f in names if not f.startswith('c')]
    return names


def print_auc_mean_std(results):
    print("    AUC: mean={:4.4f}, sd={:4.4f}".format(
        np.mean(results['metric']), np.std(results['metric'])))


def print_sorted_mean_importances(results, n=5):
    data = defaultdict(list)
    imps = results['importances']
    for d in imps:
        for fname, imp in d.items():
            data[fname].append(imp)
    mu = {fname: np.mean(vals) for fname, vals in data.items()}
    mu = sorted(mu.items(), key=itemgetter(1), reverse=True)[:n]
    print("    Importances:")
    for fname, val in mu:
        print("{:>20}: {:0.03f}".format(fname, val))


# @jit(nopython=True)
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


# @jit(nopython=True)
def get_partition(leaf_id, part, node_id, children_left, children_right, feature, threshold):
    left = np.where(children_left == leaf_id)[0]
    right = np.where(children_right == leaf_id)[0]

    if (len(left) == 0) * (len(right) == 0):
        return part, node_id

    else:
        if len(right) != 0:
            right = int(right[0])
            node_id.append(feature[right])

            part[feature[right]] = np.concatenate((part[feature[right]], np.array([[threshold[right], np.inf]])))
            part[feature[right]] = np.array([[np.max(part[feature[right]][:, 0]), np.min(part[feature[right]][:, 1])]])
            return get_partition(right, part, node_id, children_left, children_right, feature, threshold)
        else:
            left = int(left[0])
            node_id.append(feature[left])

            part[feature[left]] = np.concatenate((part[feature[left]], np.array([[-np.inf, threshold[left]]])))
            part[feature[left]] = np.array([[np.max(part[feature[left]][:, 0]), np.min(part[feature[left]][:, 1])]])

            return get_partition(left, part, node_id, children_left, children_right, feature, threshold)


def explore_partition(i, x, children_left, children_right, features, thresholds, values,
                      compatible_leaves, partition_leaves, partition_global, prob_global, s_global, S, S_bar, data,
                      down_tx, up_tx, intv=False):
    if children_left[i] < 0:
        #         tab[i] = 1
        compatible_leaves.append(i)
        partition_global[i] = partition_leaves
        partition_leaves = np.squeeze(np.array(partition_leaves))

        section_x = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0)

        section_x_bar = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S_bar], axis=0)

        section_up = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0) * up_tx

        section_down = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0) * down_tx

        prob_all = section_x * section_x_bar
        prob_up = section_up * section_x_bar
        prob_down = section_down * section_x_bar

        s_all = section_x
        s_up = section_up
        s_down = section_down

        prob_global['all'].append(prob_all.reshape(1, -1))
        prob_global['up'].append(prob_up.reshape(1, -1))
        prob_global['down'].append(prob_down.reshape(1, -1))

        s_global['all'].append(s_all.reshape(1, -1))
        s_global['up'].append(s_up.reshape(1, -1))
        s_global['down'].append(s_down.reshape(1, -1))

    else:
        if features[i] in S:
            if x[features[i]] <= thresholds[i]:
                part_left = partition_leaves.copy()
                part_left[features[i]] = np.concatenate((part_left[features[i]], np.array([[-np.inf, thresholds[i]]])))
                part_left[features[i]] = np.array(
                    [[np.max(part_left[features[i]][:, 0]), np.min(part_left[features[i]][:, 1])]])
                explore_partition(children_left[i], x, children_left, children_right, features, thresholds, values,
                                  compatible_leaves, part_left, partition_global, prob_global, s_global, S, S_bar, data,
                                  down_tx, up_tx, intv)
            else:
                part_right = partition_leaves.copy()
                part_right[features[i]] = np.concatenate((part_right[features[i]], np.array([[thresholds[i], np.inf]])))
                part_right[features[i]] = np.array(
                    [[np.max(part_right[features[i]][:, 0]), np.min(part_right[features[i]][:, 1])]])
                explore_partition(children_right[i], x, children_left, children_right, features, thresholds, values,
                                  compatible_leaves, part_right, partition_global, prob_global, s_global, S, S_bar,
                                  data, down_tx, up_tx, intv)
        else:
            part_left = partition_leaves.copy()
            part_left[features[i]] = np.concatenate((part_left[features[i]], np.array([[-np.inf, thresholds[i]]])))
            part_left[features[i]] = np.array(
                [[np.max(part_left[features[i]][:, 0]), np.min(part_left[features[i]][:, 1])]])

            part_right = partition_leaves.copy()
            part_right[features[i]] = np.concatenate((part_right[features[i]], np.array([[thresholds[i], np.inf]])))
            part_right[features[i]] = np.array(
                [[np.max(part_right[features[i]][:, 0]), np.min(part_right[features[i]][:, 1])]])

            explore_partition(children_left[i], x, children_left, children_right, features, thresholds, values,
                              compatible_leaves, part_left, partition_global, prob_global, s_global, S, S_bar, data
                              , down_tx, up_tx, intv)
            explore_partition(children_right[i], x, children_left, children_right, features, thresholds, values,
                              compatible_leaves, part_right, partition_global, prob_global, s_global, S, S_bar, data
                              , down_tx, up_tx, intv)


def get_tree_partition(x, fx, tx, tree, S, data=None, is_reg=True):
    """
    Compute the partition (L_m) of each compatible leaf of the condition X_s = x_S, then check for each
    observations in data in which leaves it falls.

    Args:
        x (array): observation
        fx (float): tree(x)
        tx (float): threshold of the classifier
        tree (DecisionTreeClassifier.tree_): model
        S (list): index of variables on which we want to compute the SDP
        algo (string): name of the estimators, recommended 'pluging'
        data (array): data used to compute the partion

    Returns:
        (array, array): binary array of shape (data_size, compatible_leaves), if [i, j] = 1 then observation i fall in
        leaf j.
        (array): return number of observations that fall in each leaf
    """

    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.features
    thresholds = tree.thresholds
    # r_w = tree.node_samples_weight
    index = range(x.shape[0])
    if is_reg:
        values = tree.values.reshape(-1) / tree.scaling
        y_pred = tree.predict(data)
        dist = (y_pred - fx) ** 2

        up_tx = np.array(dist > tx).reshape(-1)
        down_tx = np.array(dist <= tx).reshape(-1)
    else:
        values = tree.values / tree.scaling
        y_pred = tree.predict(data)

        if len(y_pred.shape) == 1:
            y_pred = np.array([1 - y_pred, y_pred]).T

        argmax_y_pred = np.argmax(y_pred, axis=1)

        up_tx = np.array(argmax_y_pred == int(fx)).reshape(-1)
        down_tx = np.array(argmax_y_pred != int(fx)).reshape(-1)

    S_bar = [i for i in index if i not in S]
    partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
    partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                        for i in range(len(tree.features))}

    prob_global = {'all': [], 'up': [], 'down': []}
    s_global = {'all': [], 'up': [], 'down': []}

    part_final = {}
    compatible_leaves = []
    explore_partition(0, x, children_left, children_right, features, thresholds, values,
                      compatible_leaves, partition_leaves, partition_global, prob_global, s_global, S, S_bar, data,
                      down_tx, up_tx, intv=False)

    part_final['all'] = np.concatenate(prob_global['all'], axis=0)
    part_final['up'] = np.concatenate(prob_global['up'], axis=0)
    part_final['down'] = np.concatenate(prob_global['down'], axis=0)

    part_final['s_all'] = np.concatenate(s_global['all'], axis=0)
    part_final['s_up'] = np.concatenate(s_global['up'], axis=0)
    part_final['s_down'] = np.concatenate(s_global['down'], axis=0)

    return part_final, values[compatible_leaves]


import_errors = {}


def assert_import(package_name):
    global import_errors
    if package_name in import_errors:
        msg, e = import_errors[package_name]
        print(msg)
        raise e


def record_import_error(package_name, msg, e):
    global import_errors
    import_errors[package_name] = (msg, e)


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def get_null_coalition(s_star, len_s_star):
    n_star = -np.ones(s_star.shape, dtype=np.long)
    index = list(range(s_star.shape[1]))

    for i in range(s_star.shape[0]):
        s_star_index = [s_star[i, j] for j in range(s_star.shape[1])]
        null_coalition = list(set(index) - set(s_star_index))
        n_star[i, len_s_star[i]:] = np.array(null_coalition)
    return s_star, n_star


def get_active_null_coalition_list(s_star, len_s_star):
    index = list(range(s_star.shape[1]))
    s_star_all = []
    n_star_all = []
    for i in range(s_star.shape[0]):
        s_star_all.append([s_star[i, j] for j in range(len_s_star[i])])
        n_star_all.append(list(set(index) - set(s_star_all[-1])))
    return s_star_all, n_star_all


class ModelW:
    def __init__(self, model, prediction='predict_proba'):
        self.model = model
        self.prediction = prediction

    def __call__(self, x):
        if self.prediction == 'predict_proba':
            if len(x.shape) == 1:
                return self.model.predict_proba(x.reshape(-1, 1))
            return self.model.predict_proba(x)
        elif self.prediction == 'predict_proba_one':
            if len(x.shape) == 1:
                return self.model.predict_proba(x.reshape(-1, 1))[:, 1]
            return self.model.predict_proba(x)[:, 1]
        else:
            if len(x.shape) == 1:
                return self.model.predict(x.reshape(-1, 1))
            return self.model.predict(x)

    def predict(self, x):
        return self.__call__(x)


def weighted_percentile(a, q, weights=None, sorter=None):
    """
    Returns the weighted percentile of a at q given weights.
    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.
    q: int
        quantile.
    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.
    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.
    Returns
    -------
    percentile: float
        Weighted percentile of a at q.
    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method
    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)
    if q > 100 or q < 0:
        raise ValueError("q should be in-between 0 and 100, "
                         "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)
    start = np.searchsorted(partial_sum, q) - 1
    if start == len(sorted_cum_weights) - 1:
        return sorted_a[-1]
    if start == -1:
        return sorted_a[0]

    # Step 3.
    fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
    return sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])


def find_nbor(rec_a, rec_b, S):
    axs = []
    dim = []
    for k in S:
        if rec_a[k, 0] == rec_b[k, 1]:
            axs.append(k)
            dim.append(0)
        elif rec_a[k, 1] == rec_b[k, 0]:
            axs.append(k)
            dim.append(1)
    return axs, dim


def extend_rec(rec_a, rec_b, S, axs, dim):
    a = 0
    for k in S:
        if k not in axs:
            if not rec_b[k, 0] <= rec_a[k, 0] and rec_b[k, 1] >= rec_a[k, 1]:
                a += 1
    if a == 0:
        for i, k in enumerate(axs):
            rec_a[k, dim[i]] = rec_b[k, dim[i]]
    return rec_a


def find_union(rec_a, list_ric, S):
    axs, dim = [], []
    for i, rec_b in enumerate(list_ric):
        axs, dim = find_nbor(rec_a, rec_b, S)
        if len(axs) != 0:
            break
    if len(axs) == 0 or len(list_ric) == 0:
        return rec_a
    else:
        del list_ric[i]
        rec_a = extend_rec(rec_a, rec_b, S, axs, dim)
        return find_union(rec_a, list_ric, S)


def extend_partition(rules, rules_data, sdp_all, pi, S):
    for i in range(rules.shape[0]):
        list_ric = [rules_data[i, j] for j in range(rules_data.shape[1]) if sdp_all[i, j] >= pi]
        find_union(rules[i], list_ric, S=S[i])


def global_rules_model(x_exp, rules, rules_output, rules_coverage, rules_acc, min_cov, S, min_acc=0.8):
    y_exp, rule_exp, rule_S = [], [], []
    for j in range(x_exp.shape[0]):
        x = x_exp[j]

        y_out, y_coverage, y_weights, y_rule, y_S = [], [], [], [], []
        for i in range(rules.shape[0]):
            rule = rules[i]

            x_in = np.prod([(x[s] <= rule[s, 1]) * (x[s] >= rule[s, 0]) for s in S[i]], axis=0).astype(bool)
            if x_in:
                y_out.append(rules_output[i])
                y_coverage.append(rules_coverage[i])
                y_weights.append(rules_acc[i])
                y_rule.append(rule)
                y_S.append(S[i])

        y_weights = np.array(y_weights)
        if len(y_out) != 0 and np.max(y_weights[np.array(y_coverage) >= min_cov]) >= min_acc:
            max_weights = np.max(y_weights[np.array(y_coverage) >= min_cov])
            best_acc = np.argmax(y_weights == max_weights)

            y_exp.append(y_out[best_acc])
            rule_exp.append(y_rule[best_acc])
            rule_S.append(y_S[best_acc])

        else:
            y_exp.append(None)
            rule_exp.append(None)
            rule_S.append(None)

    return y_exp, rule_exp, rule_S


def global_rules_model_reg(x_exp, rules, rules_output, rules_coverage, rules_acc, min_cov, S, min_mse=500):
    y_exp, rule_exp, rule_S = [], [], []
    for j in range(x_exp.shape[0]):
        x = x_exp[j]

        y_out, y_coverage, y_weights, y_rule, y_S = [], [], [], [], []
        for i in range(rules.shape[0]):
            rule = rules[i]

            x_in = np.prod([(x[s] <= rule[s, 1]) * (x[s] >= rule[s, 0]) for s in S[i]], axis=0).astype(bool)
            if x_in:
                y_out.append(rules_output[i])
                y_coverage.append(rules_coverage[i])
                y_weights.append(rules_acc[i])
                y_rule.append(rule)
                y_S.append(S[i])

        y_weights = np.array(y_weights)
        if len(y_out) != 0 and np.min(y_weights[np.array(y_coverage) >= min_cov]) <= min_mse:
            min_weights = np.min(y_weights[np.array(y_coverage) >= min_cov])
            best_acc = np.argmax(y_weights == min_weights)

            y_exp.append(y_out[best_acc])
            rule_exp.append(y_rule[best_acc])
            rule_S.append(y_S[best_acc])

        else:
            y_exp.append(None)
            rule_exp.append(None)
            rule_S.append(None)

    return y_exp, rule_exp, rule_S


def compute_rules_metrics(rules, data, y_data, S_star, classifier=True):
    d = data.shape[1]
    rules_coverage = []
    rules_var = []
    rules_acc = []
    rules_output = []
    rules_output_proba = []
    for idx in range(rules.shape[0]):
        S = S_star[idx]
        rule = rules[idx]
        where = np.prod([(data[:, s] <= rule[s, 1]) * (data[:, s] >= rule[s, 0])
                         for s in S], axis=0).astype(bool)
        if classifier:
            rules_output.append(1 * (np.mean(y_data[where]) > 0.5))
            rules_output_proba.append((np.mean(y_data[where])))
            rules_acc.append(np.mean(y_data[where] == rules_output[-1]))
        else:
            rules_output.append(np.mean(y_data[where]))
            rules_acc.append(np.mean((y_data[where] - rules_output[-1]) ** 2))

        rules_coverage.append(np.sum(where) / data.shape[0])
        rules_var.append(np.var(y_data[where]))
    if classifier:
        return rules_coverage, rules_acc, rules_var, rules_output, rules_output_proba
    return rules_coverage, rules_acc, rules_var, rules_output


def rules_frequency(rules):
    freq = np.zeros(rules.shape[0])
    for i in range(rules.shape[0]):
        rule = rules[i]
        for j in range(rules.shape[0]):
            if np.allclose(rules[j], rule):
                freq[i] += 1
    freq = freq / rules.shape[0]
    return freq


def unique_rules(rules, rules_output):
    rules_unique = np.unique(rules, axis=0)
    rules_unique_output = []
    for i in range(rules_unique.shape[0]):
        rule = rules_unique[i]
        for j in range(rules.shape[0]):
            if np.allclose(rule, rules[j]):
                rules_unique_output.append(rules_output[j])
                break
    return rules_unique, np.array(rules_unique_output)


def unique_rules_s_star(rules, rules_output):
    rules_unique = np.unique(rules, axis=0)
    rules_unique_output = []
    for i in range(rules_unique.shape[0]):
        rule = rules_unique[i]
        for j in range(rules.shape[0]):
            if np.allclose(rule, rules[j]):
                rules_unique_output.append(rules_output[j])
                break
    return rules_unique, rules_unique_output


def find_nbor_r(rec_a, rec_b, S):
    rec_a = rec_a.copy()
    rec_b = rec_b.copy()
    axs = []
    dim = []
    for k in S:
        if rec_a[k, 0] == rec_b[k, 1]:
            add = True
            for l in S:
                if k != l:
                    if not (rec_b[l, 0] <= rec_a[l, 0] and rec_b[l, 1] >= rec_a[l, 1]):
                        add = False
                        break
            if add:
                axs.append(k)
                dim.append(0)

        elif rec_a[k, 1] == rec_b[k, 0]:
            add = True
            for l in S:
                if k != l:
                    if not (rec_b[l, 0] <= rec_a[l, 0] and rec_b[l, 1] >= rec_a[l, 1]):
                        add = False
                        break
            if add:
                axs.append(k)
                dim.append(1)
    return axs, dim


def extend_rec_r(rec_a, rec_b, S, axs, dim):
    rec_a = rec_a.copy()
    rec_b = rec_b.copy()
    a = 0
    for k in S:
        if k not in axs:
            if not (rec_b[k, 0] <= rec_a[k, 0] and rec_b[k, 1] >= rec_a[k, 1]):
                a += 1
    if a == 0:
        for i, k in enumerate(axs):
            rec_a[k, dim[i]] = rec_b[k, dim[i]]

    rec_part = []
    not_axs = [i for i in S if i not in axs]
    ps = powerset(not_axs)
    for power in ps:
        rec_remain = []
        merge = True
        for k in S:
            if k in axs:
                rec_remain.append([rec_b[k, 0], rec_b[k, 1]])
            else:
                if k in power:
                    if rec_b[k, 0] == rec_a[k, 0]:
                        merge = False
                        break
                    rec_remain.append([rec_b[k, 0], rec_a[k, 0]])
                else:
                    if rec_b[k, 1] == rec_a[k, 1]:
                        merge = False
                        break
                    rec_remain.append([rec_a[k, 1], rec_b[k, 1]])

        if merge:
            rec_part.append(np.array(rec_remain))

    return rec_a, rec_part


def is_subset(rec_a, rec_b):
    if rec_a == [] or rec_b == []:
        return False
    is_subset = True
    for i in range(rec_a.shape[0]):
        if not (rec_b[i, 0] <= rec_a[i, 0] and rec_b[i, 1] >= rec_a[i, 1]):
            is_subset = False
            break
    return is_subset


def find_mergable(rec_a, list_rec, S):
    for rec_b in list_rec:
        axs, dim = find_nbor_r(rec_a, rec_b, S)
        if len(axs) != 0:
            break

    if len(axs) != 0:
        rec_union, rec_part = extend_rec_r(rec_a, rec_b, S, axs, dim)
        return True, rec_a, rec_b, rec_union, rec_part
    else:
        return False, rec_a, [], [], []


def remove_rule(rule, rule_sets):
    for i, r in enumerate(rule_sets):
        if np.allclose(r, rule):
            rule_sets.pop(i)
            break


def return_largest_rectangle(rule_p, rule_sets, S):
    rule_sets = rule_sets.copy()
    max_step = 5000
    step = 0
    i = 0
    rule_size = len(rule_sets)
    while (i <= rule_size and step <= max_step):

        step += 1
        rule = rule_sets[0]
        find, rule, rule_b, rule_union, rules_part = find_mergable(rule, rule_sets, S)

        if is_subset(rule_p, rule_b):
            find = False

        if find == True:
            remove_rule(rule, rule_sets)
            remove_rule(rule_b, rule_sets)

            random.shuffle(rule_sets)  # new add
            rule_sets.append(rule_union)
            rule_size += 1

            if rules_part != []:
                rule_sets.extend(rules_part)
                rule_size += len(rules_part)
        else:
            remove_rule(rule, rule_sets)
            random.shuffle(rule_sets)  # new add
            rule_sets.append(rule)
        i += 1
    return rule_sets


def return_best(rule_p, rule_sets, S):
    rule_p = rule_p[S]
    rule_sets = [r[S] for r in rule_sets]
    rules_part = return_largest_rectangle(rule_p, rule_sets, list(range(rule_p.shape[0])))
    for r in rules_part:
        if is_subset(rule_p, r):
            break
    return r, rules_part


def unique_rules_r(rules, sdp_all, pi):
    rules = np.array([rules[i] for i in range(rules.shape[0]) if sdp_all[i] >= pi])
    rules_unique = np.unique(rules, axis=0)
    return rules_unique


def return_adjacent_rec(adj, rule_sets, S):
    adj = adj.copy()
    # rule_sets = rule_sets.copy()
    for r in rule_sets:
        for rule in adj:
            axs, dim = find_nbor_r(rule, r, S)
            if len(axs) != 0:
                break
        if len(axs) != 0:
            adj.append(r)
            remove_rule(r, rule_sets)
            adj = return_adjacent_rec(adj, rule_sets, S)
    return adj


def overapprox_rectangle(adj):
    adj = np.array(adj)
    min_axis = np.min(adj[:, :, 0], axis=0).reshape(-1, 1)
    max_axis = np.max(adj[:, :, 1], axis=0).reshape(-1, 1)

    return np.concatenate([min_axis, max_axis], axis=-1)


def overapprox_rule(rule_p, rule_sets, S):
    rule_p = rule_p[S]
    rule_sets = [r[S] for r in rule_sets]
    adj = return_adjacent_rec([rule_p], rule_sets, list(range(rule_p.shape[0])))
    return overapprox_rectangle(adj)


def return_edges(list_rec):
    n = list_rec.shape[0]
    d = list_rec.shape[1]
    base = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            base[i, j] = list_rec[i, j, 0]

    all_points = []
    subsets = powerset(list(range(d)))

    for S in subsets:
        base_buf = base.copy()
        for s in S:
            for i in range(n):
                base_buf[i, s] = list_rec[i, s, 1]
        all_points.append(base_buf)
    return np.concatenate(all_points, axis=0)


def mc_approx_rules(rules, rules_data, S_star, strategy='random', maxitr=1000, interiorOnly=False,
                    threshold=None, verbose=False):
    #     strategy = 'random' # or 'even' or 'sequential'
    #     maxitr = 1000 # how many queries to do since last best found before satisfied
    # whether to consider only rectangles that are bounded on all sides by points rather than limits of the space
    #     interiorOnly = False # for datasets with few points, finding interior rectangles is much harder
    #     threshold = None # just find a list of the biggest

    approx_rec = rules.copy()
    for a in tqdm(range(rules.shape[0])):
        rule_p = rules[a][S_star[a]]
        rule_sets = [r[S_star[a]] for r in rules_data[a]]
        adj = np.array(return_adjacent_rec([rule_p], rule_sets, list(range(rule_p.shape[0]))))

        edges = return_edges(adj)

        d = len(S_star[a])
        vertices = []
        for i in range(edges.shape[0]):
            count = np.sum(np.prod(edges[i] == edges, axis=1))
            if count != 2 ** (d):
                vertices.append(edges[i])

        pt = np.unique(vertices, axis=0)
        hf = HoleFinder(pt, strategy, interiorOnly)

        hallOfFame = hf.findLargestMEHRs(maxitr, threshold, verbose)  # also dumps hallOfFame to disk

        approx_rec[a, S_star[a], 0] = hallOfFame[-1].L
        approx_rec[a, S_star[a], 1] = hallOfFame[-1].U
    return approx_rec


def return_xy_cnt(x, y_target, S_bar_set, x_train, y_train, w):
    """
    return the observations with y=y_target that fall in the projected leaf of x
    when we condition given S=S_bar of x.
    """

    x_train_cnt = []
    for i, wi in enumerate(w):
        if wi != 0 and y_train[i] == y_target:
            x_train_cnt.append(x_train[i].copy())

    if len(x_train_cnt) == 0:
        x_train_cnt = np.array(x).reshape(1, -1)
        y_train_cnt = np.array(y_target).reshape(1, -1)
    else:
        x_train_cnt = np.array(x_train_cnt)
        #     x_train_cnt[:, S_bar_set] = x[S_bar_set]
        y_train_cnt = np.array(x_train_cnt.shape[0] * [y_target])

    return x_train_cnt, y_train_cnt


def return_leaf_cnt(ac_explainer, S_star, x_train_cnt, y_train_cnt, x_train, y_train, pi=0.9):
    """
    return the original leaves of the observations with y=y_target that fall in the projected leaf
    when we condition given S=S_bar of x.
    """
    sdp, rules = ac_explainer.compute_sdp_rule(x_train_cnt, y_train_cnt,
                                               x_train, y_train,
                                               x_train_cnt.shape[0] * [list(range(x_train.shape[1]))]
                                               )
    if np.sum(sdp >= pi) != 0:
        rules_unique = np.unique(rules[sdp >= pi], axis=0)
    else:
        rules_unique = np.expand_dims(rules[np.argmax(sdp)], axis=0)

    r_buf = rules_unique.copy()
    for i in range(rules_unique.shape[0]):
        list_ric = [r.copy() for r in r_buf if not np.allclose(r, rules_unique[i])]
        find_union(rules_unique[i], list_ric, S=S_star)

    return rules_unique


def remove_in(ra):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                ra[i] = ra[j]
    return np.unique(ra, axis=0)


def get_compatible_leaf(acvtree, x, y_target, S_star, S_bar_set, w, x_train, y_train, pi=0.9, acc_level=0.9):
    """
    Compute the compatible leaves and order given their accuracy
    """
    x_train_cnt, y_train_cnt = return_xy_cnt(x, y_target, S_bar_set, x_train, y_train, w)

    compatible_leaves = return_leaf_cnt(acvtree, S_star, x_train_cnt, y_train_cnt, x_train, y_train, pi)
    compatible_leaves = np.unique(compatible_leaves, axis=0)
    compatible_leaves = remove_in(compatible_leaves)
    #     compatible_leaves = np.rint(compatible_leaves)
    compatible_leaves = np.round(compatible_leaves, 2)

    partition_leaf = compatible_leaves.copy()
    d = partition_leaf.shape[1]
    nb_leaf = partition_leaf.shape[0]
    leaves_acc = []
    suf_leaf = []

    for i in range(nb_leaf):
        x_in = np.prod([(x_train[:, s] <= partition_leaf[i, s, 1]) * (x_train[:, s] > partition_leaf[i, s, 0])
                        for s in range(d)], axis=0).astype(bool)
        y_in = y_train[x_in]
        acc = np.mean(y_in == y_target)

        leaves_acc.append(acc)

        if acc >= acc_level:
            suf_leaf.append(partition_leaf[i])

    best_id = np.argmax(leaves_acc)

    return suf_leaf, partition_leaf, leaves_acc, partition_leaf[best_id], leaves_acc[best_id]


def return_counterfactuals(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, x_train, y_train, pi_level):
    """
    Compute the SDP of each C_S and return the ones that has sdp >= pi_level
    """
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []

    for leaf in suf_leaf:

        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = leaf[S_star]
        sdp, w = ac_explainer.compute_ddp_cond_weights(x, y, x_train, y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    return np.unique(counterfactuals, axis=0), np.unique(counterfactuals_sdp, axis=0), \
           np.unique(counterfactuals_w, axis=0)


def return_global_counterfactuals(ac_explainer, data, y_data, s_star, n_star, x_train, y_train, w, acc_level, pi_level):
    """
    stack all to compute the C_S for each observations
    """
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf(ac_explainer, data[i], 1 - y_data[i], s_star[i], n_star[i], w[i],
                                                   x_train, y_train, pi=pi_level, acc_level=acc_level)
        suf_leaves.append(suf_leaf)
        #         print(suf_leaf)
        #         print(np.unique(suf_leaf, axis=0).shape, len(suf_leaf))
        counterfactuals, counterfactuals_sdp, w_cond = \
            return_counterfactuals(ac_explainer, suf_leaf, s_star[i], n_star[i], data[i].reshape(1, -1),
                                   y_data[i].reshape(1, -1), x_train, y_train, pi_level)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)

    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


# Fonction global explanations

def return_ge_counterfactuals(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, x_train, y_train, cond_s, pi_level):
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []

    for leaf in suf_leaf:

        cond = cond_s.copy()
        cond[:, S_star] = leaf[S_star]

        sdp, w = ac_explainer.compute_ddp_cond_weights(x, y, x_train, y_train, S=[[-6]], cond=cond)
        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    return counterfactuals, counterfactuals_sdp, counterfactuals_w


def return_ge_global_counterfactuals(ac_explainer, data, y_data, s_star, n_star, x_train, y_train, w, acc_level, cond,
                                     pi_level):
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf(ac_explainer, data[i], 1 - y_data[i], s_star[i], n_star[i], w[i],
                                                   x_train, y_train, pi=pi_level, acc_level=acc_level)
        suf_leaves.append(suf_leaf)

        counterfactuals, counterfactuals_sdp, w_cond = return_ge_counterfactuals(ac_explainer, suf_leaf, s_star[i],
                                                                                 n_star[i],
                                                                                 data[i].reshape(1, -1),
                                                                                 y_data[i].reshape(1, -1), x_train,
                                                                                 y_train,
                                                                                 np.expand_dims(cond[i], 0),
                                                                                 pi_level)
        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)

    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


def print_rule(col_name, r, decision=None, sdp=True, output=None):
    for i, col in enumerate(col_name):
        if not ((r[i, 0] <= -1e+10 and r[i, 1] >= 1e+10)):
            print('If {} in [{}, {}] and '.format(col, r[i, 0], r[i, 1]))
            print(' ')

    if sdp == True:
        print('Then the output is = {}'.format(output))
        print('SDP Probability = {}'.format(decision))
    else:
        print('Then the output is different from {}'.format(output))
        print('Counterfactual DDP Probability = {}'.format(decision))


def generate_candidate(x, S, x_train, cond, n_iterations):
    x_poss = [x_train[(cond[i, 0] <= x_train[:, i]) * (x_train[:, i] <= cond[i, 1]), i] for i in S]

    x_cand = np.repeat(x.reshape(1, -1), repeats=n_iterations, axis=0)
    for i in range(len(S)):
        rdm_id = np.random.randint(0, x_poss[i].shape[0], n_iterations)
        x_cand[:, S[i]] = x_poss[i][rdm_id]

    return x_cand


def simulated_annealing(outlier_score, x, S, x_train, cond, batch, max_iter, temp,
                        max_iter_convergence=10):
    """
    Generate sample s.t. (X | X_S \in cond) using simulated annealing and outlier score.
    Args:
        outlier_score (lambda functon): outlier_score(X) return a outlier score. If the value are negative, then the observation is an outlier.
        x (numpy.ndarray)): 1-D array, an observation
        S (list): contains the indices of the variables on which to condition
        x_train (numpy.ndarray)): 2-D array represent the training samples
        cond (numpy.ndarray)): 3-D (#variables x 2 x 1) representing the hyper-rectangle on which to condition
        batch (int): number of sample by iteration
        max_iter (int): number of iteration of the algorithm
        temp (double): the temperature of the simulated annealing algorithm
        max_iter_convergence (double): minimun number of iteration to stop the algorithm if it find an in-distribution observation

    Returns:
        The generated sample, and its outlier score
    """
    best = generate_candidate(x, S, x_train, cond, 1)
    best_eval = outlier_score(best)[0]
    curr, curr_eval = best, best_eval

    move = 0
    for i in range(max_iter):

        x_cand = generate_candidate(curr, S, x_train, cond, batch)
        score_candidates = outlier_score(x_cand)

        candidate_eval = np.max(score_candidates)
        candidate = x_cand[np.argmax(score_candidates)]

        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval
            move = 0
        else:
            move += 1

        # check convergence
        if best_eval > 0 and move > max_iter_convergence:
            break

        diff = candidate_eval - curr_eval
        t = temp / np.log(float(i + 1))
        metropolis = np.exp(-diff / t)

        if diff > 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


dataset = 'none'


def save_model(model, name='{}'.format(dataset)):
    with open('{}.pickle'.format(name), 'wb') as f:
        pickle.dump(model, f)


def load_model(name='{}'.format(dataset)):
    with open('{}.pickle'.format(name), 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


class RunExperiments:

    def __init__(self, acv_explainer, x_train, x_test, y_train, y_test, columns_name, model=None):
        self.model = None
        self.columns_name = columns_name
        self.acv_explainer = acv_explainer
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.ddp_importance_local, self.ddp_index_local, self.size_local, self.ddp_local = None, None, None, None
        self.S_star_local, self.S_bar_set_local = None, None
        self.ddp_local, self.w_local = None, None
        self.counterfactuals_samples_local, self.counterfactuals_samples_sdp_local, \
        self.counterfactuals_samples_w_local = None, None, None
        self.isolation = None, None
        self.dist_local = None
        self.score_local = None
        self.errs_local = None
        self.errs_local_original = None
        self.accuracy_local = None
        self.accuracy_local_original = None
        self.coverage_local = None
        self.sdp_importance_se, self.sdp_index_se, self.size_se, self.sdp_se = None, None, None, None
        self.S_star_se, self.N_star_se = None, None
        self.sdp_rules, self.rules, self.sdp_all, self.rules_data, self.w_rules = None, None, None, None, None

        self.ddp_importance_regional, self.ddp_index_regional, self.size_regional, self.ddp_regional = None, None, None, None
        self.S_star_regional, self.S_bar_set_regional = None, None
        self.counterfactuals_samples_regional, self.counterfactuals_samples_sdp_regional, \
        self.counterfactuals_samples_w_regional = None, None, None
        self.dist_regional = None
        self.score_regional = None
        self.errs_regional = None
        self.errs_regional_original = None
        self.accuracy_regional = None
        self.accuracy_regional_original = None
        self.coverage_regional = None

    def run_local_divergent_set(self, x, y, t=20, stop=True, pi_level=0.8):
        print('### Computing the local divergent set of (x, y)')

        self.ddp_importance_local, self.ddp_index_local, self.size_local, self.ddp_local = \
            self.acv_explainer.importance_ddp_rf(x, y, self.x_train, self.y_train, t=t,
                                                 stop=stop, pi_level=pi_level)

        self.S_star_local, self.S_bar_set_local = \
            acv_explainers.utils.get_active_null_coalition_list(self.ddp_index_local, self.size_local)
        self.ddp_local, self.w_local = self.acv_explainer.compute_ddp_weights(x, y, self.x_train, self.y_train,
                                                                              S=self.S_bar_set_local)

    def run_local_counterfactual_rules(self, x, y, acc_level=0.8, pi_level=0.8):
        print('### Computing the local counterfactual rules of (x, y)')

        self.counterfactuals_samples_local, self.counterfactuals_samples_sdp_local, \
        self.counterfactuals_samples_w_local = return_global_counterfactuals(self.acv_explainer, x, y,
                                                                             self.S_star_local, self.S_bar_set_local,
                                                                             self.x_train, self.y_train, self.w_local,
                                                                             acc_level=acc_level,
                                                                             pi_level=pi_level)

    def run_sampling_local_counterfactuals(self, x, y, batch=1000, max_iter=1000, temp=0.5):
        print('### Sampling using the local counterfactual rules of (x, y)')

        self.isolation = IsolationForest()
        self.isolation.fit(self.x_train)
        outlier_score = lambda x: self.isolation.decision_function(x)

        self.dist_local = []
        self.score_local = []
        self.errs_local = []
        self.errs_local_original = []
        for i in tqdm(range(x.shape[0])):
            if len(self.counterfactuals_samples_local[i]) != 0:
                a, sco = simulated_annealing(outlier_score, x[i], self.S_star_local[i], self.x_train,
                                             self.counterfactuals_samples_local[i][
                                                 np.argmax(self.counterfactuals_samples_sdp_local[i])][0],
                                             batch, max_iter, temp)
                self.dist_local.append(np.squeeze(a))
                self.score_local.append(sco)
                self.errs_local.append(
                    self.acv_explainer.predict(self.dist_local[-1].reshape(1, -1)) != self.acv_explainer.predict(
                        x[i].reshape(1, -1)))
                if self.model != None:
                    self.errs_local_original.append(
                        self.model.predict(self.dist_local[-1].reshape(1, -1)) != self.model.predict(
                            x[i].reshape(1, -1)))

        self.accuracy_local = np.mean(self.errs_local)
        self.accuracy_local_original = np.mean(self.errs_local_original)
        self.coverage_local = len(self.errs_local) / x.shape[0]

    def run_sufficient_rules(self, x_rule, y_rule, pi_level=0.9):
        print('### Computing the Sufficient Explanations and the Sufficient Rules')
        self.x_rules, self.y_rules = x_rule, y_rule
        self.sdp_importance_se, self.sdp_index_se, self.size_se, self.sdp_se = \
            self.acv_explainer.importance_sdp_rf(x_rule, y_rule,
                                                 self.x_train, self.y_train,
                                                 stop=False,
                                                 pi_level=pi_level)

        self.S_star_se, self.N_star_se = get_active_null_coalition_list(self.sdp_index_se, self.size_se)
        self.sdp_rules, self.rules, self.sdp_all, self.rules_data, self.w_rules = self.acv_explainer.compute_sdp_maxrules(
            x_rule, y_rule,
            self.x_train, self.y_train, self.S_star_se, verbose=True)

    def run_regional_divergent_set(self, stop=True, pi_level=0.8):
        print('### Computing the regional divergent set of (x, y)')

        self.ddp_importance_regional, self.ddp_index_regional, self.size_regional, self.ddp_regional = \
            self.acv_explainer.importance_ddp_intv(self.x_rules, self.y_rules,
                                                   self.x_train, self.y_train,
                                                   self.rules,
                                                   stop=stop,
                                                   pi_level=pi_level)

        self.S_star_regional, self.S_bar_set_regional = \
            acv_explainers.utils.get_active_null_coalition_list(self.ddp_index_regional, self.size_regional)
        self.ddp_regional, self.w_regional = self.acv_explainer.compute_ddp_weights(self.x_rules, self.y_rules,
                                                                                    self.x_train, self.y_train,
                                                                                    S=self.S_bar_set_regional)

    def run_regional_counterfactual_rules(self, acc_level=0.8, pi_level=0.8):
        print('### Computing the regional counterfactual rules of (x, y)')

        self.counterfactuals_samples_regional, self.counterfactuals_samples_sdp_regional, \
        self.counterfactuals_samples_w_regional = \
            return_ge_global_counterfactuals(self.acv_explainer, self.x_rules, self.y_rules,
                                             self.S_star_regional, self.S_bar_set_regional,
                                             self.x_train, self.y_train, self.w_regional, acc_level, self.rules,
                                             pi_level)

    def run_sampling_regional_counterfactuals(self, max_obs=2, batch=1000, max_iter=1000, temp=0.5):
        print('### Sampling using the regional counterfactual rules')

        outlier_score = lambda x: self.isolation.decision_function(x)

        self.dist_regional = []
        self.score_regional = []
        self.errs_regional = []
        self.errs_regional_original = []
        nb = 0
        for i in range(self.x_rules.shape[0]):
            if len(self.counterfactuals_samples_regional[i]) != 0:
                x_in = np.prod([(self.x_test[:, s] <= self.rules[i, s, 1]) * (self.x_test[:, s] > self.rules[i, s, 0])
                                for s in range(self.x_train.shape[1])], axis=0).astype(bool)
                nb += np.sum(x_in) if np.sum(x_in) <= max_obs else max_obs
                print('observations in rule = {}'.format(np.sum(x_in)))
                if np.sum(x_in) > 0:
                    for xi in tqdm(self.x_test[x_in][:max_obs]):
                        a, sco = simulated_annealing(outlier_score, xi, self.S_star_regional[i], self.x_train,
                                                     self.counterfactuals_samples_regional[i][
                                                         np.argmax(self.counterfactuals_samples_sdp_regional[i])][0],
                                                     batch, max_iter, temp)
                        self.dist_regional.append(np.squeeze(a))
                        self.score_regional.append(sco)
                        self.errs_regional.append(self.acv_explainer.predict(
                            self.dist_regional[-1].reshape(1, -1)) != self.acv_explainer.predict(xi.reshape(1, -1)))
                        if self.model != None:
                            self.errs_regional_original.append(self.model.predict(
                                self.dist_regional[-1].reshape(1, -1)) != self.model.predict(xi.reshape(1, -1)))

        self.accuracy_regional = np.mean(self.errs_regional)
        self.accuracy_regional_original = np.mean(self.errs_regional_original)
        self.coverage_regional = len(self.errs_regional) / nb

    def run_sampling_regional_counterfactuals_alltests(self, max_obs=2, batch=1000, max_iter=1000, temp=0.5):
        print('### Sampling using the regional counterfactual rules')

        outlier_score = lambda x: self.isolation.decision_function(x)

        self.dist_regional = []
        self.score_regional = []
        self.errs_regional = []
        self.errs_regional_original = []
        x_test_pb = []

        for i in range(self.x_rules.shape[0]):
            x_in = np.prod([(self.x_test[:, s] <= self.rules[i, s, 1]) * (self.x_test[:, s] > self.rules[i, s, 0])
                            for s in range(self.x_train.shape[1])], axis=0)
            if len(self.counterfactuals_samples_regional[i]) != 0:
                x_in = np.max(self.counterfactuals_samples_sdp_regional[i]) * x_in
            else:
                x_in = 0 * x_in
            x_test_pb.append(x_in)

        x_test_pb = np.array(x_test_pb)
        best_counterfactuals = np.argmax(x_test_pb, axis=0)

        for i in tqdm(range(self.x_test.shape[0])):
            xi = self.x_test[i]
            best_id = best_counterfactuals[i]
            if len(self.counterfactuals_samples_regional[best_id]) != 0:
                a, sco = simulated_annealing(outlier_score, xi, self.S_star_regional[best_id], self.x_train,
                                             self.counterfactuals_samples_regional[best_id][
                                                 np.argmax(self.counterfactuals_samples_sdp_regional[best_id])][0],
                                             batch, max_iter, temp)
                self.dist_regional.append(np.squeeze(a))
                self.score_regional.append(sco)
                self.errs_regional.append(self.acv_explainer.predict(
                    self.dist_regional[-1].reshape(1, -1)) != self.acv_explainer.predict(xi.reshape(1, -1)))

                if self.model != None:
                    self.errs_regional_original.append(self.model.predict(
                        self.dist_regional[-1].reshape(1, -1)) != self.model.predict(xi.reshape(1, -1)))

        self.accuracy_regional = np.mean(self.errs_regional)
        self.accuracy_regional_original = np.mean(self.errs_regional_original)
        self.coverage_regional = len(self.errs_regional) / self.x_test.shape[0]

    def show_global_counterfactuals(self):

        for idt in range(self.rules.shape[0]):
            print('Example {}'.format(idt))
            r = self.rules[idt]

            print_rule(self.columns_name, r, self.sdp_rules[idt], True, self.y_rules[idt])

            print('  ')
            print('  ')
            print('  ')
            for l in range(len(self.counterfactuals_samples_sdp_regional[idt])):
                print('Example {} - Counterfactual {}'.format(idt, l))
                print_rule(self.columns_name, self.counterfactuals_samples_regional[idt][l][0],
                           self.counterfactuals_samples_sdp_regional[idt][l], False, False)
                print('  ')
            print(' ')

    def show_local_counterfactuals(self, x, y):

        for idt in range(x.shape[0]):
            print(self.columns_name)
            print('Example {} = {}'.format(idt, x[idt]))

            print('  ')
            print('  ')
            print('  ')
            for l in range(len(self.counterfactuals_samples_sdp_local[idt])):
                print('Example {} - Counterfactual {}'.format(idt, l))
                print_rule(self.columns_name, self.counterfactuals_samples_local[idt][l][0],
                           self.counterfactuals_samples_sdp_local[idt][l], False, False)
                print('  ')
            print(' ')


def return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w):
    """
    return the observations with y=y_target that fall in the projected leaf of x
    when we condition given S=S_bar of x.
    """

    x_train_cnt = []
    y_train_cnt = []
    for i, wi in enumerate(w):
        if wi != 0 and down <= y_train[i] <= up:
            x_train_cnt.append(x_train[i].copy())
            y_train_cnt.append(y_train[i].copy())

    x_train_cnt = np.array(x_train_cnt)
    y_train_cnt = np.array(y_train_cnt)

    return x_train_cnt, y_train_cnt


def return_leaf_cnt_reg(ac_explainer, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi):
    """
    return the original leaves of the observations with y=y_target that fall in the projected leaf
    when we condition given S=S_bar of x.
    """
    size = x_train_cnt.shape[0]
    sdp, rules = ac_explainer.compute_cdp_rule(x_train_cnt, y_train_cnt, np.array(size * [down]), np.array(size * [up]),
                                               x_train, y_train,
                                               size * [list(range(x_train.shape[1]))]
                                               )
    #     print(sdp)
    if np.sum(sdp >= pi) != 0:
        rules_unique = np.unique(rules[sdp >= pi], axis=0)
    else:
        rules_unique = np.expand_dims(rules[np.argmax(sdp)], axis=0)

    r_buf = rules_unique.copy()
    for i in range(rules_unique.shape[0]):
        list_ric = [r.copy() for r in r_buf if not np.allclose(r, rules_unique[i])]
        find_union(rules_unique[i], list_ric, S=S_star)

    return rules_unique


def remove_in(ra):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                ra[i] = ra[j]
    return np.unique(ra, axis=0)


def get_compatible_leaf_reg(acvtree, x, y_target, down, up, S_star, S_bar_set, w, x_train, y_train, pi, acc_level):
    """
    Compute the compatible leaves and order given their accuracy
    """
    x_train_cnt, y_train_cnt = return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w)
    compatible_leaves = return_leaf_cnt_reg(acvtree, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi)
    compatible_leaves = np.unique(compatible_leaves, axis=0)
    compatible_leaves = remove_in(compatible_leaves)
    compatible_leaves = np.round(compatible_leaves, 2)

    partition_leaf = compatible_leaves.copy()
    d = partition_leaf.shape[1]
    nb_leaf = partition_leaf.shape[0]
    leaves_acc = []
    suf_leaf = []

    for i in range(nb_leaf):
        x_in = np.prod([(x_train[:, s] <= partition_leaf[i, s, 1]) * (x_train[:, s] > partition_leaf[i, s, 0])
                        for s in range(d)], axis=0).astype(bool)

        y_in = y_train[x_in]
        acc = np.mean((down <= y_in) * (y_in <= up))
        leaves_acc.append(acc)

        if acc >= acc_level:
            suf_leaf.append(partition_leaf[i])

    best_id = np.argmax(leaves_acc)
    return suf_leaf, partition_leaf, leaves_acc, partition_leaf[best_id], leaves_acc[best_id]


def return_counterfactuals_reg(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, down, up, x_train, y_train, pi_level):
    """
    Compute the SDP of each C_S and return the ones that has sdp >= pi_level
    """
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []

    for leaf in suf_leaf:

        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = leaf[S_star]

        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [down]), np.array(size * [up]), x_train,
                                                       y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        #         print(sdp)
        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    return np.unique(counterfactuals, axis=0), np.unique(counterfactuals_sdp, axis=0), \
           np.unique(counterfactuals_w, axis=0)


def return_global_counterfactuals_reg(ac_explainer, data, y_data, down, up, s_star, n_star, x_train, y_train, w,
                                      acc_level, pi_level):
    """
    stack all to compute the C_S for each observations
    """
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf_reg(ac_explainer, data[i], 1 - y_data[i], down[i], up[i], s_star[i],
                                                       n_star[i], w[i],
                                                       x_train, y_train, pi=pi_level, acc_level=acc_level)
        suf_leaves.append(suf_leaf)
        #         print(suf_leaf)
        #         print(np.unique(suf_leaf, axis=0).shape, len(suf_leaf))
        counterfactuals, counterfactuals_sdp, w_cond = \
            return_counterfactuals_reg(ac_explainer, suf_leaf, s_star[i], n_star[i], data[i].reshape(1, -1),
                                       y_data[i].reshape(1, -1), down[i], up[i], x_train, y_train, pi_level)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)
    #     print(counterfactuals_samples)
    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


def return_possible_values(rule, rule_data, sdp_all, pi_level):
    d = rule.shape[0]
    adj_rules = np.array([rule_data[i] for i in range(sdp_all.shape[0]) if sdp_all[i] >= pi_level])

    left_by_var = []
    right_by_var = []

    for var in range(d):
        left = set([adj_rules[i, var, 0] for i in range(adj_rules.shape[0]) if adj_rules[i, var, 0] <= rule[var, 0]])
        right = set([adj_rules[i, var, 1] for i in range(adj_rules.shape[0]) if adj_rules[i, var, 1] >= rule[var, 1]])

        left.add(rule[var, 0])
        right.add(rule[var, 1])

        left_by_var.append(list(left))
        right_by_var.append(list(right))
    return left_by_var, right_by_var


def rdm_sample_fromlist(list_values):
    return list_values[np.random.randint(0, len(list_values))]


def generate_newrule(rule, left_by_var, right_by_var, p_best=0.6):
    d = rule.shape[0]
    rule_c = rule.copy()
    if np.random.rand() > p_best:
        for i in range(d):
            rule_c[i, 0] = rdm_sample_fromlist(left_by_var[i])
            rule_c[i, 1] = rdm_sample_fromlist(right_by_var[i])
    else:
        for i in range(d):
            left = [left_by_var[i][j] for j in range(len(left_by_var[i])) if left_by_var[i][j] <= rule_c[i, 0]]
            right = [right_by_var[i][j] for j in range(len(right_by_var[i])) if right_by_var[i][j] >= rule_c[i, 1]]

            rule_c[i, 0] = rdm_sample_fromlist(left)
            rule_c[i, 1] = rdm_sample_fromlist(right)

    return rule_c


def check_rule(rule, x_train, bad_sample):
    x_in = np.prod([(x_train[:, s] <= rule[s, 1]) * (x_train[:, s] >= rule[s, 0])
                    for s in range(x_train.shape[1])], axis=0)
    return np.sum(x_in * bad_sample) == 0


def generate_valid_rule(rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.9, n_try=100):
    valid = False
    it = 0
    while valid == False and it <= n_try:
        gen_rule = generate_newrule(rule, left_by_var, right_by_var, p_best)
        valid = check_rule(gen_rule, x_train, bad_sample)
        it += 1
    if valid:
        return gen_rule
    else:
        return rule


def generate_rules(rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.6, n_try=100, n_gen=10):
    rules = [generate_valid_rule(rule, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try) for i in
             range(n_gen)]
    return rules


def convert_boundvalues(rules, max_values, min_values):
    for i in range(rules.shape[0]):
        if rules[i, 0] <= -1e+10:
            rules[i, 0] = min_values[i]
        if rules[i, 1] <= -1e+10:
            rules[i, 1] = min_values[i]
        if rules[i, 0] >= 1e+10:
            rules[i, 0] = max_values[i]
        if rules[i, 1] >= 1e+10:
            rules[i, 1] = max_values[i]
    return rules


def volume_rectangle(rec, max_values, min_values):
    d = rec.shape[0]
    rec_c = rec.copy()
    rec_c = convert_boundvalues(rec_c, max_values, min_values)
    v = 1
    for i in range(d):
        v *= (rec_c[i, 1] - rec_c[i, 0])
    return v


def volume_rectangles(recs, max_values, min_values):
    return [volume_rectangle(recs[i], max_values, min_values) for i in range(len(recs))]


def max_dens(recs, x_train, bad_sample):
    d = x_train.shape[1]
    x_in = np.prod([(x_train[:, s] <= recs[s, 1]) * (x_train[:, s] > recs[s, 0])
                    for s in range(d)], axis=0)
    return np.sum(x_in * (1 - bad_sample)) / np.sum(1 - bad_sample)


def max_denss(recs, x_train, bad_sample):
    return [max_dens(recs[i], x_train, bad_sample) for i in range(len(recs))]


def rules_simulated_annealing(value_function, rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.9,
                              n_try=100,
                              batch=100, max_iter=200, temp=1,
                              max_iter_convergence=50):
    max_values = [np.max(x_train[:, i]) for i in range(x_train.shape[1])]
    min_values = [np.min(x_train[:, i]) for i in range(x_train.shape[1])]

    best = generate_rules(rule, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try, n_gen=1)
    best_eval = value_function(best, max_values, min_values)[0]
    curr, curr_eval = np.squeeze(best[0]), best_eval

    move = 0
    for i in range(max_iter):

        x_cand = generate_rules(curr, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try, n_gen=batch)
        score_candidates = value_function(x_cand, max_values, min_values)

        candidate_eval = np.max(score_candidates)
        candidate = x_cand[np.argmax(score_candidates)]

        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval
            move = 0
        else:
            move += 1

        # check convergence
        if move > max_iter_convergence:
            break

        diff = candidate_eval - curr_eval
        t = temp / np.log(float(i + 1))
        metropolis = np.exp(-diff / t)

        if diff > 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


def rules_by_annealing(value_function, rules, rules_data, sdp_all, x_train, pi_level=0.9, p_best=0.6, n_try=50,
                       batch=100, max_iter=200, temp=1, max_iter_convergence=50):
    sr, sr_eval = [], []
    for idx in tqdm(range(rules.shape[0])):
        rule = rules[idx]
        rule_data = rules_data[idx]
        bad_sample = 1. - (sdp_all[idx] >= pi_level)
        sdp_rule = sdp_all[idx]

        left_by_var, right_by_var = return_possible_values(rule, rule_data, sdp_rule, pi_level)
        best, best_eval = rules_simulated_annealing(value_function, rule, left_by_var, right_by_var, x_train,
                                                    bad_sample,
                                                    p_best, n_try,
                                                    batch, max_iter, temp, max_iter_convergence)
        sr.append(np.squeeze(best))
        sr_eval.append(best_eval)
    return np.array(sr), sr_eval


def remove_in_wsdp(ra, sa):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                ra[i] = ra[j]
                sa[i] = sa[j]
    return np.unique(ra, axis=0), np.unique(sa, axis=0)
