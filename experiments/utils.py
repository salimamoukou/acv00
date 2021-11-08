import itertools

import math
import matplotlib.pyplot as plt

import multiprocessing as mp
import numpy as np

import pandas as pd

import random

import scipy.stats as st
import scipy.stats as stats

import seaborn as sns

import string

import sys
import time
import warnings
from collections import defaultdict
# from exp_linear import *

from operator import itemgetter

from scipy.special import comb
from scipy.stats import kendalltau
from sklearn.cluster import AffinityPropagation

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

from typing import Tuple, List



def styling_dataframe_local_sdp(instance: pd.DataFrame,
                                local_sdp: pd.DataFrame):
    """Add a style to instance dataframe.
    All columns where there's a 1 in local_sdp, the background is set to blue
    The background of column 'SDP' is set to red

    Args:
        instance (pd.DataFrame): Instance from which we got the SDP
        local_sdp (pd.DataFrame): The local sdp with a 1/0 for all variables and a SDP column

    Returns:
        pd.DataFrame: The instance with the new style
    """

    instance = (
        instance
        .assign(
            SDP=lambda df: local_sdp.iloc[0].SDP
        )
    )

    cols = list(local_sdp.columns)
    cols.remove('SDP')

    instance = instance[['SDP'] + cols]
    local_sdp = (
        local_sdp.drop(columns='SDP')
        .iloc[0]
    )

    instance = (
        instance
        .style
        .apply(highlight_importance_features,
               s=local_sdp,
               props='background-color: #3e82fc',
               axis=1,
               subset=cols)
        .set_properties(**{'background-color': '#ff073a'},
                        subset=['SDP'])
    )


    return instance


def compute_kendall(shap_values, shap_values_acv):
    shap_values_rank = shap_values.rank(axis=1, method='first', ascending=False).astype(int)
    shap_values_acv_rank = shap_values_acv.rank(axis=1, method='first', ascending=False).astype(int)
    kendall_res = []
    for i in range(len(shap_values_rank)):
        kendall_res.append(kendalltau(shap_values_rank.iloc[i].values, shap_values_acv_rank.iloc[i].values).correlation)
    return kendall_res

def plot_roc_pr_curves(y_test, y_pred):
    # Print global scores
    print(f"Roc AUC score : {roc_auc_score(y_test, y_pred)}")
    print(f"PR AUC score  : {average_precision_score(y_test, y_pred)}")

    # Plot ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             sharex=True, sharey=True)

    ax = axes[0]
    fpr_RF, tpr_RF, _ = roc_curve(y_test, y_pred)
    ax.step(fpr_RF, tpr_RF, linestyle='-.', c='g', lw=1, where='post')
    ax.set_title("ROC", fontsize=20)
    ax.legend(loc='upper center', fontsize=8)
    ax.set_xlabel('False Positive Rate', fontsize=18)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=18)

    ax = axes[1]
    precision_RF, recall_RF, _ = precision_recall_curve(y_test, y_pred)
    ax.step(recall_RF, precision_RF, linestyle='-.', c='g', lw=1, where='post')
    ax.set_title("Precision-Recall", fontsize=20)
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=18)
    ax.set_ylabel('Precision', fontsize=18)


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


def gen_data_by_cat(x_in, cat_va, prob, N, S, S_bar):
    # to do: make it real function of S, c_index, etc

    mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)

    rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key],
                         self.cov[key], S_bar, S, x_in[S]) for key in cat_va
          if np.sum(mixture_idx == key) != 0}

    rg = np.concatenate([np.concatenate([rg[key], np.tile(self.dummy[key],
                                                          (np.sum(mixture_idx == key), 1))], axis=1) for key in
                         rg.keys()])

    rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

    for val_id in S:
        rg_data[str(val_id)] = get_given_data(val_id)

    rg_data = rg_data[sorted(rg_data.columns)]
    return rg_data


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

#
# def get_tree_partition(x, fx, tx, tree, S, data=None):
#     """
#     Compute the partition (L_m) of each compatible leaf of the condition X_s = x_S, then check for each
#     observations in data in which leaves it falls.
#
#     Args:
#         x (array): observation
#         fx (float): tree(x)
#         tx (float): threshold of the classifier
#         tree (DecisionTreeClassifier.tree_): model
#         S (list): index of variables on which we want to compute the SDP
#         algo (string): name of the estimators, recommended 'pluging'
#         data (array): data used to compute the partion
#
#     Returns:
#         (array, array): binary array of shape (data_size, compatible_leaves), if [i, j] = 1 then observation i fall in
#         leaf j.
#         (array): return number of observations that fall in each leaf
#     """
#
#     a = tree.children_left
#     b = tree.children_right
#     f = tree.features
#     t = tree.thresholds
#     # r_w = tree.node_samples_weight
#     v = tree.values.reshape(-1)/tree.scaling
#     index = range(x.shape[0])
#
#     y_pred = tree.predict(data)
#     dist = (y_pred - fx) ** 2
#
#     up_tx = np.array(dist > tx).reshape(-1)
#     down_tx = np.array(dist <= tx).reshape(-1)
#
#     def explore_partition(i, tab, partition_leaves, partition_global, prob_global, s_global, S, S_bar, data,
#                           intv=False):
#         if a[i] < 0:
#             #         tab[i] = 1
#             compatible_leaves.append(i)
#             partition_global[i] = partition_leaves
#             partition_leaves = np.squeeze(np.array(partition_leaves))
#
#             section_x = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0)
#
#             section_x_bar = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S_bar], axis=0)
#
#             section_up = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0) * up_tx
#
#             section_down = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0) * down_tx
#
#             prob_all = section_x * section_x_bar
#             prob_up = section_up * section_x_bar
#             prob_down = section_down * section_x_bar
#
#             s_all = section_x
#             s_up = section_up
#             s_down = section_down
#
#             prob_global['all'].append(prob_all.reshape(1, -1))
#             prob_global['up'].append(prob_up.reshape(1, -1))
#             prob_global['down'].append(prob_down.reshape(1, -1))
#
#             s_global['all'].append(s_all.reshape(1, -1))
#             s_global['up'].append(s_up.reshape(1, -1))
#             s_global['down'].append(s_down.reshape(1, -1))
#
#         else:
#             if f[i] in S:
#                 if x[f[i]] <= t[i]:
#                     part = partition_leaves.copy()
#                     part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
#                     part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#                     explore_partition(a[i], tab, part, partition_global, prob_global, s_global, S, S_bar, data,
#                                       intv)
#                 else:
#                     part = partition_leaves.copy()
#                     part[f[i]] = np.concatenate((part[f[i]], np.array([[t[i], np.inf]])))
#                     part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#                     explore_partition(b[i], tab, part, partition_global, prob_global, s_global, S, S_bar, data,
#                                       intv)
#             else:
#                 part = partition_leaves.copy()
#                 part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
#                 part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#
#                 part_2 = partition_leaves.copy()
#                 part_2[f[i]] = np.concatenate((part_2[f[i]], np.array([[t[i], np.inf]])))
#                 part_2[f[i]] = np.array([[np.max(part_2[f[i]][:, 0]), np.min(part_2[f[i]][:, 1])]])
#
#                 explore_partition(a[i], tab, part, partition_global, prob_global, s_global, S, S_bar, data, intv)
#                 explore_partition(b[i], tab, part_2, partition_global, prob_global, s_global, S, S_bar, data, intv)
#
#     S_bar = [i for i in index if i not in S]
#     partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
#     partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
#                         for i in range(len(tree.features))}
#     prob_global = {'all': [], 'up': [], 'down': []}
#     s_global = {'all': [], 'up': [], 'down': []}
#
#     part_final = {}
#     compatible_leaves = []
#     explore_partition(0, compatible_leaves, partition_leaves, partition_global, prob_global, s_global, S,
#                       S_bar, data)
#
#     part_final['all'] = np.concatenate(prob_global['all'], axis=0)
#     part_final['up'] = np.concatenate(prob_global['up'], axis=0)
#     part_final['down'] = np.concatenate(prob_global['down'], axis=0)
#
#     part_final['s_all'] = np.concatenate(s_global['all'], axis=0)
#     part_final['s_up'] = np.concatenate(s_global['up'], axis=0)
#     part_final['s_down'] = np.concatenate(s_global['down'], axis=0)
#
#     return part_final, v[compatible_leaves]

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
                              ,down_tx, up_tx, intv)
            explore_partition(children_right[i], x, children_left, children_right, features, thresholds, values,
                              compatible_leaves, part_right, partition_global, prob_global, s_global, S, S_bar, data
                              ,down_tx, up_tx, intv)


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
        msg,e = import_errors[package_name]
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


def generate_y(x, data_type, coefs=0):
    """Generate corresponding label (y) given feature (x).

    Args:
      - x: features
      - data_type: synthetic data type (syn1 to syn6)
    Returns:
      - y: corresponding labels
    """
    # number of samples
    n = x.shape[0]

    # Logit computation
    if data_type == 'syn1':
        logit = np.exp(x[:, 0] * x[:, 1])
    elif data_type == 'syn2':
        logit = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
    elif data_type == 'synamk':
        logit = np.exp(np.sum(coefs*x, axis=1) - 4.0)
    elif data_type == 'syn3':
        logit = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                       x[:, 8] + np.exp(-x[:, 9]) - 2.4)
    elif data_type == 'syn4':
        logit1 = np.exp(x[:, 0] * x[:, 1])
        # logit2 = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
        # logit2 = np.exp(np.sum(coefs * x, axis=1) - 4.0
        logit2 = np.exp(x[:, 2] * x[:, 3])
    elif data_type == 'syn5':
        logit1 = np.exp(x[:, 0] * x[:, 1])
        logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                        x[:, 8] + np.exp(-x[:, 9]) - 2.4)
    elif data_type == 'syn6':
        logit1 = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
        logit2 = np.exp(-10 * np.sin(0.2 * x[:, 6]) + abs(x[:, 7]) + \
                        x[:, 8] + np.exp(-x[:, 9]) - 2.4)
    elif data_type == 'syn7':
        logit1 = np.sum(coefs[0:2] * x[:, 0:2], axis=1)
        # logit2 = np.exp(np.sum(x[:, 2:6] ** 2, axis=1) - 4.0)
        logit2 = np.sum(coefs[2:4] * x[:, 2:4], axis=1)
        # logit2 = np.exp(x[:, 2] * x[:, 3])

        idx1 = (x[:, 4] < 0) * 1
        idx2 = (x[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2
        return logit


        # For syn4, syn5 and syn6 only
    if data_type in ['syn4', 'syn5', 'syn6', 'syn7']:
        # Based on X[:,10], combine two logits
        idx1 = (x[:, 4] < 0) * 1
        idx2 = (x[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2

        # Compute P(Y=0|X)
    prob_0 = np.reshape((logit / (1 + logit)), [n])

    # Sampling process
    y = np.zeros([n, 2])
    # y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n, ])
    y[:, 0] = prob_0
    y[:, 1] = 1 - y[:, 0]

    return y


def generate_ground_truth(x, data_type):
    """Generate ground truth feature importance corresponding to the data type
       and feature.

    Args:
      - x: features
      - data_type: synthetic data type (syn1 to syn6)
    Returns:
      - ground_truth: corresponding ground truth feature importance
    """

    # Number of samples and features
    n, d = x.shape

    # Output initialization
    ground_truth = np.zeros([n, d])

    # For each data_type
    if data_type == 'syn1':
        ground_truth[:, :2] = 1
    elif data_type == 'syn2':
        ground_truth[:, 2:6] = 1
    elif data_type == 'syn3':
        ground_truth[:, 6:10] = 1

    # Index for syn4, syn5 and syn6
    if data_type in ['syn4', 'syn5', 'syn6']:
        idx1 = np.where(x[:, 10] < 0)[0]
        idx2 = np.where(x[:, 10] >= 0)[0]
        ground_truth[:, 10] = 1

    if data_type == 'syn4':
        ground_truth[idx1, :2] = 1
        ground_truth[idx2, 2:6] = 1
    elif data_type == 'syn5':
        ground_truth[idx1, :2] = 1
        ground_truth[idx2, 6:10] = 1
    elif data_type == 'syn6':
        ground_truth[idx1, 2:6] = 1
        ground_truth[idx2, 6:10] = 1

    return ground_truth

def feature_performance_metric(ground_truth, importance_score):
    """Performance metrics for feature importance (TPR and FDR).

    Args:
      - ground_truth: ground truth feature importance
      - importance_score: computed importance scores for each feature

    Returns:
      - mean_tpr: mean value of true positive rate
      - std_tpr: standard deviation of true positive rate
      - mean_fdr: mean value of false discovery rate
      - std_fdr: standard deviation of false discovery rate
    """

    n = importance_score.shape[0]

    tpr = np.zeros([n, ])
    fdr = np.zeros([n, ])

    # For each sample
    for i in range(n):
        # tpr
        tpr_nom = np.sum(importance_score[i, :] * ground_truth[i, :])
        tpr_den = np.sum(ground_truth[i, :])
        tpr[i] = 100 * float(tpr_nom) / float(tpr_den + 1e-8)

        # fdr
        fdr_nom = np.sum(importance_score[i, :] * (1 - ground_truth[i, :]))
        fdr_den = np.sum(importance_score[i, :])
        fdr[i] = 100 * float(fdr_nom) / float(fdr_den + 1e-8)

    mean_tpr = np.mean(tpr)
    std_tpr = np.std(tpr)
    mean_fdr = np.mean(fdr)
    std_fdr = np.std(fdr)

    return mean_tpr, std_tpr, mean_fdr, std_fdr


def quantile_discretizer(df, num, cat_cols):
    quantiles = np.round(np.linspace(0, 1, num=num), 2)
    columns_names = list(df.columns)
    q_cols = {}
    q_values = {'{}'.format(col): [] for col in columns_names if col not in cat_cols}

    for col in columns_names:
        if col not in cat_cols:
            it = 0
            for q_low, q_high in zip(quantiles[:-1], quantiles[1:]):
                it = it + 1
                q_low_v = np.quantile(df[col].values, q_low)
                q_high_v = np.quantile(df[col].values, q_high)
                q_values['{}'.format(col)].append([q_low_v, q_high_v])
                if it == num - 1:
                    q_cols['{}: q{}-q{}'.format(col, q_low, q_high)] = 1 * (q_low_v <= df[col].values) * (
                                df[col].values <= q_high_v)
                else:
                    q_cols['{}: q{}-q{}'.format(col, q_low, q_high)] = 1 * (q_low_v <= df[col].values) * (
                                df[col].values < q_high_v)
        else:
            q_cols[col] = df[col]
    return pd.DataFrame.from_dict(q_cols), q_values


def quantile_discretizer_byq(df, cat_cols, q_values):
    columns_names = list(df.columns)
    num = len(list(q_values.values())[0]) + 1
    q_cols = {}
    for col in columns_names:
        if col not in cat_cols:
            it = 0
            for q_val in q_values[col]:
                it = it + 1
                q_low_v = q_val[0]
                q_high_v = q_val[1]

                if it == num - 1:
                    q_cols['{}: I{}'.format(col, it)] = 1 * (q_low_v <= df[col].values) * (df[col].values <= q_high_v)
                else:
                    q_cols['{}: I{}'.format(col, it)] = 1 * (q_low_v <= df[col].values) * (df[col].values < q_high_v)
        else:
            q_cols[col] = df[col]
    return pd.DataFrame.from_dict(q_cols)

def cond_exp_tree(x, tree, S, mean, cov, N=10000):
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg

        y_pred = tree.predict(rg_data)

    else:
        y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

    return np.mean(y_pred, axis=0)

def cond_exp_tree_true(x, yx, tree, S, mean, cov,  N=10000):
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg

        logit1 = np.exp(rg_data[:, 0] * rg_data[:, 1])
        logit2 = np.exp(rg_data[:, 2] * rg_data[:, 3])
        idx1 = (rg_data[:, 4] < 0) * 1
        idx2 = (rg_data[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2

        # Compute P(Y=0|X)
        prob_0 = np.reshape((logit / (1 + logit)), [N, 1])

        # Sampling process
        y = np.zeros([N, 2])
        y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [N, ])
        y[:, 1] = 1 - y[:, 0]

        y_pred = y[:, 1]
        # print(prob_0)
    else:
        # x = np.expand_dims(x, axis=0)
        # logit1 = np.exp(x[:, 0] * x[:, 1])
        # logit2 = np.exp(x[:, 2] * x[:, 3])
        # idx1 = (x[:, 4] < 0) * 1
        # idx2 = (x[:, 4] >= 0) * 1
        # logit = logit1 * idx1 + logit2 * idx2
        #
        # # Compute P(Y=0|X)
        # prob_0 = np.reshape((logit / (1 + logit)), [1, 1])
        #
        # # Sampling process
        # y = np.zeros([N, 2])
        # y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [1, ])
        # y[:, 1] = 1 - y[:, 0]

        y_pred = np.expand_dims(yx, axis=0)

    return np.mean(y_pred, axis=0)


def single_sdp_true_v(x, yx, tree, S, mean, cov,  N=10000):
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg

        logit1 = np.exp(rg_data[:, 0] * rg_data[:, 1])
        logit2 = np.exp(rg_data[:, 2] * rg_data[:, 3])
        idx1 = (rg_data[:, 4] < 0) * 1
        idx2 = (rg_data[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2

        # Compute P(Y=0|X)
        prob_0 = np.reshape((logit / (1 + logit)), [N, 1])

        # Sampling process
        y = np.zeros([N, 2])
        y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [N, ])
        y[:, 1] = 1 - y[:, 0]

        y_pred = y[:, 1]
        # print(prob_0)
    else:

        y_pred = np.expand_dims(yx, axis=0)

    return np.mean(y_pred == yx, axis=0)


def shap_exp(tree, S, x):
    tree_ind = 0

    def R(node_ind):

        f = tree.features[tree_ind, node_ind]
        lc = tree.children_left[tree_ind, node_ind]
        rc = tree.children_right[tree_ind, node_ind]
        if lc < 0:
            return tree.values[tree_ind, node_ind]
        if f in S:
            if x[f] <= tree.thresholds[tree_ind, node_ind]:
                return R(lc)
            return R(rc)
        lw = tree.node_sample_weight[tree_ind, lc]
        rw = tree.node_sample_weight[tree_ind, rc]
        return (R(lc) * lw + R(rc) * rw) / (lw + rw)

    out = 0.0
    l = tree.values.shape[0] if tree.tree_limit is None else tree.tree_limit
    for i in range(l):
        tree_ind = i
        out += R(0)
    return out

def shap_cond_exp(X, S, tree):
    cond = np.zeros((X.shape[0], tree.values.shape[2]))
    for i in range(X.shape[0]):
        cond[i] = shap_exp(x=X[i], S=S, tree=tree)
    return cond

def mc_cond_exp(X, S, tree, mean, cov, N):
    cond = np.zeros((X.shape[0], tree.values.shape[2]))
    for i in range(X.shape[0]):
        cond[i] = cond_exp_tree(x=X[i], S=S, tree=tree, mean=mean, cov=cov, N=N)
    return cond

def mc_cond_exp_true(X, yX, S, tree, mean, cov, N):
    cond = np.zeros((X.shape[0], tree.values.shape[2]))
    for i in range(X.shape[0]):
        cond[i] = cond_exp_tree_true(x=X[i], yx=yX[i], S=S, tree=tree, mean=mean, cov=cov, N=N)
    return cond

def sdp_true_v(X, yX, S, tree, mean, cov, N):
    cond = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        cond[i] = single_sdp_true_v(x=X[i], yx=yX[i], S=S, tree=tree, mean=mean, cov=cov, N=N)
    return cond

def tree_sv_exact(X, C, tree, mean, cov, N):
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()

    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1], tree.values.shape[2]))

    for i in tqdm(va_id):
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in powerset(Sm):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = mc_cond_exp(X=X, S=np.array(chain_l(S) + convert_list(i)).astype(int), tree=tree, mean=mean, cov=cov, N=N)
            v_minus = mc_cond_exp(X=X, S=np.array(chain_l(S)).astype(int), tree=tree, mean=mean, cov=cov, N=N)

            for j in convert_list(i):
                phi[:, j] += weight * (v_plus - v_minus)

    return phi / m


def tree_sv_exact_true(X, yX, C, tree, mean, cov, N):
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()

    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1], tree.values.shape[2]))

    for i in tqdm(va_id):
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in powerset(Sm):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = mc_cond_exp_true(X=X, yX=yX, S=np.array(chain_l(S) + convert_list(i)).astype(int), tree=tree, mean=mean, cov=cov, N=N)
            v_minus = mc_cond_exp_true(X=X, yX=yX, S=np.array(chain_l(S)).astype(int), tree=tree, mean=mean, cov=cov, N=N)

            for j in convert_list(i):
                phi[:, j] += weight * (v_plus - v_minus)

    return phi / m

def pytree_shap_plugin(X, data, C, tree):
    N = X.shape[0]
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()

    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1], tree.values.shape[2]))

    for i in tqdm(va_id):
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in powerset(Sm):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = tree.compute_exp_normalized(X=X, S=np.array(chain_l(S) + convert_list(i)).astype(int), data=data)
            v_minus = tree.compute_exp_normalized(X=X, S=np.array(chain_l(S)).astype(int), data=data)

            for a in convert_list(i):
                phi[:, a] += weight * (v_plus - v_minus)

    return phi / m

def marMVNDiscrete(mean, cov, set):
    if set == []:
        return mean, cov
    else:
        mean_cond = mean[set]
        cov_cond = cov[set][:, set]
        return mean_cond, cov_cond


def sampleMarMVNDiscrete(n, mean, cov, set):
    mean_cond, cov_cond = marMVNDiscrete(mean, cov, set)
    sample = st.multivariate_normal(mean_cond, cov_cond).rvs(n)
    if len(set) != 0:
        sample = np.reshape(sample, (sample.shape[0], len(set)))
    return sample

def mc_exp_tree_discretized(X, tree, S, q_arr, q_values, C, mean, cov, N):
    d = len(C)
    if len(S) != X.shape[1]:
        part = np.zeros((len(S), X.shape[0], 2))
        for i in range(len(S)):
            q_idx = np.argmax(X[:, C[S[i]]], axis=1)
            for j in range(X.shape[0]):
                part[i, j, 0] = q_arr[S[i], q_idx[j], 0]
                part[i, j, 1] = q_arr[S[i], q_idx[j], 1]

        joint_samples = sampleMarMVNDiscrete(N, mean, cov, [])
        mar_samples = sampleMarMVNDiscrete(N, mean, cov, S)

        columns_name = ['X{}'.format(i) for i in range(d)]
        joint_samples_cat = pd.DataFrame(joint_samples, columns=columns_name)
        joint_samples_cat = quantile_discretizer_byq(joint_samples_cat, [], q_values).values

        y_pred = tree.predict(joint_samples_cat)
        cond_mean = np.zeros(X.shape[0])

        for j in range(X.shape[0]):
            joint_ind = np.prod([(joint_samples[:, S[i]] < part[i, j, 1]) * \
                                 (joint_samples[:, S[i]] >= part[i, j, 0]) for i in range(len(S))], axis=0)

            mar_ind = np.prod([(mar_samples[:, i] < part[i, j, 1]) * \
                               (mar_samples[:, i] >= part[i, j, 0]) for i in range(len(S))], axis=0)

            num = np.mean(y_pred * joint_ind)
            den = np.mean(mar_ind)
            cond_mean[j] = num / den

        return cond_mean
    else:
        y_pred = tree.predict(np.array(X, dtype=np.float32))

    return np.mean(y_pred, axis=0)


def tree_sv_exact_discretized(X, tree, q_arr, q_values, C, mean, cov, N=10000):
    m = len(C)
    va_id = list(range(m))
    va_buffer = va_id.copy()

    phi = np.zeros(shape=(X.shape[0], m, tree.values.shape[2]))

    for i in tqdm(va_id):
        Sm = list(set(va_buffer) - set(convert_list(i)))

        for S in powerset(Sm):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = mc_exp_tree_discretized(X=X, S=chain_l(S) + convert_list(i), tree=tree, q_arr=q_arr, q_values=q_values, C=C, mean=mean, cov=cov, N=N)
            v_minus = mc_exp_tree_discretized(X=X, S=chain_l(S), tree=tree, q_arr=q_arr, q_values=q_values, C=C, mean=mean, cov=cov, N=N)

            phi[:, i, 0] += weight * (v_plus - v_minus)

    return phi / m


def single_tree_sv_acv(X, tree, S_star=[], N_star=[], mean=0, cov=0, N=10000):
    va_id = S_star.copy()
    m = len(va_id)

    va_buffer = va_id.copy()
    phi = np.zeros((X.shape[1], tree.values.shape[2]))

    for i in va_id:
        Sm = list(set(va_buffer) - set(convert_list(i)))

        for S in powerset(Sm):
            if len(S) == 0:
                s_acv = []
                weight = comb(m - 1, len(chain_l(S))) ** (-1)
                v_plus = mc_cond_exp(X=X, S=np.array(s_acv + convert_list(i) + N_star).astype(int),
                                        tree=tree, mean=mean, cov=cov, N=N).squeeze()
                v_minus = mc_cond_exp(X=X, S=np.array(s_acv).astype(int), tree=tree,
                                      mean=mean, cov=cov, N=N).squeeze()
            else:
                s_acv = chain_l(S) + N_star

                weight = comb(m - 1, len(chain_l(S))) ** (-1)
                v_plus = mc_cond_exp(X=X, S=np.array(s_acv + convert_list(i)).astype(int), tree=tree,
                                     mean=mean, cov=cov, N=N).squeeze()
                v_minus = mc_cond_exp(X=X, S=np.array(s_acv).astype(int), tree=tree, mean=mean, cov=cov, N=N).squeeze()

            for j in convert_list(i):
                phi[j, :] += weight * (v_plus - v_minus)

    return phi / m


def tree_sv_acv(X, tree, S_star=[], N_star=[], mean=0, cov=0, N=10000):
    phi = np.zeros(shape=(X.shape[0], X.shape[1], tree.values.shape[2]))

    for i in tqdm(range(X.shape[0])):
        phi[i] = single_tree_sv_acv(X[i:i + 1], tree, S_star[i], N_star[i], mean, cov, N)
    return phi


def sdp_true(X, S, tree, mean, cov, N):
    sdp = np.zeros((X.shape[0]))
    for i in range(X.shape[0]):
        sdp[i] = single_sdp_true(X[i], S, tree, mean, cov, N)
    return sdp


def single_sdp_true(x, S, tree, mean, cov, N):
    fx = np.argmax(tree.predict(x), axis=1)
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg
        y_pred = np.argmax(tree.predict(rg_data), axis=1)
        sdp = np.mean(y_pred == fx)
        return sdp
    return 1


def importance_sdp_clf_true(X, tree, mean, cov, N_samples, C=[[]], minimal=1, pi_level=0.9):
    N = X.shape[0]
    m = X.shape[1]

    sdp = np.zeros((N))
    sdp_global = np.zeros((m))
    len_s_star = np.zeros((N), dtype=np.int)

    R, r = [], []
    for i in range(N):
        R.append(i)

    R_buf = np.zeros((N), dtype=np.int)

    va_id = [[i] for i in range(m)]

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

            sdp_b = sdp_true(X, S[:S_size], tree, mean, cov, N_samples)

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
                    for s in range(len_s_star[R_buf[i]], X.shape[1]):  # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.append(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(len(r)):
            R.remove(r[i])

        if len(R) == 0 or S_size >= X.shape[1] / 2:
            break

    return np.asarray(sdp_global) / X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star,
                                                                                          dtype=np.long), np.array(sdp)

def swing_tree_shap_clf_true(X, C,  tree, mean, cov, N_samples, threshold=0.9):
    N = X.shape[0]
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()
    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1]))

    swings = np.zeros((N, X.shape[1], 2))
    swings_prop = np.zeros((N, X.shape[1], 3))

    for i in va_id:
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in tqdm(powerset(Sm)):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = 1.*(sdp_true(X, np.array(chain_l(S) + convert_list(i)).astype(np.long), tree, mean, cov, N_samples) >= threshold)
            v_minus = 1.*(sdp_true(X, np.array(chain_l(S)).astype(np.long), tree, mean, cov, N_samples) >= threshold)

            dif_pos = (v_plus - v_minus) > 0
            dif_neg = (v_plus - v_minus) < 0
            dif_null = (v_plus - v_minus) == 0
            value = ((v_plus - v_minus) * weight) / m

            for a in convert_list(i):
                phi[:, a] += weight * (v_plus - v_minus)

                swings[:, a, 0] += dif_pos * value
                swings[:, a, 1] += dif_neg * value

                swings_prop[:, a, 0] += dif_pos
                swings_prop[:, a, 1] += dif_neg
                swings_prop[:, a, 2] += dif_null

    return phi / m, swings, swings_prop


def swing_tree_shap_clf_true_v(X, yX,  C,  tree, mean, cov, N_samples, threshold=0.9):
    N = X.shape[0]
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()
    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1]))

    swings = np.zeros((N, X.shape[1], 2))
    swings_prop = np.zeros((N, X.shape[1], 3))

    for i in va_id:
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in tqdm(powerset(Sm)):
            weight = comb(m - 1, len(S)) ** (-1)
            v_plus = 1.*(sdp_true_v(X, yX, np.array(chain_l(S) + convert_list(i)).astype(np.long), tree, mean, cov, N_samples) >= threshold)
            v_minus = 1.*(sdp_true_v(X, yX, np.array(chain_l(S)).astype(np.long), tree, mean, cov, N_samples) >= threshold)

            dif_pos = (v_plus - v_minus) > 0
            dif_neg = (v_plus - v_minus) < 0
            dif_null = (v_plus - v_minus) == 0
            value = ((v_plus - v_minus) * weight) / m

            for a in convert_list(i):
                phi[:, a] += weight * (v_plus - v_minus)

                swings[:, a, 0] += dif_pos * value
                swings[:, a, 1] += dif_neg * value

                swings_prop[:, a, 0] += dif_pos
                swings_prop[:, a, 1] += dif_neg
                swings_prop[:, a, 2] += dif_null

    return phi / m, swings, swings_prop


def cond_predict_true(x, yx, S, coefs, mean, cov,  N=10000):
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg

        logit1 = np.sum(coefs[0:2] * rg_data[:, 0:2], axis=1)
        logit2 = np.sum(coefs[2:4] * rg_data[:, 2:4], axis=1)
        idx1 = (rg_data[:, 4] < 0) * 1
        idx2 = (rg_data[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2

        y_pred = logit
        # print(prob_0)
    else:
        # x = np.expand_dims(x, axis=0)
        # logit1 = np.exp(x[:, 0] * x[:, 1])
        # logit2 = np.exp(x[:, 2] * x[:, 3])
        # idx1 = (x[:, 4] < 0) * 1
        # idx2 = (x[:, 4] >= 0) * 1
        # logit = logit1 * idx1 + logit2 * idx2
        #
        # # Compute P(Y=0|X)
        # prob_0 = np.reshape((logit / (1 + logit)), [1, 1])
        #
        # # Sampling process
        # y = np.zeros([N, 2])
        # y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [1, ])
        # y[:, 1] = 1 - y[:, 0]

        y_pred = np.expand_dims(yx, axis=0)

    # return np.mean(y_pred, axis=0)
    return y_pred - yx


def cond_predict_true_clf(x, yx, coefs, S, mean, cov,  N=10000):
    d = x.shape[0]
    index = list(range(d))
    rg_data = np.zeros(shape=(N, d))
    rg_data[:, S] = x[S]

    if len(S) != d:
        S_bar = [i for i in index if i not in S]
        rg = sampleMVN(N, mean, cov, S_bar, S, x[S])
        rg_data[:, S_bar] = rg

        logit1 = np.exp(rg_data[:, 0] * rg_data[:, 1])
        logit2 = np.exp(rg_data[:, 2] * rg_data[:, 3])
        idx1 = (rg_data[:, 4] < 0) * 1
        idx2 = (rg_data[:, 4] >= 0) * 1
        logit = logit1 * idx1 + logit2 * idx2

        # Compute P(Y=0|X)
        prob_0 = np.reshape((logit / (1 + logit)), [N])

        # Sampling process
        # y = np.zeros([N, 2])
        # y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [N, ])
        # y[:, 1] = 1 - y[:, 0]

        y_pred = 1 - prob_0
        # print(prob_0)
    else:
        # x = np.expand_dims(x, axis=0)
        # logit1 = np.exp(x[:, 0] * x[:, 1])
        # logit2 = np.exp(x[:, 2] * x[:, 3])
        # idx1 = (x[:, 4] < 0) * 1
        # idx2 = (x[:, 4] >= 0) * 1
        # logit = logit1 * idx1 + logit2 * idx2
        #
        # # Compute P(Y=0|X)
        # prob_0 = np.reshape((logit / (1 + logit)), [1, 1])
        #
        # # Sampling process
        # y = np.zeros([N, 2])
        # y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [1, ])
        # y[:, 1] = 1 - y[:, 0]

        y_pred = np.expand_dims(yx, axis=0)

    return y_pred-yx
