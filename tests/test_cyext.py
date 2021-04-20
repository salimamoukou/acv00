from acv_explainers import ACVTree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import random
import pytest
import sklearn
import sklearn.pipeline
import shap
import time
import pstats, cProfile

random.seed(2021)

np.random.seed(2021)
data_frame = pd.read_csv('/home/samoukou/Documents/ACV/data/lucas0_train.csv')

y = data_frame.Lung_cancer.values
data_frame.drop(['Lung_cancer'], axis=1, inplace=True)

forest = RandomForestClassifier(n_estimators=5, min_samples_leaf=2, random_state=212, max_depth=8)
forest.fit(data_frame, y)
acvtree = ACVTree(forest, data_frame.values)

X = np.array(data_frame.values, dtype=np.float)[:100]
data = np.array(data_frame.values, dtype=np.float)


def test_sv_cyext():
    cy = acvtree.shap_values(X, [[]], 5)
    py = acvtree.py_shap_values(X, [[]])
    assert np.allclose(cy, py)


def test_sv_cyext_coalition():
    cy = acvtree.shap_values(X, [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values(X, [[0, 1, 2, 3]])
    assert np.allclose(cy, py)


def test_sv_acv_cyext():
    cy = acvtree.shap_values_acv(X, list(range(8)), list(range(8, 11)), [[]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, 11)), [[]])
    assert np.allclose(cy, py)


def test_sv_acv_cyext_coalition():
    cy = acvtree.shap_values_acv(X, list(range(8)), list(range(8, 11)),
                                       [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, 11)), [[0, 1, 2, 3]])
    assert np.allclose(cy, py)


def test_sdp_cyext():
    cy = acvtree.compute_sdp_clf(X, S=np.array([0, 1]), data=data, num_threads=5)
    cy_cat = acvtree.compute_sdp_clf_cat(X, S=np.array([0, 1]), data=data, num_threads=5)

    py = acvtree.py_compute_sdp_clf(X, S=[0, 1], data=data)
    py_cat = acvtree.py_compute_sdp_clf_cat(X, S=[0, 1], data=data)

    assert np.allclose(cy, py)
    assert np.allclose(cy_cat, py_cat)


def test_exp_cyext():
    cy = acvtree.compute_exp(X, S=np.array([0, 1]), data=data, num_threads=5)
    cy_cat = acvtree.compute_exp_cat(X, S=np.array([0, 1]), data=data, num_threads=5)

    py = acvtree.py_compute_exp(X, S=[0, 1], data=data)
    py_cat = acvtree.py_compute_exp_cat(X, S=[0, 1], data=data)

    assert np.allclose(cy, py)
    assert np.allclose(cy_cat, py_cat)

def test_sdp_reg_cyext():
    xgboost = pytest.importorskip('xgboost')
    np.random.seed(2021)
    X, y = shap.datasets.boston()
    X = X.values
    model = xgboost.XGBRegressor(n_estimators=3)
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:2]
    cy = acvtree.compute_sdp_reg(x, 10, S=np.array([0, 1], dtype=np.int), data=X, num_threads=5)
    cy_cat = acvtree.compute_sdp_reg_cat(x, 10,  S=np.array([0, 1], dtype=np.long), data=X, num_threads=5)

    py = acvtree.py_compute_sdp_reg(X=x, tX=10, S=[0, 1], data=X)
    py_cat = acvtree.py_compute_sdp_reg_cat(x, 10,  S=[0, 1], data=X)

    assert np.allclose(cy, py)
    assert np.allclose(cy_cat, py_cat)

# X_swing = X[50:100]
# def test_swing_sv_cyext():
#     cy = acvtree.cyext_swing_sv_clf(X_swing, data=data, C=[[]], thresholds=0.8,
#                                     num_threads=5)
#     py = acvtree.swing_sv_clf(X=X_swing, data=data, C=[[]], threshold=0.8)
#     assert np.allclose(cy[0], py[0])
#     assert np.allclose(cy[1], py[1])
#     assert np.allclose(cy[2], py[2])


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


def test_cyext_all_acv():
    x = X[:100]
    global_proba = 0.9
    C = [[]]
    sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X=x, data=data, C=[[]],
                                                                            global_proba=global_proba, num_threads=5)

    s_star_all, n_star_all = get_null_coalition(sdp_index, size)
    s_star_l, n_star_l = get_active_null_coalition_list(sdp_index, size)

    sv_all = acvtree.shap_values_acv_adap(x, C=C, N_star=n_star_all,
                                                   S_star=s_star_all, size=size)
    sv = []
    i = 0
    for s, n in zip(s_star_l, n_star_l):
        sv.append(acvtree.shap_values_acv(np.expand_dims(x[i], 0), C=C, N_star=n,
                                                S_star=s))
        i += 1
    sv = np.concatenate(sv, axis=0)
    np.allclose(sv, sv_all)
