import acv_explainers
from acv_explainers import ACVTree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import random
from sklearn import datasets

random.seed(2021)
np.random.seed(2021)

data, y = datasets.load_breast_cancer(return_X_y=True)

forest = RandomForestClassifier(n_estimators=5, min_samples_leaf=2, random_state=212, max_depth=8)
forest.fit(data, y)

acvtree = ACVTree(forest, data)

X = data[:100]


def test_sv_acv():
    sv_acv = acvtree.shap_values_acv_nopa(X, np.array(list(range(8))), np.array(list(range(8, data.shape[1]))), [[]], 5)
    sv = acvtree.shap_values(X, [[]])

def test_sv_acv_cyext_nopa():
    cy = acvtree.shap_values_acv_nopa(X, np.array(list(range(8))), np.array(list(range(8, data.shape[1]))), [[]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, data.shape[1])), [[]])
    assert np.allclose(cy, py)


def test_sv_acv_cyext_coalition_nopa():
    cy = acvtree.shap_values_acv_nopa(X, np.array(list(range(8))), np.array(list(range(8, data.shape[1]))),
                                       [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, data.shape[1])), [[0, 1, 2, 3]])
    assert np.allclose(cy, py)

def test_sv_cyext():
    cy = acvtree.shap_values(X, [[]], 5)
    py = acvtree.py_shap_values(X, [[]])
    assert np.allclose(cy, py)

def test_sv_cyext_nopa():
    cy = acvtree.shap_values_nopa(X, [[]], 5)
    py = acvtree.py_shap_values(X, [[]])
    assert np.allclose(cy, py)


def test_sv_cyext_coalition():
    cy = acvtree.shap_values(X, [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values(X, [[0, 1, 2, 3]])
    assert np.allclose(cy, py)

def test_sv_cyext_coalition_nopa():
    cy = acvtree.shap_values_nopa(X, [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values(X, [[0, 1, 2, 3]])
    assert np.allclose(cy, py)


def test_sv_acv_cyext():
    cy = acvtree.shap_values_acv(X, np.array(list(range(8))), np.array(list(range(8, data.shape[1]))), [[]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, data.shape[1])), [[]])
    assert np.allclose(cy, py)


def test_sv_acv_cyext_coalition():
    cy = acvtree.shap_values_acv(X, np.array(list(range(8))), np.array(list(range(8, data.shape[1]))),
                                       [[0, 1, 2, 3]], 5)
    py = acvtree.py_shap_values_acv(X, list(range(8)), list(range(8, data.shape[1])), [[0, 1, 2, 3]])
    assert np.allclose(cy, py)


# def test_sdp_cyext():
#     cy = acvtree.compute_sdp_clf(X, S=np.array([0, 1]), data=data, num_threads=5)
#     cy_cat = acvtree.compute_sdp_clf_cat(X, S=np.array([0, 1]), data=data, num_threads=5)
#
#     py = acvtree.py_compute_sdp_clf(X, S=[0, 1], data=data)
#     py_cat = acvtree.py_compute_sdp_clf_cat(X, S=[0, 1], data=data)
#
#     assert np.allclose(cy, py)
#     assert np.allclose(cy_cat, py_cat)


# def test_exp_cyext():
#     cy = acvtree.compute_exp(X, S=np.array([0, 1]), data=data, num_threads=5)
#     cy_cat = acvtree.compute_exp_cat(X, S=np.array([0, 1]), data=data, num_threads=5)
#
#     py = acvtree.py_compute_exp(X, S=[0, 1], data=data)
#     py_cat = acvtree.py_compute_exp_cat(X, S=[0, 1], data=data)
#
#     assert np.allclose(cy, py)
#     assert np.allclose(cy_cat, py_cat)

# def test_sdp_reg_cyext():
#     xgboost = pytest.importorskip('xgboost')
#     np.random.seed(2021)
#     X, y = shap.datasets.boston()
#     X = X.values
#     model = xgboost.XGBRegressor(n_estimators=3)
#     model.fit(X, y)
#
#     acvtree = ACVTree(model, X)
#
#     x = X[:2]
#     cy = acvtree.compute_sdp_reg(x, 10, S=np.array([0, 1], dtype=np.int), data=X, num_threads=5)
#     cy_cat = acvtree.compute_sdp_reg_cat(x, 10,  S=np.array([0, 1], dtype=np.long), data=X, num_threads=5)
#
#     py = acvtree.py_compute_sdp_reg(X=x, tX=10, S=[0, 1], data=X)
#     py_cat = acvtree.py_compute_sdp_reg_cat(x, 10,  S=[0, 1], data=X)
#
#     assert np.allclose(cy, py)
#     assert np.allclose(cy_cat, py_cat)

# X_swing = X[50:100]
# def test_swing_sv_cyext():
#     cy = acvtree.swing_sv_clf(X_swing, data=data, C=[[]], thresholds=0.8,
#                                     num_threads=5)
#     py = acvtree.py_swing_sv_clf(X=X_swing, data=data, C=[[]], threshold=0.8)
#     assert np.allclose(cy[0], py[0])
#     assert np.allclose(cy[1], py[1])
#     assert np.allclose(cy[2], py[2])


def test_cyext_all_acv():
    x = data[:100]
    pi_level = 0.9
    C = [[]]
    sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X=x, data=data, C=C,
                                                                      pi_level=pi_level)

    s_star_all, n_star_all = acv_explainers.utils.get_null_coalition(sdp_index, size)
    s_star_l, n_star_l = acv_explainers.utils.get_active_null_coalition_list(sdp_index, size)

    sv_all = acvtree.shap_values_acv_adap(x, C=C, N_star=n_star_all,
                                          S_star=s_star_all, size=size)
    sv = []
    i = 0
    for s, n in zip(s_star_l, n_star_l):
        if len(s_star_l) != X.shape[1]:
            sv.append(acvtree.shap_values_acv(np.expand_dims(x[i], 0), C=C, N_star=np.array(n).astype(np.long),
                                              S_star=np.array(s).astype(np.long)))
        else:
            sv.append(acvtree.shap_values(np.expand_dims(x[i], 0), C=C))
        i += 1
    sv = np.concatenate(sv, axis=0)
    assert np.allclose(sv, sv_all)

# def test_cyext_sdp_para():
#     x = X[:100]
#     pi_level = 0.9
#     C = [[]]
#     sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X=x, data=data, C=[[]],
#                                                                             pi_level=pi_level, num_threads=5)
#
#     sdp_importance_p, sdp_index_p, size_p, sdp_p = acvtree.importance_sdp_clf_p2(X=x, data=data, C=[[]],
#                                                                       pi_level=pi_level, num_threads=5)
#
#     assert np.allclose(sdp_importance, sdp_importance_p)
#     assert np.allclose(sdp_index_p, sdp_index)
#     assert np.allclose(size, size_p)
#     assert np.allclose(sdp, sdp_p)