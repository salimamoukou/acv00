from acv_explainers import ACVTree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import random
import numpy as np
import pytest
import sklearn
import sklearn.pipeline
import shap
random.seed(2021)


def test_xgboost_binary():
    xgboost = pytest.importorskip('xgboost')
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)

    models = [
        xgboost.XGBClassifier()
    ]
    for model in models:
        model.fit(X_train.values, Y_train)

        acvtree = ACVTree(model, X_train.values)

        x = X_train.values[:10]
        shap_values = acvtree.shap_values(x, C=[[]])

        odd_means = np.mean(acvtree.predict(X_train.values), axis=0)
        odd_pred = acvtree.predict(x)

        assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), odd_pred - odd_means, atol=1e-5)


def test_lightgbm_binary():
    lightgbm = pytest.importorskip("lightgbm")
    # train lightgbm model
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)
    model = lightgbm.sklearn.LGBMClassifier(max_depth=6)
    model.fit(X_train.values, Y_train)

    acvtree = ACVTree(model, X_train.values)

    x = X_train.values[:10]
    shap_values = acvtree.shap_values(x, C=[[]])

    odd_means = np.mean(acvtree.predict(X_train.values), axis=0)
    odd_pred = acvtree.predict(x)

    assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), odd_pred - odd_means, atol=1e-5)


def test_catboost_binary():
    catboost = pytest.importorskip("catboost")
    max_features = 15
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model = catboost.CatBoostClassifier(iterations=10, learning_rate=0.5, random_seed=12, max_depth=6)
    model.fit(
        X[:, :max_features],
        y,
        verbose=False,
        plot=False
    )

    X = X[:, :max_features]
    acvtree = ACVTree(model, X)

    x = X[:10]
    shap_values = acvtree.shap_values(x, C=[[]])

    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X))

    assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)


def test_xgboost_multiclass():
    xgboost = pytest.importorskip('xgboost')
    np.random.seed(2021)
    X, y = shap.datasets.iris()
    X = X.values
    model = xgboost.XGBClassifier()
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:10]
    shap_values = acvtree.shap_values_nopa(x, C=[[]])

    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X), axis=0)

    # assert np.allclose(np.sum(shap_values, axis=1), y_pred - exp)


def test_xgboost_regressor():
    xgboost = pytest.importorskip('xgboost')
    np.random.seed(2021)
    X, y = shap.datasets.boston()
    X = X.values
    model = xgboost.XGBRegressor()
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:10]
    shap_values = acvtree.shap_values_nopa(x, C=[[]])

    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X))

    assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)
#

def test_catboost_regressor_multiclass():
    catboost = pytest.importorskip("catboost")
    # train catboost model
    # X, y = shap.datasets.boston()
    # X.drop(["RAD"], axis=1,  inplace=True)
    # # X["RAD"] = X["RAD"].astype(np.double)
    # X = X.values
    # model = catboost.CatBoostRegressor(iterations=30, learning_rate=0.1, random_seed=123)
    # p = catboost.Pool(X, y)
    # model.fit(p, verbose=False, plot=False)
    #
    # acvtree = ACVTree(model, X)
    # y_pred = acvtree.predict(X)
    # exp = np.mean(acvtree.predict(X), axis=0)
    #
    # shap_values = acvtree.shap_values(X, C=[[]])
    #
    # assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)

    # explain the model's predictions using SHAP values

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    model = catboost.CatBoostClassifier(iterations=10, learning_rate=0.5, random_seed=12)
    model.fit(
        X,
        y,
        verbose=False,
        plot=False
    )

    acvtree = ACVTree(model, X)
    y_pred = acvtree.predict(X)
    exp = np.mean(acvtree.predict(X), axis=0)

    shap_values = acvtree.shap_values(X, C=[[]])

    assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)


def test_lightgbm_regressor():
    np.random.seed(2021)
    X, y = shap.datasets.boston()
    X = X.values

    lightgbm = pytest.importorskip("lightgbm")
    model = lightgbm.sklearn.LGBMRegressor(n_estimators=10)
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:10]
    shap_values = acvtree.shap_values(x, C=[[]])

    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X))

    assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)
#
#
def test_lightgbm_multiclass():
    lightgbm = pytest.importorskip("lightgbm")

    np.random.seed(2021)
    X, y = shap.datasets.iris()
    X = X.values

    model = lightgbm.sklearn.LGBMClassifier(num_classes=3, objective="multiclass")
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:10]
    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X), axis=0)

    shap_values = acvtree.shap_values(x, C=[[]])

    assert np.allclose(np.sum(shap_values, axis=1), y_pred - exp)
#

def test_sklearn_random_forest_multiclass():
    np.random.seed(2021)
    X, y = shap.datasets.iris()
    X = X.values

    model = sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=5,
                                                    min_samples_split=2,
                                                    random_state=0)
    model.fit(X, y)

    acvtree = ACVTree(model, X)

    x = X[:10]
    y_pred = acvtree.predict(x)
    exp = np.mean(acvtree.predict(X), axis=0)

    shap_values = acvtree.shap_values(x, C=[[]])

    assert np.allclose(np.sum(shap_values, axis=1), y_pred - exp)


def test_sklearn_regressor():
    np.random.seed(2021)
    X, y = shap.datasets.boston()
    X = X.values

    models = [
        sklearn.ensemble.RandomForestRegressor(n_estimators=10, max_depth=5),
        sklearn.ensemble.ExtraTreesRegressor(n_estimators=10, max_depth=5),
    ]
    for model in models:
        model.fit(X, y)

        acvtree = ACVTree(model, X)

        x = X[:10]
        shap_values = acvtree.shap_values(x, C=[[]])

        y_pred = acvtree.predict(x)
        exp = np.mean(acvtree.predict(X))

        assert np.allclose(np.sum(shap_values, axis=1).reshape(-1), y_pred - exp)


def test_sklearn_binary():
    X_train, X_test, Y_train, _ = sklearn.model_selection.train_test_split(*shap.datasets.adult(),
                                                                           test_size=0.2,
                                                                           random_state=0)

    models = [
        sklearn.ensemble.RandomForestClassifier(n_estimators=10, max_depth=5),
        sklearn.ensemble.ExtraTreesClassifier(n_estimators=10, max_depth=5),
    ]
    for model in models:
        model.fit(X_train, Y_train)

        acvtree = ACVTree(model, X_train.values)

        x = X_train.values[:10]
        shap_values = acvtree.shap_values(x, C=[[]])

        y_pred = acvtree.predict(x)
        exp = np.mean(acvtree.predict(X_train.values), axis=0)

        assert np.allclose(np.sum(shap_values, axis=1), y_pred - exp)

