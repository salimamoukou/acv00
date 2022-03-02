from acv_explainers import ACXplainer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import datasets
import random

random.seed(2021)
np.random.seed(2021)
data, y = datasets.load_breast_cancer(return_X_y=True)


x_test = data[:100]
y_test = y[:100]
x = x_test[:1]
y_x = y_test[:1]
ac_xplainer = ACXplainer(classifier=True, n_estimators=1)


def test_acx_fit():
    ac_xplainer.fit(data, y)
    y_pred = ac_xplainer.predict(x_test)
    y_prob = ac_xplainer.predict_proba(x_test)


def test_acx_sdp_rule():
    ac_xplainer.fit(data, y)
    sdp = ac_xplainer.compute_sdp_rf(x, y_x, data, y, S=[[0, 1, 9]])
    rule = ac_xplainer.compute_sdp_rule(x, y_x, data, y, S=[[0, 1, 9]])


def test_acx_importance_sdp_rf():
    sdp, rules, sdp_all, rules_data, w = ac_xplainer.compute_sdp_maxrules(x, y_x, data, y, S=[[0, 1, 9]])

def test_acx_global_rules():
    ac_xplainer.fit(data, y)
    sdp, rules, sdp_all, rules_data, w = ac_xplainer.compute_sdp_maxrules(x_test, y_test, data, y, S=[[0, 1, 9] for i in range(x_test.shape[0])])
    ac_xplainer.fit_global_rules(x, y_x, rules, rules_s_star=[[0, 1, 9] for i in range(rules.shape[0])])
    ac_xplainer.predict_global_rules(x)