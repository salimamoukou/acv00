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


def test_run_sufficient_rules():
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from acv_explainers import ACXplainer
    from acv_explainers.utils import get_active_null_coalition_list

    seed = 2022
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    ### Train the model

    X_train = X_train.values[:50]
    y_train = y_train.values[:50]

    acv_xplainer = ACXplainer(classifier=True, n_estimators=18, max_depth=15)
    acv_xplainer.fit(X_train, y_train)

    ### 1- Compute Sufficient Explanations

    sdp_importance, sdp_index, size, sdp = acv_xplainer.importance_sdp_rf(X_train, y_train.astype(np.double),
                                                                          X_train, y_train.astype(np.double),
                                                                          stop=False,
                                                                          pi_level=0.9)

    S_star, N_star = get_active_null_coalition_list(sdp_index, size)

    ### 2- Compute Sufficient Rules

    sdp, rules, sdp_all, rules_data, w = acv_xplainer.compute_sdp_maxrules(X_train, y_train.astype(np.double),
                                                                           X_train, y_train.astype(np.double), S_star,
                                                                           verbose=True)

    ### 3- Compute Global Sufficient Rules (G-SR)

    acv_xplainer.fit_global_rules(X_train, y_train, rules, S_star)

    ### 4- All Sufficient Explanations

    max_size = 5
    sufficient_expl, sdp_expl, sdp_global = acv_xplainer.sufficient_expl_rf(X_train[:max_size], y_train[:max_size],
                                                                             X_train, y_train, pi_level=0.8)

    acv_xplainer.compute_local_sdp(X_train.shape[1], sufficient_expl)