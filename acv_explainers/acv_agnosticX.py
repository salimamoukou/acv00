from .base_agnostree import *
from .py_acv import *
from .utils_sdp import *
import numpy as np
import cyext_acv, cyext_acv_nopa, cyext_acv_cache
from acv_explainers.utils import extend_partition
from skranger.ensemble import RangerForestClassifier, RangerForestRegressor
from sklearn.utils.validation import check_is_fitted, check_X_y, column_or_1d, check_array, as_float_array, check_consistent_length


class ACXplainer:
    def __init__(
            self,
            classifier=True,
            n_estimators=100,
            verbose=False,
            mtry=0,
            importance="none",
            min_node_size=0,
            max_depth=0,
            replace=True,
            sample_fraction=None,
            keep_inbag=False,
            inbag=None,
            split_rule="gini",
            num_random_splits=1,
            seed=2021
    ):
        """
            ACXplainer is a agnostic explainer that computes two different explanations for any model or data:

                    -  Same Decision Probability and Sufficient Explanations
                    -  Minimal Sufficient Rules
        Args:
            classifier (bool): If True, the ACXplainer is on classification mode, otherwise regression mode

            n_estimators (int): Number of trees used in the forest

            verbose (bool): Enable ranger's verbose logging

            mtry (int/callable): The number of features to split on each node. When a
                        callable is passed, the function must accept a single parameter which is the
                        number of features passed, and return some value between 1 and the number of
                        features.

            importance (string):  One of one of ``none``, ``impurity``, ``impurity_corrected``, ``permutation``.

            min_node_size (int): The minimal node size.

            max_depth (int): The maximal tree depth; 0 means unlimited.

            replace (bool): Sample with replacement.

            sample_fraction (float): The fraction of observations to sample. The default is 1 when sampling with
                                     replacement, and 0.632 otherwise. This can be a list of class specific values.
            keep_inbag (bool): If true, save how often observations are in-bag in each tree.

            inbag (list): A list of size ``n_estimators``, containing inbag counts for each
                          observation. Can be used for stratified sampling.

            split_rule (string): One of 'gini', 'extratrees', 'hellinger'; default 'gini' for classification and
                                one of 'variance', 'extratrees', 'maxstat', 'beta' for regression; default 'variance'

            num_random_splits (int): The number of random splits to consider for the ``extratrees`` splitrule.

            seed (int): Random seed value
        """
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.mtry = mtry
        self.importance = importance
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.replace = replace
        self.sample_fraction = sample_fraction
        self.keep_inbag = keep_inbag
        self.inbag = inbag
        self.split_rule = split_rule
        self.num_random_splits = num_random_splits
        self.check_is_explain = False
        self.ACXplainer = None
        self.seed = seed

        if self.classifier:
            self.model = RangerForestClassifier(self.n_estimators,
                                                self.verbose,
                                                self.mtry,
                                                self.importance,
                                                self.min_node_size,
                                                self.max_depth,
                                                self.replace,
                                                self.sample_fraction,
                                                self.keep_inbag,
                                                self.inbag,
                                                self.split_rule,
                                                self.num_random_splits,
                                                seed=self.seed)
        else:
            if self.split_rule == 'gini':
                self.split_rule = 'variance'
            self.model = RangerForestRegressor(self.n_estimators,
                                               self.verbose,
                                               self.mtry,
                                               self.importance,
                                               self.min_node_size,
                                               self.max_depth,
                                               self.replace,
                                               self.sample_fraction,
                                               self.keep_inbag,
                                               self.inbag,
                                               self.split_rule,
                                               self.num_random_splits,
                                               seed=self.seed)

    def fit(self, X, y,  sample_weight=None, split_select_weights=None, always_split_features=None,
            categorical_features=None):
        """
        Fit the random forest using the trainind data for the explanations
        Args:
            X (np.array[2]): training features (# samples X # features)

            y (np.array[1]): training targets (# samples)

            sample_weight (np.array[1]): Optional weights for input samples

            split_select_weights (list): Optional Vector of weights between 0 and 1 of probabilities to select features
                                         for splitting. Can be a single vector or a vector of vectors with one vector per tree.

            always_split_features (list): Features which should always be selected for splitting. A list of column
                                          index values.

            categorical_features (list): A list of column index values which should be considered categorical, or unordered.
        """
        X, y = check_X_y(X, y, dtype=np.double)
        self.d = X.shape[1]
        self.model.fit(X=X, y=y,  sample_weight=sample_weight, split_select_weights=split_select_weights,
                       always_split_features=always_split_features, categorical_features=categorical_features)

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self.model,
                        msg="This ACXplainer instance is not fitted yet. Call 'fit' with appropriate arguments"
                            " before using this estimator")
        return self.model.predict(X)

    def predict_proba(self, X):
        X = check_array(X)
        check_is_fitted(self.model,
                        msg="This ACXplainer instance is not fitted yet. Call 'fit' with appropriate arguments"
                            " before using this estimator")
        if self.classifier:
            return self.model.predict_proba(X)

    def check_is_explainer(self):
        check_is_fitted(self.model,
                        msg="This ACXplainer instance is not fitted yet. Call 'fit' with appropriate arguments"
                            " before using this estimator")
        if not self.check_is_explain:
            self.ACXplainer = BaseAgnosTree(self.model, self.d)
            self.check_is_explain = True

    def compute_sdp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp = cyext_acv.compute_sdp_rf(x, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       classifier, t)
        return sdp

    def compute_sdp_rule(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        x, y = check_X_y(x, y, dtype=[np.double, np.double])
        data, y_data = check_X_y(data, y_data, dtype=[np.double, np.double])
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp, rules = cyext_acv.compute_sdp_rule(x, y, data, y_data, S, self.ACXplainer.features,
                                                self.ACXplainer.thresholds,
                                                self.ACXplainer.children_left,
                                                self.ACXplainer.children_right, self.ACXplainer.max_depth,
                                                min_node_size, classifier, t)
        return sdp, rules

    def compute_sdp_maxrules(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule(x, y, data, y_data, S,
                                                                           self.ACXplainer.features,
                                                                           self.ACXplainer.thresholds,
                                                                           self.ACXplainer.children_left,
                                                                           self.ACXplainer.children_right,
                                                                           self.ACXplainer.max_depth,
                                                                           min_node_size, classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data, w

    def importance_sdp_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                          C=[[]], pi_level=0.9, minimal=1, stop=True):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        self.check_is_explainer()

        if x.shape[1] > 10:
            flat_list = [item for t in self.ACXplainer.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            search_space = []
            for v in (node_idx.value_counts().keys()):
                search_space += [v]
        else:
            search_space = [i for i in range(x.shape[1])]

        sdp = cyext_acv.global_sdp_rf(x, y, data, y_data, self.ACXplainer.features, self.ACXplainer.thresholds,
                                      self.ACXplainer.children_left,
                                      self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                      classifier, t, C,
                                      pi_level, minimal, stop, search_space[:10])
        return sdp

    def sufficient_coal_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                           C=[[]], pi_level=0.9, minimal=1, stop=True):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        self.check_is_explainer()
        if x.shape[1] > 10:

            flat_list = [item for t in self.ACXplainer.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            search_space = []
            for v in (node_idx.value_counts().keys()):
                search_space += [v]
        else:
            search_space = [i for i in range(x.shape[1])]
        # TODO: Remove the unsed value in the Sufficent Coal
        sdp = cyext_acv.sufficient_coal_rf(x, y, data, y_data, self.ACXplainer.features, self.ACXplainer.thresholds,
                                           self.ACXplainer.children_left,
                                           self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                           classifier, t, C,
                                           pi_level, minimal, stop, search_space[:10])
        return sdp

    def compute_exp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        exp = cyext_acv.compute_exp_rf(x, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       classifier, t)
        return exp

    def compute_cdf_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp = cyext_acv.compute_cdf_rf(x, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       classifier, t)
        return sdp

    def compute_quantile_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        x, y = check_X_y(x, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        # TODO: Optimize the loops
        y_quantiles = cyext_acv.compute_quantile_rf(x, y, data, y_data, S, self.ACXplainer.features,
                                                    self.ACXplainer.thresholds,
                                                    self.ACXplainer.children_left,
                                                    self.ACXplainer.children_right, self.ACXplainer.max_depth,
                                                    min_node_size, classifier, t,
                                                    quantile)
        return y_quantiles

    def compute_quantile_diff_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        x, y = check_X_y(x, y, dtype=np.double)
        y, y_data = as_float_array(y), as_float_array(y_data)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        try:
            check_consistent_length(x, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        # TODO: Optimize the loops
        self.check_is_explainer()
        y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(x, y, data, y_data, S, self.ACXplainer.features,
                                                              self.ACXplainer.thresholds,
                                                              self.ACXplainer.children_left,
                                                              self.ACXplainer.children_right, self.ACXplainer.max_depth,
                                                              min_node_size,
                                                              classifier, t, quantile)
        return y_quantiles_diff



    def compute_local_sdp(d, sufficient_coal):
        flat = [item for sublist in sufficient_coal for item in sublist]
        flat = pd.Series(flat)
        flat = dict(flat.value_counts() / len(sufficient_coal))
        local_sdp = np.zeros(d)
        for key in flat.keys():
            local_sdp[key] = flat[key]
        return local_sdp

    @staticmethod
    def compute_msdp_clf(X, S, data, model=None, N=10000):
        """
        Compute marginal SDP
        """
        return msdp(X, S, model, data)

    @staticmethod
    def importance_msdp_clf_search(X, data, model=None, C=[[]], minimal=1, pi_level=0.9, r_search_space=None,
                                   stop=True):
        """
        Compute marginal S^\star of model
        """

        return importance_msdp_clf_search(X=X, rg_data=data, model=model, C=C, minimal=minimal,
                                          pi_level=pi_level, r_search_space=r_search_space, stop=stop)

    @staticmethod
    def compute_msdp_reg(X, S, data, model=None, threshold=0.2):
        """
        Compute marginal SDP of regression model
        """
        return msdp_reg(X, S, model, data, threshold)

    @staticmethod
    def importance_msdp_reg_search(X, data, model=None, C=[[]], minimal=1, pi_level=0.9, threshold=0.2,
                                   r_search_space=None, stop=True):
        """
        Compute marginal S^\star of regression model
        """
        return importance_msdp_reg_search(X, rg_data=data, model=model, C=C, minimal=minimal, pi_level=pi_level,
                                          threshold=threshold, r_search_space=r_search_space, stop=stop)