from .base_agnostree import *
from .py_acv import *
from .utils_sdp import *
from .utils import *
import numpy as np
import cyext_acv, cyext_acv_nopa, cyext_acv_cache
from acv_explainers.utils import extend_partition
from skranger.ensemble import RangerForestClassifier, RangerForestRegressor
from sklearn.utils.validation import check_is_fitted, check_X_y, column_or_1d, check_array, as_float_array, \
    check_consistent_length
from sklearn.exceptions import NotFittedError
import random

class ACXplainer:
    def __init__(
            self,
            classifier,
            n_estimators=50,
            verbose=False,
            mtry=0,
            importance="impurity",
            min_node_size=0,
            max_depth=5,
            replace=True,
            sample_fraction=None,
            keep_inbag=False,
            inbag=None,
            split_rule="gini",
            num_random_splits=1,
            seed=2021,
            quantiles=None
    ):
        """
        ACXplainer is a agnostic explainer that computes two different explanations for any model or data:

                -  Same Decision Probability and Sufficient Explanations
                -  Minimal Sufficient Rules
        It can also fit a Global Explainable Rules which is an interpretable model by design.

        Args:
            classifier (bool): If True, the ACXplainer is in classification mode, otherwise regression

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
        self.classifier = int(classifier)
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
        self.rules = None
        self.rules_output = None
        self.rules_s_star = None
        self.rules_coverage = None
        self.rules_acc = None
        self.rules_var = None
        self.d = None
        self.check_is_globalrule = False
        self.rules_output_proba = None
        self.rules_ori = None
        self.rules_s_star_ori = None

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
                                                seed=self.seed,
                                                enable_tree_details=True)
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
                                               seed=self.seed,
                                               enable_tree_details=True,
                                               quantiles=True)

    def fit(self, X, y, sample_weight=None, split_select_weights=None, always_split_features=None,
            categorical_features=None):
        """
        Fit the Random forest estimator that will be used for the different estimation using the trainind data for the explanations

        Args:
            X (numpy.ndarray): training features (# samples X # features)

            y (numpy.ndarray): training targets (# samples)

            sample_weight (numpy.ndarray): Optional weights for input samples

            split_select_weights (list): Optional Vector of weights between 0 and 1 of probabilities to select features
                                         for splitting. Can be a single vector or a vector of vectors with one vector per tree.

            always_split_features (list): Features which should always be selected for splitting. A list of column
                                          index values.

            categorical_features (list): A list of column index values which should be considered categorical, or unordered.
        """
        X, y = check_X_y(X, y, dtype=np.double)
        self.d = X.shape[1]
        self.model.fit(X=X, y=y, sample_weight=sample_weight, split_select_weights=split_select_weights,
                       always_split_features=always_split_features, categorical_features=categorical_features)

    def fit_global_rules(self, X, y, rules, rules_s_star):
        # TODO: think about removing s_star ? add complexity ?
        X, y = check_X_y(X, y, dtype=np.double)
        y = as_float_array(y)
        self.rules_ori = rules
        self.rules_s_star_ori = rules_s_star
        self.rules, self.rules_s_star = unique_rules_s_star(rules, rules_s_star)
        if self.classifier:
            self.rules_coverage, self.rules_acc, self.rules_var, self.rules_output, self.rules_output_proba = \
                compute_rules_metrics(
                    self.rules,
                    X,
                    y,
                    self.rules_s_star,
                    self.classifier)
        else:
            self.rules_coverage, self.rules_acc, self.rules_var, self.rules_output = compute_rules_metrics(self.rules,
                                                                                                           X,
                                                                                                           y,
                                                                                                           self.rules_s_star,
                                                                                                           self.classifier)
        self.check_is_globalrule = True

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

    def predict_global_rules(self, X, coverage=0.0, min_acc=0.8, min_reg=1000):
        if not self.check_is_globalrule:
            raise NotFittedError("The global Explainable Rules is not fitted yet. Call 'fit_global_rules' "
                                 "with appropriate arguments")
        X = check_array(X, dtype=np.double)
        if self.classifier:
            return global_rules_model(X, self.rules, self.rules_output, self.rules_coverage,
                                      self.rules_acc, coverage, self.rules_s_star, min_acc)
        else:
            return global_rules_model_reg(X, self.rules, self.rules_output, self.rules_coverage,
                                          self.rules_acc, coverage, self.rules_s_star, min_reg)

    def predict_proba_global_rules(self, X, coverage=0.0, min_acc=0.8):
        if not self.check_is_globalrule:
            raise NotFittedError("The global Explainable Rules is not fitted yet. Call 'fit_global_rules' "
                                 "with appropriate arguments")
        if not self.classifier:
            raise ValueError('Predict_proba function is only available for classification')

        X = check_array(X, dtype=np.double)
        return global_rules_model(X, self.rules, self.rules_output_proba, self.rules_coverage,
                                  self.rules_acc, coverage, self.rules_s_star, min_acc)

    def check_is_explainer(self):
        check_is_fitted(self.model,
                        msg="This ACXplainer instance is not fitted yet. Call 'fit' with appropriate arguments"
                            " before using this estimator")
        if not self.check_is_explain:
            self.ACXplainer = BaseAgnosTree(self.model, self.d)
            self.check_is_explain = True

    def compute_sdp_rf(self, X, y, data, y_data, S, min_node_size=5, t=20):
        """
         Estimate the Same Decision Probability (SDP) of a set of samples X given subset S using the consistent estimator
         (Projected Forest + Quantile Regression), see Paper [ref]

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            S (list[list[int]]): A list that contains the column indices of the variable on which we condition to compute the SDP
                                 for each observation

            min_node_size (int): The minimal node size of the Projected Random Forest

            t (float):  The level of variations around the prediction that defines the SDP in regression (only for regression)

        Returns:
            sdp (numpy.ndarray):  A 1-D matrix (# samples), sdp[i] contains the Same Decision Probability (SDP)
                                 of the Minimal Sufficient Explanation of observation i
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        try:
            check_consistent_length(X, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp = cyext_acv.compute_sdp_rf(X, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       self.classifier, t)
        return sdp

    def compute_sdp_rule(self, X, y, data, y_data, S, min_node_size=5, t=20):
        """
         Estimate the local rule-based explanations of a set of samples X given Sufficient Explanations S using
         the consistent estimator (Projected Forest + Quantile Regression), see Paper [ref]. For each observation x, all
         observations that falls in the rule of x has the same SDP as x.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            S (list[list[int]]): A list that contains the indices of the columns of the Sufficient Explanation for each observation

            min_node_size (int): The minimal node size of the Projected Random forest

            t (float): The level of variations around the prediction that defines the SDP in regression (only for regression)

        Returns:
            sdp (numpy.ndarray): A 1-D matrix (# samples), sdp[i] contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
            rules (numpy.ndarray): A matrix (# samples x # features x 2), rules[i] that contains the local rule of observation i
        """
        X, y = check_X_y(X, y, dtype=[np.double, np.double])
        data, y_data = check_X_y(data, y_data, dtype=[np.double, np.double])
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        try:
            check_consistent_length(X, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp, rules = cyext_acv.compute_sdp_rule(X, y, data, y_data, S, self.ACXplainer.features,
                                                self.ACXplainer.thresholds,
                                                self.ACXplainer.children_left,
                                                self.ACXplainer.children_right, self.ACXplainer.max_depth,
                                                min_node_size, self.classifier, t)
        return sdp, rules

    def compute_sdp_maxrules(self, X, y, data, y_data, S, min_node_size=5, t=20, pi_level=0.95,
                             verbose=False, algo2=False, rdm_size=20):
        """
         Estimate the maximal rule-based explanations of a set of samples X given Sufficient Explanations S using the consistent estimator
         (Projected Forest + Quantile Regression), see Paper [ref]. For each instance x, all
         observations that falls in the rule of x has its SDP given S above \pi as x.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            S (list[list[int]]): A list that contains the indices of the columns of the variable of the Sufficient
                                Explanations for each observation

            min_node_size (int): The minimal node size of the Projected Forest

            t (double): The level of variations around the prediction that defines the SDP in regression (only for regression)

            pi_level (double): The minimal level of SDP for the Sufficient Explanations

        Returns:
            sdp (numpy.ndarray): A 1-D matrix (# samples), sdp[i] contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
            rules (numpy.ndarray): A matrix (# samples x # features x 2), rules[i] that contains the maximal local rule
                                   of observation i
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        try:
            check_consistent_length(X, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        if not verbose:
            sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule(X, y, data, y_data, S,
                                                                               self.ACXplainer.features,
                                                                               self.ACXplainer.thresholds,
                                                                               self.ACXplainer.children_left,
                                                                               self.ACXplainer.children_right,
                                                                               self.ACXplainer.max_depth,
                                                                               min_node_size, self.classifier, t,
                                                                               pi_level)
        else:
            sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule_verbose(X, y, data, y_data, S,
                                                                                       self.ACXplainer.features,
                                                                                       self.ACXplainer.thresholds,
                                                                                       self.ACXplainer.children_left,
                                                                                       self.ACXplainer.children_right,
                                                                                       self.ACXplainer.max_depth,
                                                                                       min_node_size, self.classifier,
                                                                                       t, pi_level)

        if not algo2:
            extend_partition(rules, rules_data, sdp_all, pi=pi_level, S=S)
        else:
            rules_rdm = [rules.copy() for z in range(rdm_size)]
            for i in tqdm(range(rdm_size)):
                for a, ru in enumerate(rules_rdm[i]):
                    rules_comp = [rules_data[a, j] for j in range(rules_data.shape[1]) if sdp_all[a, j] >= sdp[a]]
                    random.shuffle(rules_comp)
                    r, _ = return_best(rules_rdm[i][a], rules_comp, S[a])
                    rules_rdm[i][a][S[a]] = r

            return sdp, rules_rdm, sdp_all, rules_data, w

        return sdp, rules, sdp_all, rules_data, w

    def importance_sdp_rf(self, X, y, data, y_data, min_node_size=5, t=20,
                          C=[[]], pi_level=0.9, minimal=1, stop=True):
        """
         Estimate the Minimal Sufficient Explanations and the Global sdp importance of a set of samples X using the
         consistent estimator (Projected Forest + Quantile Regression), see Paper [ref].

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            min_node_size (int): The minimal node size

            t (float): The level of variations around the prediction that defines the SDP in regression (only for regression)

            C (list[list[int]]): A list that contains a list of the indices of column for each grouped variables

            pi_level (float): The minimal value of the Same Decision Probability (SDP) of the Sufficient Explanations.
                              It should be in (0, 1).

            minimal (int): It will search the Sufficient Explanations from subsets of size "minimal" instead of 1 by default

            stop (bool): If stop=True, it will stop searching for the Sufficient Explanations if it does not find
                         any Sufficient Explanations smaller than (# features / 2), otherwise it will continues until
                         end.

        Returns:
            global_sdp_importance (numpy.ndarray): A 1-D matrix (# features) that is the global explanatory importance
                                                 based on samples X. For a given i, sdp_importance[i] corresponds to the
                                                 frequency of apparition of feature i in the Minimal Sufficient Explanations
                                                 of the set of samples X

            sdp_index (numpy.ndarray): A matrix (# samples X # features) that contains the indices of the variables in the
                                     the Minimal Sufficient Explanations for each sample. For a given i, the positive
                                     values of sdp_index[i] corresponds to the Minimal Sufficient Explanations of
                                     observation i.

            size (numpy.ndarray): A 1-D matrix (# samples) that contains the size of the Minimal Sufficient Explanation
                                for each sample.

            sdp (numpy.ndarray): A 1-D matrix (# samples) that contains the Same Decision Probability (SDP)
                               of the Sufficient Explanation for each sample.
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        self.check_is_explainer()

        if X.shape[1] > 10:
            feature_importance = -np.array(self.model.feature_importances_)
            search_space = list(np.argsort(feature_importance))
            # flat_list = [item for t in self.ACXplainer.node_idx_trees for sublist in t for item in sublist]
            # node_idx = pd.Series(flat_list)
            # search_space = []
            # for v in (node_idx.value_counts().keys()):
            #     search_space += [v]
        else:
            search_space = [i for i in range(X.shape[1])]

        return cyext_acv.global_sdp_rf(X, y, data, y_data, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       self.classifier, t, C,
                                       pi_level, minimal, stop, search_space[:10])

    def sufficient_expl_rf(self, X, y, data, y_data, min_node_size=5, t=20,
                           C=[[]], pi_level=0.9, minimal=1, stop=True):

        """
         Estimate all the Sufficient Explanations and the Global sdp importance of a set of samples X using the
         consistent estimator (Projected Forest + Quantile Regression), see Paper [ref].

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            min_node_size (int): The minimal node size

            t (float): The level of variations around the prediction that defined the SDP for regression (only for regression)

            C (list[list[int]]): A list that contains a list of the indices of the column for each grouped variables

            pi_level (float): The minimal value of the Same Decision Probability (SDP) of the Sufficient Explanations.
                              It should be in (0, 1).

            minimal (int): It will search the Sufficient Explanations from subsets of size "minimal" instead of 1 by default

            stop (bool): If stop=True, it will stop searching for the Sufficient Explanations if it does not find
                         any Sufficient Explanations smaller than (# features / 2), otherwise it will continues until
                         end.

        Returns:
            sufficient_coal (list[list[list[int]]]): a list that contains the column indices of the Sufficient Explanations
                                                     of each sample
            sdp_coal (list[list[list[int]]]): a list that contains the SDP of the Sufficient Explanations of each sample

            sdp_global (numpy.ndarray): 1-D array (# features), sdp_global[i] corresponds to the frequency of apparition
            of variable i in the **all** the sufficient explanations over the samples X.
        """

        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        self.check_is_explainer()
        if X.shape[1] > 10:

            feature_importance = -np.array(self.model.feature_importances_)
            search_space = list(np.argsort(feature_importance))
            # flat_list = [item for t in self.ACXplainer.node_idx_trees for sublist in t for item in sublist]
            # node_idx = pd.Series(flat_list)
            # search_space = []
            # for v in (node_idx.value_counts().keys()):
            #     search_space += [v]
        else:
            search_space = [i for i in range(X.shape[1])]
        # TODO: Remove the [-1] in the Sufficent Coal
        return cyext_acv.sufficient_expl_rf(X, y, data, y_data, self.ACXplainer.features, self.ACXplainer.thresholds,
                                            self.ACXplainer.children_left,
                                            self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                            self.classifier, t, C,
                                            pi_level, minimal, stop, search_space[:10])

    def compute_exp_rf(self, X, y, data, y_data, S, min_node_size=5, t=20):
        """
         Estimate the Conditional Expectation (exp) of a set of samples X given subset S using the consistent estimator
         (Projected Forest), see Paper [ref]

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            S (list[list[int]]): A list that contains the indices of the variable on which we condition to compute the SDP
                                 for each observation

            min_node_size (int): The minimal node size

            t (float): not used here

        Returns:
            exp (numpy.ndarray): A 1-D matrix (# samples), exp[i] contains the Conditional Expectation of observation i
                                 given S
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        try:
            check_consistent_length(X, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        exp = cyext_acv.compute_exp_rf(X, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       self.classifier, t)
        return exp

    def compute_cdf_rf(self, X, y, data, y_data, S, min_node_size=5, t=20):
        """
         Estimate the Projected Cumulative Distribution Function (P-CDF) of a set of samples X given subset S using the
          consistent estimator (Projected Forest + Quantile Regression), see Paper [ref]

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            y (numpy.ndarray): 1-D array (# samples) the targets of X

            data (numpy.ndarray): The background dataset that is used for the estimation of the SDP. It should be the
                                training samples.

            y_data (numpy.ndarray): The targets of the background dataset

            S (list[list[int]]): A list that contains the indices of the variable on which we condition to compute the P-CDF
                                 for each observation

            min_node_size (int): The minimal node size of the Projected Forest

            t (float): not used here

        Returns:
            exp (numpy.ndarray): A 1-D matrix (# samples), exp[i] contains the Conditional Expectation of observation i
                                 given S
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        try:
            check_consistent_length(X, S)
        except ValueError as exp:
            raise ValueError('{} for X (samples) and S (coalition)'.format(exp))

        self.check_is_explainer()

        sdp = cyext_acv.compute_cdf_rf(X, y, data, y_data, S, self.ACXplainer.features, self.ACXplainer.thresholds,
                                       self.ACXplainer.children_left,
                                       self.ACXplainer.children_right, self.ACXplainer.max_depth, min_node_size,
                                       self.classifier, t)
        return sdp

    @staticmethod
    def compute_local_sdp(d, sufficient_coal):
        """
        Estimate the local explanatory importance of each variable given the set of Sufficient Explanations of each
        sample
        Args:
            d (int): # features
            sufficient_coal (list[list[list[int]]]): a list that contains the column indices of the Sufficient Explanations
                                                     of each sample

        Returns:
            local_exp_imp (numpy.ndarray): 1-D array (# features) the local explanatory importance of each variable for each
                                           sample.
        """
        flat = [item for sublist in sufficient_coal for item in sublist]
        flat = pd.Series(flat)
        flat = dict(flat.value_counts() / len(sufficient_coal))
        local_sdp = np.zeros(d)
        for key in flat.keys():
            local_sdp[key] = flat[key]
        return local_sdp

    @staticmethod
    def compute_msdp_clf(X, S, data, model=None):
        """
        Compute marginal SDP of a set of sample X given S for classifier
        """
        return msdp(X, S, model, data)

    @staticmethod
    def importance_msdp_clf_search(X, data, model=None, C=[[]], minimal=1, pi_level=0.9, r_search_space=None,
                                   stop=True):
        """
        Compute the marginal minimal sufficient explanations of any classifier
        """

        return importance_msdp_clf_search(X=X, rg_data=data, model=model, C=C, minimal=minimal,
                                          pi_level=pi_level, r_search_space=r_search_space, stop=stop)

    @staticmethod
    def compute_msdp_reg(X, S, data, model=None, threshold=0.2):
        """
         Compute marginal SDP of a set of sample X given S for regressor
        """
        return msdp_reg(X, S, model, data, threshold)

    @staticmethod
    def importance_msdp_reg_search(X, data, model=None, C=[[]], minimal=1, pi_level=0.9, threshold=0.2,
                                   r_search_space=None, stop=True):
        """
        Compute the marginal minimal sufficient explanations of any regressor model
        """
        return importance_msdp_reg_search(X, rg_data=data, model=model, C=C, minimal=minimal, pi_level=pi_level,
                                          threshold=threshold, r_search_space=r_search_space, stop=stop)

    # def compute_quantile_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
    #     X, y = check_X_y(X, y, dtype=np.double)
    #     data, y_data = check_X_y(data, y_data, dtype=np.double)
    #     y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
    #     try:
    #         check_consistent_length(X, S)
    #     except ValueError as exp:
    #         raise ValueError('{} for X (samples) and S (coalition)'.format(exp))
    #
    #     self.check_is_explainer()
    #
    #     # TODO: Optimize the loops
    #     y_quantiles = cyext_acv.compute_quantile_rf(X, y, data, y_data, S, self.ACXplainer.features,
    #                                                 self.ACXplainer.thresholds,
    #                                                 self.ACXplainer.children_left,
    #                                                 self.ACXplainer.children_right, self.ACXplainer.max_depth,
    #                                                 min_node_size, classifier, t,
    #                                                 quantile)
    #     return y_quantiles
    #
    # def compute_quantile_diff_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
    #     X, y = check_X_y(X, y, dtype=np.double)
    #     y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
    #     data, y_data = check_X_y(data, y_data, dtype=np.double)
    #     try:
    #         check_consistent_length(X, S)
    #     except ValueError as exp:
    #         raise ValueError('{} for X (samples) and S (coalition)'.format(exp))
    #
    #     # TODO: Optimize the loops
    #     self.check_is_explainer()
    #     y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(X, y, data, y_data, S, self.ACXplainer.features,
    #                                                           self.ACXplainer.thresholds,
    #                                                           self.ACXplainer.children_left,
    #                                                           self.ACXplainer.children_right, self.ACXplainer.max_depth,
    #                                                           min_node_size,
    #                                                           classifier, t, quantile)
    #     return y_quantiles_diff
