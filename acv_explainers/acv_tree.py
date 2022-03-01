from .base_tree import *
from .py_acv import *
from .utils_exp import *
from .utils_sdp import *
import numpy as np
import cyext_acv, cyext_acv_nopa, cyext_acv_cache
from acv_explainers.utils import extend_partition
from sklearn.utils.validation import check_array, column_or_1d, check_consistent_length

from sklearn.utils.validation import check_is_fitted, check_X_y, column_or_1d, check_array, as_float_array, \
    check_consistent_length
from sklearn.exceptions import NotFittedError

class ACVTree(BaseTree):

    def shap_values(self, X, C=[[]], num_threads=10):
        """
        Estimate the Shapley Values of a set of samples using the Leaf estimator

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to explain the model's output

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            num_threads (int): not used, deprecated

        Returns:
            shapley_values (numpy.ndarray): The Shapley Values of each sample (# samples X # features X # model's output)
        """
        X = check_array(X, dtype=[np.double])
        if not self.cache:
            return cyext_acv.shap_values_leaves_pa(X, self.data, self.values,
                                                   self.partition_leaves_trees,
                                                   self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                   self.node_idx_trees, C, num_threads)
        return self.shap_values_cache(X, C)

    def shap_values_acv_adap(self, X, S_star, N_star, size, C=[[]], num_threads=10):
        """
        Estimate the **Active** Shapley Values for a set of samples given **the corresponding** S_star and N_star of
        **each sample** using the Leaf estimator.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to explain the model's output

            S_star (numpy.ndarray): A matrix (# samples X # features) that contains the indices of the sufficient variables
                                  of each sample. For each row, the first elements should corresponds to the indices of
                                  the Sufficient variables.

            N_star (numpy.ndarray): A matrix (# samples X # features) that contains the indices of the null variables
                                  of each sample. For each row, the first elements should corresponds to the indices of
                                  the null variables.

            size (numpy.ndarray): A 1-D matrix (# samples) that contains the size of the corresponding sufficient variables
                                for each samples

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            num_threads (int): not used, deprecated

        Returns:
            active_shapley_values (numpy.ndarray): The **Active** Shapley Values of each sample (# samples X # features X # model's output)
        """
        X = check_array(X, dtype=[np.double])
        S_star = check_array(S_star, dtype=[np.long])
        N_star = check_array(N_star, dtype=[np.long])
        check_consistent_length(X, S_star)
        size = column_or_1d(size)
        return cyext_acv.shap_values_acv_leaves_adap(X, self.data, self.values,
                                                     self.partition_leaves_trees,
                                                     self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                     self.node_idx_trees, S_star, N_star, size, C, num_threads)

    def shap_values_acv(self, X, S_star, N_star, C=[[]], num_threads=10):
        """
        Estimate the **Active** Shapley Values for a set of samples given a **single** S_star and N_star **for all**
        the observations using the Leaf estimator.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to explain the model's output

            S_star (list): A list that contains the columns indices of the active variables

            N_star (list): A list that contains the columns indices of the null variables

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            num_threads (int): not used, deprecated

        Returns:
            numpy.ndarray: The **Active** Shapley Values of each sample (# samples X # features X # model's output)
        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv.shap_values_acv_leaves(X, self.data, self.values,
                                                self.partition_leaves_trees,
                                                self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                self.node_idx_trees, S_star, N_star, C, num_threads)

    def importance_sdp_clf(self, X, data, C=[[]], pi_level=0.9, minimal=1, stop=True):
        """
        Estimate the Minimal-Sufficient Explanations of a set of samples for tree-based classifier models. It searches
        the Sufficient Explanations in the subspace of the 10-variables frequently selected in the tree-based model to
        reduce the complexity from 2**(features) to 2**(10).

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the Sufficient Explanations

            data (numpy.ndarray): The background dataset to use for the estimation of the explanations. It should be the
                                training samples.

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            pi_level (float): The minimal value of the Same Decision Probability (SDP) of the Sufficient Explanations

            minimal (int): It will search the Sufficient Explanations from subsets of size "minimal" instead of 1 by default

            stop (bool): If stop=True, it will stop searching for the Sufficient Explanations, if it does not find
                         any Sufficient Explanations smaller than (# features / 2), otherwise it will continues until
                         end.

        Returns:
            global_sdp_importance (numpy.ndarray): A 1-D matrix (# features) that is the global explanatory importance
                                                 based on samples X. For a given i, sdp_importance[i] corresponds to the
                                                 frequency of apparition of feature i in the Minimal Sufficient Explanations
                                                 of the set of samples X

            sdp_index (numpy.ndarray): A matrix (# samples X # features) that contains the column indices of the variables
                                       in the the Minimal Sufficient Explanations for each sample. For a given i, the positive
                                       value of sdp_index[i] corresponds to the Minimal Sufficient Explanations of observation i.

            size (numpy.ndarray): A 1-D matrix (# samples), size[i] corresponds to the size of the Minimal Sufficient Explanation
                                  of sample i.

            sdp (numpy.ndarray): A 1-D matrix (# samples) that contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation for each sample.


        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        if X.shape[1] > 10:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_sdp_clf_search(X, data, C, pi_level, minimal, list(order_va[:10]), stop)
        else:
            return self.importance_sdp_clf_greedy(X, data, C, pi_level, minimal, stop=stop)

    def importance_sdp_clf_greedy(self, X, data, C=[[]], pi_level=0.9, minimal=1, stop=True):
        """
        Estimate the Minimal-Sufficient Explanations of a set of samples for tree-based classifier models. It searches
        the Sufficient Explanations in the space of all the variables thus it has a complexity of 2**(# features).

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the Sufficient Explanations

            data (numpy.ndarray): The background dataset to use for the estimation of the explanations. It should be the
                                training samples.

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            pi_level (float): The minimal value of the Same Decision Probability (SDP) of the Sufficient Explanations

            minimal (int): It will search the Sufficient Explanations from subsets of size "minimal" instead of 1 by default

            stop (bool): If stop=True, it will stop searching for the Sufficient Explanations, if it does not find
                         any Sufficient Explanations smaller than (# features / 2), otherwise it will continues until
                         end.

        Returns:
            global_sdp_importance (numpy.ndarray): A 1-D matrix (# features) that is the global explanatory importance
                                                 based on samples X. For a given i, sdp_importance[i] corresponds to the
                                                 frequency of apparition of feature i in the Minimal Sufficient Explanations
                                                 of the set of samples X

            sdp_index (numpy.ndarray): A matrix (# samples X # features) that contains the indices of the variables in the
                                     the Minimal Sufficient Explanations for each sample. For a given i, the positive
                                     value of sdp_index[i] corresponds to the Minimal Sufficient Explanations of
                                     observation i.

            size (numpy.ndarray): A 1-D matrix (# samples), size[i] corresponds to the size of the Minimal Sufficient Explanation
                                  of observation i

            sdp (numpy.ndarray): A 1-D matrix (# samples), sdp[i] that contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier",
                                        "sklearn.ensemble.GradientBoostingClassifier",
                                        "sklearn.ensemble._gb.GradientBoostingClassifier",
                                        "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"
                                        ]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf(X, fX, y_pred, data, self.values_binary,
                                            self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                            self.scalings, C, pi_level, minimal, stop)

        return cyext_acv.global_sdp_clf(X, fX, y_pred, data, self.values,
                                        self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                        self.scalings, C, pi_level, minimal, stop)

    def importance_sdp_clf_search(self, X, data, C=[[]], pi_level=0.9, minimal=1, search_space=[], stop=True):
        """
        Estimate the Minimal-Sufficient Explanations of a set of samples for tree-based classifier models. It searches
        the Sufficient Explanations in the given subspace (search_space). Thus, it has a complexity of 2**(# search_space).

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the Sufficient Explanations

            data (numpy.ndarray): The background dataset to use for the estimation of the explanations. It should be the
                                training samples.

            C (list[list[int]]): A list that contains a list of columns indices for each grouped variables

            pi_level (float): The minimal value of the Same Decision Probability (SDP) of the Sufficient Explanations

            minimal (int): It will search the Sufficient Explanations from subsets of size "minimal" instead of 1 by default

            search_space (list): A list containing the variables (columns indices) on which to search the Sufficient Explanations.

            stop (bool): If stop=True, it will stop searching for the Sufficient Explanations, if it does not find
                         any Sufficient Explanations smaller than (# features / 2), otherwise it will continues until
                         end.

        Returns:
            global_sdp_importance (numpy.ndarray): A 1-D matrix (# features) that is the global explanatory importance
                                                 based on samples X. For a given i, sdp_importance[i] corresponds to the
                                                 frequency of apparition of feature i in the Minimal Sufficient Explanations
                                                 of the set of samples X

            sdp_index (numpy.ndarray): A matrix (# samples X # features) that contains the indices of the variables in the
                                     the Minimal Sufficient Explanations for each sample. For a given i, the positive
                                     value of sdp_index[i] corresponds to the Minimal Sufficient Explanations of
                                     observation i.

            size (numpy.ndarray): A 1-D matrix (# samples), size[i] corresponds to the size of the Minimal Sufficient Explanation
                                  of observation i

            sdp (numpy.ndarray): A 1-D matrix (# samples), sdp[i] that contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier",
                                        "sklearn.ensemble.GradientBoostingClassifier",
                                        "sklearn.ensemble._gb.GradientBoostingClassifier",
                                        "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf_approx(np.array(X, dtype=np.float), fX, y_pred, data, self.values_binary,
                                                   self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                   self.scalings, C, pi_level, minimal, search_space, stop)

        return cyext_acv.global_sdp_clf_approx(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                               self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                               self.scalings, C, pi_level, minimal, search_space, stop)

    def compute_sdp_clf(self, X, S, data, num_threads=10):
        """
        Estimate the Same Decision Probability (SDP) of a set of samples X given subset S using the Leaf estimator.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            S (numpy.ndarray): A 1-D that contains the indices of the variable on which we condition to compute the SDP

            data (numpy.ndarray): The background dataset to use for the estimation of the SDP. It should be the
                                  training samples.

            num_threads (int): not used, deprecated

        Returns:
            sdp (numpy.ndarray)  A 1-D matrix (# samples), sdp[i] that contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        S = column_or_1d(S)
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier",
                                        "sklearn.ensemble.GradientBoostingClassifier",
                                        "sklearn.ensemble._gb.GradientBoostingClassifier",
                                        "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.compute_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values_binary,
                                             self.partition_leaves_trees,
                                             self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

        return cyext_acv.compute_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values,
                                         self.partition_leaves_trees,
                                         self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_clf_cat(self, X, S, data, num_threads=10):
        """
        Estimate the Same Decision Probability (SDP) of a set of samples X given subset S using the Discrete estimator.

        Args:
            X (numpy.ndarray): A matrix of samples (# samples X # features) on which to compute the SDP

            S (numpy.ndarray): A 1-D that contains the indices of the variable on which we condition to compute the SDP

            data (numpy.ndarray): The background dataset to use for the estimation of the SDP. It should be the
                                training samples.

            num_threads (int): not used, deprecated

        Returns:
            sdp (numpy.ndarray)  A 1-D matrix (# samples), sdp[i] that contains the Same Decision Probability (SDP)
                                 of the Sufficient Explanation of observation i
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        S = column_or_1d(S)
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier",
                                        "sklearn.ensemble.GradientBoostingClassifier",
                                        "sklearn.ensemble._gb.GradientBoostingClassifier",
                                        "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.compute_sdp_clf_cat(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values_binary,
                                                 self.partition_leaves_trees,
                                                 self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

        return cyext_acv.compute_sdp_clf_cat(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values,
                                             self.partition_leaves_trees,
                                             self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    # ----------------------------------------------------RELATED FUNCTION---------------------------------------------#

    def shap_values_nopa(self, X, C=[[]], num_threads=10):
        """
                Same as **shap_values** function, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv_nopa.shap_values_leaves_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                      self.partition_leaves_trees,
                                                      self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                      self.node_idx_trees, C, num_threads)

    def shap_values_acv_nopa(self, X, S_star, N_star, C=[[]], num_threads=10):
        """
                Same as **shap_values_acv** function, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv_nopa.shap_values_acv_leaves_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                          self.partition_leaves_trees,
                                                          self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                          self.node_idx_trees, S_star, N_star, C, num_threads)

    def shap_values_acv_adap_nopa(self, X, S_star, N_star, size, C=[[]], num_threads=10):
        """
                Same as **shap_values_acv_adap** function, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        S_star = check_array(S_star, dtype=[np.long])
        N_star = check_array(N_star, dtype=[np.long])
        check_consistent_length(X, S_star)
        size = column_or_1d(size)
        return cyext_acv_nopa.shap_values_acv_leaves_adap_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                               self.partition_leaves_trees,
                                                               self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                               self.node_idx_trees, S_star, N_star, size, C,
                                                               num_threads)

    def shap_values_cache(self, X, C=[[]], num_threads=10):
        """
                Same as **shap_values**, but use cached values to speed up computation

        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv_cache.shap_values_leaves_cache(np.array(X, dtype=np.float), self.data, self.values,
                                                        self.partition_leaves_trees,
                                                        self.leaf_idx_trees, self.leaves_nb, self.lm, self.lm_s,
                                                        self.lm_si,
                                                        self.max_var,
                                                        self.node_idx_trees, C, num_threads)

    def shap_values_cache_nopa(self, X, C=[[]], num_threads=10):
        """
                Same as **shap_values** but use cached values and no parallelism
        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv_nopa.shap_values_leaves_cache_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                            self.partition_leaves_trees,
                                                            self.leaf_idx_trees, self.leaves_nb, self.lm, self.lm_s,
                                                            self.lm_si,
                                                            self.max_var,
                                                            self.node_idx_trees, C, num_threads)

    def importance_sdp_clf_nopa(self, X, data, C=[[]], pi_level=0.9, minimal=0):
        """
                Same as importance_sdp_clf, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.global_sdp_clf_nopa(X, fX, y_pred, data, self.values,
                                                  self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                  self.scalings, C, pi_level, minimal)

    def compute_sdp_clf_nopa(self, X, S, data, num_threads=10):
        """
                Same as compute_sdp_clf, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        data = check_array(data, dtype=[np.double])
        S = column_or_1d(S)
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.compute_sdp_clf_nopa(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values,
                                                   self.partition_leaves_trees,
                                                   self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_clf_cat_nopa(self, X, S, data, num_threads=10):
        """
                Same as compute_sdp_clf_cat, but does not use parallelism
        """
        X = check_array(X, dtype=[np.double])
        S = column_or_1d(S)
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.compute_sdp_clf_cat_nopa(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values,
                                                       self.partition_leaves_trees,
                                                       self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    # ------------------------------------------------PYTHON VERSION---------------------------------------------------#

    def py_shap_values(self, x, C=[[]]):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves(x, self.partition_leaves_trees[i], self.data,
                                      self.node_idx_trees[i],
                                      self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i],
                                      self.values[i], C,
                                      self.num_outputs)
        return out

    def py_shap_valuesv2(self, x, C=[[]]):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves_v2(x, self.partition_leaves_trees[i], self.data,
                                         self.node_idx_trees[i],
                                         self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i],
                                         self.values[i], C,
                                         self.num_outputs)
        return out

    def py_shap_values_acv(self, x, S_star, N_star, C=[[]]):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_acv_leaves(x, self.partition_leaves_trees[i], self.data,
                                          self.node_idx_trees[i],
                                          self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i],
                                          self.values[i], C, S_star,
                                          N_star,
                                          self.num_outputs)
        return out

    def py_compute_sdp_reg(self, X, tX, S, data):
        return compute_sdp_reg(X, tX, self, S, data=data)

    def py_compute_sdp_clf(self, X, S, data, tX=0):
        return compute_sdp_clf(X, tX, self, S, data=data)

    def py_compute_sdp_reg_cat(self, X, tX, S, data):
        return compute_sdp_reg_cat(X, tX, model=self, S=S, data=data)

    def py_compute_sdp_clf_cat(self, X, S, data, tX=0):
        return compute_sdp_clf_cat(X, tX, model=self, S=S, data=data)

    def py_compute_exp(self, X, S, data):
        return compute_exp(X=X, model=self, S=S, data=data)

    def py_compute_exp_cat(self, X, S, data):
        return compute_exp_cat(X=X, model=self, S=S, data=data)

    def py_compute_local_sdp_clf(self, X, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(X, threshold, proba, index, data, final_coal, decay, C, verbose,
                         self.compute_sdp_clf)

    def py_compute_local_sdp_reg(self, X, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(X, threshold, proba, index, data, final_coal, decay, C, verbose, self.compute_sdp_reg)

    def py_swing_values_clf(self, X, S, data, threshold, tX=0):
        return np.array(self.compute_sdp_clf(X=X, tX=tX, S=S, data=data) >= threshold, dtype=float)

    def py_swing_values_reg(self, X, tX, S, data, threshold):
        return np.array(self.compute_sdp_reg(X=X, tX=tX, S=S, data=data) >= threshold, dtype=float)

    def py_swing_sv_clf(self, X, data, threshold, C, tX=0):
        return swing_tree_shap(X, tX, threshold, data, C, self.swing_values_clf)

    def py_swing_sv_reg(self, X, data, threshold, C, tX=0):
        return swing_tree_shap(X, tX, threshold, data, C, self.swing_values_reg)

    def py_global_sdp_importance_clf(self, data, data_bground, columns_names, pi_level, decay, threshold,
                                     proba, C, verbose):

        return global_sdp_importance(data, data_bground, columns_names, pi_level, decay, threshold,
                                     proba, C, verbose, self.compute_sdp_clf)

    def py_global_sdp_importance_reg(self, data, data_bground, columns_names, pi_level, decay, threshold,
                                     proba, C, verbose):
        return global_sdp_importance(data, data_bground, columns_names, pi_level, decay, threshold,
                                     proba, C, verbose, self.compute_sdp_reg)

    def py_shap_values_notoptimized(self, X, data, C=[[]]):
        return shap_values_leaves_notoptimized(X, data, C, self)

    def py_shap_values_discrete_notoptimized(self, X, data, C=[[]]):
        return shap_values_discrete_notoptimized(X, data, C, self)

    # ------------------------------------------------FOR DEVELOPMENT--------------------------------------------------#

    def importance_sdp_clf_ptrees(self, X, data, C=[[]], pi_level=0.9, minimal=0, stop=True):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier",
                                        "sklearn.ensemble.GradientBoostingClassifier",
                                        "sklearn.ensemble._gb.GradientBoostingClassifier",
                                        "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf_ptrees(np.array(X, dtype=np.float), fX, y_pred, data, self.values_binary,
                                                   self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                   self.scalings, C, pi_level, minimal, stop)

        return cyext_acv.global_sdp_clf_ptrees(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                               self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                               self.scalings, C, pi_level, minimal, stop)

    def compute_sdp_reg(self, X, tX, S, data, num_threads=10):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values,
                                         self.partition_leaves_trees,
                                         self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_cat(self, X, tX, S, data, num_threads=10):
        # raise Warning('The current implementation may take a long time if n_trees and depth are large. The number of '
        #               'operation is 2**(depth*n_trees)')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg_cat(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values,
                                             self.partition_leaves_trees,
                                             self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def swing_sv_clf(self, X, data, C=[[]], thresholds=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv.swing_sv_clf_direct(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                             self.partition_leaves_trees,
                                             self.leaf_idx_trees, self.leaves_nb, self.scalings, C, thresholds,
                                             num_threads)

    def swing_sv_clf_nopa(self, X, data, C=[[]], thresholds=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.swing_sv_clf_direct_nopa(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                       self.partition_leaves_trees,
                                                       self.leaf_idx_trees, self.leaves_nb, self.scalings, C,
                                                       thresholds, num_threads)

    def demo_swing_sv_clf(self, X, data, C):
        return swing_tree_shap_clf(X, data, C, self.compute_sdp_clf)

    def demo_swing_sv_clf_cat(self, X, data, C):
        return swing_tree_shap_clf(X, data, C, self.compute_sdp_clf_cat)

    def swing_sv_clf_slow(self, X, data, C=[[]], thresholds=0.9, num_threads=5):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv.swing_sv_clf(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                      self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                      C, thresholds, num_threads)

    def importance_sdp_reg_cat(self, X, tX, data, C=[[]], pi_level=0.9, minimal=0, stop=True):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.global_sdp_reg_cat(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                            self.partition_leaves_trees,
                                            self.leaf_idx_trees, self.leaves_nb, self.scalings, C, pi_level,
                                            minimal, stop)

    def importance_sdp_reg(self, X, tX, data, C=[[]], pi_level=0.9, minimal=0, stop=True):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.global_sdp_reg(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                        self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                        self.scalings, C, pi_level, minimal, stop)

    def importance_sdp_reg_cat_nopa(self, X, tX, data, C=[[]], pi_level=0.9, minimal=0):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.global_sdp_reg_cat_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                                      self.partition_leaves_trees,
                                                      self.leaf_idx_trees, self.leaves_nb, self.scalings, C,
                                                      pi_level, minimal)

    def importance_sdp_reg_nopa(self, X, tX, data, C=[[]], pi_level=0.9, minimal=0):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.global_sdp_reg_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                                  self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                  self.scalings, C, pi_level, minimal)

    def compute_exp(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_exp_cat(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_cat(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                         self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                         num_threads)

    def compute_exp_normalized(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_normalized(np.array(X, dtype=np.float), S, data, self.values,
                                                self.partition_leaves_trees,
                                                self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                                num_threads)

    def compute_exp_normalized_nopa(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_normalized_nopa(np.array(X, dtype=np.float), S, data, self.values,
                                                     self.partition_leaves_trees,
                                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                                     num_threads)

    def shap_values_normalized_cache(self, X, C=[[]], num_threads=10):
        return cyext_acv_cache.shap_values_leaves_normalized_cache(np.array(X, dtype=np.float), self.data, self.values,
                                                                   self.partition_leaves_trees,
                                                                   self.leaf_idx_trees, self.leaves_nb, self.lm_n,
                                                                   self.lm_s_n, self.lm_si_n,
                                                                   self.max_var,
                                                                   self.node_idx_trees, C, num_threads)

    def shap_values_normalized_cache_nopa(self, X, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_leaves_normalized_cache_nopa(np.array(X, dtype=np.float), self.data,
                                                                       self.values,
                                                                       self.partition_leaves_trees,
                                                                       self.leaf_idx_trees, self.leaves_nb, self.lm_n,
                                                                       self.lm_s_n, self.lm_si_n,
                                                                       self.max_var,
                                                                       self.node_idx_trees, C, num_threads)

    def shap_values_normalized(self, X, C=[[]], num_threads=10):
        if not self.cache_normalized:
            return cyext_acv.shap_values_leaves_normalized(np.array(X, dtype=np.float), self.data, self.values,
                                                           self.partition_leaves_trees,
                                                           self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                           self.node_idx_trees, C, num_threads)
        return self.shap_values_normalized_cache(X, C)

    def leaves_cache(self, C=[[]], num_threads=10):
        return cyext_acv_cache.leaves_cache(self.data, self.values, self.partition_leaves_trees,
                                            self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                            self.node_idx_trees, C, num_threads)

    def leaves_cache_normalized(self, C=[[]], num_threads=10):
        return cyext_acv_cache.leaves_cache_normalized(self.data, self.values, self.partition_leaves_trees,
                                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                       self.node_idx_trees, C, num_threads)

    def leaves_cache_nopa(self, C=[[]], num_threads=10):
        return cyext_acv_nopa.leaves_cache_nopa(self.data, self.values, self.partition_leaves_trees,
                                                self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                self.node_idx_trees, C, num_threads)

    def leaves_cache_normalized_nopa(self, C=[[]], num_threads=10):
        return cyext_acv_nopa.leaves_cache_normalized_nopa(self.data, self.values, self.partition_leaves_trees,
                                                           self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                           self.node_idx_trees, C, num_threads)

    def compute_exp_nopa(self, X, S, data, num_threads=10):
        return cyext_acv_nopa.compute_exp_nopa(np.array(X, dtype=np.float), S, data, self.values,
                                               self.partition_leaves_trees,
                                               self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_exp_cat_nopa(self, X, S, data, num_threads=10):
        return cyext_acv_nopa.compute_exp_cat_nopa(np.array(X, dtype=np.float), S, data, self.values,
                                                   self.partition_leaves_trees,
                                                   self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                                   num_threads)

    def compute_sdp_reg_nopa(self, X, tX, S, data, num_threads=10):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.compute_sdp_reg_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values,
                                                   self.partition_leaves_trees,
                                                   self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_cat_nopa(self, X, tX, S, data, num_threads=10):
        # raise Warning('The current implementation may take a long time if n_trees > 10 and depth > 6')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.compute_sdp_reg_cat_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, S, data,
                                                       self.values, self.partition_leaves_trees,
                                                       self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_msdp_clf(self, X, S, data, model=None, N=10000):
        """
        Compute marginal SDP of a set of sample X given S for classifier
        """
        # if data:
        #     return msdp_true(X, S, self.model, self.data)

        if model == None:
            return msdp(X, S, self.model, data)
        return msdp(X, S, model, data)

    def importance_msdp_clf_search(self, X, data, model=None, C=[[]], minimal=1, pi_level=0.9, r_search_space=None,
                                   stop=True):
        """
        Compute the marginal minimal sufficient explanations of any classifier
        """
        # if data:
        #     return importance_msdp_clf_true(X.values, self.model, self.data, C=C,
        #                                          minimal=minimal,
        #                                          pi_level=pi_level)
        if model == None:
            return importance_msdp_clf_search(X=X, rg_data=data, model=self.model, C=C, minimal=minimal,
                                              pi_level=pi_level, r_search_space=r_search_space, stop=stop)
        return importance_msdp_clf_search(X=X, rg_data=data, model=model, C=C, minimal=minimal,
                                          pi_level=pi_level, r_search_space=r_search_space, stop=stop)

    def importance_msdp_clf(self, X, data, model=None, C=[[]], pi_level=0.9, minimal=1, stop=True):
        if X.shape[1] > 15:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_msdp_clf_search(X=X, data=data, model=model, C=C, pi_level=pi_level,
                                                   minimal=minimal,
                                                   r_search_space=order_va[:15], stop=stop)
        else:
            return self.importance_msdp_clf_search(X=X, data=data, model=model, C=C, pi_level=pi_level,
                                                   minimal=minimal, stop=stop)

    def compute_msdp_reg(self, X, S, data, model=None, N=10000, threshold=0.2):
        """
         Compute marginal SDP of a set of sample X given S for regressor
        """
        # if data:
        #     return msdp_true(X, S, self.model, self.data)

        if model == None:
            return msdp_reg(X, S, self.model, data, threshold)
        return msdp_reg(X, S, model, data, threshold)

    def importance_msdp_reg_search(self, X, data, model=None, C=[[]], minimal=1, pi_level=0.9, threshold=0.2,
                                   r_search_space=None, stop=True):
        """
        Compute the marginal minimal sufficient explanations of any regressor model
        """
        # if data:
        #     return importance_msdp_clf_true(X.values, self.model, self.data, C=C,
        #                                          minimal=minimal,
        #                                          pi_level=pi_level)
        if model == None:
            return importance_msdp_reg_search(X, rg_data=data, model=self.model, C=C, minimal=minimal,
                                              pi_level=pi_level, threshold=threshold,
                                              r_search_space=r_search_space, stop=stop)
        return importance_msdp_reg_search(X, rg_data=data, model=model, C=C, minimal=minimal, pi_level=pi_level,
                                          threshold=threshold, r_search_space=r_search_space, stop=stop)

    def importance_msdp_reg(self, X, data, model=None, C=[[]], pi_level=0.9, minimal=1, threshold=0.2, stop=True):
        """
        Compute the marginal minimal sufficient explanations of any regressor model
        """
        if X.shape[1] > 15:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_msdp_reg_search(X=X, data=data, model=model, C=C, pi_level=pi_level,
                                                   minimal=minimal,
                                                   r_search_space=order_va[:15], threshold=threshold, stop=stop)
        else:
            return self.importance_msdp_reg_search(X=X, data=data, model=model, C=C, pi_level=pi_level,
                                                   minimal=minimal, threshold=threshold, stop=stop)

    def compute_exp_shaff(self, X, data, y_data, S, min_node_size=5):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_exp(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size)
        return exp

    def compute_sdp_clf_shaff(self, X, y_X, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp_clf(X=X[i], y_X=y_X[i], S=S, model=self, data=data, Y=y_data,
                                           min_node_size=min_node_size)
        return sdp

    def compute_sdp_shaff(self, X, y_X, t, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp(X=X[i], t=t, y_X=y_X[i], S=S, model=self, data=data, Y=y_data,
                                       min_node_size=min_node_size)
        return sdp

    def compute_quantile_shaff(self, X, data, y_data, S, min_node_size=5, quantile=95):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_quantile(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size,
                                            quantile=quantile)
        return exp

    def compute_sdp_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        """
         Estimate the Same Decision Probability (SDP) of a set of samples X given subset S using the consistent estimator

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

        sdp = cyext_acv.compute_sdp_rf(X, y, data, y_data, S, self.features, self.thresholds,
                                       self.children_left,
                                       self.children_right, self.max_depth, min_node_size,
                                       classifier, t)
        return sdp

    def compute_cdf_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        """
         Estimate the Projected Cumulative Distribution Function (P-CDF) of a set of samples X given subset S using the
          consistent estimator

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


        sdp = cyext_acv.compute_cdf_rf(X, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def importance_sdp_rf(self, X, y, data, y_data, min_node_size=5, classifier=1, t=20,
                          C=[[]], pi_level=0.9, minimal=1, stop=True):

        """
         Estimate the Minimal Sufficient Explanations and the Global sdp importance of a set of samples X using the
         consistent estimator

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

        if X.shape[1] > 10:
            try:
                feature_importance = -np.array(self.model.feature_importances_)
                search_space = list(np.argsort(feature_importance))
            except AttributeError:
                flat_list = [item for t in self.ACXplainer.node_idx_trees for sublist in t for item in sublist]
                node_idx = pd.Series(flat_list)
                search_space = []
                for v in (node_idx.value_counts().keys()):
                    search_space += [v]
        else:
            search_space = [i for i in range(X.shape[1])]

        return cyext_acv.global_sdp_rf(X, y, data, y_data, self.features, self.thresholds,
                                       self.children_left,
                                       self.children_right, self.max_depth, min_node_size,
                                       classifier, t, C,
                                       pi_level, minimal, stop, search_space[:10])

    def sufficient_expl_rf(self, X, y, data, y_data, min_node_size=5, classifier=1, t=20,
                           C=[[]], pi_level=0.9, minimal=1, stop=True):

        """
         Estimate all the Sufficient Explanations and the Global sdp importance of a set of samples X using the
         consistent estimator

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
        if X.shape[1] > 10:

            try:
                feature_importance = -np.array(self.model.feature_importances_)
                search_space = list(np.argsort(feature_importance))
            except AttributeError:
                flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
                node_idx = pd.Series(flat_list)
                search_space = []
                for v in (node_idx.value_counts().keys()):
                    search_space += [v]
        else:
            search_space = [i for i in range(X.shape[1])]
        # TODO: Remove the [-1] in the Sufficent Coal
        return cyext_acv.sufficient_expl_rf(X, y, data, y_data, self.features, self.thresholds,
                                            self.children_left,
                                            self.children_right, self.max_depth, min_node_size,
                                            classifier, t, C,
                                            pi_level, minimal, stop, search_space[:10])

    def compute_exp_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        """
         Estimate the Conditional Expectation (exp) of a set of samples X given subset S using the consistent estimator

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

        exp = cyext_acv.compute_exp_rf(X, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return exp

    def compute_quantile_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles = cyext_acv.compute_quantile_rf(X, y, data, y_data, S, self.features, self.thresholds,
                                                    self.children_left,
                                                    self.children_right, self.max_depth, min_node_size, classifier, t,
                                                    quantile)
        return y_quantiles

    def compute_quantile_diff_rf(self, X, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(X, y, data, y_data, S, self.features, self.thresholds,
                                                              self.children_left,
                                                              self.children_right, self.max_depth, min_node_size,
                                                              classifier, t, quantile)
        return y_quantiles_diff


class ACVTreeAgnostic(BaseTree):

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

    def compute_cdf_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_cdf_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def compute_sdp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_sdp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def compute_sdp_rf_v0(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_sdp_rf_v0(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                          self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def compute_sdp_rule(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp, rules = cyext_acv.compute_sdp_rule(x, y, data, y_data, S, self.features, self.thresholds,
                                                self.children_left,
                                                self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp, rules

    def compute_sdp_rule_v0(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp, rules = cyext_acv.compute_sdp_rule_v0(x, y, data, y_data, S, self.features, self.thresholds,
                                                   self.children_left,
                                                   self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp, rules

    def compute_sdp_rule_biased(self, x, y, data, S, min_node_size=5, classifier=1, t=20):
        sdp, rules = cyext_acv.compute_sdp_rule_biased(x, y, data, S, self.features, self.thresholds,
                                                       self.children_left,
                                                       self.children_right, self.max_depth, min_node_size, classifier,
                                                       t)
        return sdp, rules

    def compute_sdp_maxrules(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule(x, y, data, y_data, S, self.features,
                                                                           self.thresholds, self.children_left,
                                                                           self.children_right, self.max_depth,
                                                                           min_node_size, classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data, w

    def compute_sdp_maxrules_biased(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        sdp, rules, sdp_all, rules_data = cyext_acv.compute_sdp_maxrule_biased(x, y, data, y_data, S, self.features,
                                                                               self.thresholds, self.children_left,
                                                                               self.children_right, self.max_depth,
                                                                               min_node_size, classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data

    def compute_sdp_maxrules_opti(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule_opti(x, y, data, y_data, S, self.features,
                                                                                self.thresholds, self.children_left,
                                                                                self.children_right, self.max_depth,
                                                                                min_node_size, classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data, w

    def compute_sdp_maxrules_v0(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        sdp, rules, sdp_all, rules_data = cyext_acv.compute_sdp_maxrule_v0(x, y, data, y_data, S, self.features,
                                                                           self.thresholds, self.children_left,
                                                                           self.children_right, self.max_depth,
                                                                           min_node_size, classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data

    def compute_sdp_maxrules_biased_v2(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, pi=0.95):
        sdp, rules, sdp_all, rules_data, w = cyext_acv.compute_sdp_maxrule_biased_v2(x, y, data, y_data, S,
                                                                                     self.features, self.thresholds,
                                                                                     self.children_left,
                                                                                     self.children_right,
                                                                                     self.max_depth, min_node_size,
                                                                                     classifier, t, pi)
        extend_partition(rules, rules_data, sdp_all, pi=pi, S=S)
        return sdp, rules, sdp_all, rules_data, w

    def importance_sdp_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                          C=[[]], pi_level=0.9, minimal=1, stop=True):

        if x.shape[1] > 10:

            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            search_space = []
            for v in (node_idx.value_counts().keys()):
                search_space += [v]
        else:
            search_space = [i for i in range(x.shape[1])]

        sdp = cyext_acv.global_sdp_rf(x, y, data, y_data, self.features, self.thresholds, self.children_left,
                                      self.children_right, self.max_depth, min_node_size, classifier, t, C,
                                      pi_level, minimal, stop, search_space[:10])
        return sdp

    def importance_sdp_rf_v0(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                             C=[[]], pi_level=0.9, minimal=1, stop=True):

        if x.shape[1] > 10:

            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            search_space = []
            for v in (node_idx.value_counts().keys()):
                search_space += [v]
        else:
            search_space = [i for i in range(x.shape[1])]

        sdp = cyext_acv.global_sdp_rf_v0(x, y, data, y_data, self.features, self.thresholds, self.children_left,
                                         self.children_right, self.max_depth, min_node_size, classifier, t, C,
                                         pi_level, minimal, stop, search_space[:10])
        return sdp

    def compute_exp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        exp = cyext_acv.compute_exp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return exp

    def compute_quantile_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles = cyext_acv.compute_quantile_rf(x, y, data, y_data, S, self.features, self.thresholds,
                                                    self.children_left,
                                                    self.children_right, self.max_depth, min_node_size, classifier, t,
                                                    quantile)
        return y_quantiles

    def compute_quantile_diff_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(x, y, data, y_data, S, self.features, self.thresholds,
                                                              self.children_left,
                                                              self.children_right, self.max_depth, min_node_size,
                                                              classifier, t, quantile)
        return y_quantiles_diff

    def sufficient_expl_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                           C=[[]], pi_level=0.9, minimal=1, stop=True):

        if x.shape[1] > 10:

            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            search_space = []
            for v in (node_idx.value_counts().keys()):
                search_space += [v]
        else:
            search_space = [i for i in range(x.shape[1])]

        sdp = cyext_acv.sufficient_expl_rf(x, y, data, y_data, self.features, self.thresholds, self.children_left,
                                           self.children_right, self.max_depth, min_node_size, classifier, t, C,
                                           pi_level, minimal, stop, search_space[:10])
        return sdp

    def compute_local_sdp(d, sufficient_coal):
        flat = [item for sublist in sufficient_coal for item in sublist]
        flat = pd.Series(flat)
        flat = dict(flat.value_counts() / len(sufficient_coal))
        local_sdp = np.zeros(d)
        for key in flat.keys():
            local_sdp[key] = flat[key]
        return local_sdp

    def compute_exp_shaff(self, X, data, y_data, S, min_node_size=5):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_exp(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size)
        return exp

    def compute_sdp_clf_shaff(self, X, y_X, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp_clf(X=X[i], y_X=y_X[i], S=S, model=self, data=data, Y=y_data,
                                           min_node_size=min_node_size)
        return sdp

    def compute_sdp_shaff(self, X, y_X, t, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp(X=X[i], t=t, y_X=y_X[i], S=S, model=self, data=data, Y=y_data,
                                       min_node_size=min_node_size)
        return sdp

    def compute_quantile_shaff(self, X, data, y_data, S, min_node_size=5, quantile=95):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_quantile(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size,
                                            quantile=quantile)
        return exp
