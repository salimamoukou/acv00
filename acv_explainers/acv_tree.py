from .base_tree import *
from .py_acv import *
from .utils_exp import *
from .utils_sdp import *
import numpy as np
import cyext_acv, cyext_acv_nopa, cyext_acv_cache


class ACVTree(BaseTree):

    def shap_values(self, X, C=[[]], num_threads=10):
        if not self.cache:
            return cyext_acv.shap_values_leaves_pa(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, C, num_threads)
        return self.shap_values_cache(X, C)

    def shap_values_acv(self, X, S_star, N_star, C=[[]], num_threads=10):
        return cyext_acv.shap_values_acv_leaves(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, S_star, N_star, C, num_threads)

    def shap_values_acv_adap(self, X, S_star, N_star, size, C=[[]], num_threads=10):
        return cyext_acv.shap_values_acv_leaves_adap(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, S_star, N_star, size, C, num_threads)

    def importance_sdp_clf_greedy(self, X, data, C=[[]], global_proba=0.9, minimal=1, stop=True):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier", "lightgbm.sklearn.LGBMClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, data, self.values_binary,
                                            self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                            self.scalings, C, global_proba, minimal, stop)

        return cyext_acv.global_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal, stop)

    def importance_sdp_clf_search(self, X, data, C=[[]], global_proba=0.9, minimal=1, search_space=[], stop=True):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf_approx(np.array(X, dtype=np.float), fX, y_pred, data, self.values_binary,
                                                   self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                   self.scalings, C, global_proba, minimal, search_space, stop)

        return cyext_acv.global_sdp_clf_approx(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal, search_space, stop)

    def importance_sdp_clf(self, X, data, C=[[]], global_proba=0.9, minimal=1, stop=True):
        if X.shape[1] > 15:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_sdp_clf_search(X, data, C, global_proba, minimal, list(order_va[:15]), stop)
        else:
            return self.importance_sdp_clf_greedy(X, data, C, global_proba, minimal, stop=stop)

    def importance_sdp_clf_ptrees(self, X, data, C=[[]], global_proba=0.9, minimal=0, stop=True):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.global_sdp_clf_ptrees(np.array(X, dtype=np.float), fX, y_pred, data, self.values_binary,
                                                   self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                   self.scalings, C, global_proba, minimal, stop)

        return cyext_acv.global_sdp_clf_ptrees(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal, stop)

    def compute_exp(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_exp_cat(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_cat(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                      num_threads)

    def compute_sdp_clf(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.compute_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values_binary,
                                             self.partition_leaves_trees,
                                             self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

        return cyext_acv.compute_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_clf_cat(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        if safe_isinstance(self.model, ["xgboost.sklearn.XGBClassifier", "catboost.core.CatBoostClassifier",
                                        "lightgbm.sklearn.LGBMClassifier"]) and \
                self.num_outputs == 1:
            return cyext_acv.compute_sdp_clf_cat(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values_binary,
                                                 self.partition_leaves_trees,
                                                 self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

        return cyext_acv.compute_sdp_clf_cat(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg(self, X, tX,  S, data, num_threads=10):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_cat(self, X, tX, S, data, num_threads=10):
        # raise Warning('The current implementation may take a long time if n_trees and depth are large. The number of '
        #               'operation is 2**(depth*n_trees)')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg_cat(np.array(X, dtype=np.float), fX, tX,  y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def swing_sv_clf(self, X, data, C=[[]], thresholds=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv.swing_sv_clf_direct(np.array(X, dtype=np.float), fX, y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, thresholds, num_threads)

    def swing_sv_clf_nopa(self, X, data, C=[[]], thresholds=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.swing_sv_clf_direct_nopa(np.array(X, dtype=np.float), fX, y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, thresholds, num_threads)

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

    def shap_values_nopa(self, X, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_leaves_nopa(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, C, num_threads)

    def shap_values_acv_nopa(self, X, S_star, N_star, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_acv_leaves_nopa(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, S_star, N_star, C, num_threads)

    def shap_values_acv_adap_nopa(self, X, S_star, N_star, size, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_acv_leaves_adap_nopa(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, S_star, N_star, size, C, num_threads)


    def importance_sdp_reg_cat(self, X, tX, data, C=[[]], global_proba=0.9, minimal=0, stop=True):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.global_sdp_reg_cat(np.array(X, dtype=np.float), fX, tX,  y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, global_proba, minimal, stop)

    def importance_sdp_reg(self, X, tX, data, C=[[]], global_proba=0.9, minimal=0, stop=True):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.global_sdp_reg(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal, stop)

    def importance_sdp_reg_cat_nopa(self, X, tX, data, C=[[]], global_proba=0.9, minimal=0):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.global_sdp_reg_cat_nopa(np.array(X, dtype=np.float), fX, tX,  y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, global_proba, minimal)

    def importance_sdp_reg_nopa(self, X, tX, data, C=[[]], global_proba=0.9, minimal=0):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.global_sdp_reg_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal)


    def compute_exp_normalized(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_normalized(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                  num_threads)

    def compute_exp_normalized_nopa(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_normalized_nopa(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                  num_threads)

    def shap_values_normalized(self, X, C=[[]], num_threads=10):
        if not self.cache_normalized:
            return cyext_acv.shap_values_leaves_normalized(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                       self.node_idx_trees, C, num_threads)
        return self.shap_values_normalized_cache(X, C)

    def shap_values_cache(self, X, C=[[]], num_threads=10):
        return cyext_acv_cache.shap_values_leaves_cache(np.array(X, dtype=np.float), self.data, self.values,
                                                  self.partition_leaves_trees,
                                                  self.leaf_idx_trees, self.leaves_nb, self.lm, self.lm_s, self.lm_si,
                                                  self.max_var,
                                                  self.node_idx_trees, C, num_threads)

    def shap_values_normalized_cache(self, X, C=[[]], num_threads=10):
        return cyext_acv_cache.shap_values_leaves_normalized_cache(np.array(X, dtype=np.float), self.data, self.values,
                                                  self.partition_leaves_trees,
                                                  self.leaf_idx_trees, self.leaves_nb, self.lm_n, self.lm_s_n, self.lm_si_n,
                                                  self.max_var,
                                                  self.node_idx_trees, C, num_threads)

    def shap_values_cache_nopa(self, X, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_leaves_cache_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                  self.partition_leaves_trees,
                                                  self.leaf_idx_trees, self.leaves_nb, self.lm, self.lm_s, self.lm_si,
                                                  self.max_var,
                                                  self.node_idx_trees, C, num_threads)

    def shap_values_normalized_cache_nopa(self, X, C=[[]], num_threads=10):
        return cyext_acv_nopa.shap_values_leaves_normalized_cache_nopa(np.array(X, dtype=np.float), self.data, self.values,
                                                  self.partition_leaves_trees,
                                                  self.leaf_idx_trees, self.leaves_nb, self.lm_n, self.lm_s_n, self.lm_si_n,
                                                  self.max_var,
                                                  self.node_idx_trees, C, num_threads)

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

    def importance_sdp_clf_nopa(self, X, data, C=[[]], global_proba=0.9, minimal=0):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.global_sdp_clf_nopa(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, minimal)

    def compute_exp_nopa(self, X, S, data, num_threads=10):
        return cyext_acv_nopa.compute_exp_nopa(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_exp_cat_nopa(self, X, S, data, num_threads=10):
        return cyext_acv_nopa.compute_exp_cat_nopa(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                      num_threads)

    def compute_sdp_clf_nopa(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.compute_sdp_clf_nopa(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_clf_cat_nopa(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1).astype(np.long)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1).astype(np.long)
        return cyext_acv_nopa.compute_sdp_clf_cat_nopa(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_nopa(self, X, tX,  S, data, num_threads=10):
        # if self.partition_leaves_trees.shape[0] > 1:
        #     raise NotImplementedError('Continuous SDP is currently available only for trees with n_trees=1')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.compute_sdp_reg_nopa(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_cat_nopa(self, X, tX, S, data, num_threads=10):
        # raise Warning('The current implementation may take a long time if n_trees > 10 and depth > 6')
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv_nopa.compute_sdp_reg_cat_nopa(np.array(X, dtype=np.float), fX, tX,  y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)


    def py_shap_values(self, x, C=[[]]):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves(x, self.partition_leaves_trees[i], self.data,
                                      self.node_idx_trees[i],
                                      self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i], self.values[i], C,
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
                                          self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i], self.values[i], C, S_star,
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

    def py_swing_sv_reg(self, X, data, threshold, C,  tX=0):
        return swing_tree_shap(X, tX, threshold, data, C, self.swing_values_reg)

    def py_global_sdp_importance_clf(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):

        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_clf)

    def py_global_sdp_importance_reg(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):
        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_reg)

    def py_shap_values_notoptimized(self, X, data, C=[[]]):
        return shap_values_leaves_notoptimized(X, data, C, self)

    def py_shap_values_discrete_notoptimized(self, X, data, C=[[]]):
        return shap_values_discrete_notoptimized(X, data, C, self)

    def compute_msdp_clf(self, X, S, data, model=None, N=10000):
        """
        Compute marginal SDP
        """
        # if data:
        #     return msdp_true(X, S, self.model, self.data)

        if model == None:
            return msdp(X, S, self.model, data)
        return msdp(X, S, model, data)

    def importance_msdp_clf_search(self, X, data, model=None, C=[[]], minimal=1, global_proba=0.9, r_search_space=None,
                                   stop=True):
        """
        Compute marginal S^\star of model
        """
        # if data:
        #     return importance_msdp_clf_true(X.values, self.model, self.data, C=C,
        #                                          minimal=minimal,
        #                                          global_proba=global_proba)
        if model == None:
            return importance_msdp_clf_search(X=X, rg_data=data, model=self.model, C=C, minimal=minimal, global_proba=global_proba, r_search_space=r_search_space, stop=stop)
        return importance_msdp_clf_search(X=X, rg_data=data, model=model, C=C, minimal=minimal, global_proba=global_proba, r_search_space=r_search_space, stop=stop)

    def importance_msdp_clf(self, X, data, model=None, C=[[]], global_proba=0.9, minimal=1, stop=True):
        if X.shape[1] > 15:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_msdp_clf_search(X=X, data=data, model=model, C=C, global_proba=global_proba, minimal=minimal,
                                                   r_search_space=order_va[:15], stop=stop)
        else:
            return self.importance_msdp_clf_search(X=X, data=data, model=model, C=C, global_proba=global_proba, minimal=minimal, stop=stop)

    def compute_msdp_reg(self, X, S, data, model=None, N=10000, threshold=0.2):
        """
        Compute marginal SDP of regression model
        """
        # if data:
        #     return msdp_true(X, S, self.model, self.data)

        if model == None:
            return msdp_reg(X, S, self.model, data, threshold)
        return msdp_reg(X, S, model, data, threshold)

    def importance_msdp_reg_search(self, X, data, model=None, C=[[]], minimal=1, global_proba=0.9, threshold=0.2, r_search_space=None, stop=True):
        """
        Compute marginal S^\star of regression model
        """
        # if data:
        #     return importance_msdp_clf_true(X.values, self.model, self.data, C=C,
        #                                          minimal=minimal,
        #                                          global_proba=global_proba)
        if model == None:
            return importance_msdp_reg_search(X, rg_data=data, model=self.model, C=C, minimal=minimal, global_proba=global_proba, threshold=threshold, r_search_space=r_search_space, stop=stop)
        return importance_msdp_reg_search(X, rg_data=data, model=model,  C=C, minimal=minimal, global_proba=global_proba, threshold=threshold, r_search_space=r_search_space, stop=stop)

    def importance_msdp_reg(self, X, data, model=None, C=[[]], global_proba=0.9, minimal=1, threshold=0.2, stop=True):
        if X.shape[1] > 15:
            flat_list = [item for t in self.node_idx_trees for sublist in t for item in sublist]
            node_idx = pd.Series(flat_list)
            order_va = []
            for v in (node_idx.value_counts().keys()):
                order_va += [v]
            return self.importance_msdp_reg_search(X=X, data=data, model=model, C=C, global_proba=global_proba, minimal=minimal,
                                                   r_search_space=order_va[:15], threshold=threshold, stop=stop)
        else:
            return self.importance_msdp_reg_search(X=X, data=data, model=model, C=C, global_proba=global_proba, minimal=minimal, threshold=threshold, stop=stop)

    def compute_exp_shaff(self, X, data, y_data, S, min_node_size=5):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_exp(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size)
        return exp

    def compute_sdp_clf_shaff(self, X, y_X, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp_clf(X=X[i], y_X=y_X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size)
        return sdp

    def compute_sdp_shaff(self, X, y_X, t, data, y_data, S, min_node_size=5):
        sdp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            sdp[i] = compute_shaff_sdp(X=X[i], t=t, y_X=y_X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size)
        return sdp

    def compute_quantile_shaff(self, X, data, y_data, S, min_node_size=5, quantile=95):
        exp = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            exp[i] = compute_shaff_quantile(X=X[i], S=S, model=self, data=data, Y=y_data, min_node_size=min_node_size, quantile=quantile)
        return exp

    def compute_sdp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_sdp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def compute_cdf_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_cdf_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def importance_sdp_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                          C=[[]], global_proba=0.9, minimal=1, stop=True):

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
                                      global_proba, minimal, stop, search_space[:10])
        return sdp

    def compute_exp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        exp = cyext_acv.compute_exp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return exp

    def compute_quantile_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles = cyext_acv.compute_quantile_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t, quantile)
        return y_quantiles

    def compute_quantile_diff_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t, quantile)
        return y_quantiles_diff


class ACVTreeAgnostic(BaseTree):

    @staticmethod
    def compute_msdp_clf(X, S, data, model=None, N=10000):
        """
        Compute marginal SDP
        """
        return msdp(X, S, model, data)

    @staticmethod
    def importance_msdp_clf_search(X, data, model=None, C=[[]], minimal=1, global_proba=0.9, r_search_space=None,
                                   stop=True):
        """
        Compute marginal S^\star of model
        """

        return importance_msdp_clf_search(X=X, rg_data=data, model=model, C=C, minimal=minimal,
                                          global_proba=global_proba, r_search_space=r_search_space, stop=stop)

    @staticmethod
    def compute_msdp_reg(self, X, S, data, model=None, N=10000, threshold=0.2):
        """
        Compute marginal SDP of regression model
        """
        return msdp_reg(X, S, model, data, threshold)

    @staticmethod
    def importance_msdp_reg_search(X, data, model=None, C=[[]], minimal=1, global_proba=0.9, threshold=0.2,
                                   r_search_space=None, stop=True):
        """
        Compute marginal S^\star of regression model
        """
        return importance_msdp_reg_search(X, rg_data=data, model=model, C=C, minimal=minimal, global_proba=global_proba,
                                          threshold=threshold, r_search_space=r_search_space, stop=stop)

    def compute_cdf_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_cdf_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def compute_sdp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        sdp = cyext_acv.compute_sdp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return sdp

    def importance_sdp_rf(self, x, y, data, y_data, min_node_size=5, classifier=1, t=20,
                          C=[[]], global_proba=0.9, minimal=1, stop=True):

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
                                      global_proba, minimal, stop, search_space[:10])
        return sdp

    def compute_exp_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20):
        exp = cyext_acv.compute_exp_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t)
        return exp

    def compute_quantile_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles = cyext_acv.compute_quantile_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t, quantile)
        return y_quantiles

    def compute_quantile_diff_rf(self, x, y, data, y_data, S, min_node_size=5, classifier=1, t=20, quantile=95):
        y_quantiles_diff = cyext_acv.compute_quantile_diff_rf(x, y, data, y_data, S, self.features, self.thresholds, self.children_left,
                                       self.children_right, self.max_depth, min_node_size, classifier, t, quantile)
        return y_quantiles_diff

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
