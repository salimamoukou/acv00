from .base_tree import *
from .py_acv import *
from .utils_exp import *
from .utils_sdp import *
import numpy as np
import exp_co


class ACVTree(BaseTree):

    def shap_values(self, x, C):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves(x, self.partition_leaves_trees[i], self.data,
                                      self.node_idx_trees[i],
                                      self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i], self.values[i], C,
                                      self.num_outputs)
        return out

    def shap_valuesv2(self, x, C):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves_v2(x, self.partition_leaves_trees[i], self.data,
                                      self.node_idx_trees[i],
                                      self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i],
                                      self.values[i], C,
                                      self.num_outputs)
        return out

    def shap_values_acv(self, x, C, S_star, N_star):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_acv_leaves(x, self.partition_leaves_trees[i], self.data_leaves_trees[i],
                                          self.node_idx_trees[i],
                                          self.leaf_idx_trees[i], self.leaves_nb[i], self.node_sample_weight[i], self.values[i], C, S_star,
                                          N_star,
                                          self.num_outputs)
        return out

    def compute_sdp_reg(self, X, tX, S, data):
        return compute_sdp_reg(X, tX, self, S, data=data)

    def compute_sdp_clf(self, X, tX, S, data):
        return compute_sdp_clf(X, tX, self, S, data=data)

    def cyext_compute_sdp_clf(self, X, S, data):
        fX = np.argmax(self.predict(X), axis=1)
        y_pred = np.argmax(self.predict(data), axis=1)
        return exp_co.compute_sdp_clf(X, fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.trees[0].scaling)

    def cyext_compute_sdp_clf_cat(self, X, S, data):
        fX = np.argmax(self.predict(X), axis=1)
        y_pred = np.argmax(self.predict(data), axis=1)
        return exp_co.compute_sdp_clf_cat(X, fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.trees[0].scaling)

    def cyext_compute_exp(self, X, S, data):
        return exp_co.compute_exp(X, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.trees[0].scaling)

    def cyext_compute_exp_cat(self, X, S, data):
        return exp_co.compute_exp_cat(X, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.trees[0].scaling)

    def compute_sdp_reg_cat(self, X, tX, S, data):
        return compute_sdp_reg_cat(X, tX, model=self, S=S, data=data)

    def compute_sdp_clf_cat(self, X, tX, S, data):
        return compute_sdp_clf_cat(X, tX, model=self, S=S, data=data)

    def compute_cond_exp(self, X, S, data):
        return compute_exp(X=X, model=self, S=S, data=data)

    def compute_cond_exp_cat(self, X, S, data):
        return compute_exp_cat(X=X, model=self, S=S, data=data)

    def compute_local_sdp_clf(self, x, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(x, threshold, proba, index, data, final_coal, decay, C, verbose,
                         self.compute_sdp_clf)

    def compute_local_sdp_reg(self, x, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(x, threshold, proba, index, data, final_coal, decay, C, verbose, self.compute_sdp_reg)

    def swing_values_clf(self, x,  tx, S, data, threshold):
        return np.array(self.compute_sdp_clf(x, tx, S, data) >= threshold, dtype=float)

    def swing_values_reg(self, x, tx, S, data, threshold):
        return np.array(self.compute_sdp_reg(x, tx, S, data) >= threshold, dtype=float)

    def shap_values_swing_clf(self, x, tx, data, threshold, C):
        return swing_tree_shap(x, tx, threshold, data, C, self.swing_values_clf)

    def shap_values_swing_reg(self, x, tx, data, threshold, C):
        return swing_tree_shap(x, tx, threshold, data, C, self.swing_values_reg)

    def global_sdp_importance_clf(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):

        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_clf)

    def global_sdp_importance_reg(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):
        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_reg)



