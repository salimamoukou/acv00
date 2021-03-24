from .base_tree import *
from .py_acv import *
import numpy as np


class ACVTree(BaseTree):

    def shap_values(self, x, C):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_leaves(x, self.partition_leaves_trees[i], self.data_leaves_trees[i],
                                      self.node_idx_trees[i],
                                      self.leaf_idx_trees[i], self.node_sample_weight[i], self.values[i], C,
                                      self.num_outputs)
        return out

    def shap_values_acv(self, x, C, S_star, N_star):
        out = np.zeros((x.shape[0], x.shape[1], self.num_outputs))
        for i in range(len(self.trees)):
            out += shap_values_acv_leaves(x, self.partition_leaves_trees[i], self.data_leaves_trees[i],
                                          self.node_idx_trees[i],
                                          self.leaf_idx_trees[i], self.node_sample_weight[i], self.values[i], C, S_star,
                                          N_star,
                                          self.num_outputs)
        return out

    # def compute_sdp_clf_v1(self, x, fx, tx, S, data):
    #     sdp = 0
    #     ntrees = len(self.trees)
    #     for i in range(ntrees):
    #         sdp += cond_sdp_tree_clf_v1(x, fx, tx, self.trees[i], S, data, ntrees=ntrees)
    #     return sdp / ntrees

    def compute_sdp_reg(self, x, fx, tx, S, data):
        sdp = cond_sdp_forest(x, fx, tx, self.trees, S, data=data)
        return sdp

    def compute_sdp_clf(self, x, fx, tx, S, data):
        sdp = cond_sdp_forest_clf(x, fx, tx, self.trees, S, data=data)
        return sdp

    def compute_local_sdp_clf(self, x, f, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(x, f, threshold, proba, index, data, final_coal, decay, C, verbose,
                         self.compute_sdp_clf)

    def compute_local_sdp_reg(self, x, f, threshold, proba, index, data, final_coal, decay, C, verbose):
        return local_sdp(x, f, threshold, proba, index, data, final_coal, decay, C, verbose, self.compute_sdp_reg)

    def swing_values_clf(self, x, fx, tx, S, data, threshold):
        return int(self.compute_sdp_clf(x, fx, tx, S, data) >= threshold)

    def swing_values_reg(self, x, fx, tx, S, data, threshold):
        return int(self.compute_sdp_reg(x, fx, tx, S, data) >= threshold)

    def shap_values_swing_clf(self, x, fx, tx, data, C, threshold):
        kwargs = {'x': x, 'fx': fx, 'tx': tx, 'data': data, 'threshold': threshold}
        return brute_force_tree_shap(x, self.num_outputs, C, self.swing_values_clf, kwargs, swing=True)

    def shap_values_swing_reg(self, x, fx, tx, data, C, threshold):
        kwargs = {'x': x, 'fx': fx, 'tx': tx, 'data': data, 'threshold': threshold}
        return brute_force_tree_shap(x, self.num_outputs, C, self.swing_values_reg, kwargs, swing=True)

    def global_sdp_importance_clf(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):

        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_clf, self.predict)

    def global_sdp_importance_reg(self, data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose):
        return global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, self.compute_sdp_reg, self.predict)



