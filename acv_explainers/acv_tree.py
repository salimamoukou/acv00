from .base_tree import *
from .py_acv import *
from .utils_exp import *
from .utils_sdp import *
import numpy as np
import cyext_acv


class ACVTree(BaseTree):

    def shap_values(self, X, C=[[]], num_threads=10):
        return cyext_acv.shap_values_leaves_pa(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                       self.node_idx_trees, C, num_threads)

    def shap_values_acv(self, X, S_star, N_star, C=[[]], num_threads=10):
        return cyext_acv.shap_values_acv_leaves(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                       self.node_idx_trees, S_star, N_star, C, num_threads)

    def shap_values_acv_adap(self, X, S_star, N_star, size, C=[[]], num_threads=10):
        return cyext_acv.shap_values_acv_leaves_data_cpp(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                       self.node_idx_trees, S_star, N_star, size, C, num_threads)


    def compute_exp(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                  num_threads)

    def compute_exp_cat(self, X, S, data, num_threads=10):
        return cyext_acv.compute_exp_cat(np.array(X, dtype=np.float), S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                      num_threads)

    def compute_sdp_clf(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.compute_sdp_clf(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_clf_cat(self, X, S, data, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.compute_sdp_clf_cat(np.array(X, dtype=np.float), fX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg(self, X, tX,  S, data, num_threads=10):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg(np.array(X, dtype=np.float), fX, tX, y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def compute_sdp_reg_cat(self, X, tX, S, data, num_threads=10):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.compute_sdp_reg_cat(np.array(X, dtype=np.float), fX, tX,  y_pred, S, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, num_threads)

    def swing_sv_clf(self, X, data, C=[[]], thresholds=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.swing_sv_clf_direct(np.array(X, dtype=np.float), fX, y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, thresholds, num_threads)

    def swing_sv_clf_slow(self, X, data, C=[[]], thresholds=0.9, num_threads=5):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.swing_sv_clf(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                      self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                      C, thresholds, num_threads)

    def importance_sdp_clf(self, X, data, C=[[]], global_proba=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.global_sdp_clf_cpp_pa_coal(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, num_threads)

    def importance_sdp_reg(self, X, tX, data, C=[[]], global_proba=0.9, num_threads=10):
        fX = self.predict(X)
        y_pred = self.predict(data)
        return cyext_acv.global_sdp_reg_cpp_pa_coal(np.array(X, dtype=np.float), fX, tX, y_pred, data, self.values,
                                                    self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                    self.scalings, C, global_proba, num_threads)

    def importance_sdp_clf_r(self, X, data, C=[[]], global_proba=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.global_sdp_clf_cpp_pa_coal_r(np.array(X, dtype=np.float), fX, y_pred, data, self.values,
                                                      self.partition_leaves_trees, self.leaf_idx_trees, self.leaves_nb,
                                                      self.scalings, C, global_proba, num_threads)

    def importance_sdp_clf_slow(self, X, data, C=[[]], global_proba=0.9, num_threads=10):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.global_sdp_clf_pa_coal(np.array(X, dtype=np.float), fX, y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, global_proba, num_threads)

    def importance_sdp_clf_slowv2(self, X, data, C=[[]], global_proba=0.9):
        fX = np.argmax(self.model.predict_proba(X), axis=1)
        y_pred = np.argmax(self.model.predict_proba(data), axis=1)
        return cyext_acv.global_sdp_clf_coal(np.array(X, dtype=np.float), fX, y_pred, data, self.values, self.partition_leaves_trees,
                                     self.leaf_idx_trees, self.leaves_nb, self.scalings, C, global_proba)

    def shap_values_acv_all(self, X, S_star, N_star, C=[[]], num_threads=10):
        return cyext_acv.shap_values_acv_leaves_data(np.array(X, dtype=np.float), self.data, self.values, self.partition_leaves_trees,
                                       self.leaf_idx_trees, self.leaves_nb, self.scalings,
                                       self.node_idx_trees, S_star, N_star, C, num_threads)

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



