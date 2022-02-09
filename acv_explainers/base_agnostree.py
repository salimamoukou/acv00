from abc import abstractmethod
import scipy.special
import json
import struct
import cext_acv
import warnings
from .py_acv import *
import cyext_acv, cyext_acv_nopa, cyext_acv_cache
from .py_acv import *
from sklearn.utils.validation import check_array
from .utils import rebuild_tree
from distutils.version import LooseVersion


# This is based on https://github.com/slundberg/shap/blob/master/shap/explainers/_tree.py
class BaseAgnosTree:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data_dim=None, data=None, data_missing=None, cache=False, cache_normalized=False, multi_threads=True,
                 C=[[]]):
        self.model_type = "internal"
        self.trees = None
        self.base_offset = 0
        self.model_output = None
        self.objective = None  # what we explain when explaining the loss of the model
        self.tree_output = None  # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = np.float64  # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True  # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None  # used for limiting the number of trees we use by default (like from early stopping)
        self.num_stacked_models = 1  # If this is greater than 1 it means we have multiple stacked models with the same number of trees in each model (XGBoost multi-output style)
        self.cat_feature_indices = None  # If this is set it tells us which features are treated categorically
        self.model = model
        self.cache = cache
        self.cache_normalized = cache_normalized
        self.C = C
        self.multi_threads = multi_threads
        self.data_dim = data_dim

        if safe_isinstance(model, ["skranger.ensemble.classifier.RangerForestClassifier"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            # self.scaling = scaling
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for
                          e in model.estimators_]
            self.tree_output = "probability"

        elif safe_isinstance(model, ["skranger.ensemble.regressor.RangerForestRegressor"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_)  # output is average of trees
            # self.scaling = scaling
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in
                          model.estimators_]
            self.tree_output = "raw_value"

        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in
                                  self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            num_trees = len(self.trees)
            if self.num_stacked_models > 1:
                assert len(
                    self.trees) % self.num_stacked_models == 0, "Only stacked models with equal numbers of trees are supported!"
                assert self.trees[0].values.shape[
                           1] == 1, "Only stacked models with single outputs per model are supported!"
                self.num_outputs = self.num_stacked_models
            else:
                self.num_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.features = -np.ones((num_trees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((num_trees, max_nodes, self.num_outputs), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)

            self.partition_leaves_trees = []
            self.node_idx_trees = []
            # self.data_leaves_trees = []
            self.leaf_idx_trees = []
            self.leaves_nb = []
            self.scalings = []
            for i in tqdm(range(num_trees)):
                self.scalings.append(self.trees[i].scaling)
                self.children_left[i, :len(self.trees[i].children_left)] = self.trees[i].children_left
                self.children_right[i, :len(self.trees[i].children_right)] = self.trees[i].children_right
                self.children_default[i, :len(self.trees[i].children_default)] = self.trees[i].children_default
                self.features[i, :len(self.trees[i].features)] = self.trees[i].features
                self.thresholds[i, :len(self.trees[i].thresholds)] = self.trees[i].thresholds
                if self.num_stacked_models > 1:
                    # stack_pos = int(i // (num_trees / self.num_stacked_models))
                    stack_pos = i % self.num_stacked_models
                    self.values[i, :len(self.trees[i].values[:, 0]), stack_pos] = self.trees[i].values[:, 0]
                else:
                    self.values[i, :len(self.trees[i].values)] = self.trees[i].values
                self.node_sample_weight[i, :len(self.trees[i].node_sample_weight)] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False

                self.leaf_idx = [idx for idx in range(len(self.trees[i].features))
                                 if self.trees[i].children_left[idx] < 0]

                self.leaves_nb.append(len(self.leaf_idx))

                self.partition_leaves = []
                self.node_idx = []
                self.max_var = []
                # self.data_leaves = []
                for leaf_id in self.leaf_idx:
                    node_id = [-1]
                    partition_leaf = [np.array([[-np.inf, np.inf]]) for idx2 in range(self.data_dim)]
                    _ = get_partition(leaf_id, partition_leaf, node_id, self.trees[i].children_left,
                                      self.trees[i].children_right, self.trees[i].features, self.trees[i].thresholds)

                    self.partition_leaves.append(np.squeeze(np.array(partition_leaf)))
                    self.node_idx.append(list(set(node_id[1:])))
                    self.max_var.append(len(self.node_idx[-1]))
                    # self.data_leaves.append(np.array([(self.data[:, s] <= self.partition_leaves[-1][s, 1]) * \
                    #                                       (self.data[:, s] > self.partition_leaves[-1][s, 0])
                    #                                       for s in range(self.data.shape[1])], dtype=np.int).transpose())

                self.partition_leaves_trees.append(self.partition_leaves)
                # self.data_leaves_trees.append(self.data_leaves)

                self.node_idx_trees.append(self.node_idx)
                self.leaf_idx_trees.append(self.leaf_idx)

            leaf_idx_trees = -np.ones(shape=(len(self.leaves_nb), np.max(self.leaves_nb)), dtype=np.int)
            partition_leaves_trees = -np.ones(
                shape=(len(self.leaves_nb), np.max(self.leaves_nb), self.data_dim, 2))
            # data_leaves_trees = -np.ones(shape=(len(self.leaves_nb), np.max(self.leaves_nb), self.data.shape[0], self.data.shape[1]), dtype=np.int)
            for i in range(len(self.leaves_nb)):
                leaf_idx_trees[i, :self.leaves_nb[i]] = np.array(self.leaf_idx_trees[i], dtype=np.int)
                partition_leaves_trees[i, :self.leaves_nb[i]] = np.array(self.partition_leaves_trees[i])
                # data_leaves_trees[i, :self.leaves_nb[i]] = np.array(self.data_leaves_trees[i], dtype=np.int)

            self.leaf_idx_trees = leaf_idx_trees
            self.partition_leaves_trees = partition_leaves_trees
            self.leaves_nb = np.array(self.leaves_nb, dtype=np.int)
            self.scalings = np.array(self.scalings, dtype=np.float)
            # self.data = np.array(self.data, dtype=np.float)
            self.max_var = np.max(self.max_var)
            # self.data_leaves_trees = data_leaves_trees

            # if safe_isinstance(model, ["xgboost.sklearn.XGBClassifier",
            #                            "catboost.core.CatBoostClassifier", "lightgbm.sklearn.LGBMClassifier"]) and \
            #         self.num_outputs == 1:
            #     p = np.exp(self.values)/(1 + np.exp(self.values))
            #     print(np.max(p), np.min(1-p))
            #     self.values = np.concatenate([1-p, p], axis=2)
            #     self.num_outputs = 2

            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

    def shap_values_cache(self, X, data_bground, C=[[]], num_threads=10):
        """
                Same as **shap_values**, but use cached values to speed up computation

        """
        X = check_array(X, dtype=[np.double])
        return cyext_acv_cache.shap_values_leaves_cache(X, data_bground, self.values,
                                                        self.partition_leaves_trees,
                                                        self.leaf_idx_trees, self.leaves_nb, self.lm, self.lm_s,
                                                        self.lm_si,
                                                        self.max_var,
                                                        self.node_idx_trees, C, num_threads)

    def shap_values(self, X, data_bground, C=[[]], num_threads=10):
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
            return cyext_acv.shap_values_leaves_pa(X, data_bground, self.values,
                                                   self.partition_leaves_trees,
                                                   self.leaf_idx_trees, self.leaves_nb, self.max_var,
                                                   self.node_idx_trees, C, num_threads)
        return self.shap_values_cache(X, data_bground, C)

class SingleTree:
    """ A single decision tree.

    The primary point of this object is to parse many different tree types into a common format.
    """

    def __init__(self, tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        self.scaling = scaling
        self.children_left = tree.children_left.astype(np.int32)
        self.children_right = tree.children_right.astype(np.int32)
        self.children_default = self.children_left  # missing values not supported in sklearn
        self.features = tree.feature.astype(np.int32)
        self.thresholds = tree.threshold.astype(np.float64)
        # corrected rangers features handle
        for i in range(self.features.shape[0]):
            if self.features[i] == -2:
                if self.children_left[i] != -1 or self.children_right[i] != -1:
                    self.features[i] = 0

        self.values = tree.value.reshape(tree.value.shape[0], tree.value.shape[1] * tree.value.shape[2])
        if normalize:
            self.values = (self.values.T / self.values.sum(1)).T
        self.values = self.values * scaling
        self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

        self.max_depth = cext_acv.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )

