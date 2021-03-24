/**
 * Fast computation of classic SHAP values, SDP, Swing SV in trees.
 */

#include <algorithm>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
using namespace std;

typedef double tfloat;
//typedef tfloat (* transform_f)(const tfloat margin, const tfloat y);

struct TreeEnsemble {
    int *children_left;
    int *children_right;
    int *children_default;
    int *features;
    tfloat *thresholds;
    tfloat *values;
    tfloat *node_sample_weights;
    unsigned max_depth;
    unsigned tree_limit;
    tfloat *base_offset;
    unsigned max_nodes;
    unsigned num_outputs;

    TreeEnsemble() {}
    TreeEnsemble(int *children_left, int *children_right, int *children_default, int *features,
                 tfloat *thresholds, tfloat *values, tfloat *node_sample_weights,
                 unsigned max_depth, unsigned tree_limit, tfloat *base_offset,
                 unsigned max_nodes, unsigned num_outputs) :
        children_left(children_left), children_right(children_right),
        children_default(children_default), features(features), thresholds(thresholds),
        values(values), node_sample_weights(node_sample_weights),
        max_depth(max_depth), tree_limit(tree_limit),
        base_offset(base_offset), max_nodes(max_nodes), num_outputs(num_outputs) {}

    void get_tree(TreeEnsemble &tree, const unsigned i) const {
        const unsigned d = i * max_nodes;

        tree.children_left = children_left + d;
        tree.children_right = children_right + d;
        tree.children_default = children_default + d;
        tree.features = features + d;
        tree.thresholds = thresholds + d;
        tree.values = values + d * num_outputs;
        tree.node_sample_weights = node_sample_weights + d;
        tree.max_depth = max_depth;
        tree.tree_limit = 1;
        tree.base_offset = base_offset;
        tree.max_nodes = max_nodes;
        tree.num_outputs = num_outputs;
    }

    bool is_leaf(unsigned pos)const {
        return children_left[pos] < 0;
    }

    void allocate(unsigned tree_limit_in, unsigned max_nodes_in, unsigned num_outputs_in) {
        tree_limit = tree_limit_in;
        max_nodes = max_nodes_in;
        num_outputs = num_outputs_in;
        children_left = new int[tree_limit * max_nodes];
        children_right = new int[tree_limit * max_nodes];
        children_default = new int[tree_limit * max_nodes];
        features = new int[tree_limit * max_nodes];
        thresholds = new tfloat[tree_limit * max_nodes];
        values = new tfloat[tree_limit * max_nodes * num_outputs];
        node_sample_weights = new tfloat[tree_limit * max_nodes];
    }

    void free() {
        delete[] children_left;
        delete[] children_right;
        delete[] children_default;
        delete[] features;
        delete[] thresholds;
        delete[] values;
        delete[] node_sample_weights;
    }
};

struct ExplanationDataset {
    tfloat *X;
    bool *X_missing;
//    tfloat *y;
    tfloat *R;
    bool *R_missing;
    unsigned num_X;
    unsigned M;
    unsigned num_R;

    ExplanationDataset() {}
    ExplanationDataset(tfloat *X, bool *X_missing, tfloat *R, bool *R_missing, unsigned num_X,
                       unsigned M, unsigned num_R) :
        X(X), X_missing(X_missing), R(R), R_missing(R_missing), num_X(num_X), M(M), num_R(num_R) {}

    void get_x_instance(ExplanationDataset &instance, const unsigned i) const {
        instance.M = M;
        instance.X = X + i * M;
        instance.X_missing = X_missing + i * M;
        instance.num_X = 1;
    }
};


//inline tfloat logistic_transform(const tfloat margin, const tfloat y) {
//    return 1 / (1 + exp(-margin));
//}
//
//inline tfloat logistic_nlogloss_transform(const tfloat margin, const tfloat y) {
//    return log(1 + exp(margin)) - y * margin; // y is in {0, 1}
//}
//
//inline tfloat squared_loss_transform(const tfloat margin, const tfloat y) {
//    return (margin - y) * (margin - y);
//}
//inline void leaves_partition(const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void multi_game_sv(tfloat *out_contribs, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void multi_game_sv_acv(tfloat *out_contribs, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void cond_exp_tree(tfloat *out_contribs, const tfloat *x,  const int *S, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void cond_sdp_tree(tfloat *out_contribs, const tfloat *x,  const int *S, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void swing_sdp_tree(tfloat *out_contribs, const tfloat *x,  const int *S, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void tree_sv(tfloat *out_contribs, const tfloat *x,  tfloat (*value_func)(tfloat, int), const TreeEnsemble& trees, const ExplanationDataset &data, const int *S_star,
//                    const int *N) {}
//
//inline void tree_sv_swing(tfloat *out_contribs, const tfloat *x,  tfloat (*value_func)(tfloat, int), const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void local_sdp(tfloat *out_contribs, const tfloat *x, const TreeEnsemble& trees, const ExplanationDataset &data) {}
//
//inline void global_sdp(tfloat *out_contribs, const tfloat *x, const TreeEnsemble& trees, const ExplanationDataset &data) {}

// example
inline int compute_expectations(TreeEnsemble &tree, int i = 0, int depth = 0) {
    unsigned max_depth = 0;

    if (tree.children_right[i] >= 0) {
        const unsigned li = tree.children_left[i];
        const unsigned ri = tree.children_right[i];
        const unsigned depth_left = compute_expectations(tree, li, depth + 1);
        const unsigned depth_right = compute_expectations(tree, ri, depth + 1);
        const tfloat left_weight = tree.node_sample_weights[li];
        const tfloat right_weight = tree.node_sample_weights[ri];
        const unsigned li_offset = li * tree.num_outputs;
        const unsigned ri_offset = ri * tree.num_outputs;
        const unsigned i_offset = i * tree.num_outputs;
        for (unsigned j = 0; j < tree.num_outputs; ++j) {
            if ((left_weight == 0) && (right_weight == 0)) {
//                tree.values[i_offset + j] = 0.0;
            } else {
//                const tfloat v = (left_weight * tree.values[li_offset + j] + right_weight * tree.values[ri_offset + j]) / (left_weight + right_weight);
//                tree.values[i_offset + j] = v;
            }
        }
        max_depth = std::max(depth_left, depth_right) + 1;
    }
    
    if (depth == 0) tree.max_depth = max_depth;
    
    return max_depth;
}

//namespace MODEL_TRANSFORM {
//    const unsigned identity = 0;
//    const unsigned logistic = 1;
//    const unsigned logistic_nlogloss = 2;
//    const unsigned squared_loss = 3;
//}

//inline transform_f get_transform(unsigned model_transform) {
//    transform_f transform = NULL;
//    switch (model_transform) {
//        case MODEL_TRANSFORM::logistic:
//            transform = logistic_transform;
//            break;
//
//        case MODEL_TRANSFORM::logistic_nlogloss:
//            transform = logistic_nlogloss_transform;
//            break;
//
//        case MODEL_TRANSFORM::squared_loss:
//            transform = squared_loss_transform;
//            break;
//    }
//
//    return transform;
//}

inline tfloat *tree_predict(unsigned i, const TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];

        // we hit a leaf so return a pointer to the values
        if (trees.is_leaf(pos)) {
            return trees.values + pos * trees.num_outputs;
        }

        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else {
            node = trees.children_right[pos];
        }
    }
}
//
//inline tfloat *tree_predict_proba(tfloat *out, const TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
////    const unsigned offset = i * trees.max_nodes;
//    unsigned node = 0;
//    while (true) {
//        const unsigned pos = node;
//        const unsigned feature = trees.features[pos];
//
//        // we hit a leaf so return a pointer to the values
//        if (trees.is_leaf(pos)) {
//             tfloat *leaf_value = trees.values + pos * trees.num_outputs;
//             for (unsigned k = 0; k < trees.num_outputs; ++k) {
//                out[k] += leaf_value[k];
//            }
//            return leaf_value;
//        }
//
//        // otherwise we are at an internal node and need to recurse
//        if (x_missing[feature]) {
//            node = trees.children_default[pos];
//        } else if (x[feature] <= trees.thresholds[pos]) {
//            node = trees.children_left[pos];
//        } else {
//            node = trees.children_right[pos];
//        }
//    }
//}

inline void dense_tree_predict(tfloat *out, const TreeEnsemble &trees, const ExplanationDataset &data) {
    tfloat *row_out = out;
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    // see what transform (if any) we have
//    transform_f transform = get_transform(model_transform);

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the base offset
        for (unsigned k = 0; k < trees.num_outputs; ++k) {
            row_out[k] += trees.base_offset[k];
        }

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            const tfloat *leaf_value = tree_predict(j, trees, x, x_missing);

            for (unsigned k = 0; k < trees.num_outputs; ++k) {
                row_out[k] += leaf_value[k];
            }
        }

        // apply any needed transform
//        if (transform != NULL) {
////            const tfloat y_i = data.y == NULL ? 0 : data.y[i];
//            for (unsigned k = 0; k < trees.num_outputs; ++k) {
//                row_out[k] = transform(row_out[k], y_i);
//            }
//        }

        x += data.M;
        x_missing += data.M;
        row_out += trees.num_outputs;
    }
}

inline void single_tree_predict(tfloat *out, const TreeEnsemble &trees, const ExplanationDataset &data, const unsigned j) {
    tfloat *row_out = out;
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    // see what transform (if any) we have
//    transform_f transform = get_transform(model_transform);

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the base offset
        for (unsigned k = 0; k < trees.num_outputs; ++k) {
            row_out[k] += trees.base_offset[k];
        }

        // add the leaf values from each tree
        const tfloat *leaf_value = tree_predict(j, trees, x, x_missing);
//        for (unsigned j = 0; j < trees.tree_limit; ++j) {


        for (unsigned k = 0; k < trees.num_outputs; ++k) {
            row_out[k] += leaf_value[k];
        }
//        }

        // apply any needed transform
//        if (transform != NULL) {
////            const tfloat y_i = data.y == NULL ? 0 : data.y[i];
//            for (unsigned k = 0; k < trees.num_outputs; ++k) {
//                row_out[k] = transform(row_out[k], y_i);
//            }
//        }

        x += data.M;
        x_missing += data.M;
        row_out += trees.num_outputs;
    }
}

inline void tree_update_weights(unsigned i, TreeEnsemble &trees, const tfloat *x, const bool *x_missing) {
    const unsigned offset = i * trees.max_nodes;
    unsigned node = 0;
    while (true) {
        const unsigned pos = offset + node;
        const unsigned feature = trees.features[pos];

        // Record that a sample passed through this node
        trees.node_sample_weights[pos] += 1.0;

        // we hit a leaf so return a pointer to the values
        if (trees.children_left[pos] < 0) break;

        // otherwise we are at an internal node and need to recurse
        if (x_missing[feature]) {
            node = trees.children_default[pos];
        } else if (x[feature] <= trees.thresholds[pos]) {
            node = trees.children_left[pos];
        } else {
            node = trees.children_right[pos];
        }
    }
}

inline void dense_tree_update_weights(TreeEnsemble &trees, const ExplanationDataset &data) {
    const tfloat *x = data.X;
    const bool *x_missing = data.X_missing;

    for (unsigned i = 0; i < data.num_X; ++i) {

        // add the leaf values from each tree
        for (unsigned j = 0; j < trees.tree_limit; ++j) {
            tree_update_weights(j, trees, x, x_missing);
        }

        x += data.M;
        x_missing += data.M;
    }
}
