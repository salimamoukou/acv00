#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "acv_tree.h"
#include <iostream>

//static PyObject *_cext_multi_game_sv(PyObject *self, PyObject *args);
//static PyObject *_cext_cond_exp_tree(PyObject *self, PyObject *args);
//static PyObject *_cext_cond_sdp_tree(PyObject *self, PyObject *args);
//static PyObject *_cext_swing_sdp_tree(PyObject *self, PyObject *args);
//static PyObject *_cext_tree_sv(PyObject *self, PyObject *args);
//static PyObject *_cext_tree_sv_acv(PyObject *self, PyObject *args);
//static PyObject *_cext_local_sdp(PyObject *self, PyObject *args);
//static PyObject *_cext_global_sdp(PyObject *self, PyObject *args);
static PyObject *_cext_dense_tree_predict(PyObject *self, PyObject *args);
static PyObject *_cext_tree_predict(PyObject *self, PyObject *args);
static PyObject *_cext_single_tree_predict(PyObject *self, PyObject *args);
static PyObject *_cext_compute_expectations(PyObject *self, PyObject *args);


static PyMethodDef module_methods[] = {
//    {"multi_game_sv", _cext_multi_game_sv, METH_VARARGS, "C implementation of Multi-Game Shapley Values."},
//    {"cond_exp_tree", _cext_cond_exp_tree, METH_VARARGS, "C implementation of Plug-In estimators"},
//    {"cond_sdp_tree", _cext_cond_sdp_tree, METH_VARARGS, "C implementation of tree SDP compuatations."},
//    {"swing_sdp_tree", _cext_swing_sdp_tree, METH_VARARGS, "C implementation of Swing Shapley values."},
//    {"tree_sv", _cext_tree_sv, METH_VARARGS, "C implementation of Tree SV given any value function."},
//    {"tree_sv_acv", _cext_tree_sv_acv, METH_VARARGS, "C implementation of Tree Active SV given any value function."},
//    {"local_sdp", _cext_local_sdp, METH_VARARGS, "C implementation of the algorithm that find S_STAR."},
//    {"global_sdp", _cext_global_sdp, METH_VARARGS, "C implementation of tree Global SV"},
    {"dense_tree_predict", _cext_dense_tree_predict, METH_VARARGS, "C implementation of trees predictions."},
//    {"tree_predict", _cext_tree_predict, METH_VARARGS, "C implementation of single tree predictions."},
    {"single_tree_predict", _cext_single_tree_predict, METH_VARARGS, "C implementation of single tree predictions."},
    {"compute_expectations", _cext_compute_expectations, METH_VARARGS, "(A refaire, pour donner juste le max_depth) Compute expectations of internal nodes."},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "cext_acv",
    "ACV function optimized in C !",
    -1,
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_cext_acv(void)
#else
PyMODINIT_FUNC init_cext_acv(void)
#endif
{
    #if PY_MAJOR_VERSION >= 3
        PyObject *module = PyModule_Create(&moduledef);
        if (!module) return NULL;
    #else
        PyObject *module = Py_InitModule("cext_acv", module_methods);
        if (!module) return;
    #endif

    /* Load `numpy` functionality. */
    import_array();

    #if PY_MAJOR_VERSION >= 3
        return module;
    #endif
}


// Example of C function wrapping in Python
static PyObject *_cext_compute_expectations(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *node_sample_weight_obj;
    PyObject *values_obj;
    
    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOO", &children_left_obj, &children_right_obj, &node_sample_weight_obj, &values_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_sample_weight_array = (PyArrayObject*)PyArray_FROM_OTF(node_sample_weight_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. */
    if (children_left_array == NULL || children_right_array == NULL ||
        values_array == NULL || node_sample_weight_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        //PyArray_ResolveWritebackIfCopy(values_array);
        Py_XDECREF(values_array);
        Py_XDECREF(node_sample_weight_array);
        return NULL;
    }

    TreeEnsemble tree;

    // number of outputs
    tree.num_outputs = PyArray_DIM(values_array, 1);

    /* Get pointers to the data as C-types. */
    tree.children_left = (int*)PyArray_DATA(children_left_array);
    tree.children_right = (int*)PyArray_DATA(children_right_array);
    tree.values = (tfloat*)PyArray_DATA(values_array);
    tree.node_sample_weights = (tfloat*)PyArray_DATA(node_sample_weight_array);

    const int max_depth = compute_expectations(tree);

    // clean up the created python objects
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    //PyArray_ResolveWritebackIfCopy(values_array);
    Py_XDECREF(values_array);
    Py_XDECREF(node_sample_weight_array);

    PyObject *ret = Py_BuildValue("i", max_depth);
    return ret;
}


static PyObject *_cext_dense_tree_predict(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int max_depth;
    int tree_limit;
    PyObject *base_offset_obj;
//    int model_output;
    PyObject *X_obj;
    PyObject *X_missing_obj;
//    PyObject *y_obj;
    PyObject *out_pred_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiiOOOO", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &max_depth, &tree_limit, &base_offset_obj,
        &X_obj, &X_missing_obj, &out_pred_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *base_offset_array = (PyArrayObject*)PyArray_FROM_OTF(base_offset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *y_array = NULL;
//    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_pred_array = (PyArrayObject*)PyArray_FROM_OTF(out_pred_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || X_array == NULL ||
        X_missing_array == NULL || out_pred_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        Py_XDECREF(base_offset_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
//        if (y_array != NULL) Py_XDECREF(y_array);
        //PyArray_ResolveWritebackIfCopy(out_pred_array);
        Py_XDECREF(out_pred_array);
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);
    const unsigned num_outputs = PyArray_DIM(values_array, 2);

    const unsigned num_offsets = PyArray_DIM(base_offset_array, 0);
    if (num_offsets != num_outputs) {
        std::cerr << "The passed base_offset array does that have the same number of outputs as the values array: " << num_offsets << " vs. " << num_outputs << std::endl;
        return NULL;
    }

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *base_offset = (tfloat*)PyArray_DATA(base_offset_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
//    tfloat *y = NULL;
//    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
    tfloat *out_pred = (tfloat*)PyArray_DATA(out_pred_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        NULL, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, NULL, NULL, num_X, M, 0);

    dense_tree_predict(out_pred, trees, data);

    // clean up the created python objects
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    Py_XDECREF(base_offset_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);
//    if (y_array != NULL) Py_XDECREF(y_array);
    //PyArray_ResolveWritebackIfCopy(out_pred_array);
    Py_XDECREF(out_pred_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", (double)values[0]);
    return ret;
}

//
//static PyObject *_cext_tree_predict(PyObject *self, PyObject *args)
//{
//    PyObject *children_left_obj;
//    PyObject *children_right_obj;
//    PyObject *children_default_obj;
//    PyObject *features_obj;
//    PyObject *thresholds_obj;
//    PyObject *values_obj;
//    int max_depth;
//    int tree_limit;
//    PyObject *base_offset_obj;
////    int model_output;
//    PyObject *X_obj;
//    PyObject *X_missing_obj;
////    PyObject *y_obj;
//    PyObject *out_pred_obj;
//
//    /* Parse the input tuple */
//    if (!PyArg_ParseTuple(
//        args, "OOOOOOiiOOOO", &children_left_obj, &children_right_obj, &children_default_obj,
//        &features_obj, &thresholds_obj, &values_obj, &max_depth, &tree_limit, &base_offset_obj,
//        &X_obj, &X_missing_obj, &out_pred_obj
//    )) return NULL;
//
//    /* Interpret the input objects as numpy arrays. */
//    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *base_offset_array = (PyArrayObject*)PyArray_FROM_OTF(base_offset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
////    PyArrayObject *y_array = NULL;
////    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *out_pred_array = (PyArrayObject*)PyArray_FROM_OTF(out_pred_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
//
//    /* If that didn't work, throw an exception. Note that R and y are optional. */
//    if (children_left_array == NULL || children_right_array == NULL ||
//        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
//        values_array == NULL || X_array == NULL ||
//        X_missing_array == NULL || out_pred_array == NULL) {
//        Py_XDECREF(children_left_array);
//        Py_XDECREF(children_right_array);
//        Py_XDECREF(children_default_array);
//        Py_XDECREF(features_array);
//        Py_XDECREF(thresholds_array);
//        Py_XDECREF(values_array);
//        Py_XDECREF(base_offset_array);
//        Py_XDECREF(X_array);
//        Py_XDECREF(X_missing_array);
////        if (y_array != NULL) Py_XDECREF(y_array);
//        //PyArray_ResolveWritebackIfCopy(out_pred_array);
//        Py_XDECREF(out_pred_array);
//        return NULL;
//    }
//
//    const unsigned num_X = PyArray_DIM(X_array, 0);
//    const unsigned M = PyArray_DIM(X_array, 1);
//    const unsigned max_nodes = PyArray_DIM(values_array, 1);
//    const unsigned num_outputs = PyArray_DIM(values_array, 2);
//
//    const unsigned num_offsets = PyArray_DIM(base_offset_array, 0);
//    if (num_offsets != num_outputs) {
//        std::cerr << "The passed base_offset array does that have the same number of outputs as the values array: " << num_offsets << " vs. " << num_outputs << std::endl;
//        return NULL;
//    }
//
//    // Get pointers to the data as C-types
//    int *children_left = (int*)PyArray_DATA(children_left_array);
//    int *children_right = (int*)PyArray_DATA(children_right_array);
//    int *children_default = (int*)PyArray_DATA(children_default_array);
//    int *features = (int*)PyArray_DATA(features_array);
//    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
//    tfloat *values = (tfloat*)PyArray_DATA(values_array);
//    tfloat *base_offset = (tfloat*)PyArray_DATA(base_offset_array);
//    tfloat *X = (tfloat*)PyArray_DATA(X_array);
//    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
////    tfloat *y = NULL;
////    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
//    tfloat *out_pred = (tfloat*)PyArray_DATA(out_pred_array);
//
//    // these are just wrapper objects for all the pointers and numbers associated with
//    // the ensemble tree model and the datset we are explaing
//    TreeEnsemble trees = TreeEnsemble(
//        children_left, children_right, children_default, features, thresholds, values,
//        NULL, max_depth, tree_limit, base_offset,
//        max_nodes, num_outputs
//    );
////    ExplanationDataset data = ExplanationDataset(X, X_missing, NULL, NULL, num_X, M, 0);
//
//    tree_predict_proba(out_pred, trees, X, X_missing);
//
//    // clean up the created python objects
//    Py_XDECREF(children_left_array);
//    Py_XDECREF(children_right_array);
//    Py_XDECREF(children_default_array);
//    Py_XDECREF(features_array);
//    Py_XDECREF(thresholds_array);
//    Py_XDECREF(values_array);
//    Py_XDECREF(base_offset_array);
//    Py_XDECREF(X_array);
//    Py_XDECREF(X_missing_array);
////    if (y_array != NULL) Py_XDECREF(y_array);
//    //PyArray_ResolveWritebackIfCopy(out_pred_array);
//    Py_XDECREF(out_pred_array);
//
//    /* Build the output tuple */
//    PyObject *ret = Py_BuildValue("d", (double)values[0]);
//    return ret;
//}


static PyObject *_cext_single_tree_predict(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int max_depth;
    int tree_limit;
    int j;
    PyObject *base_offset_obj;
//    int model_output;
    PyObject *X_obj;
    PyObject *X_missing_obj;
//    PyObject *y_obj;
    PyObject *out_pred_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiiOOOOi", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &max_depth, &tree_limit, &base_offset_obj,
        &X_obj, &X_missing_obj, &out_pred_obj, &j
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *base_offset_array = (PyArrayObject*)PyArray_FROM_OTF(base_offset_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);
//    PyArrayObject *y_array = NULL;
//    if (y_obj != Py_None) y_array = (PyArrayObject*)PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *out_pred_array = (PyArrayObject*)PyArray_FROM_OTF(out_pred_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);

    /* If that didn't work, throw an exception. Note that R and y are optional. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || X_array == NULL ||
        X_missing_array == NULL || out_pred_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        Py_XDECREF(base_offset_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
//        if (y_array != NULL) Py_XDECREF(y_array);
        //PyArray_ResolveWritebackIfCopy(out_pred_array);
        Py_XDECREF(out_pred_array);
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);
    const unsigned num_outputs = PyArray_DIM(values_array, 2);

    const unsigned num_offsets = PyArray_DIM(base_offset_array, 0);
    if (num_offsets != num_outputs) {
        std::cerr << "The passed base_offset array does that have the same number of outputs as the values array: " << num_offsets << " vs. " << num_outputs << std::endl;
        return NULL;
    }

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *base_offset = (tfloat*)PyArray_DATA(base_offset_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);
//    tfloat *y = NULL;
//    if (y_array != NULL) y = (tfloat*)PyArray_DATA(y_array);
    tfloat *out_pred = (tfloat*)PyArray_DATA(out_pred_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        NULL, max_depth, tree_limit, base_offset,
        max_nodes, num_outputs
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, NULL, NULL, num_X, M, 0);

    single_tree_predict(out_pred, trees, data, j);

    // clean up the created python objects
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    Py_XDECREF(base_offset_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);
//    if (y_array != NULL) Py_XDECREF(y_array);
    //PyArray_ResolveWritebackIfCopy(out_pred_array);
    Py_XDECREF(out_pred_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", (double)values[0]);
    return ret;
}

static PyObject *_cext_dense_tree_update_weights(PyObject *self, PyObject *args)
{
    PyObject *children_left_obj;
    PyObject *children_right_obj;
    PyObject *children_default_obj;
    PyObject *features_obj;
    PyObject *thresholds_obj;
    PyObject *values_obj;
    int tree_limit;
    PyObject *node_sample_weight_obj;
    PyObject *X_obj;
    PyObject *X_missing_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(
        args, "OOOOOOiOOO", &children_left_obj, &children_right_obj, &children_default_obj,
        &features_obj, &thresholds_obj, &values_obj, &tree_limit, &node_sample_weight_obj, &X_obj, &X_missing_obj
    )) return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyArrayObject *children_left_array = (PyArrayObject*)PyArray_FROM_OTF(children_left_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_right_array = (PyArrayObject*)PyArray_FROM_OTF(children_right_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *children_default_array = (PyArrayObject*)PyArray_FROM_OTF(children_default_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *features_array = (PyArrayObject*)PyArray_FROM_OTF(features_obj, NPY_INT, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *thresholds_array = (PyArrayObject*)PyArray_FROM_OTF(thresholds_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject*)PyArray_FROM_OTF(values_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *node_sample_weight_array = (PyArrayObject*)PyArray_FROM_OTF(node_sample_weight_obj, NPY_DOUBLE, NPY_ARRAY_INOUT_ARRAY);
    PyArrayObject *X_array = (PyArrayObject*)PyArray_FROM_OTF(X_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *X_missing_array = (PyArrayObject*)PyArray_FROM_OTF(X_missing_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (children_left_array == NULL || children_right_array == NULL ||
        children_default_array == NULL || features_array == NULL || thresholds_array == NULL ||
        values_array == NULL || node_sample_weight_array == NULL || X_array == NULL ||
        X_missing_array == NULL) {
        Py_XDECREF(children_left_array);
        Py_XDECREF(children_right_array);
        Py_XDECREF(children_default_array);
        Py_XDECREF(features_array);
        Py_XDECREF(thresholds_array);
        Py_XDECREF(values_array);
        //PyArray_ResolveWritebackIfCopy(node_sample_weight_array);
        Py_XDECREF(node_sample_weight_array);
        Py_XDECREF(X_array);
        Py_XDECREF(X_missing_array);
        std::cerr << "Found a NULL input array in _cext_dense_tree_update_weights!\n";
        return NULL;
    }

    const unsigned num_X = PyArray_DIM(X_array, 0);
    const unsigned M = PyArray_DIM(X_array, 1);
    const unsigned max_nodes = PyArray_DIM(values_array, 1);

    // Get pointers to the data as C-types
    int *children_left = (int*)PyArray_DATA(children_left_array);
    int *children_right = (int*)PyArray_DATA(children_right_array);
    int *children_default = (int*)PyArray_DATA(children_default_array);
    int *features = (int*)PyArray_DATA(features_array);
    tfloat *thresholds = (tfloat*)PyArray_DATA(thresholds_array);
    tfloat *values = (tfloat*)PyArray_DATA(values_array);
    tfloat *node_sample_weight = (tfloat*)PyArray_DATA(node_sample_weight_array);
    tfloat *X = (tfloat*)PyArray_DATA(X_array);
    bool *X_missing = (bool*)PyArray_DATA(X_missing_array);

    // these are just wrapper objects for all the pointers and numbers associated with
    // the ensemble tree model and the datset we are explaing
    TreeEnsemble trees = TreeEnsemble(
        children_left, children_right, children_default, features, thresholds, values,
        node_sample_weight, 0, tree_limit, 0, max_nodes, 0
    );
    ExplanationDataset data = ExplanationDataset(X, X_missing, NULL, NULL, num_X, M, 0);

    dense_tree_update_weights(trees, data);

    // clean up the created python objects
    Py_XDECREF(children_left_array);
    Py_XDECREF(children_right_array);
    Py_XDECREF(children_default_array);
    Py_XDECREF(features_array);
    Py_XDECREF(thresholds_array);
    Py_XDECREF(values_array);
    //PyArray_ResolveWritebackIfCopy(node_sample_weight_array);
    Py_XDECREF(node_sample_weight_array);
    Py_XDECREF(X_array);
    Py_XDECREF(X_missing_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", 1);
    return ret;
}
