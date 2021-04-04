import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from scipy.special import comb
import itertools


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]
    cdef int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N, values.shape[2]))
    elif S_size == 0:
        return np.zeros(shape=(N, values.shape[2]))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef dtype_t value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, values.shape[2]))

    cdef int it, it_s
    cdef dtype_t lm, p_ss
    cdef unsigned int b, leaf_numb, nb_leaf, i, s

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling
            lm = 0
            p_ss = 0
            for i in range(data.shape[0]):
                it = 0
                it_s = 0
                for s in range(m):
                    if((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                        it += 1
                for s in range(S_size):
                    if((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                        it_s += 1
                if it == m:
                    lm += 1
                if it_s == S_size:
                    p_ss += 1

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                mean_forest[i] += (lm * value) / p_ss if p_ss != 0 else 0

    return mean_forest / n_trees


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]
    cdef int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef dtype_t value

    cdef np.ndarray[dtype_t, ndim=1] mean_forest
    mean_forest = np.zeros((N))

    cdef np.ndarray[long, ndim=1] lm
    lm = np.zeros(data.shape[0], dtype=np.int)

    cdef int it, it_s, a_it, b_it
    cdef dtype_t p, p_s
    cdef unsigned int b, leaf_numb, nb_leaf, i, s

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb], 0] / scaling

            for i in range(data.shape[0]):
                a_it = 0
                for s in range(m):
                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                        a_it += 1
                if a_it == m:
                    lm[i] = 1
                else:
                    lm[i] = 0

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                p = 0
                p_s = 0
                for j in range(data.shape[0]):
                    b_it = 0
                    for s in range(S_size):
                        if data[j, S[s]] == X[i, S[s]]:
                            b_it += 1
                    if b_it == S_size:
                        p += lm[i]
                        p_s += 1

                mean_forest[i] += (p * value) / p_s if p_s != 0 else 0

    return mean_forest / n_trees

cpdef np.ndarray[long, ndim=1] compute_sdp_clf_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]
    cdef int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef np.ndarray[dtype_t, ndim=1] sdp
    sdp = np.zeros((N))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d
    cdef int it, it_s, a_it, b_it, o_all
    cdef dtype_t p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

            lm = np.zeros(data.shape[0], dtype=np.int)

            for i in range(data.shape[0]):
                a_it = 0
                for s in range(m):
                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                        a_it += 1
                if a_it == m:
                    lm[i] = 1

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                p = 0
                p_u = 0
                p_d = 0
                p_s = 0
                p_su = 0
                p_sd = 0

                for j in range(data.shape[0]):
                    up = 0
                    down = 0
                    b_it = 0
                    for s in range(S_size):
                        if data[j, S[s]] == X[i, S[s]]:
                            b_it += 1

                    if b_it == S_size:
                        if fX[i] == y_pred[j]:
                            up = 1
                        else:
                            down = 1

                        p += lm[j]
                        p_u += lm[j] * up
                        p_d += lm[j] * down
                        p_s += 1
                        p_su += up
                        p_sd += down

                mean_forest[i, 0] += (p * value[fX[i]]) / p_s if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * value[fX[i]]) / p_su if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * value[fX[i]]) / p_sd if p_sd != 0 else 0

    for i in range(N):
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2]))
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp

cpdef np.ndarray[long, ndim=1] compute_sdp_clf(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]
    cdef int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef np.ndarray[dtype_t, ndim=1] sdp
    sdp = np.zeros((N))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef dtype_t p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

            lm = np.zeros(data.shape[0], dtype=np.int)
            lm_s = np.zeros(data.shape[0], dtype=np.int)

            for i in range(data.shape[0]):
                a_it = 0
                b_it = 0
                for s in range(m):
                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                        a_it += 1
                for s in range(S_size):
                    if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                        b_it +=1

                if a_it == m:
                    lm[i] = 1

                if b_it == S_size:
                    lm_s[i] = 1

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                p = 0
                p_u = 0
                p_d = 0
                p_s = 0
                p_su = 0
                p_sd = 0

                for j in range(data.shape[0]):
                    p += lm[j]
                    p_s += lm_s[j]
                    if fX[i] == y_pred[j]:
                        p_u += lm[j]
                        p_su += lm_s[j]
                    else:
                        p_d += lm[j]
                        p_sd += lm_s[j]

                mean_forest[i, 0] += (p * value[fX[i]]) / p_s if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * value[fX[i]]) / p_su if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * value[fX[i]]) / p_sd if p_sd != 0 else 0

    for i in range(N):
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2]))
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp


cpdef np.ndarray[long, ndim=1] global_sdp_clf(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling, float global_proba):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef np.ndarray[dtype_t, ndim=1] sdp
    cdef np.ndarray[dtype_t, ndim=1] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef dtype_t p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up, s_1
    cdef np.ndarray[long, ndim=1] S
    cdef list power, va_id, R, r


    R = list(range(N))
    va_id = list(range(m))
    power = [np.array(co, dtype=np.int) for co in powerset(va_id)]

    for s_1 in range(1, len(power)-1):
        S = power[s_1]
        S_size = S.shape[0]
        r = []
        N = len(R)
        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees):
            leaves_tree = partition_leaves_trees[b]
            nb_leaf = leaves_nb[b]

            for leaf_numb in range(nb_leaf):
                leaf_part = leaves_tree[leaf_numb]
                value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

                lm = np.zeros(data.shape[0], dtype=np.int)
                lm_s = np.zeros(data.shape[0], dtype=np.int)

                for i in range(data.shape[0]):
                    a_it = 0
                    b_it = 0
                    for s in range(m):
                        if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                            a_it += 1
                    for s in range(S_size):
                        if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                            b_it +=1

                    if a_it == m:
                        lm[i] = 1

                    if b_it == S_size:
                        lm_s[i] = 1

                for i in range(N):
                    o_all = 0
                    for s in range(S_size):
                        if ((X[R[i], S[s]] > leaf_part[S[s], 1]) or (X[R[i], S[s]] < leaf_part[S[s], 0])):
                            o_all +=1
                    if o_all > 0:
                        continue

                    p = 0
                    p_u = 0
                    p_d = 0
                    p_s = 0
                    p_su = 0
                    p_sd = 0
                    for j in range(data.shape[0]):
                        p += lm[j]
                        p_s += lm_s[j]
                        if fX[R[i]] == y_pred[j]:
                            p_u += lm[j]
                            p_su += lm_s[j]
                        else:
                            p_d += lm[j]
                            p_sd += lm_s[j]

                    mean_forest[R[i], 0] += (p * value[fX[R[i]]]) / p_s if p_s != 0 else 0
                    mean_forest[R[i], 1] += (p_u * value[fX[R[i]]]) / p_su if p_su != 0 else 0
                    mean_forest[R[i], 2] += (p_d * value[fX[R[i]]]) / p_sd if p_sd != 0 else 0

        for i in range(N):
            ss = (mean_forest[R[i], 0] - mean_forest[R[i], 2])/(n_trees*(mean_forest[R[i], 1] - mean_forest[R[i], 2]))
            if((ss <= 1) and (ss>=0)):
                sdp[R[i]] = ss
                if ss >= global_proba:
                    r.append(R[i])
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

            elif(ss > 1):
                sdp[R[i]] = 1
                r.append(R[i])
                for s in range(S_size):
                    sdp_global[S[s]] += 1
            else:
                sdp[R[i]] = 0

        for i in range(len(r)):
            R.remove(r[i])

        if len(R) == 0:
            continue

    return sdp_global/X.shape[0]

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
