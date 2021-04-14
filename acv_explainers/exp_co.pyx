import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from scipy.special import comb
import itertools
from cython.parallel cimport prange, parallel

cdef extern from "limits.h":
    unsigned long ULONG_MAX


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]

    if S_size == m:
        return np.ones(shape=(N, values.shape[2]))
    elif S_size == 0:
        return np.zeros(shape=(N, values.shape[2]))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, d))

    cdef unsigned int it, it_s, nb_leaf, lm, p_ss, o_all
    cdef Py_ssize_t b, leaf_numb, i, s, j

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]
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
                for j in range(d):
                    mean_forest[i, j] += (lm * value[j]) / p_ss if p_ss != 0 else 0

    return mean_forest / n_trees



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees,  np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]

    if S_size == m:
        return np.ones(shape=(N, values.shape[2]))
    elif S_size == 0:
        return np.zeros(shape=(N, values.shape[2]))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, d))

    cdef unsigned int it, it_s, p_ss, o_all, nb_leaf, lm
    cdef Py_ssize_t b, leaf_numb, i, s, j1, j

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                lm = 0
                p_ss = 0
                for j in range(data.shape[0]):

                    it = 0
                    it_s = 0
                    for s in range(m):
                        if((data[j, s] <= leaf_part[s, 1]) and (data[j, s] >= leaf_part[s, 0])):
                            it += 1

                    for s in range(S_size):
                        if(data[j, S[s]] == X[i, S[s]]):
                            it_s += 1

                    if it_s == S_size:
                        p_ss += 1
                        if it == m:
                            lm += 1

                for j1 in range(d):
                    mean_forest[i, j1] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j1]) / (scaling[b]*p_ss) if p_ss != 0 else 0

    return mean_forest / n_trees

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] compute_sdp_clf_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

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
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, down, up
    cdef dtype_t ss
    cdef Py_ssize_t b, leaf_numb, i, s, j

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] compute_sdp_clf(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

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
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up
    cdef dtype_t ss
    cdef Py_ssize_t b, leaf_numb, i, s, j

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] global_sdp_clf(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling, float global_proba):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

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
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef dtype_t ss
    cdef Py_ssize_t b, leaf_numb, i, s, s_1, S_size, j

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
                value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
            ss = (mean_forest[R[i], 0] - mean_forest[R[i], 2])/(n_trees*(mean_forest[R[i], 1] - mean_forest[R[i], 2])) if mean_forest[R[i], 1] - mean_forest[R[i], 2] !=0 else 0
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] global_sdp_clf_coal(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling,
            float global_proba, list C):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

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
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s , p_u, p_d, p_su, p_sd, ci, cj, nb_leaf, down, up, S_size
    cdef dtype_t ss
    cdef Py_ssize_t b, leaf_numb, i, s, s_1, j
    cdef np.ndarray[long, ndim=1] S
    cdef list power, va_id, R, r

    R = list(range(N))
    if C[0] != []:
        remove_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    power = [co for co in powerset(va_id)]

    S = np.zeros((m), dtype=np.int)
    for s_1 in range(1, len(power)-1):
        S_size = 0
        for ci in range(len(power[s_1])):
            for cj in range(len(power[s_1][ci])):
                S[S_size] = power[s_1][ci][cj]
                S_size += 1

        r = []
        N = len(R)
        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees):
            leaves_tree = partition_leaves_trees[b]
            nb_leaf = leaves_nb[b]

            for leaf_numb in range(nb_leaf):
                leaf_part = leaves_tree[leaf_numb]
                value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
            ss = (mean_forest[R[i], 0] - mean_forest[R[i], 2])/(n_trees*(mean_forest[R[i], 1] - mean_forest[R[i], 2])) if mean_forest[R[i], 1] - mean_forest[R[i], 2] !=0 else 0
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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] global_sdp_clf_dist(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling,
            float global_proba):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    # cdef np.ndarray[dtype_t, ndim=1] sdp
    cdef np.ndarray[dtype_t, ndim=1] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef dtype_t p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up, s_1, set_size, pow_set_size, counter, ci, cj, ite, S_size
    cdef list power, va_id, R, r
    cdef np.ndarray[long, ndim=1] S

    va_id = [[i] for i in range(m)]
    set_size = len(va_id)
    m = len(va_id)
    S = -np.ones(shape=(m), dtype=np.int)
    pow_set_size = 2**set_size-1

    for counter in range(1, pow_set_size):
        S_size = 0
        for ci in range(set_size):
            if((counter & (1 << ci)) > 0):
                for cj in range(len(va_id[ci])):
                    S[S_size] = va_id[ci][cj]
                    S_size += 1

        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees):
            leaves_tree = partition_leaves_trees[b]
            nb_leaf = leaves_nb[b]

            for leaf_numb in range(nb_leaf):
                leaf_part = leaves_tree[leaf_numb]
                value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            if ss >= global_proba:
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

    return sdp_global/(X.shape[0] * pow_set_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] global_sdp_clf_dist_coal(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling,
            float global_proba, list C):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    # cdef np.ndarray[dtype_t, ndim=1] sdp
    cdef np.ndarray[dtype_t, ndim=1] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef dtype_t p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up, s_1, set_size, pow_set_size, counter, ci, cj, ite
    cdef list power, va_id, R, r

    cdef np.ndarray[long, ndim=1] S
    S = -np.ones(shape=(m), dtype=np.int)

    if C[0] != []:
        remove_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    set_size = len(va_id)
    pow_set_size = 2**set_size-1

    for counter in range(1, pow_set_size):
        S_size = 0
        for ci in range(set_size):
            if((counter & (1 << ci)) > 0):
                for cj in range(len(va_id[ci])):
                    S[S_size] = va_id[ci][cj]
                    S_size += 1

        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees):
            leaves_tree = partition_leaves_trees[b]
            nb_leaf = leaves_nb[b]

            for leaf_numb in range(nb_leaf):
                leaf_part = leaves_tree[leaf_numb]
                value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling[b]

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
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            if ss >= global_proba:
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

    return sdp_global/(X.shape[0] * pow_set_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] compute_sdp_reg(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=1] fX, dtype_t tX,
            np.ndarray[dtype_t, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree_l
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=2] leaf_part_l
    cdef dtype_t value
    cdef dtype_t value_l

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef np.ndarray[dtype_t, ndim=1] sdp
    sdp = np.zeros((N))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, nb_leaf_l, o_all, down, up
    cdef dtype_t ss
    cdef Py_ssize_t b, l, leaf_numb, i, s, j, leaf_numb_l

    for b in range(n_trees):
        for l in range(n_trees):
            if b == l:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]

                for leaf_numb in range(nb_leaf):
                    leaf_part = leaves_tree[leaf_numb]
                    value = values[b, leaf_idx_trees[b, leaf_numb], 0] / scaling[b]

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
                            if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                p_u += lm[j]
                                p_su += lm_s[j]
                            else:
                                p_d += lm[j]
                                p_sd += lm_s[j]

                        mean_forest[i, 0] += (p * value*value) / (p_s * n_trees*n_trees) - (2 * fX[i] * p * value)/(p_s * n_trees) if p_s != 0 else 0
                        mean_forest[i, 1] += (p_u * value*value) / (p_su * n_trees*n_trees) - (2 * fX[i] * p_u * value)/(p_su * n_trees) if p_su != 0 else 0
                        mean_forest[i, 2] += (p_d * value*value) / (p_sd * n_trees*n_trees) - (2 * fX[i] * p_d * value)/(p_sd * n_trees) if p_sd != 0 else 0
            else:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]
                leaves_tree_l = partition_leaves_trees[l]
                nb_leaf_l = leaves_nb[l]

                for leaf_numb in range(nb_leaf):
                    for leaf_numb_l in range(nb_leaf_l):

                        leaf_part = leaves_tree[leaf_numb]
                        leaf_part_l = leaves_tree_l[leaf_numb_l]
                        value = values[b, leaf_idx_trees[b, leaf_numb], 0] / scaling[b]
                        value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0] / scaling[l]

                        lm = np.zeros(data.shape[0], dtype=np.int)
                        lm_s = np.zeros(data.shape[0], dtype=np.int)

                        for i in range(data.shape[0]):
                            a_it = 0
                            b_it = 0
                            for s in range(m):
                                if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                    a_it += 1
                            for s in range(S_size):
                                if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                    b_it +=1

                            if a_it == m:
                                lm[i] = 1

                            if b_it == S_size:
                                lm_s[i] = 1

                        for i in range(N):
                            o_all = 0
                            for s in range(S_size):
                                if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0]) or (X[i, S[s]] > leaf_part_l[S[s], 1]) or (X[i, S[s]] < leaf_part_l[S[s], 0])):
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
                                if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                    p_u += lm[j]
                                    p_su += lm_s[j]
                                else:
                                    p_d += lm[j]
                                    p_sd += lm_s[j]

                            mean_forest[i, 0] += (p * value*value_l) / (p_s * n_trees*n_trees)  if p_s != 0 else 0
                            mean_forest[i, 1] += (p_u * value*value_l) / (p_su * n_trees*n_trees)  if p_su != 0 else 0
                            mean_forest[i, 2] += (p_d * value*value_l) / (p_sd * n_trees*n_trees) if p_sd != 0 else 0

    for i in range(N):
        ss = (mean_forest[i, 1] - mean_forest[i, 0])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[long, ndim=1] compute_sdp_reg_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=1] fX, dtype_t tX,
            np.ndarray[dtype_t, ndim=1] y_pred, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree_l
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=2] leaf_part_l
    cdef dtype_t value
    cdef dtype_t value_l

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef np.ndarray[dtype_t, ndim=1] sdp
    sdp = np.zeros((N))

    cdef np.ndarray[long, ndim=1] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, nb_leaf_l, o_all, down, up
    cdef dtype_t ss
    cdef Py_ssize_t b, l, leaf_numb, i, s, j, leaf_numb_l

    for b in range(n_trees):
        for l in range(n_trees):
            if b == l:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]

                for leaf_numb in range(nb_leaf):
                    leaf_part = leaves_tree[leaf_numb]
                    value = values[b, leaf_idx_trees[b, leaf_numb], 0] / scaling[b]

                    lm = np.zeros(data.shape[0], dtype=np.int)

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
                                    if (fX[i] - y_pred[j]) * (fX[i] - y_pred[j]) > tX:
                                        up = 1
                                    else:
                                        down = 1

                                    p += lm[j]
                                    p_u += lm[j] * up
                                    p_d += lm[j] * down
                                    p_s += 1
                                    p_su += up
                                    p_sd += down

                        mean_forest[i, 0] += (p * value*value) / (p_s * n_trees*n_trees) - (2 * fX[i] * p * value)/(p_s * n_trees) if p_s != 0 else 0
                        mean_forest[i, 1] += (p_u * value*value) / (p_su * n_trees*n_trees) - (2 * fX[i] * p_u * value)/(p_su * n_trees) if p_su != 0 else 0
                        mean_forest[i, 2] += (p_d * value*value) / (p_sd * n_trees*n_trees) - (2 * fX[i] * p_d * value)/(p_sd * n_trees) if p_sd != 0 else 0
            else:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]
                leaves_tree_l = partition_leaves_trees[l]
                nb_leaf_l = leaves_nb[l]

                for leaf_numb in range(nb_leaf):
                    for leaf_numb_l in range(nb_leaf_l):

                        leaf_part = leaves_tree[leaf_numb]
                        leaf_part_l = leaves_tree_l[leaf_numb_l]
                        value = values[b, leaf_idx_trees[b, leaf_numb], 0] / scaling[b]
                        value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0] / scaling[l]

                        lm = np.zeros(data.shape[0], dtype=np.int)
                        lm_s = np.zeros(data.shape[0], dtype=np.int)

                        for i in range(data.shape[0]):
                            a_it = 0
                            b_it = 0
                            for s in range(m):
                                if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                    a_it += 1

                            if a_it == m:
                                lm[i] = 1

                        for i in range(N):
                            o_all = 0
                            for s in range(S_size):
                                if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0]) or (X[i, S[s]] > leaf_part_l[S[s], 1]) or (X[i, S[s]] < leaf_part_l[S[s], 0])):
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
                                    if (fX[i] - y_pred[j]) * (fX[i] - y_pred[j]) > tX:
                                        up = 1
                                    else:
                                        down = 1

                                    p += lm[j]
                                    p_u += lm[j] * up
                                    p_d += lm[j] * down
                                    p_s += 1
                                    p_su += up
                                    p_sd += down

                            mean_forest[i, 0] += (p * value*value_l) / (p_s * n_trees*n_trees)  if p_s != 0 else 0
                            mean_forest[i, 1] += (p_u * value*value_l) / (p_su * n_trees*n_trees)  if p_su != 0 else 0
                            mean_forest[i, 2] += (p_d * value*value_l) / (p_sd * n_trees*n_trees) if p_sd != 0 else 0

    for i in range(N):
        ss = (mean_forest[i, 1] - mean_forest[i, 0])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef shap_values_leaves(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, double[:] scaling,
            list node_idx_trees, list C):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef np.ndarray[dtype_t, ndim=3] phi
    phi = np.zeros((N, m, d))


    cdef unsigned int S_size
    cdef Py_ssize_t n_trees = values.shape[0]
    cdef np.ndarray[dtype_t, ndim=3] leaves_tree
    cdef np.ndarray[dtype_t, ndim=2] leaf_part
    cdef np.ndarray[dtype_t, ndim=1] value
    cdef np.ndarray[long, ndim=1] S

    S = -np.ones((m), dtype=np.int)

    cdef unsigned int a_it, b_it, o_all, nb_leaf, cs, csi, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add, nv_bool
    cdef Py_ssize_t b, leaf_numb, i, s, j, i1, i2
    cdef list va_id, node_id, Sm, coal_va, remove_va, C_b, node_id_v2
    cdef dtype_t coef, p_s, p_si, lm, lm_s, lm_si


    if C[0] != []:
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        remove_va = [[i] for i in range(m) if i not in coal_va]
        va_id = remove_va + C
        m = len(va_id)
    else:
        va_id = [[i] for i in range(m)]


    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in range(nb_leaf):
            leaf_part = partition_leaves_trees[b, leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]]
            node_id = node_idx_trees[b][leaf_numb]
            node_id_v2 = []

            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] > leaf_part[s, 0])):
                        a_it += 1
                if a_it == data.shape[1]:
                    lm += 1

            if C[0] != []:
                C_b = C.copy()
                for nv in range(len(node_id)):
                    add = 0
                    for ns in range(len(remove_va)):
                        if node_id[nv] == remove_va[ns]:
                            add = 1
                            node_id_v2 += [[node_id[nv]]]
                            continue
                    if add == 0:
                        for ci in range(len(C_b)):
                            for cj in range(len(C_b[ci])):
                                if C_b[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += [C_b[ci]]
                                    continue
                            if add == 1:
                                C_b.remove(C_b[ci])
                                continue

                node_id = node_id_v2
            else:
                node_id = [[i] for i in node_id]

            for va in range(len(va_id)):
                nv_bool = 0
                for i in range(len(node_id)):
                    if va_id[va] == node_id[i]:
                        nv_bool += 1
                        continue

                if nv_bool == 0:
                    continue

                Sm = node_id.copy()
                Sm.remove(va_id[va])

                set_size = len(Sm)
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    S_size = 0
                    va_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(len(Sm[ci])):
                                S[S_size] = Sm[ci][cj]
                                S_size += 1
                            va_size += 1

                    lm_s = 0
                    lm_si = 0
                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] > leaf_part[S[s], 0])):
                                b_it +=1

                        if b_it == S_size:
                            lm_s += 1

                            nv_bool = 0
                            for nv in range(len(va_id[va])):
                                if ((data[i, va_id[va][nv]] > leaf_part[va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= leaf_part[va_id[va][nv], 0])):
                                    nv_bool += 1
                                    continue

                            if nv_bool == 0:
                                lm_si += 1

                    for i in range(N):

                        p_s = 0
                        p_si = 0

                        cs = 0
                        csi = 0

                        o_all = 0
                        for s in range(S_size):
                            if ((X[i, S[s]] <= leaf_part[S[s], 1]) and (X[i, S[s]] > leaf_part[S[s], 0])):
                                o_all +=1

                        if o_all == S_size:
                            cs = 1
                            nv_bool = 0
                            for nv in range(len(va_id[va])):
                                if ((X[i, va_id[va][nv]] > leaf_part[va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= leaf_part[va_id[va][nv], 0])):
                                    nv_bool += 1
                                    continue

                            if nv_bool == 0:
                                csi = 1

                        coef = 0
                        for l in range(1, m - len(Sm)):
                            coef += binomialC(m - len(Sm) - 1, l)/binomialC(m - 1, l + va_size)

                        if S_size == 0:
                            p_s = lm/data.shape[0]
                        else:
                            p_s = (cs * lm)/lm_s

                        p_si = (csi * lm)/lm_si
                        coef_0 = 1/binomialC(m-1, va_size)
                        for nv in range(len(va_id[va])):
                            for i2 in range(d):
                                phi[i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * value[i2]

    return phi / m


cdef unsigned long binomialC(unsigned long N, unsigned long k) nogil:
    cdef unsigned long r
    r = _comb_int_long(N, k)
    if r != 0:
        return r


cdef unsigned long _comb_int_long(unsigned long N, unsigned long k) nogil:
    """
    Compute binom(N, k) for integers.
    Returns 0 if error/overflow encountered.
    """
    cdef unsigned long val, j, M, nterms

    if k > N or N == ULONG_MAX:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        if val > ULONG_MAX // (M - j):
            return 0

        val *= M - j
        val //= j

    return val




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
# @cython.cdivision(True)
def shap_values_leaves_pa(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    double[:] scaling,
    list node_idx_trees,
    list C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))


#     cdef unsigned int S_size
    cdef unsigned int n_trees = values.shape[0]
    cdef long[::1] S
    S = np.zeros((m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef list va_id, node_id, Sm, coal_va, remove_va, C_b, node_id_v2
    cdef double lm

    cdef long[:] va_c
    cdef int len_va_c, S_size, va_size,  b_it, nv_bool
    cdef long set_size
    va_c = np.zeros((m), dtype=np.int)

    if C[0] != []:
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        remove_va = [[i] for i in range(m) if i not in coal_va]
        va_id = remove_va + C
        m = len(va_id)

    else:
        va_id = [[i] for i in range(m)]

    cdef long[:, :] sm_buf
    sm_buf = np.zeros((2**m, data.shape[1]), dtype=np.int)

    cdef long[:] sm_size
    sm_size = np.zeros((2**m), dtype=np.int)

    cdef double p_s, p_si, lm_s, lm_si, coef, coef_0

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in range(nb_leaf):
            node_id = node_idx_trees[b][leaf_numb]
            node_id_v2 = []

            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it += 1
                if a_it == data.shape[1]:
                    lm += 1

            if C[0] != []:
                C_b = C.copy()
                for nv in range(len(node_id)):
                    add = 0
                    for ns in range(len(remove_va)):
                        if node_id[nv] == remove_va[ns]:
                            add = 1
                            node_id_v2 += [[node_id[nv]]]
                            continue
                    if add == 0:
                        for ci in range(len(C_b)):
                            for cj in range(len(C_b[ci])):
                                if C_b[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += [C_b[ci]]
                                    continue
                            if add == 1:
                                C_b.remove(C_b[ci])
                                continue

                node_id = node_id_v2
            else:
                node_id = [[i] for i in node_id]

            for va in range(len(va_id)):
                na_bool = 0
                for i in range(len(node_id)):
                    if va_id[va] == node_id[i]:
                        na_bool += 1
                        continue

                if na_bool == 0:
                    continue

                len_va_c = len(va_id[va])
                for i in range(len_va_c):
                    va_c[i] = va_id[va][i]


                Sm = node_id.copy()
                Sm.remove(va_id[va])

                set_size = len(Sm)
                pow_set_size = 2**set_size

                for i in range(set_size):
                    sm_size[i] = len(Sm[i])
                    for j in range(len(Sm[i])):
                        sm_buf[i, j] = Sm[i][j]

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(sm_size[ci]):
                                S[S_size] = sm_buf[ci, cj]
                                S_size += 1
                            va_size += 1

                    lm_s = 0
                    lm_si = 0

                    for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size:
                            lm_s += 1

                            nv_bool = 0
                            for nv in range(len_va_c):
                                if ((data[i, va_c[nv]] > partition_leaves_trees[b, leaf_numb, va_c[nv], 1]) or (data[i, va_c[nv]] <= partition_leaves_trees[b, leaf_numb, va_c[nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si += 1

                    for i in prange(N, nogil=True, num_threads=num_threads):

                        p_s = 0
                        p_si = 0

                        o_all = 0
                        csi = 0
                        cs = 0

                        for s in range(S_size):
                            if ((X[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                o_all = o_all + 1

                        if o_all == S_size:
                            cs = 1
                            nv_bool = 0
                            for nv in range(len_va_c):
                                if ((X[i, va_c[nv]] > partition_leaves_trees[b, leaf_numb, va_c[nv], 1]) or (X[i, va_c[nv]] <= partition_leaves_trees[b, leaf_numb, va_c[nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                csi = 1

                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0

                        if S_size == 0:
                            p_s = lm/data.shape[0]
                        else:
                            p_s = (cs * lm)/lm_s

                        p_si = (csi * lm)/lm_si
                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[i, va_c[nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

    return np.asarray(phi)/m