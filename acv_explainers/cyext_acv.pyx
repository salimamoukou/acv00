# distutils: language = c++

from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
ctypedef np.float64_t double
cimport cython
from scipy.special import comb
import itertools
from cython.parallel cimport prange, parallel

cdef extern from "<algorithm>" namespace "std" nogil:
     iter std_remove "std::remove" [iter, T](iter first, iter last, const T& val)

cdef extern from "limits.h":
    unsigned long ULONG_MAX



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp(double[:, :] X, long[:] S, double[:, :] data, double[:, :, :] values,
        double[:, :, :, :] partition_leaves_trees, long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
        int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]

    if S_size == m:
        return np.ones(shape=(N, values.shape[2]))
    elif S_size == 0:
        return np.zeros(shape=(N, values.shape[2]))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, d))

    cdef unsigned int it, it_s, nb_leaf, lm, p_ss, o_all
    cdef int b, leaf_numb, i, s, j

    for b in range(n_trees):
        for leaf_numb in range(leaves_nb[b]):
            lm = 0
            p_ss = 0
            for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                it = 0
                it_s = 0
                for s in range(m):
                    if((data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                        it = it + 1
                for s in range(S_size):
                    if((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        it_s = it_s + 1
                if it == m:
                    lm += 1
                if it_s == S_size:
                    p_ss += 1

            for i in prange(N, nogil=True, num_threads=num_threads):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue
                for j in range(d):
                    mean_forest[i, j] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j]) / p_ss if p_ss != 0 else 0

    return np.asarray(mean_forest)



@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_cat(double[:, :] X, long[:] S, double[:, :] data, double[:, :, :] values,
        double[:, :, :, :] partition_leaves_trees, long[:, :] leaf_idx_trees,  long[:] leaves_nb, double[:] scaling,
        int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]

    if S_size == m:
        return np.ones(shape=(N, values.shape[2]))
    elif S_size == 0:
        return np.zeros(shape=(N, values.shape[2]))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, d))

    cdef unsigned int it, it_s, p_ss, o_all, nb_leaf, lm
    cdef int b, leaf_numb, i, s, j1, j

    for b in range(n_trees ):

        for leaf_numb in range(leaves_nb[b]):

            for i in prange(N, nogil=True, num_threads=num_threads):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue

                lm = 0
                p_ss = 0
                for j in range(data.shape[0]):

                    it = 0
                    it_s = 0
                    for s in range(m):
                        if((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                            it = it +  1

                    for s in range(S_size):
                        if(data[j, S[s]] == X[i, S[s]]):
                            it_s = it_s +  1

                    if it_s == S_size:
                        p_ss += 1
                        if it == m:
                            lm += 1

                for j1 in range(d):
                    mean_forest[i, j1] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j1]) / (p_ss) if p_ss != 0 else 0

    return np.asarray(mean_forest)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_sdp_clf_cat(double[:, :] X, long[:] fX,
            long[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, j, lm

    for b in range(n_trees):

        for leaf_numb in range(leaves_nb[b]):

            for i in prange(N, nogil=True, num_threads=num_threads):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
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
                    lm = 0
                    a_it = 0
                    for s in range(m):
                        if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                            a_it = a_it + 1
                    if a_it == m:
                        lm = 1

                    for s in range(S_size):
                        if data[j, S[s]] == X[i, S[s]]:
                            b_it = b_it + 1

                    if b_it == S_size:
                        if fX[i] == y_pred[j]:
                            up = 1
                        else:
                            down = 1

                        p += lm
                        p_u += lm * up
                        p_d += lm * down
                        p_s += 1
                        p_su += up
                        p_sd += down

                mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_s) if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_su) if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

    for i in prange(N, nogil=True, num_threads=num_threads):
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return np.asarray(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_sdp_clf(double[:, :] X, long[:] fX,
            long[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, j, lm, lm_s

    for b in range(n_trees ):

        for leaf_numb in range(leaves_nb[b]):
            for i in prange(N,  nogil=True, num_threads=num_threads):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue

                p = 0
                p_u = 0
                p_d = 0
                p_s = 0
                p_su = 0
                p_sd = 0

                for j in range(data.shape[0]):
                    a_it = 0
                    b_it = 0
                    lm = 0
                    lm_s = 0
                    for s in range(m):
                        if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                            a_it = a_it + 1
                    for s in range(S_size):
                        if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            b_it = b_it + 1

                    if a_it == m:
                        lm = 1

                    if b_it == S_size:
                        lm_s = 1

                    p += lm
                    p_s += lm_s
                    if fX[i] == y_pred[j]:
                        p_u += lm
                        p_su += lm_s
                    else:
                        p_d += lm
                        p_sd += lm_s

                mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_s) if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_su) if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

    for i in prange(N, nogil=True, num_threads=num_threads):
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef global_sdp_clf(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, double global_proba,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, s_1, S_size, j, lm, lm_s

    cdef long[:] S
    cdef list power, va_id, R, r


    R = list(range(N))
    cdef int[:] R_buf
    R_buf = np.array(R)

    va_id = list(range(m))
    power = [np.array(co, dtype=np.int) for co in powerset(va_id)]

    for s_1 in range(1, len(power)-1):
        S = power[s_1]
        S_size = S.shape[0]
        r = []
        N = len(R)
        for i in range(N):
            R_buf[i] = R[i]
        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees ):

            for leaf_numb in range(leaves_nb[b]):
                for i in range(N):
                    o_all = 0
                    for s in range(S_size):
                        if ((X[R_buf[i], S[s]] > leaf_part[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part[S[s], 0])):
                            o_all = o_all + 1
                    if o_all > 0:
                        continue

                    p = 0
                    p_u = 0
                    p_d = 0
                    p_s = 0
                    p_su = 0
                    p_sd = 0
                    for j in range(data.shape[0]):
                        a_it = 0
                        b_it = 0
                        lm = 0
                        lm_s = 0
                        for s in range(m):
                            if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                a_it = a_it + 1
                        for s in range(S_size):
                            if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it = b_it + 1

                        if a_it == m:
                            lm = 1

                        if b_it == S_size:
                            lm_s = 1

                        p += lm
                        p_s += lm_s
                        if fX[R_buf[i]] == y_pred[j]:
                            p_u += lm
                            p_su += lm_s
                        else:
                            p_d += lm
                            p_sd += lm_s

                    mean_forest[R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] * p_s) if p_s != 0 else 0
                    mean_forest[R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_su) if p_su != 0 else 0
                    mean_forest[R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

        for i in range(N):
            ss = (mean_forest[R_buf[i], 0] - mean_forest[R_buf[i], 2])/(mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2]) if mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2] !=0 else 0
            if((ss <= 1) and (ss>=0)):
                sdp[R_buf[i]] = ss
                if ss >= global_proba:
                    r.append(R[i])
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

            elif(ss > 1):
                sdp[R_buf[i]] = 1
                r.append(R_buf[i])
                for s in range(S_size):
                    sdp_global[S[s]] += 1
            else:
                sdp[R_buf[i]] = 0

        for i in range(len(r)):
            R.remove(r[i])

        if len(R) == 0:
            continue

    return np.asarray(sdp_global)/X.shape[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef global_sdp_clf_coal(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            list C, double global_proba):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s , p_u, p_d, p_su, p_sd, ci, cj, nb_leaf, down, up, S_size
    cdef double ss
    cdef int b, leaf_numb, i, s, s_1, j
    cdef long[:] S
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

                    mean_forest[R[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R[i]]]) /(scaling[b]* p_s) if p_s != 0 else 0
                    mean_forest[R[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R[i]]]) / (scaling[b]*p_su) if p_su != 0 else 0
                    mean_forest[R[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R[i]]]) / (scaling[b]*p_sd) if p_sd != 0 else 0

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

    return np.asarray(sdp_global)/X.shape[0]


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef global_sdp_clf_dist(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            float global_proba):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    # cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef double p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up, s_1, set_size, pow_set_size, counter, ci, cj, ite, S_size
    cdef list power, va_id, R, r
    cdef long[:] S

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

                    mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_s) if p_s != 0 else 0
                    mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_su) if p_su != 0 else 0
                    mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_sd) if p_sd != 0 else 0

        for i in range(N):
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            if ss >= global_proba:
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

    return np.asarray(sdp_global)/(X.shape[0] * pow_set_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef global_sdp_clf_dist_coal(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            float global_proba, list C):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    # cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef int it, it_s, a_it, b_it, o_all
    cdef double p, p_s, ss, p_u, p_d, p_su, p_sd
    cdef unsigned int b, leaf_numb, nb_leaf, i, s, down, up, s_1, set_size, pow_set_size, counter, ci, cj, ite
    cdef list power, va_id, R, r

    cdef long[:] S
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

                    mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_s) if p_s != 0 else 0
                    mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_su) if p_su != 0 else 0
                    mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b]*p_sd) if p_sd != 0 else 0

        for i in range(N):
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(n_trees*(mean_forest[i, 1] - mean_forest[i, 2])) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            if ss >= global_proba:
                    for s in range(S_size):
                        sdp_global[S[s]] += 1

    return np.asarray(sdp_global)/(X.shape[0] * pow_set_size)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_sdp_reg(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, nb_leaf_l, o_all, down, up
    cdef double ss
    cdef int b, l, leaf_numb, i, s, j, leaf_numb_l

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

    return np.asarray(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_sdp_reg_cat(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, nb_leaf_l, o_all, down, up
    cdef double ss
    cdef int b, l, leaf_numb, i, s, j, leaf_numb_l

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

    return np.asarray(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef shap_values_leaves(double[:, :] X, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            list node_idx_trees, list C):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))


    cdef unsigned int S_size
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value
    cdef long[:] S

    S = -np.ones((m), dtype=np.int)

    cdef unsigned int a_it, b_it, o_all, nb_leaf, cs, csi, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add, nv_bool
    cdef int b, leaf_numb, i, s, j, i1, i2
    cdef list va_id, node_id, Sm, coal_va, remove_va, C_b, node_id_v2
    cdef double coef, p_s, p_si, lm, lm_s, lm_si


    if C[0] != []:
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        remove_va = [[i] for i in range(m) if i not in coal_va]
        va_id = remove_va + C
        m = len(va_id)
    else:
        va_id = [[i] for i in range(m)]
        remove_va = [[i] for i in range(m)]


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

    return np.asarray(phi) / m


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
cpdef shap_values_leaves_pa(const double[:, :] X,
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
            for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
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
                            break
                    if add == 0:
                        for ci in range(len(C_b)):
                            for cj in range(len(C_b[ci])):
                                if C_b[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += [C_b[ci]]
                                    break
                            if add == 1:
                                C_b.remove(C_b[ci])
                                break

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
                            p_s = (cs * lm)/lm_s if lm_s !=0 else 0
                        p_si = (csi * lm)/lm_si if lm_si !=0 else 0
                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[i, va_c[nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

    return np.asarray(phi)/m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef shap_values_acv_leaves(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    double[:] scaling,
    list node_idx_trees, list S_star, list N_star,
    list C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef unsigned int n_trees = values.shape[0]
    cdef long[::1] S
    S = np.zeros((m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs, len_n_star
    cdef list va_id, node_id, Sm, coal_va, remove_va, C_b, node_id_v2
    cdef double lm, lm_star

    cdef long[:] va_c, N_star_a
    cdef int len_va_c, S_size, va_size,  b_it, nv_bool
    cdef long set_size
    va_c = np.zeros((m), dtype=np.int)

    N_star_a = np.array(N_star)
    len_n_star = N_star_a.shape[0]
    if C[0] != []:
        C_buff = C.copy()
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = []
        remove_va = []

        for i in S_star:
            if i not in coal_va:
                remove_va.append([i])
                va_id.append([i])
            else:
                for c in C_buff:
                    if i in c:
                        va_id.append(c)
                        C_buff.remove(c)
                        break

    else:
        va_id = [[i] for i in S_star]

    m = len(va_id)

    cdef long[:, :] sm_buf
    sm_buf = np.zeros((data.shape[1], data.shape[1]), dtype=np.int)

    cdef long[:] sm_size
    sm_size = np.zeros((data.shape[1]), dtype=np.int)

    cdef double p_s, p_si, lm_s, lm_si, coef, coef_0, p_off

    for b in range(n_trees):
        for leaf_numb in range(leaves_nb[b]):
            node_id = node_idx_trees[b][leaf_numb]
            node_id = list(set(node_id) - set(N_star))
            node_id_v2 = []

            lm = 0
            for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
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
                            break
                    if add == 0:
                        for ci in range(len(C_b)):
                            for cj in range(len(C_b[ci])):
                                if C_b[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += [C_b[ci]]
                                    break
                            if add == 1:
                                C_b.remove(C_b[ci])
                                break

                node_id = node_id_v2
            else:
                node_id = [[i] for i in node_id]

            for va in range(len(va_id)):
                na_bool = 0
                for i in range(len(node_id)):
                    if va_id[va] == node_id[i]:
                        na_bool += 1
                        continue

                len_va_c = len(va_id[va])
                for i in range(len_va_c):
                    va_c[i] = va_id[va][i]


                if na_bool == 0:
                    lm_star = 0
                    for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                        b_it = 0
                        for s in range(len_n_star):
                            if ((data[i, N_star_a[s]] <= partition_leaves_trees[b, leaf_numb, N_star_a[s], 1]) * (data[i, N_star_a[s]] > partition_leaves_trees[b, leaf_numb, N_star_a[s], 0])):
                                b_it = b_it + 1

                        if b_it == len_n_star:
                            lm_star += 1

                    for i in prange(N, nogil=True, num_threads=num_threads):
                        p_s = 0
                        p_si = 0

                        o_all = 0
                        csi = 0
                        for s in range(len_n_star):
                            if ((X[i, N_star_a[s]] <= partition_leaves_trees[b, leaf_numb, N_star_a[s], 1]) * (X[i, N_star_a[s]] > partition_leaves_trees[b, leaf_numb, N_star_a[s], 0])):
                                o_all = o_all + 1

                        if o_all == len_n_star:
                            csi = 1

                        p_s = lm/data.shape[0]
                        p_si = (csi * lm)/lm_star
                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[i, va_c[nv], i2] += (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]
                    continue


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
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s = 0
                    lm_si = 0

                    for nv in range(len_n_star):
                        S[S_size+nv] = N_star_a[nv]

                    for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                        b_it = 0
                        for s in range(S_size + len_n_star):
                            if ((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size + len_n_star:
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


                        csi = 0
                        cs = 0
                        o_all = 0
                        for s in range(S_size + len_n_star):
                            if ((X[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                o_all = o_all + 1

                        if o_all == S_size + len_n_star:
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


                        p_s = (cs * lm)/lm_s
                        p_si = (csi * lm)/lm_si

                        if S_size != 0:
                            for nv in range(len_va_c):
                                for i2 in range(d):
                                    phi[i, va_c[nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

                        else:
                            p_off = lm/data.shape[0]
                            for nv in range(len_va_c):
                                for i2 in range(d):
                                    phi[i, va_c[nv], i2] += (p_si-p_off)*values[b, leaf_idx_trees[b, leaf_numb], i2] + coef * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

    return np.asarray(phi)/m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef compute_sdp_swing(const double[:, :] X, const long[:] fX,
            const long[:] y_pred, long[::1] S, unsigned long S_size, const double[:, :] data,
            const double[:, :, :] values,const  double[:, :, :, :] partition_leaves_trees,
            const long[:, :] leaf_idx_trees, const long[:] leaves_nb, const double[:] scaling,
            const double thresholds, int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef double[:] out

    if S_size == m:
        out = np.ones(shape=(N))
        return out
    elif S_size == 0:
        out = np.zeros(shape=(N))
        return out

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, j, lm, lm_s

    for b in range(n_trees):
        for leaf_numb in range(leaves_nb[b]):
            for i in prange(N, nogil=True, num_threads=num_threads):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue

                p = 0
                p_u = 0
                p_d = 0
                p_s = 0
                p_su = 0
                p_sd = 0

                for j in range(data.shape[0]):
                    a_it = 0
                    b_it = 0
                    lm = 0
                    lm_s = 0
                    for s in range(m):
                        if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                            a_it = a_it + 1
                    for s in range(S_size):
                        if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            b_it = b_it + 1

                    if a_it == m:
                        lm = 1

                    if b_it == S_size:
                        lm_s = 1

                    p += lm
                    p_s += lm_s
                    if fX[i] == y_pred[j]:
                        p_u += lm
                        p_su += lm_s
                    else:
                        p_d += lm
                        p_sd += lm_s

                mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_s) if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_su) if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

    for i in prange(N, nogil=True, num_threads=num_threads):
        ss = (mean_forest[i, 0] - mean_forest[i, 2])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
        if ss >= thresholds:
            sdp[i] = 1
        else:
            sdp[i] = 0

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef swing_sv_clf(const double[:, :] X,
    const long[:] fX,
    const long[:] y_pred,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const double[:] scaling, list C, const double thresholds, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :] phi
    phi = np.zeros((N, m))

    cdef double[:] v_plus, v_minus
    v_plus = np.zeros((N))
    v_minus = np.zeros((N))

    cdef long[::1] S
    S = np.zeros((m), dtype=np.int)
    cdef unsigned int S_size, dif_pos, dif_neg, dif_null, va_size, pow_set_size
    cdef unsigned int counter, i, va, j, ci, cj, set_size
#     cdef list Sm
    cdef vector[vector[int]] Sm, va_id_cpp
    cdef double weight

    cdef long[:, :, :] swings_prop
    swings_prop = np.zeros((N, m, 3), dtype=np.int)

    cdef double[:, :, :] swings,
    swings = np.zeros((N, m, 2))

    cdef long[:, :] sm_buf
    sm_buf = np.zeros((data.shape[1], data.shape[1]), dtype=np.int)

    cdef long[:] sm_size
    sm_size = np.zeros((data.shape[1]), dtype=np.int)
    if C[0] != []:
        C_buff = C.copy()
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = []
        remove_va = []

        for i in range(m):
            if i not in coal_va:
                remove_va.append([i])
                va_id.append([i])
            else:
                for c in C_buff:
                    if i in c:
                        va_id.append(c)
                        C_buff.remove(c)
                        continue

    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    va_id_cpp = va_id

    cdef long[:, :] va_id_a
    cdef long[:] len_va_id

    va_id_a = np.empty((data.shape[1], data.shape[1]), dtype=np.int)
    len_va_id = np.empty((data.shape[1]), dtype=np.int)

    for i in range(m):
        len_va_id[i] = va_id_cpp[i].size()
        for j in range(len_va_id[i]):
            va_id_a[i, j] = va_id_cpp[i][j]

    set_size = m - 1
    pow_set_size = 2**set_size
    for va in range(m):
        Sm.assign(va_id_cpp.begin(), va_id_cpp.end())
        Sm.erase(Sm.begin() + va)

        for i in range(set_size):
            sm_size[i] = Sm[i].size()
            for j in range(sm_size[i]):
                sm_buf[i, j] = Sm[i][j]

        for counter in range(0, pow_set_size):
            va_size = 0
            S_size = 0
            for ci in range(set_size):
                if((counter & (1 << ci)) > 0):
                    for cj in range(sm_size[ci]):
                        S[S_size] = sm_buf[ci, cj]
                        S_size = S_size + 1
                    va_size = va_size + 1

            for i in range(len_va_id[va]):
                S[S_size + i] = va_id_a[va, i]

            weight = 1./binomialC(m - 1, va_size)
            v_plus = compute_sdp_swing(X, fX, y_pred, S, S_size+len_va_id[va], data, values,
                      partition_leaves_trees, leaf_idx_trees, leaves_nb, scaling,
                      thresholds, num_threads)

            v_minus = compute_sdp_swing(X, fX, y_pred, S, S_size, data, values,
                      partition_leaves_trees, leaf_idx_trees, leaves_nb, scaling,
                      thresholds, num_threads)

            for i in prange(N, nogil=True, num_threads=num_threads):
                dif_pos = 1 if (v_plus[i] - v_minus[i]) > 0 else 0
                dif_neg = 1 if (v_plus[i] - v_minus[i]) < 0 else 0
                dif_null = 1 if (v_plus[i] - v_minus[i]) == 0 else 0

                for j in range(len_va_id[va]):
                    phi[i, va_id_a[va, j]] += weight * (v_plus[i] - v_minus[i])


                    swings[i, va_id_a[va, j], 0] += (dif_pos * (v_plus[i] - v_minus[i]) * weight) / m
                    swings[i, va_id_a[va, j], 1] += (dif_neg * (v_plus[i] - v_minus[i]) * weight) / m

                    swings_prop[i, va_id_a[va, j], 0] += dif_pos
                    swings_prop[i, va_id_a[va, j], 1] += dif_neg
                    swings_prop[i, va_id_a[va, j], 2] += dif_null

    return np.array(phi)/m, np.array(swings), np.array(swings_prop)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_clf_cpp_pa_coal(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double global_proba,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value
    cdef vector[int].iterator t

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, s_1, S_size, j, lm, lm_s

    cdef long[:] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    if C[0] != []:
        remove_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    power = [np.array(sum(list(co),[])) for co in powerset(va_id)]

    cdef long[:, :] s_star
    s_star = -1*np.ones((N, m), dtype=np.int)

    cdef vector[vector[long]] power_cpp = power
    cdef long power_set_size = 2**m

    S = np.zeros((m), dtype=np.int)
    mean_forest = np.zeros((X.shape[0], 3))

    for s_1 in range(1, power_set_size-1):
        for i in range(power_cpp[s_1].size()):
            S[i] = power_cpp[s_1][i]

        S_size = power_cpp[s_1].size()

        r.clear()
        N = R.size()

        for i in range(N):
            R_buf[i] = R[i]
            for j in range(3):
                mean_forest[R_buf[i], j] = 0

        for b in range(n_trees):

            for leaf_numb in range(leaves_nb[b]):
                for i in prange(N, nogil=True, num_threads=num_threads):
                    o_all = 0
                    for s in range(S_size):
                        if ((X[R_buf[i], S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[R_buf[i], S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            o_all = o_all + 1
                    if o_all > 0:
                        continue

                    p = 0
                    p_u = 0
                    p_d = 0
                    p_s = 0
                    p_su = 0
                    p_sd = 0
                    for j in range(data.shape[0]):
#                         print('debug', j)
                        a_it = 0
                        b_it = 0
                        lm = 0
                        lm_s = 0
                        for s in range(m):
                            if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                a_it = a_it + 1
                        for s in range(S_size):
                            if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it = b_it + 1

                        if a_it == m:
                            lm = 1

                        if b_it == S_size:
                            lm_s = 1

                        p += lm
                        p_s += lm_s
                        if fX[R_buf[i]] == y_pred[j]:
                            p_u += lm
                            p_su += lm_s
                        else:
                            p_d += lm
                            p_sd += lm_s

                    mean_forest[R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] * p_s) if p_s != 0 else 0
                    mean_forest[R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_su) if p_su != 0 else 0
                    mean_forest[R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

        for i in prange(N, nogil=True, num_threads=num_threads):
            ss = (mean_forest[R_buf[i], 0] - mean_forest[R_buf[i], 2])/(mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2]) if mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2] !=0 else 0
            if((ss <= 1) and (ss>=0)):
                sdp[R_buf[i]] = ss
                if ss >= global_proba:
                    r.push_back(R[i])
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                        sdp_global[S[s]] += 1

            elif(ss > 1):
                sdp[R_buf[i]] = 1
                r.push_back(R[i])

                len_s_star[R_buf[i]] = S_size
                for s in range(S_size):
                    s_star[R_buf[i], s] = S[s]
                    sdp_global[S[s]] += 1
            else:
                sdp[R_buf[i]] = 0

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if R.size() == 0:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star), np.array(len_s_star)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_clf_pa_coal(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double global_proba,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss
    cdef int b, leaf_numb, i, s, s_1, S_size, j, lm, lm_s

    cdef long[:] S, len_s_star
    len_s_star = np.zeros((N), np.int)

    cdef list power, va_id, R, r
    R = list(range(N))
    cdef long[:] R_buf
    R_buf = np.array(R)

    if C[0] != []:
        remove_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    power = [np.array(sum(list(co),[])) for co in powerset(va_id)]

    cdef long[:, :] s_star
    s_star = -1*np.ones((N, m), dtype=np.int)

    for s_1 in range(1, len(power)-1):
        S = power[s_1]
        S_size = S.shape[0]
        r = []
        N = len(R)
        for i in range(N):
            R_buf[i] = R[i]
        mean_forest = np.zeros((X.shape[0], 3))

        for b in range(n_trees):

            for leaf_numb in range(leaves_nb[b]):
                for i in prange(N, nogil=True, num_threads=num_threads):
                    o_all = 0
                    for s in range(S_size):
                        if ((X[R_buf[i], S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[R_buf[i], S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            o_all = o_all + 1
                    if o_all > 0:
                        continue

                    p = 0
                    p_u = 0
                    p_d = 0
                    p_s = 0
                    p_su = 0
                    p_sd = 0
                    for j in range(data.shape[0]):
                        a_it = 0
                        b_it = 0
                        lm = 0
                        lm_s = 0
                        for s in range(m):
                            if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                a_it = a_it + 1
                        for s in range(S_size):
                            if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it = b_it + 1

                        if a_it == m:
                            lm = 1

                        if b_it == S_size:
                            lm_s = 1

                        p += lm
                        p_s += lm_s
                        if fX[R_buf[i]] == y_pred[j]:
                            p_u += lm
                            p_su += lm_s
                        else:
                            p_d += lm
                            p_sd += lm_s

                    mean_forest[R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] * p_s) if p_s != 0 else 0
                    mean_forest[R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_su) if p_su != 0 else 0
                    mean_forest[R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (scaling[b] *p_sd) if p_sd != 0 else 0

        for i in range(N):
            ss = (mean_forest[R_buf[i], 0] - mean_forest[R_buf[i], 2])/(mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2]) if mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2] !=0 else 0
            if((ss <= 1) and (ss>=0)):
                sdp[R_buf[i]] = ss
                if ss >= global_proba:
                    r.append(R[i])
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                        sdp_global[S[s]] += 1

            elif(ss > 1):
                sdp[R_buf[i]] = 1
                r.append(R_buf[i])
                len_s_star[R_buf[i]] = S_size
                for s in range(S_size):
                    s_star[R_buf[i], s] = S[s]
                    sdp_global[S[s]] += 1
            else:
                sdp[R_buf[i]] = 0

        for i in range(len(r)):
            R.remove(r[i])


        if len(R) == 0:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star), np.array(len_s_star)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef compute_sdp_swing_diff(const double[:, :] X, const long[:] fX,
            const long[:] y_pred, long[::1] S, long[:, :] va_id, long va, long[:] len_va_id, unsigned long S_size, const double[:, :] data,
            const double[:, :, :] values,const  double[:, :, :, :] partition_leaves_trees,
            const long[:, :] leaf_idx_trees, const long[:] leaves_nb, const double[:] scaling,
            const double thresholds,int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef double[:] out

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :] mean_forest_m
    mean_forest_m = np.zeros((N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up, nv, csm
    cdef double ss, ss_m
    cdef unsigned int b, leaf_numb, i, s, j, lm, lm_m, lm_s, lm_s_m,  p_m, p_s_m, p_u_m, p_d_m, p_su_m, p_sd_m, nv_bool

    for b in range(n_trees):
        for leaf_numb in range(leaves_nb[b]):
            for i in prange(N, nogil=True, num_threads=num_threads):
                csm = 0
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue
                else:
                    nv_bool = 0
                    for nv in range(len_va_id[va]):
                        if (X[i, va_id[va, nv]] >  partition_leaves_trees[b, leaf_numb, va_id[va, nv], 1]) or (X[i, va_id[va, nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va, nv], 0]):
                            nv_bool = nv_bool + 1
                            continue

                    if nv_bool == 0:
                        csm = 1


                p = 0
                p_u = 0
                p_d = 0
                p_s = 0
                p_su = 0
                p_sd = 0

                p_s_m = 0
                p_su_m = 0
                p_sd_m = 0

                for j in range(data.shape[0]):
                    a_it = 0
                    b_it = 0

                    lm = 0
                    lm_s = 0

                    lm_m = 0
                    lm_s_m = 0

                    for s in range(m):
                        if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                            a_it = a_it + 1
                    for s in range(S_size):
                        if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            b_it = b_it + 1

                    if a_it == m:
                        lm = 1

                    if b_it == S_size:
                        lm_s = 1

                        nv_bool = 0
                        for nv in range(len_va_id[va]):
                            if (data[j, va_id[va, nv]] >  partition_leaves_trees[b, leaf_numb, va_id[va, nv], 1]) or (data[j, va_id[va, nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va, nv], 0]):
                                nv_bool = nv_bool + 1
                                continue

                        if nv_bool == 0:
                            lm_s_m = 1


                    p += lm
                    p_s += lm_s
                    p_s_m += lm_s_m

                    if fX[i] == y_pred[j]:
                        p_u += lm
                        p_su += lm_s
                        p_su_m += lm_s_m
                    else:
                        p_d += lm
                        p_sd += lm_s
                        p_sd_m += lm_s_m

                mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_s) if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_su) if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_sd) if p_sd != 0 else 0

                mean_forest_m[i, 0] += csm*(p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_s_m) if p_s_m != 0 else 0
                mean_forest_m[i, 1] += csm*(p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_su_m) if p_su_m != 0 else 0
                mean_forest_m[i, 2] += csm*(p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (scaling[b] * p_sd_m) if p_sd_m != 0 else 0


    if S_size !=0 and S_size + len_va_id[va] != m:
        for i in prange(N, nogil=True, num_threads=num_threads):
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            ss_m = (mean_forest_m[i, 0] - mean_forest_m[i, 2])/(mean_forest_m[i, 1] - mean_forest_m[i, 2]) if mean_forest_m[i, 1] - mean_forest_m[i, 2] !=0 else 0


            if (ss < thresholds and ss_m < thresholds):
                sdp[i] = 0
            elif (ss < thresholds and ss_m >= thresholds):
                sdp[i] = 1
            elif (ss >= thresholds and ss_m >= thresholds):
                sdp[i] = 0
            elif (ss >= thresholds and ss_m < thresholds):
                sdp[i] = -1
    elif S_size == 0:
        ss = 0
        for i in prange(N, nogil=True, num_threads=num_threads):
            ss_m = (mean_forest_m[i, 0] - mean_forest_m[i, 2])/(mean_forest_m[i, 1] - mean_forest_m[i, 2]) if mean_forest_m[i, 1] - mean_forest_m[i, 2] !=0 else 0

            if (ss < thresholds and ss_m < thresholds):
                sdp[i] = 0
            elif (ss < thresholds and ss_m >= thresholds):
                sdp[i] = 1
            elif (ss >= thresholds and ss_m >= thresholds):
                sdp[i] = 0
            elif (ss >= thresholds and ss_m < thresholds):
                sdp[i] = -1

    elif S_size + len_va_id[va] == m:
        ss_m = 1
        for i in prange(N, nogil=True, num_threads=num_threads):
            ss = (mean_forest[i, 0] - mean_forest[i, 2])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0
            if (ss < thresholds and ss_m < thresholds):
                sdp[i] = 0
            elif (ss < thresholds and ss_m >= thresholds):
                sdp[i] = 1
            elif (ss >= thresholds and ss_m >= thresholds):
                sdp[i] = 0
            elif (ss >= thresholds and ss_m < thresholds):
                sdp[i] = -1
    else:
        raise ValueError('Check condition in the computation of SDP')

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef swing_sv_clf_direct(const double[:, :] X,
    const long[:] fX,
    const long[:] y_pred,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const double[:] scaling, list C, const double thresholds, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :] phi
    phi = np.zeros((N, m))

    cdef double[:] v_value
    v_value = np.zeros((N))

    cdef long[::1] S
    S = np.zeros((m), dtype=np.int)
    cdef unsigned int S_size, dif_pos, dif_neg, dif_null, va_size, pow_set_size
    cdef unsigned int counter, i, va, j, ci, cj, set_size, nv
#     cdef list Sm
    cdef vector[vector[int]] Sm, va_id_cpp
    cdef double weight

    cdef long[:, :, :] swings_prop
    swings_prop = np.zeros((N, m, 3), dtype=np.int)

    cdef double[:, :, :] swings,
    swings = np.zeros((N, m, 2))

    cdef long[:, :] sm_buf
    sm_buf = np.zeros((2**data.shape[1], data.shape[1]), dtype=np.int)

    cdef long[:] sm_size
    sm_size = np.zeros((2**data.shape[1]), dtype=np.int)
    if C[0] != []:
        C_buff = C.copy()
        coal_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = []
        remove_va = []

        for i in range(m):
            if i not in coal_va:
                remove_va.append([i])
                va_id.append([i])
            else:
                for c in C_buff:
                    if i in c:
                        va_id.append(c)
                        C_buff.remove(c)
                        continue

    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    va_id_cpp = va_id

    cdef long[:, :] va_id_a
    cdef long[:] len_va_id

    va_id_a = np.empty((data.shape[1], data.shape[1]), dtype=np.int)
    len_va_id = np.empty((data.shape[1]), dtype=np.int)

    for i in range(m):
        len_va_id[i] = va_id_cpp[i].size()
        for j in range(len_va_id[i]):
            va_id_a[i, j] = va_id_cpp[i][j]

    set_size = m - 1
    pow_set_size = 2**set_size

    for va in range(m):
#         Sm = va_id.copy()
#         Sm.remove(va_id[va])
        Sm.assign(va_id_cpp.begin(), va_id_cpp.end())
        Sm.erase(Sm.begin() + va)

        for i in range(set_size):
            sm_size[i] = Sm[i].size()
            for j in range(sm_size[i]):
                sm_buf[i, j] = Sm[i][j]

        for counter in range(0, pow_set_size):
            va_size = 0
            S_size = 0
            for ci in range(set_size):
                if((counter & (1 << ci)) > 0):
                    for cj in range(sm_size[ci]):
                        S[S_size] = sm_buf[ci, cj]
                        S_size = S_size + 1
                    va_size = va_size + 1

            weight = 1./binomialC(m - 1, va_size)
            v_value = compute_sdp_swing_diff(X, fX, y_pred, S, va_id_a, va, len_va_id, S_size, data, values,
                      partition_leaves_trees, leaf_idx_trees, leaves_nb, scaling,
                      thresholds, num_threads)


            for i in prange(N, nogil=True, num_threads=num_threads):
                dif_pos = 1 if v_value[i] > 0 else 0
                dif_neg = 1 if v_value[i] < 0 else 0
                dif_null = 1 if v_value[i] == 0 else 0

                for j in range(len_va_id[va]):
                    phi[i, va_id_a[va, j]] += weight * v_value[i]


                    swings[i, va_id_a[va, j], 0] += (dif_pos * (v_value[i]) * weight) / m
                    swings[i, va_id_a[va, j], 1] += (dif_neg * (v_value[i]) * weight) / m

                    swings_prop[i, va_id_a[va, j], 0] += dif_pos
                    swings_prop[i, va_id_a[va, j], 1] += dif_neg
                    swings_prop[i, va_id_a[va, j], 2] += dif_null

    return np.array(phi)/m, np.array(swings), np.array(swings_prop)