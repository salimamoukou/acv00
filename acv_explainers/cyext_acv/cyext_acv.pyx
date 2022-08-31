# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.set cimport set
import numpy as np
cimport numpy as np
ctypedef np.float64_t double
cimport cython
from scipy.special import comb
import itertools
from tqdm import tqdm
from acv_explainers.utils import weighted_percentile
from cython.parallel cimport prange, parallel, threadid
from cython.operator cimport dereference as deref, preincrement as inc
cimport openmp

# TO DO: change every vector into set

cdef extern from "<algorithm>" namespace "std" nogil:
     iter std_remove "std::remove" [iter, T](iter first, iter last, const T& val)
     iter std_find "std::find" [iter, T](iter first, iter last, const T& val)

cdef extern from "limits.h":
    unsigned long ULONG_MAX



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_exp(double[:, :] X, long[:] S, double[:, :] data, double[:, :, :] values,
        double[:, :, :, :] partition_leaves_trees, long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
        int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, d))

    cdef double[:, :, :] mean_forest_b
    mean_forest_b = np.zeros((max_leaves, N, d))

    cdef unsigned int it, it_s, nb_leaf, lm, p_ss, o_all
    cdef int b, leaf_numb, i, s, j

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            lm = 0
            p_ss = 0
            for i in range(data.shape[0]):
                it = 0
                it_s = 0
                for s in range(m):
                    if((data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                        it = it + 1
                for s in range(S_size):
                    if((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        it_s = it_s + 1
                if it == m:
                    lm = lm + 1
                if it_s == S_size:
                    p_ss = p_ss + 1

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue
                for j in range(d):
                    mean_forest_b[leaf_numb, i, j] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j]) / p_ss if p_ss != 0 else 0

    for i in range(N):
        for j in range(d):
            for leaf_numb in range(max_leaves):
                mean_forest[i, j] += mean_forest_b[leaf_numb, i, j]

    return np.asarray(mean_forest)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_exp_normalized(double[:, :] X, long[:] S, double[:, :] data, double[:, :, :] values,
        double[:, :, :, :] partition_leaves_trees, long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
        int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, d))

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, N, d))

    cdef unsigned int it, it_s, nb_leaf, lm, p_ss, o_all
    cdef int b, leaf_numb, i, s, j
    cdef double[:, :, :] p_n
    p_n = np.zeros((n_trees, max_leaves, N))
    cdef double z

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            lm = 0
            p_ss = 0
            for i in range(data.shape[0]):
                it = 0
                it_s = 0
                for s in range(m):
                    if((data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                        it = it + 1
                for s in range(S_size):
                    if((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        it_s = it_s + 1
                if it == m:
                    lm = lm + 1
                if it_s == S_size:
                    p_ss = p_ss + 1

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] < partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue

                p_n[b, leaf_numb, i] += (1.*lm)/p_ss

                for j in range(d):
                    mean_forest_b[b, leaf_numb, i, j] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j]) / p_ss if p_ss != 0 else 0


    for i in range(N):
        for b in range(n_trees):
            z = 0
            for leaf_numb in range(leaves_nb[b]):
                z += p_n[b, leaf_numb, i]
            for leaf_numb in range(leaves_nb[b]):
                for j in range(d):
                    mean_forest[i, j] += mean_forest_b[b, leaf_numb, i, j]/z if z !=0 else 0


    return np.asarray(mean_forest)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_exp_cat(double[:, :] X, long[:] S, double[:, :] data, double[:, :, :] values,
        double[:, :, :, :] partition_leaves_trees, long[:, :] leaf_idx_trees,  long[:] leaves_nb, double[:] scaling,
        int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int d = values.shape[2]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, d))

    cdef double[:, :, :] mean_forest_b
    mean_forest_b = np.zeros((max_leaves, N, d))

    cdef unsigned int it, it_s, p_ss, o_all, nb_leaf, lm
    cdef int b, leaf_numb, i, s, j1, j

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 1]) or (X[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                        o_all = o_all + 1
                if o_all > 0:
                    continue

                lm = 0
                p_ss = 0
                for j in range(data.shape[0]):

                    it = 0
                    it_s = 0
                    for s in range(m):
                        if((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] > partition_leaves_trees[b, leaf_numb, s, 0])):
                            it = it +  1

                    for s in range(S_size):
                        if(data[j, S[s]] == X[i, S[s]]):
                            it_s = it_s +  1

                    if it_s == S_size:
                        p_ss =  p_ss + 1
                        if it == m:
                            lm = lm + 1

                for j1 in range(d):
                    mean_forest_b[leaf_numb, i, j1] += (lm * values[b, leaf_idx_trees[b, leaf_numb], j1]) / (p_ss) if p_ss != 0 else 0

    for i in range(N):
        for j in range(d):
            for leaf_numb in range(max_leaves):
                mean_forest[i, j] += mean_forest_b[leaf_numb, i, j]

    return np.asarray(mean_forest)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_clf_cat(double[:, :] X, long[:] fX,
            long[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int max_leaves =partition_leaves_trees.shape[1]
    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :, :] mean_forest_b
    mean_forest_b = np.zeros((max_leaves, N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, j, lm

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            for i in range(N):
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

                        p = p + lm
                        p_u = p_u + lm * up
                        p_d = p_d + lm * down
                        p_s = p_s + 1
                        p_su = p_su + up
                        p_sd = p_sd + down

                mean_forest_b[leaf_numb, i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_s) if p_s != 0 else 0
                mean_forest_b[leaf_numb, i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_su) if p_su != 0 else 0
                mean_forest_b[leaf_numb, i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_sd) if p_sd != 0 else 0

    for i in range(N):
        ss_u = 0
        ss_d = 0
        ss_a = 0
        for leaf_numb in range(max_leaves):
            ss_a += mean_forest_b[leaf_numb, i, 0]
            ss_u += mean_forest_b[leaf_numb, i, 1]
            ss_d += mean_forest_b[leaf_numb, i, 2]
        ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
        if((ss <= 1) and (ss>=0)):
            sdp[i] = ss
        elif(ss > 1):
            sdp[i] = 1
        else:
            sdp[i] = 0

    return np.asarray(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_clf(double[:, :] X, long[:] fX,
            long[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling,
            int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    if S_size == m:
        return np.ones(shape=(N))
    elif S_size == 0:
        return np.zeros(shape=(N))

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up
    cdef double ss, ss_u, ss_a, ss_d
    cdef int b, leaf_numb, i, s, j, lm, lm_s

    cdef double[:, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, N))
    cdef double n, n_u, n_d

    for b in tqdm(range(n_trees)):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            for i in range(N):
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

                    p = p + lm
                    p_s = p_s + lm_s
                    if fX[i] == y_pred[j]:
                        p_u = p_u + lm
                        p_su = p_su + lm_s
                    else:
                        p_d = p_d + lm
                        p_sd = p_sd + lm_s

                mean_forest_b[b, leaf_numb, i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_s) if p_s != 0 else 0
                mean_forest_b[b, leaf_numb, i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_su) if p_su != 0 else 0
                mean_forest_b[b, leaf_numb, i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_sd) if p_sd != 0 else 0

                p_n[b, leaf_numb, i] += (1.*p)/p_s if p_s != 0 else 0
                p_u_n[b, leaf_numb, i] += (1.*p_u)/p_su if p_su != 0 else 0
                p_d_n[b, leaf_numb, i] += (1.*p_d)/p_sd if p_sd != 0 else 0

    for i in range(N):
        ss_u = 0
        ss_d = 0
        ss_a = 0
        for b in range(n_trees):
            nb_leaf = leaves_nb[b]
            n = 0
            n_u = 0
            n_d = 0
            for leaf_numb in range(nb_leaf):
                n += p_n[b, leaf_numb, i]
                n_u += p_u_n[b, leaf_numb, i]
                n_d += p_d_n[b, leaf_numb, i]

            for leaf_numb in range(max_leaves):
                ss_a += mean_forest_b[b, leaf_numb, i, 0]/n if n !=0 else 0
                ss_u += mean_forest_b[b, leaf_numb, i, 1]/n_u if n_u !=0 else 0
                ss_d += mean_forest_b[b, leaf_numb, i, 2]/n_d if n_d !=0 else 0

        sdp[i] = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0

    return np.array(sdp)

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_reg(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
    cdef int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, max_leaves, N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef double[:] norm, norm_u, norm_d
    norm = np.zeros((n_trees))
    norm_u = np.zeros((n_trees))
    norm_d = np.zeros((n_trees))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, nb_leaf_l, o_all, down, up
    cdef double ss
    cdef int b, l, leaf_numb, i, s, j, leaf_numb_l

    cdef double[:, :, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    cdef double n, n_u, n_d

    for b in range(n_trees):
        for l in range(n_trees):
            if b == l:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]

                for leaf_numb in range(nb_leaf):
                    leaf_part = leaves_tree[leaf_numb]
                    value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                    lm = np.zeros(data.shape[0], dtype=np.int)
                    lm_s = np.zeros(data.shape[0], dtype=np.int)

                    for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                        a_it = 0
                        b_it = 0
                        for s in range(m):
                            if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                a_it = a_it + 1
                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                b_it = b_it + 1

                        if a_it == m:
                            lm[i] = 1

                        if b_it == S_size:
                            lm_s[i] = 1

                    for i in prange(N, nogil=True, num_threads=num_threads):
                        o_all = 0
                        for s in range(S_size):
                            if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
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
                            p += lm[j]
                            p_s += lm_s[j]
                            if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                p_u += lm[j]
                                p_su += lm_s[j]
                            else:
                                p_d += lm[j]
                                p_sd += lm_s[j]

                        mean_forest_b[b, leaf_numb, leaf_numb, i, 0] += (p * value*value) / (p_s) - (2 * fX[i] * p * value)/(p_s) if p_s != 0 else 0
                        mean_forest_b[b, leaf_numb, leaf_numb, i, 1] += (p_u * value*value) / (p_su) - (2 * fX[i] * p_u * value)/(p_su) if p_su != 0 else 0
                        mean_forest_b[b, leaf_numb, leaf_numb, i, 2] += (p_d * value*value) / (p_sd) - (2 * fX[i] * p_d * value)/(p_sd) if p_sd != 0 else 0

                        p_n[b, leaf_numb, leaf_numb, i] += (1.*p)/p_s if p_s != 0 else 0
                        p_u_n[b, leaf_numb, leaf_numb, i] += (1.*p_u)/p_su if p_su != 0 else 0
                        p_d_n[b, leaf_numb, leaf_numb, i] += (1.*p_d)/p_sd if p_sd != 0 else 0
            else:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]
                leaves_tree_l = partition_leaves_trees[l]
                nb_leaf_l = leaves_nb[l]

                for leaf_numb in range(nb_leaf):
                    for leaf_numb_l in range(nb_leaf_l):

                        leaf_part = leaves_tree[leaf_numb]
                        leaf_part_l = leaves_tree_l[leaf_numb_l]
                        value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                        value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                        lm = np.zeros(data.shape[0], dtype=np.int)
                        lm_s = np.zeros(data.shape[0], dtype=np.int)

                        for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                            a_it = 0
                            b_it = 0
                            for s in range(m):
                                if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                    a_it = a_it + 1
                            for s in range(S_size):
                                if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                    b_it = b_it + 1

                            if a_it == m:
                                lm[i] = 1

                            if b_it == S_size:
                                lm_s[i] = 1

                        for i in prange(N, nogil=True, num_threads=num_threads):
                            o_all = 0
                            for s in range(S_size):
                                if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0]) or (X[i, S[s]] > leaf_part_l[S[s], 1]) or (X[i, S[s]] < leaf_part_l[S[s], 0])):
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
                                p += lm[j]
                                p_s += lm_s[j]
                                if (fX[i] - y_pred[j])*(fX[i] - y_pred[j]) > tX:
                                    p_u += lm[j]
                                    p_su += lm_s[j]
                                else:
                                    p_d += lm[j]
                                    p_sd += lm_s[j]

                            mean_forest_b[b, leaf_numb, leaf_numb_l, i, 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                            mean_forest_b[b, leaf_numb, leaf_numb_l, i, 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                            mean_forest_b[b, leaf_numb, leaf_numb_l, i, 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

                            p_n[b, leaf_numb, leaf_numb_l, i] += (1.*p)/p_s if p_s != 0 else 0
                            p_u_n[b, leaf_numb, leaf_numb_l, i] += (1.*p_u)/p_su if p_su != 0 else 0
                            p_d_n[b, leaf_numb, leaf_numb_l, i] += (1.*p_d)/p_sd if p_sd != 0 else 0
    for i in range(N):
        ss_u = 0
        ss_d = 0
        ss_a = 0
        for b in range(n_trees):
            for l in range(n_trees):
                if b == l:
                    n = 0
                    n_u = 0
                    n_d = 0
                    nb_leaf = leaves_nb[b]
                    for leaf_numb in range(nb_leaf):
                        n += p_n[b, leaf_numb, leaf_numb, i]
                        n_u += p_u_n[b, leaf_numb, leaf_numb, i]
                        n_d += p_d_n[b, leaf_numb, leaf_numb, i]
                    norm[b] = n
                    norm_u[b] = n_u
                    norm_d[b] = n_d
                    for leaf_numb in range(nb_leaf):
                        ss_a += mean_forest_b[b, leaf_numb, leaf_numb,  i, 0]/norm[b] if norm[b] !=0 else 0
                        ss_u += mean_forest_b[b, leaf_numb, leaf_numb, i, 1]/norm_u[b] if norm_u[b] !=0 else 0
                        ss_d += mean_forest_b[b, leaf_numb, leaf_numb, i, 2]/norm_d[b] if norm_d[b] !=0 else 0
                else:
                    nb_leaf = leaves_nb[b]
                    nb_leaf_l = leaves_nb[l]

                    for leaf_numb in range(nb_leaf):
                        n = 0
                        n_u = 0
                        n_d = 0
                        for leaf_numb_l in range(nb_leaf_l):
                            n += p_n[b, leaf_numb, leaf_numb_l, i]
                            n_u += p_u_n[b, leaf_numb, leaf_numb_l, i]
                            n_d += p_d_n[b, leaf_numb, leaf_numb_l, i]

                        for leaf_numb_l in range(nb_leaf_l):
                            ss_a += (p_n[b, leaf_numb, leaf_numb, i]/norm[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l,  i, 0]/n) if n*norm[b] !=0  else 0
                            ss_u += (p_u_n[b, leaf_numb, leaf_numb, i]/norm_u[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, i, 1]/n_u) if n_u*norm_u[b] !=0 else 0
                            ss_d += (p_d_n[b, leaf_numb, leaf_numb, i]/norm_d[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, i, 2]/n_d) if n_d*norm_d[b] !=0 else 0


        sdp[i] = (ss_u - ss_a)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0

    return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_reg_cat(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, long[:] S, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int S_size = S.shape[0]
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
                    value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                    lm = np.zeros(data.shape[0], dtype=np.int)

                    for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                        a_it = 0
                        b_it = 0
                        for s in range(m):
                            if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                a_it = a_it + 1
                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                b_it = b_it + 1

                        if a_it == m:
                            lm[i] = 1

                    for i in prange(N, nogil=True):
                        o_all = 0
                        for s in range(S_size):
                            if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
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
                                for s in range(S_size):
                                    if data[j, S[s]] == X[i, S[s]]:
                                        b_it = b_it + 1

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

                        mean_forest[i, 0] += (p * value*value) / (p_s) - (2 * fX[i] * p * value)/(p_s) if p_s != 0 else 0
                        mean_forest[i, 1] += (p_u * value*value) / (p_su) - (2 * fX[i] * p_u * value)/(p_su) if p_su != 0 else 0
                        mean_forest[i, 2] += (p_d * value*value) / (p_sd) - (2 * fX[i] * p_d * value)/(p_sd) if p_sd != 0 else 0
            else:
                leaves_tree = partition_leaves_trees[b]
                nb_leaf = leaves_nb[b]
                leaves_tree_l = partition_leaves_trees[l]
                nb_leaf_l = leaves_nb[l]

                for leaf_numb in range(nb_leaf):
                    for leaf_numb_l in range(nb_leaf_l):

                        leaf_part = leaves_tree[leaf_numb]
                        leaf_part_l = leaves_tree_l[leaf_numb_l]
                        value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                        value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                        lm = np.zeros(data.shape[0], dtype=np.int)
                        lm_s = np.zeros(data.shape[0], dtype=np.int)

                        for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                            a_it = 0
                            b_it = 0
                            for s in range(m):
                                if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                    a_it = a_it + 1

                            if a_it == m:
                                lm[i] = 1

                        for i in prange(N, nogil=True):
                            o_all = 0
                            for s in range(S_size):
                                if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0]) or (X[i, S[s]] > leaf_part_l[S[s], 1]) or (X[i, S[s]] < leaf_part_l[S[s], 0])):
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
                                for s in range(S_size):
                                    if data[j, S[s]] == X[i, S[s]]:
                                        b_it = b_it + 1

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

                            mean_forest[i, 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                            mean_forest[i, 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                            mean_forest[i, 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

    for i in range(N):
        sdp[i]  = (mean_forest[i, 1] - mean_forest[i, 0])/(mean_forest[i, 1] - mean_forest[i, 2]) if mean_forest[i, 1] - mean_forest[i, 2] !=0 else 0

    return np.asarray(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_reg_cat(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef vector[int].iterator t
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, leaf_idx_trees.shape[1], N, 3))

    cdef double[:] sdp, sdp_b
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

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
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            sdp_b = compute_sdp_reg_cat(X, fX, tX, y_pred, S[:S_size], data, values, partition_leaves_trees, leaf_idx_trees,
                        leaves_nb, scaling, 0)
            for i in range(N):
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_reg(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef vector[int].iterator t
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, max_leaves, max_leaves, N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, leaf_numb_l

    cdef long[:] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)
    cdef double[:, :, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, max_leaves, N))
    cdef double n, n_u, n_d
    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    cdef double[:] norm, norm_u, norm_d
    norm = np.zeros((n_trees))
    norm_u = np.zeros((n_trees))
    norm_d = np.zeros((n_trees))

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            for b in range(n_trees):
                for leaf_numb in range(max_leaves):
                    for leaf_numb_l in range(max_leaves):
                        for i in range(N):
                            p_n[b, leaf_numb, leaf_numb_l, R_buf[i]] = 0
                            p_u_n[b, leaf_numb, leaf_numb_l, R_buf[i]] = 0
                            p_d_n[b, leaf_numb, leaf_numb_l, R_buf[i]] = 0
                            for j in range(3):
                                mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], j] = 0
            for b in range(n_trees):
                for l in range(n_trees):
                    if b == l:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]

                        for leaf_numb in range(nb_leaf):
                            leaf_part = leaves_tree[leaf_numb]
                            value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                            lm = np.zeros(data.shape[0], dtype=np.int)
                            lm_s = np.zeros(data.shape[0], dtype=np.int)

                            for i in prange(data.shape[0], nogil=True):
                                a_it = 0
                                b_it = 0
                                for s in range(m):
                                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                        a_it = a_it + 1
                                for s in range(S_size):
                                    if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                        b_it = b_it + 1

                                if a_it == m:
                                    lm[i] = 1

                                if b_it == S_size:
                                    lm_s[i] = 1

                            for i in prange(N, nogil=True):
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
                                    p += lm[j]
                                    p_s += lm_s[j]
                                    if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                        p_u += lm[j]
                                        p_su += lm_s[j]
                                    else:
                                        p_d += lm[j]
                                        p_sd += lm_s[j]

                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 0] += (p * value*value) / (p_s) - (2 * fX[R_buf[i]] * p * value)/(p_s) if p_s != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 1] += (p_u * value*value) / (p_su) - (2 * fX[R_buf[i]] * p_u * value)/(p_su) if p_su != 0 else 0
                                mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 2] += (p_d * value*value) / (p_sd) - (2 * fX[R_buf[i]] * p_d * value)/(p_sd) if p_sd != 0 else 0

                                p_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                                p_u_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                                p_d_n[b, leaf_numb, leaf_numb, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0
                    else:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]
                        leaves_tree_l = partition_leaves_trees[l]
                        nb_leaf_l = leaves_nb[l]

                        for leaf_numb in range(nb_leaf):
                            for leaf_numb_l in range(nb_leaf_l):

                                leaf_part = leaves_tree[leaf_numb]
                                leaf_part_l = leaves_tree_l[leaf_numb_l]
                                value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                                value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                                lm = np.zeros(data.shape[0], dtype=np.int)
                                lm_s = np.zeros(data.shape[0], dtype=np.int)

                                for i in prange(data.shape[0], nogil=True):
                                    a_it = 0
                                    b_it = 0
                                    for s in range(m):
                                        if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                            a_it = a_it + 1
                                    for s in range(S_size):
                                        if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                            b_it = b_it + 1

                                    if a_it == m:
                                        lm[i] = 1

                                    if b_it == S_size:
                                        lm_s[i] = 1

                                for i in prange(N, nogil=True):
                                    o_all = 0
                                    for s in range(S_size):
                                        if ((X[R_buf[i], S[s]] > leaf_part[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part[S[s], 0]) or (X[R_buf[i], S[s]] > leaf_part_l[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part_l[S[s], 0])):
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
                                        p += lm[j]
                                        p_s += lm_s[j]
                                        if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                            p_u += lm[j]
                                            p_su += lm_s[j]
                                        else:
                                            p_d += lm[j]
                                            p_sd += lm_s[j]

                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                                    mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

                                    p_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                                    p_u_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                                    p_d_n[b, leaf_numb, leaf_numb_l, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0
            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    for l in range(n_trees):
                        if b == l:
                            n = 0
                            n_u = 0
                            n_d = 0
                            nb_leaf = leaves_nb[b]
                            for leaf_numb in range(nb_leaf):
                                n += p_n[b, leaf_numb, leaf_numb, R_buf[i]]
                                n_u += p_u_n[b, leaf_numb, leaf_numb, R_buf[i]]
                                n_d += p_d_n[b, leaf_numb, leaf_numb, R_buf[i]]
                            norm[b] = n
                            norm_u[b] = n_u
                            norm_d[b] = n_d
                            for leaf_numb in range(nb_leaf):
                                ss_a += mean_forest_b[b, leaf_numb, leaf_numb,  R_buf[i], 0]/norm[b] if norm[b] !=0 else 0
                                ss_u += mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 1]/norm_u[b] if norm_u[b] !=0 else 0
                                ss_d += mean_forest_b[b, leaf_numb, leaf_numb, R_buf[i], 2]/norm_d[b] if norm_d[b] !=0 else 0
                        else:
                            nb_leaf = leaves_nb[b]
                            nb_leaf_l = leaves_nb[l]

                            for leaf_numb in range(nb_leaf):
                                n = 0
                                n_u = 0
                                n_d = 0
                                for leaf_numb_l in range(nb_leaf_l):
                                    n += p_n[b, leaf_numb, leaf_numb_l, R_buf[i]]
                                    n_u += p_u_n[b, leaf_numb, leaf_numb_l, R_buf[i]]
                                    n_d += p_d_n[b, leaf_numb, leaf_numb_l, R_buf[i]]

                                for leaf_numb_l in range(nb_leaf_l):
                                    ss_a += (p_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l,  R_buf[i], 0]/n) if n*norm[b] !=0  else 0
                                    ss_u += (p_u_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm_u[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 1]/n_u) if n_u*norm_u[b] !=0 else 0
                                    ss_d += (p_d_n[b, leaf_numb, leaf_numb, R_buf[i]]/norm_d[b])*(mean_forest_b[b, leaf_numb, leaf_numb_l, R_buf[i], 2]/n_d) if n_d*norm_d[b] !=0 else 0


                ss = (ss_u - ss_a)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
                if ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef shap_values_leaves_pa(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[ :, :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, N, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef double lm

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0
    cdef double[:, :, :] lm_s, lm_si
    lm_s = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, 2**scaling))

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm = lm + 1

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s[va, leaf_numb, counter] = 0
                    lm_si[va, leaf_numb, counter] = 0
                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size:
                            lm_s[va, leaf_numb, counter] = lm_s[va, leaf_numb, counter] + 1

                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[va, leaf_numb, counter] = lm_si[va, leaf_numb, counter] + 1

                    for i in range(N):

                        csi = 0
                        cs = 0

                        o_all = 0
                        for s in range(S_size):
                            if ((X[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                o_all = o_all + 1

                        if o_all == S_size:
                            cs = 1
                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((X[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
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
                            p_s = (cs * lm)/lm_s[va, leaf_numb, counter] if lm_s[va, leaf_numb, counter] !=0 else 0
                        p_si = (csi * lm)/lm_si[va, leaf_numb, counter] if lm_si[va, leaf_numb, counter] !=0 else 0

                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for i in range(N):
            for j in range(m):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        for counter in range(phi_b.shape[1]):
                            phi[i, j, i2] += phi_b[leaf_numb, counter, i, j, i2]

    return np.asarray(phi)/m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef shap_values_leaves_normalized(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[ :, :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, N, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, leaf_n
    cdef double lm

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0, n_s, n_si
    cdef double[:, :, :, :] lm_s, lm_si, lm_n, csi, cs
    lm_n = np.zeros((va_id.size(), max_leaves, max_leaves, 2**scaling))
    lm_s = np.zeros((va_id.size(), max_leaves, max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, max_leaves, 2**scaling))
    csi = np.zeros((va_id.size(), max_leaves, max_leaves, 2**scaling))
    cs = np.zeros((va_id.size(), max_leaves, max_leaves, 2**scaling))

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm = lm + 1

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    for leaf_n in range(nb_leaf):
                        lm_s[va, leaf_numb, leaf_n, counter] = 0
                        lm_si[va, leaf_numb, leaf_n,  counter] = 0
                        lm_n[va, leaf_numb, leaf_n, counter] = 0
                        for i in range(data.shape[0]):
                            a_it = 0
                            for s in range(data.shape[1]):
                                if (data[i, s] <= partition_leaves_trees[b, leaf_n, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_n, s, 0]):
                                    a_it = a_it + 1
                            if a_it == data.shape[1]:
                                lm_n[va, leaf_numb, leaf_n, counter] = lm_n[va, leaf_numb, leaf_n, counter] + 1

                            b_it = 0
                            for s in range(S_size):
                                if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                    b_it = b_it + 1

                            if b_it == S_size:
                                lm_s[va, leaf_numb, leaf_n, counter] = lm_s[va, leaf_numb, leaf_n, counter] + 1

                                nv_bool = 0
                                for nv in range(va_id[va].size()):
                                    if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                        nv_bool = nv_bool + 1
                                        continue

                                if nv_bool == 0:
                                    lm_si[va, leaf_numb, leaf_n, counter] = lm_si[va, leaf_numb, leaf_n, counter] + 1

                    for i in range(N):

                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0

                        n_s = 0
                        n_si = 0
                        if S_size == 0:
                            p_s = lm/data.shape[0]
                            n_s = 0
                            n_si = 0
                            for leaf_n in range(nb_leaf):
                                csi[va, leaf_numb, leaf_n, counter] = 0
                                cs[va, leaf_numb, leaf_n, counter] = 0

                                o_all = 0
                                for s in range(S_size):
                                    if ((X[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                        o_all = o_all + 1

                                if o_all == S_size:
                                    cs[va, leaf_numb, leaf_n, counter] = 1
                                    nv_bool = 0
                                    for nv in range(va_id[va].size()):
                                        if ((X[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                            nv_bool = nv_bool + 1
                                            continue

                                    if nv_bool == 0:
                                        csi[va, leaf_numb, leaf_n, counter] = 1

                                n_si = n_si + (csi[va, leaf_numb, leaf_n, counter]*lm_n[va, leaf_numb, leaf_n, counter])/lm_si[va, leaf_numb, leaf_n, counter] if lm_si[va, leaf_numb, leaf_n, counter] !=0 else 0

                            p_si = (csi[va, leaf_numb, leaf_numb, counter] * lm)/(lm_si[va, leaf_numb, leaf_numb, counter] * n_si) if n_si*lm_si[va, leaf_numb, leaf_numb, counter] !=0 else 0
                        else:

                            for leaf_n in range(nb_leaf):
                                csi[va, leaf_numb, leaf_n, counter] = 0
                                cs[va, leaf_numb, leaf_n, counter] = 0

                                o_all = 0
                                for s in range(S_size):
                                    if ((X[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                        o_all = o_all + 1

                                if o_all == S_size:
                                    cs[va, leaf_numb, leaf_n, counter] = 1
                                    nv_bool = 0
                                    for nv in range(va_id[va].size()):
                                        if ((X[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                            nv_bool = nv_bool + 1
                                            continue

                                    if nv_bool == 0:
                                        csi[va, leaf_numb, leaf_n, counter] = 1

                                n_s = n_s + (cs[va, leaf_numb, leaf_n, counter]*lm_n[va, leaf_numb, leaf_n, counter])/lm_s[va, leaf_numb, leaf_n, counter] if lm_s[va, leaf_numb, leaf_n, counter] !=0 else 0
                                n_si = n_si + (csi[va, leaf_numb, leaf_n, counter]*lm_n[va, leaf_numb, leaf_n, counter])/lm_si[va, leaf_numb, leaf_n, counter] if lm_si[va, leaf_numb, leaf_n, counter] !=0 else 0

                            p_s = (cs[va, leaf_numb, leaf_numb, counter] * lm)/(lm_s[va, leaf_numb, leaf_numb, counter] * n_s) if n_s*lm_s[va, leaf_numb, leaf_numb, counter] !=0 else 0
                            p_si = (csi[va, leaf_numb, leaf_numb, counter] * lm)/(lm_si[va, leaf_numb,  leaf_numb, counter] * n_si) if n_si*lm_si[va, leaf_numb, leaf_numb, counter] !=0 else 0

                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for i in range(N):
            for j in range(m):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        for counter in range(phi_b.shape[1]):
                            phi[i, j, i2] += phi_b[leaf_numb, counter, i, j, i2]

    return np.asarray(phi)/m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef shap_values_acv_leaves(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    int scaling,
    vector[vector[vector[long]]] node_idx_trees, const long[:] S_star, const long[:] N_star,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[ :, :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, N, m, d))


    cdef double[:, :, :,  :] phi_n
    phi_n = np.zeros((max_leaves, N, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs, len_n_star
    cdef double lm

    cdef long[:] va_c, N_star_a
    cdef int len_va_c, S_size, va_size,  b_it, nv_bool
    cdef long set_size
    va_c = np.zeros((m), dtype=np.int)

    cdef vector[vector[long]] C_buff, va_id, buff, Sm
    cdef vector[long] coal_va, remove_va, node_id_b
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)


    cdef vector[vector[vector[long]]] node_id_v2, C_b, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    # Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    N_star_a = np.array(N_star)
    len_n_star = N_star_a.shape[0]

    if C[0].size() != 0:
        C_buff = C
        coal_va = [C[ci][cj] for ci in range(C.size()) for cj in range(C[ci].size())]

        for i in range(S_star.shape[0]):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), S_star[i]) != coal_va.end():
                remove_va.push_back(S_star[i])
                buff[0][0] = S_star[i]
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), S_star[i]) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        va_id = [[S_star[i]] for i in range(S_star.shape[0])]

    m = va_id.size()
    cdef double p_s, p_si, coef, coef_0, p_off
    cdef double[:, :, :] lm_s, lm_si
    cdef double[:, :] lm_star
    lm_s = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_star = np.zeros((va_id.size(), max_leaves))

    for b in range(n_trees):
        for i in range(N):
            for j in range(data.shape[1]):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        phi_n[leaf_numb, i, j, i2] = 0
                        for counter in range(phi_b.shape[1]):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            for i in range(N_star.shape[0]):
                if (std_find[vector[long].iterator, long](node_idx_trees[b][leaf_numb].begin(), node_idx_trees[b][leaf_numb].end(), N_star[i]) != node_idx_trees[b][leaf_numb].end()):
                    std_remove[vector[long].iterator, long](node_idx_trees[b][leaf_numb].begin(), node_idx_trees[b][leaf_numb].end(), N_star[i])
                    node_idx_trees[b][leaf_numb].pop_back()
            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm = lm + 1


            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])

            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    lm_star[va, leaf_numb] = 0
                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(len_n_star):
                            if ((data[i, N_star_a[s]] <= partition_leaves_trees[b, leaf_numb, N_star_a[s], 1]) * (data[i, N_star_a[s]] > partition_leaves_trees[b, leaf_numb, N_star_a[s], 0])):
                                b_it = b_it + 1

                        if b_it == len_n_star:
                            lm_star[va, leaf_numb] = lm_star[va, leaf_numb] +  1

                    for i in range(N):
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
                        p_si = (csi * lm)/lm_star[va, leaf_numb]
                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_n[leaf_numb, i, va_id[va][nv], i2] += (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s[va, leaf_numb, counter] = 0
                    lm_si[va, leaf_numb, counter] = 0

                    for nv in range(len_n_star):
                        S[va, counter, leaf_numb, S_size+nv] = N_star_a[nv]

                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size + len_n_star):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size + len_n_star:
                            lm_s[va, leaf_numb, counter] = lm_s[va, leaf_numb, counter] + 1
                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[va, leaf_numb, counter] = lm_si[va, leaf_numb, counter] + 1

                    for i in range(N):

                        p_s = 0
                        p_si = 0


                        csi = 0
                        cs = 0
                        o_all = 0
                        for s in range(S_size + len_n_star):
                            if ((X[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                o_all = o_all + 1

                        if o_all == S_size + len_n_star:
                            cs = 1
                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((X[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                csi = 1

                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0


                        p_s = (cs * lm)/lm_s[va, leaf_numb, counter]
                        p_si = (csi * lm)/lm_si[va, leaf_numb, counter]

                        if S_size != 0:
                            for nv in range(va_id[va].size()):
                                for i2 in range(d):
                                    phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

                        else:
                            p_off = lm/data.shape[0]
                            for nv in range(va_id[va].size()):
                                for i2 in range(d):
                                    phi_b[leaf_numb, counter, i, va_id[va][nv], i2] += (p_si-p_off)*values[b, leaf_idx_trees[b, leaf_numb], i2] + coef * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for i in range(N):
            for j in range(data.shape[1]):
                for i2 in range(d):
                    for leaf_numb in range(phi_b.shape[0]):
                        phi[i, j, i2] += phi_n[leaf_numb, i, j, i2]
                        for counter in range(phi_b.shape[1]):
                            phi[i, j, i2] += phi_b[leaf_numb, counter, i, j, i2]
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

                mean_forest[i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_s) if p_s != 0 else 0
                mean_forest[i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_su) if p_su != 0 else 0
                mean_forest[i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_sd) if p_sd != 0 else 0

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
cdef compute_sdp_swing_diff(const double[:, :] X, const long[:] fX,
            const long[:] y_pred, long[::1] S, long[:, :] va_id, long va, long[:] len_va_id, unsigned long S_size, const double[:, :] data,
            const double[:, :, :] values,const  double[:, :, :, :] partition_leaves_trees,
            const long[:, :] leaf_idx_trees, const long[:] leaves_nb, const double[:] scaling,
            const double thresholds,int num_threads):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef double[:] out

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value

    cdef double[:, :, :, :] mean_forest
    mean_forest = np.zeros((n_trees, max_leaves, N, 3))

    cdef double[:, :, :, :] mean_forest_m
    mean_forest_m = np.zeros((n_trees, max_leaves, N, 3))

    cdef double[:] sdp
    sdp = np.zeros((N))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, p, p_s, p_u, p_d, p_su, p_sd, nb_leaf, o_all, down, up, nv, csm
    cdef unsigned int b, leaf_numb, i, s, j, lm, lm_m, lm_s, lm_s_m,  p_m, p_s_m, p_u_m, p_d_m, p_su_m, p_sd_m, nv_bool

    cdef double[:, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, N))
    cdef double n, n_u, n_d
    cdef double ss, ss_u, ss_a, ss_d

    cdef double[:, :, :] p_n_m, p_u_n_m, p_d_n_m
    p_n_m = np.zeros((n_trees, max_leaves, N))
    p_u_n_m = np.zeros((n_trees, max_leaves, N))
    p_d_n_m = np.zeros((n_trees, max_leaves, N))
    cdef double n_m, n_u_m, n_d_m
    cdef double ss_m, ss_u_m, ss_a_m, ss_d_m

    for b in range(n_trees):
        for leaf_numb in range(leaves_nb[b]):
            for i in prange(N, nogil=True, schedule='dynamic'):
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
                    else:
                        continue


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

                mean_forest[b, leaf_numb, i, 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_s) if p_s != 0 else 0
                mean_forest[b, leaf_numb, i, 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_su) if p_su != 0 else 0
                mean_forest[b, leaf_numb, i, 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_sd) if p_sd != 0 else 0

                p_n[b, leaf_numb, i] += (1.*p)/p_s if p_s != 0 else 0
                p_u_n[b, leaf_numb, i] += (1.*p_u)/p_su if p_su != 0 else 0
                p_d_n[b, leaf_numb, i] += (1.*p_d)/p_sd if p_sd != 0 else 0

                mean_forest_m[b, leaf_numb, i, 0] += csm*(p * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_s_m) if p_s_m != 0 else 0
                mean_forest_m[b, leaf_numb, i, 1] += csm*(p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_su_m) if p_su_m != 0 else 0
                mean_forest_m[b, leaf_numb, i, 2] += csm*(p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[i]]) / (p_sd_m) if p_sd_m != 0 else 0

                p_n_m[b, leaf_numb, i] += csm*(1.*p)/p_s_m if p_s_m != 0 else 0
                p_u_n_m[b, leaf_numb, i] += csm*(1.*p_u)/p_su_m if p_su_m != 0 else 0
                p_d_n_m[b, leaf_numb, i] += csm*(1.*p_d)/p_sd_m if p_sd_m != 0 else 0


    if S_size !=0 and S_size + len_va_id[va] != m:
        for i in range(N):
            ss_u = 0
            ss_d = 0
            ss_a = 0
            ss_u_m = 0
            ss_d_m = 0
            ss_a_m = 0
            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                n = 0
                n_u = 0
                n_d = 0
                n_m = 0
                n_u_m = 0
                n_d_m = 0
                for leaf_numb in range(nb_leaf):
                    n += p_n[b, leaf_numb, i]
                    n_u += p_u_n[b, leaf_numb, i]
                    n_d += p_d_n[b, leaf_numb, i]

                    n_m += p_n_m[b, leaf_numb, i]
                    n_u_m += p_u_n_m[b, leaf_numb, i]
                    n_d_m += p_d_n_m[b, leaf_numb, i]

                for leaf_numb in range(nb_leaf):
                    ss_a += mean_forest[b, leaf_numb, i, 0]/n if n !=0 else 0
                    ss_u += mean_forest[b, leaf_numb, i, 1]/n_u if n_u !=0 else 0
                    ss_d += mean_forest[b, leaf_numb, i, 2]/n_d if n_d !=0 else 0

                    ss_a_m += mean_forest_m[b, leaf_numb, i, 0]/n_m if n !=0 else 0
                    ss_u_m += mean_forest_m[b, leaf_numb, i, 1]/n_u_m if n_u !=0 else 0
                    ss_d_m += mean_forest_m[b, leaf_numb, i, 2]/n_d_m if n_d !=0 else 0

            ss_m = (ss_a_m - ss_d_m)/(ss_u_m - ss_d_m) if ss_u_m - ss_d_m !=0 else 0
            ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0


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
        for i in range(N):
            ss_u_m = 0
            ss_d_m = 0
            ss_a_m = 0
            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                n_m = 0
                n_u_m = 0
                n_d_m = 0
                for leaf_numb in range(nb_leaf):
                    n_m += p_n_m[b, leaf_numb, i]
                    n_u_m += p_u_n_m[b, leaf_numb, i]
                    n_d_m += p_d_n_m[b, leaf_numb, i]

                for leaf_numb in range(max_leaves):
                    ss_a_m += mean_forest_m[b, leaf_numb, i, 0]/n_m if n !=0 else 0
                    ss_u_m += mean_forest_m[b, leaf_numb, i, 1]/n_u_m if n_u !=0 else 0
                    ss_d_m += mean_forest_m[b, leaf_numb, i, 2]/n_d_m if n_d !=0 else 0
            ss_m = (ss_a_m - ss_d_m)/(ss_u_m - ss_d_m) if ss_u_m - ss_d_m !=0 else 0

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
        for i in range(N):
            ss_u = 0
            ss_d = 0
            ss_a = 0
            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                n = 0
                n_u = 0
                n_d = 0
                for leaf_numb in range(nb_leaf):
                    n += p_n[b, leaf_numb, i]
                    n_u += p_u_n[b, leaf_numb, i]
                    n_d += p_d_n[b, leaf_numb, i]

                for leaf_numb in range(max_leaves):
                    ss_a += mean_forest[b, leaf_numb, i, 0]/n if n !=0 else 0
                    ss_u += mean_forest[b, leaf_numb, i, 1]/n_u if n_u !=0 else 0
                    ss_d += mean_forest[b, leaf_numb, i, 2]/n_d if n_d !=0 else 0
            ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_reg_cpp_pa_coal(double[:, :] X, double[:] fX, double tX,
            double[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int num_threads, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :, :] leaves_tree_l
    cdef double[:, :] leaf_part
    cdef double[:, :] leaf_part_l
    cdef double value
    cdef double value_l

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef vector[int].iterator t

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm, lm_u, lm_d, lm_s
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, nb_leaf_l, p_u, p_d, p_su, p_sd, down, up
    cdef double ss
    cdef int b, l,  leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, leaf_numb_l

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
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**25:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    # cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)
    mean_forest = np.zeros((X.shape[0], 3))

    for s_0 in range(m + 1):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]
                for j in range(3):
                    mean_forest[R_buf[i], j] = 0

            for b in range(n_trees):
                for l in range(n_trees):
                    if b == l:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]

                        for leaf_numb in range(nb_leaf):
                            leaf_part = leaves_tree[leaf_numb]
                            value = values[b, leaf_idx_trees[b, leaf_numb], 0]

                            lm = np.zeros(data.shape[0], dtype=np.int)
                            lm_s = np.zeros(data.shape[0], dtype=np.int)

                            for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                                a_it = 0
                                b_it = 0
                                for s in range(data.shape[1]):
                                    if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0])):
                                        a_it = a_it + 1
                                for s in range(S_size):
                                    if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0])):
                                        b_it = b_it + 1

                                if a_it == data.shape[1]:
                                    lm[i] = 1

                                if b_it == S_size:
                                    lm_s[i] = 1

                            for i in prange(N, nogil=True, num_threads=num_threads):
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
                                    p += lm[j]
                                    p_s += lm_s[j]
                                    if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                        p_u += lm[j]
                                        p_su += lm_s[j]
                                    else:
                                        p_d += lm[j]
                                        p_sd += lm_s[j]

                                mean_forest[R_buf[i], 0] += (p * value*value) / (p_s) - (2 * fX[R_buf[i]] * p * value)/(p_s) if p_s != 0 else 0
                                mean_forest[R_buf[i], 1] += (p_u * value*value) / (p_su) - (2 * fX[R_buf[i]] * p_u * value)/(p_su) if p_su != 0 else 0
                                mean_forest[R_buf[i], 2] += (p_d * value*value) / (p_sd) - (2 * fX[R_buf[i]] * p_d * value)/(p_sd) if p_sd != 0 else 0
                    else:
                        leaves_tree = partition_leaves_trees[b]
                        nb_leaf = leaves_nb[b]
                        leaves_tree_l = partition_leaves_trees[l]
                        nb_leaf_l = leaves_nb[l]

                        for leaf_numb in range(nb_leaf):
                            for leaf_numb_l in range(nb_leaf_l):

                                leaf_part = leaves_tree[leaf_numb]
                                leaf_part_l = leaves_tree_l[leaf_numb_l]
                                value = values[b, leaf_idx_trees[b, leaf_numb], 0]
                                value_l = values[l, leaf_idx_trees[l, leaf_numb_l], 0]

                                lm = np.zeros(data.shape[0], dtype=np.int)
                                lm_s = np.zeros(data.shape[0], dtype=np.int)

                                for i in prange(data.shape[0], nogil=True, num_threads=num_threads):
                                    a_it = 0
                                    b_it = 0
                                    for s in range(data.shape[1]):
                                        if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] >= leaf_part[s, 0]) and (data[i, s] <= leaf_part_l[s, 1]) and (data[i, s] >= leaf_part_l[s, 0])):
                                            a_it = a_it + 1
                                    for s in range(S_size):
                                        if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] >= leaf_part[S[s], 0]) and (data[i, S[s]] <= leaf_part_l[S[s], 1]) and (data[i, S[s]] >= leaf_part_l[S[s], 0])):
                                            b_it = b_it + 1

                                    if a_it == data.shape[1]:
                                        lm[i] = 1

                                    if b_it == S_size:
                                        lm_s[i] = 1

                                for i in prange(N, nogil=True, num_threads=num_threads):
                                    o_all = 0
                                    for s in range(S_size):
                                        if ((X[R_buf[i], S[s]] > leaf_part[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part[S[s], 0]) or (X[R_buf[i], S[s]] > leaf_part_l[S[s], 1]) or (X[R_buf[i], S[s]] < leaf_part_l[S[s], 0])):
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
                                        p += lm[j]
                                        p_s += lm_s[j]
                                        if (fX[R_buf[i]] - y_pred[j])*(fX[R_buf[i]] - y_pred[j]) > tX:
                                            p_u += lm[j]
                                            p_su += lm_s[j]
                                        else:
                                            p_d += lm[j]
                                            p_sd += lm_s[j]

                                    mean_forest[R_buf[i], 0] += (p * value*value_l) / (p_s)  if p_s != 0 else 0
                                    mean_forest[R_buf[i], 1] += (p_u * value*value_l) / (p_su)  if p_su != 0 else 0
                                    mean_forest[R_buf[i], 2] += (p_d * value*value_l) / (p_sd) if p_sd != 0 else 0

            for i in range(N):
                ss = (mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 0])/(mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2]) if mean_forest[R_buf[i], 1] - mean_forest[R_buf[i], 2] !=0 else 0
                if ss >= pi_level and ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])

                for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                    s_star[R_buf[i], s] = -1
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_clf(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value
    cdef vector[int].iterator t

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, leaf_idx_trees.shape[1], N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, lm, lm_s, max_size, size

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
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(minimal, m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)

    cdef double[:, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, N))
    cdef double n, n_u, n_d

    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(len(power))):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in range(nb_leaf):
                    for i in range(N):
                        p_n[b, leaf_numb, R_buf[i]] = 0
                        p_u_n[b, leaf_numb, R_buf[i]] = 0
                        p_d_n[b, leaf_numb, R_buf[i]] = 0
                        for j in range(3):
                            mean_forest_b[b, leaf_numb, R_buf[i], j] = 0

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
                    for i in range(N):
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
                            for s in range(data.shape[1]):
                                if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                    a_it = a_it + 1
                            for s in range(S_size):
                                if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                    b_it = b_it + 1

                            if a_it == data.shape[1]:
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

                        mean_forest_b[b, leaf_numb, R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) /p_s if p_s != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]])/p_su if p_su != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]])/p_sd if p_sd != 0 else 0

                        p_n[b, leaf_numb, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                        p_u_n[b, leaf_numb, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                        p_d_n[b, leaf_numb, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0

            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    nb_leaf = leaves_nb[b]
                    n = 0
                    n_u = 0
                    n_d = 0
                    for leaf_numb in range(nb_leaf):
                        n += p_n[b, leaf_numb, R_buf[i]]
                        n_u += p_u_n[b, leaf_numb, R_buf[i]]
                        n_d += p_d_n[b, leaf_numb, R_buf[i]]

                    for leaf_numb in range(nb_leaf):
                        ss_a += mean_forest_b[b, leaf_numb, R_buf[i], 0]/n if n !=0 else 0
                        ss_u += mean_forest_b[b, leaf_numb, R_buf[i], 1]/n_u if n_u !=0 else 0
                        ss_d += mean_forest_b[b, leaf_numb, R_buf[i], 2]/n_d if n_d !=0 else 0

                ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
                if ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_clf_ptrees(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int N_b = X.shape[0]

    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value
    cdef vector[int].iterator t

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, leaf_idx_trees.shape[1], N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, lm, lm_s, max_size, size

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
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in range(m) if i not in remove_va] + C
    else:
        va_id = [[i] for i in range(m)]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]
                # for j in range(3):
                #   mean_forest[R_buf[i], j] = 0

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in range(nb_leaf):
                    for i in range(N):
                        for j in range(3):
                            mean_forest_b[b, leaf_numb, R_buf[i], j] = 0

            for b in prange(n_trees, nogil=True, schedule='dynamic'):
                nb_leaf = leaves_nb[b]
                for leaf_numb in range(nb_leaf):
                    for i in range(N):
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
                            for s in range(data.shape[1]):
                                if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                    a_it = a_it + 1
                            for s in range(S_size):
                                if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                    b_it = b_it + 1

                            if a_it == data.shape[1]:
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

                        mean_forest_b[b, leaf_numb, R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (p_s) if p_s != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (p_su) if p_su != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) / (p_sd) if p_sd != 0 else 0

            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    nb_leaf = leaves_nb[b]
                    for leaf_numb in range(nb_leaf):
                        ss_a += mean_forest_b[b, leaf_numb, R_buf[i], 0]
                        ss_u += mean_forest_b[b, leaf_numb, R_buf[i], 1]
                        ss_d += mean_forest_b[b, leaf_numb, R_buf[i], 2]
                ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
                if ss >= pi_level and ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                    s_star[R_buf[i], s] = -1
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if R.size() == 0 and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef shap_values_acv_leaves_adap(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    int & scaling,
    vector[vector[vector[long]]] node_idx_trees, const long[:, :] S_star, const long[:, :] N_star, const long[:] size,
    const vector[vector[long]] C, int num_threads):

    cdef unsigned int d = values.shape[2]
    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:, :, :] phi
    phi = np.zeros((N, m, d))

    cdef double[:, :] phi_x
    cdef int i, j , k
    cdef const long[:] s_star_i, n_star_i

    for i in range(N):
        if size[i] != m:
            s_star_i = S_star[i, :size[i]]
            n_star_i = N_star[i, size[i]:]
            phi_x = single_shap_values_acv_leaves(X[i], data, values, partition_leaves_trees, leaf_idx_trees,
                            leaves_nb, scaling, node_idx_trees, s_star_i, n_star_i, C, num_threads)
        else:
            # Warning here....
            phi_x = single_shap_values(X[i], data, values, partition_leaves_trees, leaf_idx_trees,
                            leaves_nb, scaling, node_idx_trees, C, num_threads)

        for j in range(m):
            for k in range(d):
                phi[i, j, k] =  phi_x[j, k]
    return np.array(phi)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef single_shap_values_acv_leaves(const double[:] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    int scaling,
    vector[vector[vector[long]]] node_idx_trees, const long[:] S_star, const long[:] N_star,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :] phi
    phi = np.zeros((m, d))

    cdef double[ :, :,  :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, m, d))


    cdef double[:, :, :] phi_n
    phi_n = np.zeros((max_leaves, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs, len_n_star
    cdef double lm

    cdef long[:] va_c, N_star_a
    cdef int len_va_c, S_size, va_size,  b_it, nv_bool
    cdef long set_size

    cdef vector[vector[long]] C_buff, va_id, buff, Sm
    cdef vector[long] coal_va, remove_va, node_id_b
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)


    cdef vector[vector[vector[long]]] node_id_v2, C_b, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    # Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    N_star_a = np.array(N_star)
    len_n_star = N_star_a.shape[0]

    if C[0].size() != 0:
        C_buff = C
        coal_va = [C[ci][cj] for ci in range(C.size()) for cj in range(C[ci].size())]

        for i in range(S_star.shape[0]):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), S_star[i]) != coal_va.end():
                remove_va.push_back(S_star[i])
                buff[0][0] = S_star[i]
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), S_star[i]) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        va_id = [[S_star[i]] for i in range(S_star.shape[0])]

    m = va_id.size()

    cdef double p_s, p_si, coef, coef_0, p_off
    cdef double[:, :, :] lm_s, lm_si
    cdef double[:, :] lm_star
    lm_s = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_star = np.zeros((va_id.size(), max_leaves))

    for b in range(n_trees):
        for j in range(data.shape[1]):
            for i2 in range(d):
                for leaf_numb in range(phi_b.shape[0]):
                    phi_n[leaf_numb, j, i2] = 0
                    for counter in range(phi_b.shape[1]):
                        phi_b[leaf_numb, counter, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            for i in range(N_star.shape[0]):
                if (std_find[vector[long].iterator, long](node_idx_trees[b][leaf_numb].begin(), node_idx_trees[b][leaf_numb].end(), N_star[i]) != node_idx_trees[b][leaf_numb].end()):
                    std_remove[vector[long].iterator, long](node_idx_trees[b][leaf_numb].begin(), node_idx_trees[b][leaf_numb].end(), N_star[i])
                    node_idx_trees[b][leaf_numb].pop_back()
            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm = lm + 1


            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])

            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():

                    lm_star[va, leaf_numb] = 0
                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(len_n_star):
                            if ((data[i, N_star_a[s]] <= partition_leaves_trees[b, leaf_numb, N_star_a[s], 1]) * (data[i, N_star_a[s]] > partition_leaves_trees[b, leaf_numb, N_star_a[s], 0])):
                                b_it = b_it + 1

                        if b_it == len_n_star:
                            lm_star[va, leaf_numb] = lm_star[va, leaf_numb] +  1


                    p_s = 0
                    p_si = 0

                    o_all = 0
                    csi = 0
                    for s in range(len_n_star):
                        if ((X[N_star_a[s]] <= partition_leaves_trees[b, leaf_numb, N_star_a[s], 1]) * (X[N_star_a[s]] > partition_leaves_trees[b, leaf_numb, N_star_a[s], 0])):
                            o_all = o_all + 1

                    if o_all == len_n_star:
                        csi = 1

                    p_s = lm/data.shape[0]
                    p_si = (csi * lm)/lm_star[va, leaf_numb]
                    for nv in range(va_id[va].size()):
                        for i2 in range(d):
                            phi_n[leaf_numb, va_id[va][nv], i2] += (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s[va, leaf_numb, counter] = 0
                    lm_si[va, leaf_numb, counter] = 0

                    for nv in range(len_n_star):
                        S[va, counter, leaf_numb, S_size+nv] = N_star_a[nv]

                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size + len_n_star):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size + len_n_star:
                            lm_s[va, leaf_numb, counter] = lm_s[va, leaf_numb, counter] + 1
                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[va, leaf_numb, counter] = lm_si[va, leaf_numb, counter] + 1



                    p_s = 0
                    p_si = 0


                    csi = 0
                    cs = 0
                    o_all = 0
                    for s in range(S_size + len_n_star):
                        if ((X[S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (X[S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                            o_all = o_all + 1

                    if o_all == S_size + len_n_star:
                        cs = 1
                        nv_bool = 0
                        for nv in range(va_id[va].size()):
                            if ((X[va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (X[va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                nv_bool = nv_bool + 1
                                continue

                        if nv_bool == 0:
                            csi = 1

                    coef = 0
                    for l in range(1, m - set_size):
                        coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                    coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0


                    p_s = (cs * lm)/lm_s[va, leaf_numb, counter]
                    p_si = (csi * lm)/lm_si[va, leaf_numb, counter]

                    if S_size != 0:
                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

                    else:
                        p_off = lm/data.shape[0]
                        for nv in range(va_id[va].size()):
                            for i2 in range(d):
                                phi_b[leaf_numb, counter, va_id[va][nv], i2] += (p_si-p_off)*values[b, leaf_idx_trees[b, leaf_numb], i2] + coef * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for j in range(data.shape[1]):
            for i2 in range(d):
                for leaf_numb in range(phi_b.shape[0]):
                    phi[j, i2] += phi_n[leaf_numb, j, i2]
                    for counter in range(phi_b.shape[1]):
                        phi[j, i2] += phi_b[leaf_numb, counter, j, i2]

    return np.asarray(phi)/m



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef single_shap_values(const double[:] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    int scaling,
    vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int m = X.shape[0]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef double[:, :] phi
    phi = np.zeros((m, d))

    cdef double[ :, :, :, :] phi_b
    phi_b = np.zeros((max_leaves, 2**scaling, m, d))

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef double lm

    cdef vector[vector[long]] C_buff, va_id, buff
    cdef vector[long] coal_va, remove_va, node_id_b, buff_l
    buff.resize(max_leaves)
    for i in range(max_leaves):
        buff[i].resize(1)

    cdef vector[vector[vector[long]]] node_id_v2, C_b, Sm, node_id
    node_id_v2.resize(max_leaves)
    C_b.resize(max_leaves)
    Sm.resize(max_leaves)
    node_id.resize(max_leaves)

    cdef vector[vector[vector[vector[long]]]] node_id_va
    node_id_va.resize(m)
    for i in range(m):
        node_id_va[i].resize(max_leaves)

    cdef int S_size, va_size,  b_it, nv_bool

    cdef long set_size

    if C[0].size() != 0:
        C_buff = C
        for ci in range(C.size()):
            for cj in range(C[ci].size()):
                coal_va.push_back(C[ci][cj])

        for i in range(m):
            if not std_find[vector[long].iterator, long](coal_va.begin(), coal_va.end(), i) != coal_va.end():
                remove_va.push_back(i)
                buff[0][0] = i
                va_id.push_back(buff[0])
            else:
                for ci in range(C_buff.size()):
                    if (std_find[vector[long].iterator, long](C_buff[ci].begin(), C_buff[ci].end(), i) != C_buff[ci].end()):
                        va_id.push_back(C_buff[ci])
                        std_remove[vector[vector[long]].iterator, vector[long]](C_buff.begin(), C_buff.end(), C_buff[ci])
                        C_buff.pop_back()
                        break

    else:
        for i in range(m):
            buff[0][0] = i
            va_id.push_back(buff[0])

    cdef double p_s, p_si, coef, coef_0
    cdef double[:, :, :] lm_s, lm_si
    lm_s = np.zeros((va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((va_id.size(), max_leaves, 2**scaling))

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for j in range(m):
                    for i2 in range(d):
                        phi_b[leaf_numb, counter, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            lm = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm = lm + 1

            if C[0].size() != 0:
                C_b[leaf_numb] = C
                for nv in range(node_idx_trees[b][leaf_numb].size()):
                    add = 0
                    for ns in range(remove_va.size()):
                        if node_idx_trees[b][leaf_numb][nv] == remove_va[ns]:
                            add = 1
                            buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][nv]
                            node_id_v2[leaf_numb].push_back(buff[leaf_numb])
                            break
                    if add == 0:
                        for ci in range(C_b[leaf_numb].size()):
                            for cj in range(C_b[leaf_numb][ci].size()):
                                if C_b[leaf_numb][ci][cj] == node_idx_trees[b][leaf_numb][nv]:
                                    add = 1
                                    node_id_v2[leaf_numb].push_back(C_b[leaf_numb][ci])
                                    break
                            if add == 1:
                                std_remove[vector[vector[long]].iterator, vector[long]](C_b[leaf_numb].begin(), C_b[leaf_numb].end(), C_b[leaf_numb][ci])
                                C_b[leaf_numb].pop_back()
                                break
                node_id[leaf_numb] = node_id_v2[leaf_numb]
            else:
                node_id[leaf_numb].clear()
                for i in range(node_idx_trees[b][leaf_numb].size()):
                    buff[leaf_numb][0] = node_idx_trees[b][leaf_numb][i]
                    node_id[leaf_numb].push_back(buff[leaf_numb])


            for va in range(va_id.size()):
                node_id_va[va][leaf_numb] = node_id[leaf_numb]
                if not std_find[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va]) != node_id_va[va][leaf_numb].end():
                    continue

                std_remove[vector[vector[long]].iterator, vector[long]](node_id_va[va][leaf_numb].begin(), node_id_va[va][leaf_numb].end(), va_id[va])
                node_id_va[va][leaf_numb].pop_back()

                set_size = node_id_va[va][leaf_numb].size()
                pow_set_size = 2**set_size

                for counter in range(0, pow_set_size):
                    va_size = 0
                    S_size = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(node_id_va[va][leaf_numb][ci].size()):
                                S[va, counter, leaf_numb, S_size] = node_id_va[va][leaf_numb][ci][cj]
                                S_size = S_size + 1
                            va_size = va_size + 1

                    lm_s[va, leaf_numb, counter] = 0
                    lm_si[va, leaf_numb, counter] = 0
                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size:
                            lm_s[va, leaf_numb, counter] = lm_s[va, leaf_numb, counter] + 1

                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[va, leaf_numb, counter] = lm_si[va, leaf_numb, counter] + 1

                    csi = 0
                    cs = 0

                    o_all = 0
                    for s in range(S_size):
                        if ((X[S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (X[S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                            o_all = o_all + 1

                    if o_all == S_size:
                        cs = 1
                        nv_bool = 0
                        for nv in range(va_id[va].size()):
                            if ((X[va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (X[va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
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
                        p_s = (cs * lm)/lm_s[va, leaf_numb, counter] if lm_s[va, leaf_numb, counter] !=0 else 0
                    p_si = (csi * lm)/lm_si[va, leaf_numb, counter] if lm_si[va, leaf_numb, counter] !=0 else 0

                    for nv in range(va_id[va].size()):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, va_id[va][nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

        for j in range(m):
            for i2 in range(d):
                for leaf_numb in range(phi_b.shape[0]):
                    for counter in range(phi_b.shape[1]):
                        phi[j, i2] += phi_b[leaf_numb, counter, j, i2]

    return np.asarray(phi)/m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_clf_approx(double[:, :] X, long[:] fX,
            long[:] y_pred, double[:, :] data,
            double[:, :, :] values, double[:, :, :, :] partition_leaves_trees,
            long[:, :] leaf_idx_trees, long[:] leaves_nb, double[:] scaling, list C, double pi_level,
            int minimal, list search_space, bint stop):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]
    cdef int n_trees = values.shape[0]
    cdef double[:, :, :] leaves_tree
    cdef double[:, :] leaf_part
    cdef double[:] value
    cdef vector[int].iterator t

    cdef double[:, :] mean_forest
    mean_forest = np.zeros((N, 3))

    cdef double[:, :, :, :] mean_forest_b
    mean_forest_b = np.zeros((n_trees, leaf_idx_trees.shape[1], N, 3))

    cdef double[:] sdp
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef long[:] lm_u, lm_d
    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, lm, lm_s, max_size, size

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
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(minimal, m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break
    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)

    cdef double[:, :, :] p_n, p_u_n, p_d_n
    p_n = np.zeros((n_trees, max_leaves, N))
    p_u_n = np.zeros((n_trees, max_leaves, N))
    p_d_n = np.zeros((n_trees, max_leaves, N))
    cdef double n, n_u, n_d

    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(len(power))):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            for i in range(N):
                R_buf[i] = R[i]

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in range(nb_leaf):
                    for i in range(N):
                        p_n[b, leaf_numb, R_buf[i]] = 0
                        p_u_n[b, leaf_numb, R_buf[i]] = 0
                        p_d_n[b, leaf_numb, R_buf[i]] = 0
                        for j in range(3):
                            mean_forest_b[b, leaf_numb, R_buf[i], j] = 0

            for b in range(n_trees):
                nb_leaf = leaves_nb[b]
                for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
                    for i in range(N):
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
                            for s in range(data.shape[1]):
                                if ((data[j, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[j, s] >= partition_leaves_trees[b, leaf_numb, s, 0])):
                                    a_it = a_it + 1
                            for s in range(S_size):
                                if ((data[j, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) and (data[j, S[s]] >= partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                    b_it = b_it + 1

                            if a_it == data.shape[1]:
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

                        mean_forest_b[b, leaf_numb, R_buf[i], 0] += (p * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]]) /p_s if p_s != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 1] += (p_u * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]])/p_su if p_su != 0 else 0
                        mean_forest_b[b, leaf_numb, R_buf[i], 2] += (p_d * values[b, leaf_idx_trees[b, leaf_numb], fX[R_buf[i]]])/p_sd if p_sd != 0 else 0

                        p_n[b, leaf_numb, R_buf[i]] += (1.*p)/p_s if p_s != 0 else 0
                        p_u_n[b, leaf_numb, R_buf[i]] += (1.*p_u)/p_su if p_su != 0 else 0
                        p_d_n[b, leaf_numb, R_buf[i]] += (1.*p_d)/p_sd if p_sd != 0 else 0

            for i in range(N):
                ss_u = 0
                ss_d = 0
                ss_a = 0
                for b in range(n_trees):
                    nb_leaf = leaves_nb[b]
                    n = 0
                    n_u = 0
                    n_d = 0
                    for leaf_numb in range(nb_leaf):
                        n += p_n[b, leaf_numb, R_buf[i]]
                        n_u += p_u_n[b, leaf_numb, R_buf[i]]
                        n_d += p_d_n[b, leaf_numb, R_buf[i]]

                    for leaf_numb in range(nb_leaf):
                        ss_a += mean_forest_b[b, leaf_numb, R_buf[i], 0]/n if n !=0 else 0
                        ss_u += mean_forest_b[b, leaf_numb, R_buf[i], 1]/n_u if n_u !=0 else 0
                        ss_d += mean_forest_b[b, leaf_numb, R_buf[i], 2]/n_d if n_d !=0 else 0

                ss = (ss_a - ss_d)/(ss_u - ss_d) if ss_u - ss_d !=0 else 0
                if ss >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = ss
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)



cpdef double binomialC(unsigned long N, unsigned long k) nogil:
    cdef double r
    r = _comb_int_long(N, k)
    if r != 0:
        return r

cpdef double _comb_int_long(unsigned long N, unsigned long k) nogil:
    """
    Compute binom(N, k) for integers.
    Returns 0 if error/overflow encountered.
    """
    cdef double val
    cdef unsigned long long j, M, nterms

    if k > N or N == ULONG_MAX:
        return 0

    M = N + 1
    nterms = min(k, N - k)

    val = 1

    for j in range(1, nterms + 1):
        # Overflow check
        # if val > ULONG_MAX // (M - j):
        #   return 0

        val *= M - j
        val //= j

    return val

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rf_v0(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    sdp = 0

    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()
        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.push_back(i)
            in_data_b.push_back(i)

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):

                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])
                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                if in_data.size() < min_node_size:
                    break

            nodes_level = nodes_child

        if classifier == 1:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[in_data[i]])
        else:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * ((y_x - y_data[in_data[i]])*(y_x - y_data[in_data[i]]) <= t)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rf_v0(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_sdp_rf_v0(X[i], y_X[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t)
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_rf_v0(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_sdp_rf_v0(X[R_buf[i]],  y_X[R_buf[i]], data, y_data, S[:S_size], features, thresholds,  children_left, children_right,
                #                 max_depth, min_node_size, classifier, t)

            sdp_ba = compute_sdp_rf_v0(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t)

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_exp_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * y_data[deref(it)]
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * y_data[deref(it)]
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_exp_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_exp_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_quantile_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t, double[::1] & weights, double[::1] & samples, double[::1] & samples_child, double & quantile):

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    sdp = 0


    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()

        for i in range(N):
            samples[i] = 1
            samples_child[i] = 1

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):
                s = 0
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        for i in range(N):
                            samples_child[i] = samples[i] * (data[i, features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]])
                            s += samples_child[i]
                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])

                        for i in range(N):
                            samples_child[i] = samples[i] * (data[i, features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]])
                            s += samples_child[i]
                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                     for i in range(N):
                        samples_child[i] = samples[i]
                        s += samples_child[i]

                if s < min_node_size:
                    break
                else:
                    for i in range(N):
                        samples[i] = samples_child[i]

            nodes_level = nodes_child

        s = 0
        for i in range(N):
            s += samples[i]

        for i in range(N):
            weights[i] += samples[i] * (1/s)

    weights_py = np.array(weights)/n_trees
    sorter = np.argsort(y_data)
    y_quantile = weighted_percentile(y_data, quantile, weights_py, sorter)
    return y_quantile

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_quantile_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & quantile):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, ::1] weights, samples, samples_child
        weights = np.zeros((N, data.shape[0]))
        samples = np.ones((N, data.shape[0]))
        samples_child = np.ones((N, data.shape[0]))
        cdef int i
        for i in range(N):
            sdp[i] = single_compute_quantile_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, weights[i, :], samples[i, :], samples_child[i, :], quantile)
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_quantile_diff_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t, double[::1] & weights, double[::1] & samples, double[::1] & samples_child, double & quantile):

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    sdp = 0

    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    cdef double[::1] local_diff = np.zeros(N)

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()
        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.push_back(i)
            in_data_b.push_back(i)

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):

                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])
                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                if in_data.size() < min_node_size:
                    break

            nodes_level = nodes_child

        for i in range(in_data.size()):
            weights[in_data[i]] +=  1./in_data.size()
            # local_diff[i] = (y_data[i] - y_x)*(y_data[i] - y_x)
            local_diff[in_data[i]] = y_x - y_data[in_data[i]]
    weights_py = np.array(weights)/n_trees
    sorter = np.argsort(local_diff)
    y_quantile = weighted_percentile(local_diff, quantile, weights_py, sorter)
    return y_quantile

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_quantile_diff_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & quantile):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, ::1] weights, samples, samples_child
        weights = np.zeros((N, data.shape[0]))
        samples = np.ones((N, data.shape[0]))
        samples_child = np.ones((N, data.shape[0]))
        cdef int i
        for i in range(N):
            sdp[i] = single_compute_quantile_diff_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, weights[i, :], samples[i, :], samples_child[i, :], quantile)
        return np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdf_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    sdp = 0

    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()
        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.push_back(i)
            in_data_b.push_back(i)

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):

                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])
                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                if in_data.size() < min_node_size:
                    break

            nodes_level = nodes_child

        for i in range(in_data.size()):
            sdp  += (1./(n_trees*in_data.size())) * (y_data[in_data[i]] <= t)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdf_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True):
            sdp[i] = single_compute_cdf_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sufficient_expl_rf_v0(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d, K
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, subset, a, k
    K = 0

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    cdef vector[vector[vector[long]]] sufficient = [[[-1]] for i in range(N)]
    cdef vector[vector[double]] sufficient_sdp = [[-1] for i in range(N)]

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            sdp = compute_sdp_rf_v0(X, y_X, data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t)

            for i in range(N):
                if sdp[i] >= pi_level:
                    subset = 0
                    for j in range(sufficient[i].size()):
                        a = 0
                        for k in range(sufficient[i][j].size()):
                            if std_find[vector[long].iterator, long](power_cpp[s_0][s_1].begin(), power_cpp[s_0][s_1].end(), sufficient[i][j][k]) != power_cpp[s_0][s_1].end():
                                a += 1
                        if a == sufficient[i][j].size():
                            subset = 1
                            break

                    if subset == 0:
                        sufficient[i].push_back(power_cpp[s_0][s_1])
                        sufficient_sdp[i].push_back(sdp[i])
                        for s in range(S_size):
                            sdp_global[S[s]] += 1
                        K += 1

    return sufficient, sufficient_sdp, np.asarray(sdp_global)/K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rule_v0(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t, double[:, :] & partition) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()
        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.push_back(i)
            in_data_b.push_back(i)

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):

                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], 1] >= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], 1] = thresholds[b, nodes_level[it_node]]

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], 0] <= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], 0] = thresholds[b, nodes_level[it_node]]

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                if in_data.size() < min_node_size:
                    break

            nodes_level = nodes_child

        if classifier == 1:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[in_data[i]])
        else:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * ((y_x - y_data[in_data[i]])*(y_x - y_data[in_data[i]]) <= t)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rule_v0(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs
        partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_sdp_rule_v0(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, partition_byobs[i])

        return np.array(sdp), np.array(partition_byobs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule_v0(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs
        partition_byobs = buffer

        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], 2))
        buffer_data[:, :, :, 0] = -1e+10

        cdef double [:, :, :, :] partition_byobs_data
        partition_byobs_data = buffer_data

        cdef int N = X.shape[0]
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))
        cdef double[:] sdp = np.zeros(N)
        cdef int i, j, k


        for i in range(N):
            sdp[i] = single_compute_sdp_rule_v0(X[i], y_X[i], data, y_data, S[i],
                            features, thresholds, children_left, children_right,
                            max_depth, min_node_size, classifier, t, partition_byobs[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                sdp_data[i, j] = single_compute_sdp_rule_v0(data[j], y_X[i], data, y_data, S[i],
                            features, thresholds, children_left, children_right,
                            max_depth, min_node_size, classifier, t, partition_byobs_data[i, j])

        return np.array(sdp), np.array(partition_byobs), np.array(sdp_data), np.array(partition_byobs_data)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rule_biased(double[:] & x, double & y_x, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t, double[:, :, :, :] & partition) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:

                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], b, level, 1] >= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], b, level,  1] = thresholds[b, nodes_level[it_node]]

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], b, level,  0] <= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], b, level, 0] = thresholds[b, nodes_level[it_node]]

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

            nodes_level = nodes_child

    return sdp

cpdef compile_rules(rule_data, data, S, min_node_size):
    ntree = rule_data.shape[1]
    rule_trees_up = []
    rule_trees_dn = []
    for i in range(ntree):
        tn = 0
        depth = rule_data.shape[2]
        while(tn <= min_node_size):
            part_up = np.min(rule_data[:, i, :depth, 1], axis=-1)
            part_dn = np.max(rule_data[:, i, :depth, 0], axis=-1)
            where = np.prod([(data[:, s] <= part_up[s]) * (data[:, s] >= part_dn[s])
                                             for s in S], axis=0).astype(bool)
            tn = np.sum(where)
            depth -= 1

        rule_trees_up.append(np.expand_dims(part_up, -1))
        rule_trees_dn.append(np.expand_dims(part_dn, -1))
    rule_trees_up = np.min(np.concatenate(rule_trees_up, axis=-1), axis=-1)
    rule_trees_dn = np.max(np.concatenate(rule_trees_dn, axis=-1), axis=-1)
    rule_trees = np.concatenate([np.expand_dims(rule_trees_dn, axis=-1), np.expand_dims(rule_trees_up, axis=-1)],
                                axis=-1)
    return rule_trees


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rule_biased(double[:, :] X, double[::1] y_X, double[:, :] data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], children_left.shape[0], max_depth , 2))
        buffer[:, :, :, :, 0] = -1e+10

        cdef double [:, :, :, :, :] partition_byobs
        partition_byobs = buffer

        rules = np.zeros(shape=(X.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in range(N):
            sdp[i] = single_compute_sdp_rule_biased(X[i], y_X[i], S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, partition_byobs[i])

        for i in range(N):
            rules[i] = compile_rules(partition_byobs[i], data, S[i], min_node_size)


        return np.array(sdp), rules


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule_biased(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], children_left.shape[0], max_depth , 2))
        buffer[:, :, :, :, 0] = -1e+10

        cdef double [:, :, :, :, :] partition_byobs
        partition_byobs = buffer

        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], children_left.shape[0], max_depth , 2))
        buffer_data[:, :, :, :, :, 0] = -1e+10

        cdef double [:, :, :, :, :, :] partition_byobs_data
        partition_byobs_data = buffer_data

        rules = np.zeros(shape=(X.shape[0], X.shape[1], 2))
        rules_data = np.zeros(shape=(X.shape[0], data.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))

        cdef int i, j
        for i in range(N):
            sdp[i] = single_compute_sdp_rule_biased(X[i], y_X[i], S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, partition_byobs[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                sdp_data[i, j] = single_compute_sdp_rule_biased(data[j], y_X[i], S[i],
                            features, thresholds, children_left, children_right,
                            max_depth, min_node_size, classifier, t, partition_byobs_data[i, j])

        for i in range(N):
            rules[i] = compile_rules(partition_byobs[i], data, S[i], min_node_size)
            for j in range(data.shape[0]):
                rules_data[i, j] = compile_rules(partition_byobs_data[i, j], data, S[i], min_node_size)

        return np.array(sdp), rules, np.array(sdp_data), rules_data

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rule_v0_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double & t, double[:, :] & partition, double[:] & w) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0
    cdef unsigned int b, level, it_node, i
    cdef vector[int] nodes_level, nodes_child, in_data, in_data_b

    for b in range(n_trees):
        nodes_level.clear()
        nodes_level.push_back(0)
        nodes_child.clear()
        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.push_back(i)
            in_data_b.push_back(i)

        for level in range(max_depth):
            for it_node in range(nodes_level.size()):

                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, nodes_level[it_node]]) != S.end():
                    if x[features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                        nodes_child.push_back(children_left[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], 1] >= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], 1] = thresholds[b, nodes_level[it_node]]

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] > thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                    else:
                        nodes_child.push_back(children_right[b, nodes_level[it_node]])

                        if partition[features[b, nodes_level[it_node]], 0] <= thresholds[b, nodes_level[it_node]]:
                            partition[features[b, nodes_level[it_node]], 0] = thresholds[b, nodes_level[it_node]]

                        for i in range(in_data.size()):
                            if data[in_data[i], features[b, nodes_level[it_node]]] <= thresholds[b, nodes_level[it_node]]:
                                std_remove[vector[int].iterator, int](in_data_b.begin(), in_data_b.end(), in_data[i])
                                in_data_b.pop_back()
                        in_data = in_data_b

                else:
                     nodes_child.push_back(children_left[b, nodes_level[it_node]])
                     nodes_child.push_back(children_right[b, nodes_level[it_node]])

                if in_data.size() < min_node_size:
                    break

            nodes_level = nodes_child

        if classifier == 1:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[in_data[i]])
                w[in_data[i]] += (1./(n_trees*in_data.size()))
        else:
            for i in range(in_data.size()):
                sdp  += (1./(n_trees*in_data.size())) * ((y_x - y_data[in_data[i]])*(y_x - y_data[in_data[i]]) <= t)
                w[in_data[i]] += (1./(n_trees*in_data.size()))
    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule_opti(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs = buffer

        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], 2))
        buffer_data[:, :, :, 0] = -1e+10

        cdef double [:, :, :, :] partition_byobs_data = buffer_data

        rules_data = np.zeros(shape=(X.shape[0], data.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))

        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in range(N):
            sdp[i] = single_compute_sdp_rule_v0_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, partition_byobs[i], weights[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                if weights[i, j] != 0:
                    sdp_data[i, j] = single_compute_sdp_rule_v0(data[j], y_X[i], data, y_data, S[i],
                                features, thresholds, children_left, children_right,
                                max_depth, min_node_size, classifier, t, partition_byobs_data[i, j])

        return np.array(sdp), np.array(partition_byobs), np.array(sdp_data), np.array(partition_byobs_data), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule_biased_v2(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs
        partition_byobs = buffer


        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], children_left.shape[0], max_depth , 2))
        buffer_data[:, :, :, :, :, 0] = -1e+10

        cdef double [:, :, :, :, :, :] partition_byobs_data
        partition_byobs_data = buffer_data

        rules_data = np.zeros(shape=(X.shape[0], data.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))

        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in range(N):
            sdp[i] = single_compute_sdp_rule_v0_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t, partition_byobs[i], weights[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                if weights[i, j] != 0:
                    sdp_data[i, j] = single_compute_sdp_rule_biased(data[j], y_X[i], S[i],
                                features, thresholds, children_left, children_right,
                                max_depth, min_node_size, classifier, t, partition_byobs_data[i, j])

        for i in range(N):
            for j in range(data.shape[0]):
                if weights[i, j] != 0:
                    rules_data[i, j] = compile_rules(partition_byobs_data[i, j], data, S[i], min_node_size)

        return np.array(sdp), np.array(partition_byobs), np.array(sdp_data), rules_data, np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rule(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1])
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rule(double[:, :] & X, double[::1] & y_X,  double[:, :] & data, double[::1] & y_data, vector[vector[int]] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_sdp_rule(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i])

        return np.array(sdp), np.array(partition_byobs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rule_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs = buffer

        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], 2))
        buffer_data[:, :, :, 0] = -1e+10

        cdef double [:, :, :, :] partition_byobs_data = buffer_data

        rules_data = np.zeros(shape=(X.shape[0], data.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))

        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in range(N):
            sdp[i] = single_compute_sdp_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                if weights[i, j] != 0:
                    sdp_data[i, j] = single_compute_sdp_rule(data[j], y_X[i], data, y_data, S[i],
                                features, thresholds, children_left, children_right,
                                max_depth, min_node_size, classifier, t[i], partition_byobs_data[i, j])

        return np.array(sdp), np.array(partition_byobs), np.array(sdp_data), np.array(partition_byobs_data), np.array(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_maxrule_verbose(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10

        cdef double [:, :, :] partition_byobs = buffer

        buffer_data = 1e+10 * np.ones(shape=(X.shape[0], data.shape[0], X.shape[1], 2))
        buffer_data[:, :, :, 0] = -1e+10

        cdef double [:, :, :, :] partition_byobs_data = buffer_data

        rules_data = np.zeros(shape=(X.shape[0], data.shape[0], X.shape[1], 2))

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] sdp_data = np.zeros((N, data.shape[0]))

        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in tqdm(range(N)):
            sdp[i] = single_compute_sdp_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])

            for j in prange(data.shape[0], nogil=True, schedule='dynamic'):
                if weights[i, j] != 0:
                    sdp_data[i, j] = single_compute_sdp_rule(data[j], y_X[i], data, y_data, S[i],
                                features, thresholds, children_left, children_right,
                                max_depth, min_node_size, classifier, t[i], partition_byobs_data[i, j])

        return np.array(sdp), np.array(partition_byobs), np.array(sdp_data), np.array(partition_byobs_data), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_ddp_rule(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (1-(t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_rule(double[:, :] & X, double[::1] & y_X,  double[:, :] & data, double[::1] & y_data, vector[vector[int]] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_rule(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i])

        return np.array(sdp), np.array(partition_byobs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_ddp_rule_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (1-(t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_weights(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])

        return np.array(sdp), np.array(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_weights_verbose(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in tqdm(range(N)):
            sdp[i] = single_compute_ddp_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])


        return np.array(sdp), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_ddp_cond_rule_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w, const double[:, :] & cond) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])

                    if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                        partition[features[b, it_node], 1] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 0] > thresholds[b, it_node]:
                    nodes_child.insert(children_right[b, it_node])

                    if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                        partition[features[b, it_node], 0] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] >= thresholds[b, it_node] and cond[features[b, it_node], 0] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                    # if partition[features[b, it_node], 0] <= thresholds[b, it_node]: # TO DO: compute correctly the partition in this case
                    #     partition[features[b, it_node], 0] = cond[features[b, it_node], 0]

                    # if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                    #     partition[features[b, it_node], 1] = cond[features[b, it_node], 1]

                    # it = in_data.begin()
                    # while(it != in_data.end()):
                    #     if data[deref(it), features[b, it_node]] < cond[features[b, it_node], 0] or data[deref(it), features[b, it_node]] > cond[features[b, it_node], 1]:
                    #         in_data_b.erase(deref(it))
                    #     inc(it)
                    # in_data = in_data_b

                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (1-(t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_cond_weights(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_cond_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_sdp_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1])
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_sdp_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_ddp_rf(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (1-(t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_rf(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_rf_same_set(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_rf(X[i], y_X[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_ddp_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)
    t_arr = np.array(t, dtype=np.double)

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)

    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)
    S_bar = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m+1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            S_bar_set = list(range(data.shape[1]))
            for i in range(S_size):
                S_bar_set.remove(S[i])

            for i in range(data.shape[1]-S_size):
                S_bar[i] = S_bar_set[i]

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_ddp_rf(X[R_buf[i]],  y_X[R_buf[i]], data, y_data, S_bar[:m-S_size], features, thresholds,  children_left, children_right,
                #                 max_depth, min_node_size, classifier, t)

            sdp_ba = compute_ddp_rf_same_set(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S_bar[:data.shape[1]-S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t_arr[np.array(in_data, dtype=bool)])

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_sdp_rf_same_set(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_sdp_rf(X[i], y_X[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_sdp_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)
    t_arr = np.array(t, dtype=np.double)

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_sdp_rf(X[R_buf[i]],  y_X[R_buf[i]], data, y_data, S[:S_size], features, thresholds,  children_left, children_right,
                #                 max_depth, min_node_size, classifier, t)

            sdp_ba = compute_sdp_rf_same_set(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t_arr[np.array(in_data, dtype=bool)])

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sufficient_expl_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d, K
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, subset, a, k
    K = 0

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    cdef vector[vector[vector[long]]] sufficient = [[[-1]] for i in range(N)]
    cdef vector[vector[double]] sufficient_sdp = [[-1] for i in range(N)]

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            sdp = compute_sdp_rf_same_set(X, y_X, data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t)

            for i in range(N):
                if sdp[i] >= pi_level:
                    subset = 0
                    for j in range(sufficient[i].size()):
                        a = 0
                        for k in range(sufficient[i][j].size()):
                            if std_find[vector[long].iterator, long](power_cpp[s_0][s_1].begin(), power_cpp[s_0][s_1].end(), sufficient[i][j][k]) != power_cpp[s_0][s_1].end():
                                a += 1
                        if a == sufficient[i][j].size():
                            subset = 1
                            break

                    if subset == 0:
                        sufficient[i].push_back(power_cpp[s_0][s_1])
                        sufficient_sdp[i].push_back(sdp[i])
                        for s in range(S_size):
                            sdp_global[S[s]] += 1
                        K += 1

    return sufficient, sufficient_sdp, np.asarray(sdp_global)/K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef sufficient_cxpl_rf(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    sdp = np.zeros((N))
    sdp_global = np.zeros((m))

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d, K
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size, subset, a, k
    K = 0

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    cdef vector[vector[vector[long]]] sufficient = [[[-1]] for i in range(N)]
    cdef vector[vector[double]] sufficient_sdp = [[-1] for i in range(N)]

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            sdp = compute_ddp_rf_same_set(X, y_X, data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t)

            for i in range(N):
                if sdp[i] >= pi_level:
                    subset = 0
                    for j in range(sufficient[i].size()):
                        a = 0
                        for k in range(sufficient[i][j].size()):
                            if std_find[vector[long].iterator, long](power_cpp[s_0][s_1].begin(), power_cpp[s_0][s_1].end(), sufficient[i][j][k]) != power_cpp[s_0][s_1].end():
                                a += 1
                        if a == sufficient[i][j].size():
                            subset = 1
                            break

                    if subset == 0:
                        sufficient[i].push_back(power_cpp[s_0][s_1])
                        sufficient_sdp[i].push_back(sdp[i])
                        for s in range(S_size):
                            sdp_global[S[s]] += 1
                        K += 1

    return sufficient, sufficient_sdp, np.asarray(sdp_global)/K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_ddp_intv_rule_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w, const double[:, :] & cond) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])

                    if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                        partition[features[b, it_node], 1] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 0] > thresholds[b, it_node]:
                    nodes_child.insert(children_right[b, it_node])

                    if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                        partition[features[b, it_node], 0] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] >= thresholds[b, it_node] and cond[features[b, it_node], 0] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                    # if partition[features[b, it_node], 0] <= thresholds[b, it_node]: # TO DO: compute correctly the partition in this case
                    #    partition[features[b, it_node], 0] = cond[features[b, it_node], 0]

                    # if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                    #    partition[features[b, it_node], 1] = cond[features[b, it_node], 1]

                    # it = in_data.begin()
                    # while(it != in_data.end()):
                    #     if data[deref(it), features[b, it_node]] < cond[features[b, it_node], 0] or data[deref(it), features[b, it_node]] > cond[features[b, it_node], 1]:
                    #         in_data_b.erase(deref(it))
                    #     inc(it)
                    # in_data = in_data_b

                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (1-(t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_intv_weights(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_intv_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_ddp_intv_same_set(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_ddp_intv_rule_weights(X[i], y_X[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_ddp_intv(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)
    cond_arr = np.array(cond)
    t_arr = np.array(t, dtype=np.double)

    # buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
    # buffer[:, :, 0] = -1e+10
    # cdef double [:, :, :] partition_byobs = buffer
    # cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_ddp_intv_rule_weights(X[R_buf[i]],  y_X[R_buf[i]], data, y_data, S[:S_size],
                #                                          features, thresholds,  children_left, children_right,
                #                                          max_depth, min_node_size, classifier, t,
                #                                          partition_byobs[R_buf[i]], weights[R_buf[i]], cond[R_buf[i]])

            sdp_ba = compute_ddp_intv_same_set(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, cond_arr[np.array(in_data, dtype=bool)], max_depth, min_node_size, classifier, t_arr[np.array(in_data, dtype=bool)])

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdp_rule(double[:] & x, double & y_x, double & down, double & up, double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (down<=y_data[deref(it)]<=up)
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_rule(double[:, :] & X, double[::1] & y_X,  double[::1] down, double[::1] up, double[:, :] & data, double[::1] & y_data, vector[vector[int]] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_rule(X[i], y_X[i], down[i], up[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i])

        return np.array(sdp), np.array(partition_byobs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdp_cond_rule_weights(double[:] & x, double & y_x, double & down, double & up,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w, const double[:, :] & cond) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])

                    if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                        partition[features[b, it_node], 1] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 0] > thresholds[b, it_node]:
                    nodes_child.insert(children_right[b, it_node])

                    if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                        partition[features[b, it_node], 0] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] >= thresholds[b, it_node] and cond[features[b, it_node], 0] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                    # if partition[features[b, it_node], 0] <= thresholds[b, it_node]: # TO DO: compute correctly the partition in this case
                    #     partition[features[b, it_node], 0] = cond[features[b, it_node], 0]

                    # if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                    #     partition[features[b, it_node], 1] = cond[features[b, it_node], 1]

                    # it = in_data.begin()
                    # while(it != in_data.end()):
                    #     if data[deref(it), features[b, it_node]] < cond[features[b, it_node], 0] or data[deref(it), features[b, it_node]] > cond[features[b, it_node], 1]:
                    #         in_data_b.erase(deref(it))
                    #     inc(it)
                    # in_data = in_data_b

                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (down <=y_data[deref(it)]<= up)
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_cond_weights(double[:, :] X, double[::1] y_X, double[::1] down, double[::1] up,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_cond_rule_weights(X[i], y_X[i], down[i], up[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp), np.array(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdp_rf(double[:] & x, double & y_x,  double & down, double & up, double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (down<=y_data[deref(it)]<= up)
                inc(it)

    return sdp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_rf(double[:, :] X, double[::1] y_X, double[::1] down, double[::1] up,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_rf(X[i], y_X[i], down[i], up[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_rf_same_set(double[:, :] X, double[::1] y_X, double[::1] down, double[::1] up,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef int i
        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_rf(X[i], y_X[i], down[i], up[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i])
        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_cdp_rf(double[:, :] X, double[::1] y_X, double[::1] DOWN, double[::1] UP, double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)
    down_arr = np.array(DOWN, dtype=np.double)
    up_arr = np.array(UP, dtype=np.double)
    t_arr = np.array(t, dtype=np.double)

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)

    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)
    S_bar = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m+1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            S_bar_set = list(range(data.shape[1]))
            for i in range(S_size):
                S_bar_set.remove(S[i])

            for i in range(data.shape[1]-S_size):
                S_bar[i] = S_bar_set[i]

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_ddp_rf(X[R_buf[i]],  y_X[R_buf[i]], down[R_buf[i]], up[R[buf[i]],  data, y_data, S_bar[:m-S_size], features, thresholds,  children_left, children_right,
                #                 max_depth, min_node_size, classifier, t)

            sdp_ba = compute_cdp_rf_same_set(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)], down_arr[np.array(in_data, dtype=bool)],
                                     up_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S_bar[:data.shape[1]-S_size], features, thresholds,  children_left,
                                    children_right, max_depth, min_node_size, classifier, t_arr[np.array(in_data, dtype=bool)])

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdp_rule_weights(double[:] & x, double & y_x, double & down, double & up,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    if x[features[b, it_node]] <= thresholds[b, it_node]:
                        nodes_child.insert(children_left[b, it_node])

                        if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                            partition[features[b, it_node], 1] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b

                    else:
                        nodes_child.insert(children_right[b, it_node])

                        if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                            partition[features[b, it_node], 0] = thresholds[b, it_node]

                        it = in_data.begin()
                        while(it != in_data.end()):
                            if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                                in_data_b.erase(deref(it))
                            inc(it)
                        in_data = in_data_b
                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x == y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (down<=y_data[deref(it)]<=up)
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_weights(double[:, :] X, double[::1] y_X, double[::1] down,  double[::1] up,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_rule_weights(X[i], y_X[i], down[i], up[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])

        return np.array(sdp), np.array(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_weights_verbose(double[:, :] X, double[::1] y_X, double[::1] down,  double[::1] up,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, double & pi):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in tqdm(range(N)):
            sdp[i] = single_compute_cdp_rule_weights(X[i], y_X[i], down[i], up[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i])


        return np.array(sdp), np.array(weights)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_cdp_intv_rule_weights(double[:] & x, double & y_x,  double[:, :] & data, double[::1] & y_data, vector[int] & S,
        int[:, :] & features, double[:, :] & thresholds,  int[:, :] & children_left, int[:, :] & children_right,
        int & max_depth, int & min_node_size, int & classifier, double[:] & t, double[:, :] & partition,
        double[:] & w, const double[:, :] & cond) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] nodes_level, nodes_child, in_data, in_data_b
    cdef set[int].iterator it, it_point

    for b in range(n_trees):
        nodes_child.clear()
        nodes_level.clear()
        nodes_level.insert(0)

        in_data.clear()
        in_data_b.clear()
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        for level in range(max_depth):
            it_point = nodes_level.begin()
            while(it_point != nodes_level.end()):
                it_node = deref(it_point)
                if std_find[vector[int].iterator, int](S.begin(), S.end(), features[b, it_node]) != S.end():
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])

                    if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                        partition[features[b, it_node], 1] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 0] > thresholds[b, it_node]:
                    nodes_child.insert(children_right[b, it_node])

                    if partition[features[b, it_node], 0] <= thresholds[b, it_node]:
                        partition[features[b, it_node], 0] = thresholds[b, it_node]

                    it = in_data.begin()
                    while(it != in_data.end()):
                        if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                            in_data_b.erase(deref(it))
                        inc(it)
                    in_data = in_data_b

                elif features[b, it_node] >=0 and cond[features[b, it_node], 1] >= thresholds[b, it_node] and cond[features[b, it_node], 0] <= thresholds[b, it_node]:
                    nodes_child.insert(children_left[b, it_node])
                    nodes_child.insert(children_right[b, it_node])

                    # if partition[features[b, it_node], 0] <= thresholds[b, it_node]: # TO DO: compute correctly the partition in this case
                    #    partition[features[b, it_node], 0] = cond[features[b, it_node], 0]

                    # if partition[features[b, it_node], 1] >= thresholds[b, it_node]:
                    #    partition[features[b, it_node], 1] = cond[features[b, it_node], 1]

                    # it = in_data.begin()
                    # while(it != in_data.end()):
                    #     if data[deref(it), features[b, it_node]] < cond[features[b, it_node], 0] or data[deref(it), features[b, it_node]] > cond[features[b, it_node], 1]:
                    #         in_data_b.erase(deref(it))
                    #     inc(it)
                    # in_data = in_data_b

                else:
                     nodes_child.insert(children_left[b, it_node])
                     nodes_child.insert(children_right[b, it_node])

                if in_data.size() < min_node_size:
                    break
                inc(it_point)

            nodes_level = nodes_child

        if classifier == 1:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * (y_x != y_data[deref(it)])
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)
        else:
            it = in_data.begin()
            while(it != in_data.end()):
                sdp  += (1./(n_trees*in_data.size())) * ((t[0]<=y_data[deref(it)])*(y_data[deref(it)] <= t[1]))
                w[deref(it)] += (1./(n_trees*in_data.size()))
                inc(it)

    return sdp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_intv_weights(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[vector[int]] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_intv_rule_weights(X[i], y_X[i], data, y_data, S[i],
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp), np.array(weights)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_cdp_intv_same_set(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data, vector[int] S,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t):

        buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
        buffer[:, :, 0] = -1e+10
        cdef double [:, :, :] partition_byobs = buffer

        cdef int N = X.shape[0]
        cdef double[::1] sdp = np.zeros(N)
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef int i, j

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_cdp_intv_rule_weights(X[i], y_X[i], data, y_data, S,
                        features, thresholds, children_left, children_right,
                        max_depth, min_node_size, classifier, t[i], partition_byobs[i], weights[i], cond[i])

        return np.array(sdp)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef global_cdp_intv(double[:, :] X, double[::1] y_X,  double[:, :] data, double[::1] y_data,
        int[:, :] features, double[:, :] thresholds,  int[:, :] children_left, int[:, :] children_right,
        const double [:, :, :] & cond,
        int max_depth, int min_node_size, int & classifier, double[:, :] & t, list C, double pi_level,
            int minimal, bint stop, list search_space):

    cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[1]

    cdef double[:] sdp, sdp_b, sdp_ba
    cdef double[:] sdp_global
    cdef double[:] in_data
    sdp = np.zeros((N))
    sdp_b = np.zeros((N))
    sdp_global = np.zeros((m))
    in_data = np.zeros(N)

    cdef unsigned int it, it_s, a_it, b_it, o_all, p, p_s, nb_leaf, p_u, p_d, p_su, p_sd, down, up
    cdef double ss, ss_a, ss_u, ss_d
    cdef int b, leaf_numb, i, s, s_0, s_1, S_size, j, max_size, size

    cdef vector[int] S, len_s_star
    len_s_star = np.zeros((N), dtype=np.int)

    cdef list power, va_id

    cdef vector[long] R, r
    R.resize(N)
    for i in range(N):
        R[i] = i
    r.resize(N)

    cdef long[:] R_buf
    R_buf = np.zeros((N), dtype=np.int)

    X_arr = np.array(X, dtype=np.double)
    y_X_arr = np.array(y_X, dtype=np.double)
    cond_arr = np.array(cond)
    t_arr = np.array(t, dtype=np.double)

    # buffer = 1e+10 * np.ones(shape=(X.shape[0], X.shape[1], 2))
    # buffer[:, :, 0] = -1e+10
    # cdef double [:, :, :] partition_byobs = buffer
    # cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))

    if C[0] != []:
        remove_va = [C[ci][cj] for ci in range(len(C)) for cj in range(len(C[ci]))]
        va_id = [[i] for i in search_space if i not in remove_va]
        for ci in range(len(C)):
            i = 0
            for cj in range(len(C[ci])):
                if C[ci][cj] in search_space:
                    i += 1
                    break
            if i != 0:
                va_id += [C[ci]]
    else:
        va_id = [[i] for i in search_space]

    m = len(va_id)
    power = []
    max_size = 0
    for size in range(m + 1):
        power_b = []
        for co in itertools.combinations(va_id, size):
            power_b.append(np.array(sum(list(co),[])))
            max_size += 1
        power.append(power_b)
        if max_size >= 2**15:
            break

    cdef vector[vector[vector[long]]] power_cpp = power
    cdef long[:, :] s_star
    s_star = -1*np.ones((N, X.shape[1]), dtype=np.int)


    cdef long power_set_size = 2**m
    S = np.zeros((data.shape[1]), dtype=np.int)

    for s_0 in tqdm(range(minimal, m + 1)):
        for s_1 in range(0, power_cpp[s_0].size()):
            for i in range(power_cpp[s_0][s_1].size()):
                S[i] = power_cpp[s_0][s_1][i]

            S_size = power_cpp[s_0][s_1].size()
            r.clear()
            N = R.size()
            in_data = np.zeros(X.shape[0])

            for i in range(N):
                R_buf[i] = R[i]
                in_data[R_buf[i]] = 1

                # sdp_b[R_buf[i]] = single_compute_cdp_intv_rule_weights(X[R_buf[i]],  y_X[R_buf[i]], data, y_data, S[:S_size],
                #                                          features, thresholds,  children_left, children_right,
                #                                          max_depth, min_node_size, classifier, t,
                #                                          partition_byobs[R_buf[i]], weights[R_buf[i]], cond[R_buf[i]])

            sdp_ba = compute_cdp_intv_same_set(X_arr[np.array(in_data, dtype=bool)],
                                    y_X_arr[np.array(in_data, dtype=bool)],
                                    data, y_data, S[:S_size], features, thresholds,  children_left,
                                    children_right, cond_arr[np.array(in_data, dtype=bool)], max_depth, min_node_size, classifier, t_arr[np.array(in_data, dtype=bool)])

            for i in range(N):
                sdp_b[R_buf[i]] = sdp_ba[i]
                if sdp_b[R_buf[i]] >= sdp[R_buf[i]]:
                    sdp[R_buf[i]] = sdp_b[R_buf[i]]
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]

                if S_size == X.shape[1]:
                    sdp[R_buf[i]] = 1
                    len_s_star[R_buf[i]] = S_size
                    for s in range(S_size):
                        s_star[R_buf[i], s] = S[s]
                    for s in range(len_s_star[R_buf[i]], X.shape[1]): # to filter (important for coalition)
                        s_star[R_buf[i], s] = -1

        for i in range(N):
            if sdp[R_buf[i]] >= pi_level:
                r.push_back(R[i])
                for s in range(len_s_star[R_buf[i]]):
                    sdp_global[s_star[R_buf[i], s]] += 1

        for i in range(r.size()):
            std_remove[vector[long].iterator, long](R.begin(), R.end(), r[i])
            R.pop_back()

        if (R.size() == 0 or S_size >= X.shape[1]/2) and stop:
            break

    return np.asarray(sdp_global)/X.shape[0], np.array(s_star, dtype=np.long), np.array(len_s_star, dtype=np.long), np.array(sdp)
