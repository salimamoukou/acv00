# distutils: language = c++

from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
ctypedef np.float64_t double
cimport cython
from scipy.special import comb
import itertools
from tqdm import tqdm
from cython.parallel cimport prange, parallel, threadid
cimport openmp

cdef extern from "<algorithm>" namespace "std" nogil:
     iter std_remove "std::remove" [iter, T](iter first, iter last, const T& val)
     iter std_find "std::find" [iter, T](iter first, iter last, const T& val)

cdef extern from "limits.h":
    unsigned long ULONG_MAX


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
cpdef leaves_cache(
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int m = data.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, csi, cs
    cdef double[:, :] lm
    lm = np.zeros((n_trees, max_leaves))

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
    cdef double[:, :, :, :] lm_s, lm_si
    lm_s = np.zeros((n_trees, va_id.size(), max_leaves, 2**scaling))
    lm_si = np.zeros((n_trees, va_id.size(), max_leaves, 2**scaling))

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()
            lm[b, leaf_numb] = 0
            for i in range(data.shape[0]):
                a_it = 0
                for s in range(data.shape[1]):
                    if (data[i, s] <= partition_leaves_trees[b, leaf_numb, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_numb, s, 0]):
                        a_it = a_it + 1
                if a_it == data.shape[1]:
                    lm[b, leaf_numb] = lm[b, leaf_numb] + 1

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

                    for i in range(data.shape[0]):
                        b_it = 0
                        for s in range(S_size):
                            if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_numb, S[va, counter, leaf_numb, s], 0])):
                                b_it = b_it + 1

                        if b_it == S_size:
                            lm_s[b, va, leaf_numb, counter] = lm_s[b, va, leaf_numb, counter] + 1

                            nv_bool = 0
                            for nv in range(va_id[va].size()):
                                if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_numb, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_numb, va_id[va][nv], 0])):
                                    nv_bool = nv_bool + 1
                                    continue

                            if nv_bool == 0:
                                lm_si[b, va, leaf_numb, counter] = lm_si[b, va, leaf_numb, counter] + 1

    return np.asarray(lm), np.asarray(lm_s), np.asarray(lm_si)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef shap_values_leaves_cache(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const double [:, :] lm,
    const double[:, :, :, :] lm_s,
    const double[:, :, :, :] lm_si,
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

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True):
            node_id_v2[leaf_numb].clear()

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
                            p_s = lm[b, leaf_numb]/data.shape[0]
                        else:
                            p_s = (cs * lm[b, leaf_numb])/lm_s[b, va, leaf_numb, counter] if lm_s[b, va, leaf_numb, counter] !=0 else 0
                        p_si = (csi * lm[b, leaf_numb])/lm_si[b, va, leaf_numb, counter] if lm_si[b, va, leaf_numb, counter] !=0 else 0

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
cpdef shap_values_leaves_normalized_cache(const double[:, :] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const double[:, :, :, :, :] lm,
    const double[:, :, :, :, :] lm_s,
    const double[:, :, :, :, :] lm_si,
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

    cdef double p_s, p_si, coef, coef_0, n_s, n_si, csi, cs, csi_n, cs_n

    for b in range(n_trees):
        for leaf_numb in range(phi_b.shape[0]):
            for counter in range(phi_b.shape[1]):
                for i in range(N):
                    for j in range(m):
                        for i2 in range(d):
                            phi_b[leaf_numb, counter, i, j, i2] = 0
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True):
            node_id_v2[leaf_numb].clear()

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

                    for i in range(N):

                        coef = 0
                        for l in range(1, m - set_size):
                            coef = coef + (1.*binomialC(m - set_size - 1, l))/binomialC(m - 1, l + va_size) if binomialC(m - 1, l + va_size) !=0 else 0

                        coef_0 = 1./binomialC(m-1, va_size) if binomialC(m-1, va_size) !=0 else 0

                        n_s = 0
                        n_si = 0
                        csi = 0
                        cs = 0

                        if S_size == 0:
                            p_s = lm[b, va, leaf_numb, leaf_numb, counter]/data.shape[0]
                            for leaf_n in range(nb_leaf):
                                csi_n = 0
                                cs_n = 0
                                o_all = 0

                                for s in range(S_size):
                                    if ((X[i, S[va, counter, leaf_numb, s]] < partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] >= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                        o_all = o_all + 1

                                if o_all == S_size:
                                    cs_n = 1
                                    nv_bool = 0
                                    for nv in range(va_id[va].size()):
                                        if ((X[i, va_id[va][nv]] >= partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] < partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                            nv_bool = nv_bool + 1
                                            continue

                                    if nv_bool == 0:
                                        csi_n = 1

                                n_si = n_si + (csi_n * lm[b, va, leaf_numb, leaf_n, counter])/lm_si[b, va, leaf_numb, leaf_n, counter] if lm_si[b, va, leaf_numb, leaf_n, counter] !=0 else 0

                                if leaf_n == leaf_numb:
                                    csi = csi_n
                                    cs = cs_n

                            p_si = (csi * lm[b, va, leaf_numb, leaf_numb, counter])/(lm_si[b, va, leaf_numb, leaf_numb, counter] * n_si) if n_si*lm_si[b, va, leaf_numb, leaf_numb, counter] !=0 else 0
                        else:

                            for leaf_n in range(nb_leaf):
                                csi_n = 0
                                cs_n = 0
                                o_all = 0

                                for s in range(S_size):
                                    if ((X[i, S[va, counter, leaf_numb, s]] < partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (X[i, S[va, counter, leaf_numb, s]] >= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                        o_all = o_all + 1

                                if o_all == S_size:
                                    cs_n = 1
                                    nv_bool = 0
                                    for nv in range(va_id[va].size()):
                                        if ((X[i, va_id[va][nv]] >= partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (X[i, va_id[va][nv]] < partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                            nv_bool = nv_bool + 1
                                            continue

                                    if nv_bool == 0:
                                        csi_n = 1

                                n_s = n_s + (cs_n * lm[b, va, leaf_numb, leaf_n, counter])/lm_s[b, va, leaf_numb, leaf_n, counter] if lm_s[b, va, leaf_numb, leaf_n, counter] !=0 else 0
                                n_si = n_si + (csi_n * lm[b, va, leaf_numb, leaf_n, counter])/lm_si[b, va, leaf_numb, leaf_n, counter] if lm_si[b, va, leaf_numb, leaf_n, counter] !=0 else 0
                                if leaf_n == leaf_numb:
                                    csi = csi_n
                                    cs = cs_n

                            p_s = (cs * lm[b, va, leaf_numb, leaf_numb, counter])/(lm_s[b, va, leaf_numb, leaf_numb, counter] * n_s) if n_s*lm_s[b, va, leaf_numb, leaf_numb, counter] !=0 else 0
                            p_si = (csi * lm[b, va, leaf_numb, leaf_numb, counter])/(lm_si[b, va, leaf_numb, leaf_numb, counter] * n_si) if n_si*lm_si[b, va, leaf_numb, leaf_numb, counter] !=0 else 0

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
cpdef leaves_cache_normalized(
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    const long scaling,
    const vector[vector[vector[long]]] node_idx_trees,
    const vector[vector[long]] C, int num_threads):


    cdef unsigned int d = values.shape[2]
    cdef unsigned int m = data.shape[1]
    cdef unsigned int n_trees = values.shape[0]
    cdef unsigned int max_leaves = partition_leaves_trees.shape[1]

    cdef long[:, :, :, ::1] S
    S = np.zeros((m, 2**scaling, max_leaves, m), dtype=np.int)

    cdef unsigned int a_it, nb_leaf, va,  counter, ci, cj, ite, pow_set_size, nv, ns, add
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool, o_all, leaf_n

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
    cdef double[:, :, :, :, :] lm_s, lm_si, lm_n, csi, cs
    lm_n = np.zeros((n_trees, va_id.size(), max_leaves, max_leaves, 2**scaling))
    lm_s = np.zeros((n_trees, va_id.size(), max_leaves, max_leaves, 2**scaling))
    lm_si = np.zeros((n_trees, va_id.size(), max_leaves, max_leaves, 2**scaling))

    for b in range(n_trees):
        nb_leaf = leaves_nb[b]
        for leaf_numb in prange(nb_leaf, nogil=True, schedule='dynamic'):
            node_id_v2[leaf_numb].clear()

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
                        for i in range(data.shape[0]):
                            a_it = 0
                            for s in range(data.shape[1]):
                                if (data[i, s] <= partition_leaves_trees[b, leaf_n, s, 1]) and (data[i, s] > partition_leaves_trees[b, leaf_n, s, 0]):
                                    a_it = a_it + 1
                            if a_it == data.shape[1]:
                                lm_n[b, va, leaf_numb, leaf_n, counter] = lm_n[b, va, leaf_numb, leaf_n, counter] + 1

                            b_it = 0
                            for s in range(S_size):
                                if ((data[i, S[va, counter, leaf_numb, s]] <= partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 1]) * (data[i, S[va, counter, leaf_numb, s]] > partition_leaves_trees[b, leaf_n, S[va, counter, leaf_numb, s], 0])):
                                    b_it = b_it + 1

                            if b_it == S_size:
                                lm_s[b, va, leaf_numb, leaf_n, counter] = lm_s[b, va, leaf_numb, leaf_n, counter] + 1

                                nv_bool = 0
                                for nv in range(va_id[va].size()):
                                    if ((data[i, va_id[va][nv]] > partition_leaves_trees[b, leaf_n, va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= partition_leaves_trees[b, leaf_n, va_id[va][nv], 0])):
                                        nv_bool = nv_bool + 1
                                        continue

                                if nv_bool == 0:
                                    lm_si[b, va, leaf_numb, leaf_n, counter] = lm_si[b, va, leaf_numb, leaf_n, counter] + 1


    return lm_n, lm_s, lm_si

