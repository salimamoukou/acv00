@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_cat(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, float scaling):

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
    cdef dtype_t p_ss
    cdef unsigned int b, leaf_numb, nb_leaf, i, s

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaf_idx_trees[b].shape[0]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

            lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in range(m)], axis=0)

            for i in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[i, S[s]] > leaf_part[S[s], 1]) or (X[i, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                p_s = np.prod([data[:, S[s]] == X[i, S[s]] for s in range(S_size)], axis=0)
                p_ss = np.sum(p_s)

                mean_forest[i] += (np.sum(lm * p_s) * value) / p_ss if p_ss != 0 else 0
    return mean_forest / n_trees


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_cat_opti(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, float scaling):

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
        nb_leaf = leaf_idx_trees[b].shape[0]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

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
                    if it == m and it_s == S_size:
                        lm += 1
                    if it_s == S_size:
                        p_ss += 1

                mean_forest[i] += (lm * value) / p_ss if p_ss != 0 else 0

    return mean_forest / n_trees

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, float scaling):

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

    cdef unsigned int b, leaf_numb, i, nb_leaf
    cdef dtype_t p_ss

    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, values.shape[2]))

    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaf_idx_trees[b].shape[0]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling

            leaf_bool = np.prod([(X[:, S[s]] <= leaf_part[S[s], 1]) * (X[:, S[s]] >= leaf_part[S[s], 0]) for s in range(S_size)], axis=0)

            if np.sum(leaf_bool) == 0:
                continue

            lm = np.prod([(data[:, s] <= leaf_part[s, 1]) * (data[:, s] >= leaf_part[s, 0]) for s in range(m)], axis=0)
            p_ss = np.sum(np.prod([(data[:, S[s]] <= leaf_part[S[s], 1]) * (data[:, S[s]] >= leaf_part[S[s], 0])
                        for s in range(S_size)], axis=0))

            for i in range(N):
                if leaf_bool[i] == 0:
                    continue

                mean_forest[i] += (np.sum(lm) * value) / p_ss if p_ss != 0 else 0

    return mean_forest / n_trees


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_exp_opti_v2(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] S, np.ndarray[dtype_t, ndim=2] data, np.ndarray[dtype_t, ndim=3] values,
        np.ndarray[dtype_t, ndim=4] partition_leaves_trees, np.ndarray[long, ndim=2] leaf_idx_trees, float scaling):

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
    cdef dtype_t lm, p_ss
    cdef np.ndarray[dtype_t, ndim=2] mean_forest
    mean_forest = np.zeros((N, values.shape[2]))

    cdef unsigned int b, leaf_numb, i, nb_leaf, j, k, v, s,  it, it_s


    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaf_idx_trees[b].shape[0]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]] / scaling
            lm = 0
            p_ss = 0
            for i in range(data.shape[0]):
                it = 0
                it_s = 0
                for j in range(m):
                    if((data[i, j] <= leaf_part[j, 1]) and (data[i, j] >= leaf_part[j, 0])):
                        it += 1
                for k in range(S_size):
                    if((data[i, S[k]] <= leaf_part[S[k], 1]) and (data[i, S[k]] >= leaf_part[S[k], 0])):
                        it_s += 1
                if it == m:
                    lm += 1
                if it_s == S_size:
                    p_ss += 1

            for v in range(N):
                o_all = 0
                for s in range(S_size):
                    if ((X[v, S[s]] > leaf_part[S[s], 1]) or (X[v, S[s]] < leaf_part[S[s], 0])):
                        o_all +=1
                if o_all > 0:
                    continue

                mean_forest[v] += (lm * value) / p_ss if p_ss != 0 else 0

    return mean_forest / n_trees


cpdef swing_tree_shap_clf(np.ndarray[dtype_t, ndim=2] X, np.ndarray[long, ndim=1] fX,
            np.ndarray[long, ndim=1] y_pred, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, float scaling, dtype_t threshold, list C):

    cdef int N = X.shape[0]
    cdef int m = X.shape[1]

    cdef int i_0, i_1, i_2, i_3, ix, iy
    remove_va = [C[ix][iy] for iy in range(len(C[ix])) for ix in range(len(C))]

    cdef list va_id = [[i_0] for i_0 in range(m) if i_0 not in remove_va]
    if C[0] != []:
        va_id += C

    m = len(va_id)
    cdef list va_buffer = va_id.copy()
    cdef list c
    cdef dtype_t weight

    m = len(va_id)

    cdef np.ndarray[dtype_t, ndim=2] phi
    phi = np.zeros(shape=(X.shape[0], X.shape[1]))

    cdef np.ndarray[dtype_t, ndim=3] swings
    swings = np.zeros((N, m, 2))

    cdef np.ndarray[dtype_t, ndim=3] swings_prop
    swings_prop = np.zeros((N, m, 3))

    cdef list S, Sm, buffer_Sm, power, i

    for i_1 in range(m):
        print(i_1)
        i = va_id[i_1]
        print(i)
        Sm = va_id.copy()
        Sm.remove(i)
        power = [list(co) for co in powerset(Sm)]
        for i_2 in range(len(power)):
            S = power[i_2]
            weight = comb(m - 1, len(S), exact=True) ** (-1)

            for i_3 in range(N):
                for i_4 in range(len(i)):
                    phi[i_1, i_4] += 10
        print(phi)
    return phi / m

