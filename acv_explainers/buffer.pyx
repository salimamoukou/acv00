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


cdef int binomialC(int n,  int r):
    if (r > n):
        return 0
    cdef int i
    cdef long long int m = 1000000007
    cdef long long int[::1] inv
    inv = np.zeros(r+1, dtype=np.longlong)
    inv[1] = 1

    for i in range(2, r+1):
        inv[i] =  m - (m / i) * inv[m % i] % m

    cdef long long int ans = 1

    for i in range(2, r+1):
        ans = ((ans % m) * (inv[i] % m)) % m;

    for i in range(n, n-r, -1):
        ans = ((ans % m) * (i % m)) % m

    return ans



cpdef shap_values_leaves(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling,
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
    cdef list va_id, node_id, Sm, remove_va
    cdef dtype_t coef, p_s, p_si, lm, lm_s, lm_si


    if C[0] != []:
        coal_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
        remove_va = [[i] for i in range(m) if i not in coal_va]
        va_id = remove_va + C
        m = len(va_id)
    else:
        va_id = [[i] for i in range(m)]


    for b in range(n_trees):
        leaves_tree = partition_leaves_trees[b]
        nb_leaf = leaves_nb[b]

        for leaf_numb in range(nb_leaf):
            leaf_part = leaves_tree[leaf_numb]
            value = values[b, leaf_idx_trees[b, leaf_numb]]
            node_id = node_idx_trees[b][leaf_numb]
            node_id_v2 = []

            if C[0] != []:
                for nv in range(len(node_id)):
                    if node_id[nv] in node_id_v2:
                        continue
                    add = 0
                    for ns in range(len(remove_va)):
                        if node_id[nv] == remove_va[ns]:
                            add = 1
                            node_id_v2 += [node_id[nv]]
                            continue
                    if add == 0:
                        for ci in range(len(C)):
                            for cj in range(len(C[ci])):
                                if C[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += C[ci]
                                    continue
                            if add == 1:
                                continue
                node_id = list(set(node_id_v2))

            for va in range(len(va_id)):

                if va_id[va][0] not in node_id:
                    continue

                buff = []
                if C[0] != []:
                    Sm = []
                    for nv in range(len(node_id)):
                        add = 0
                        if node_id[nv] in buff:
                            continue
                        for ns in range(len(remove_va)):
                            if node_id[nv] == remove_va[ns]:
                                add = 1
                                Sm += [[node_id[nv]]]
                                continue
                        if add == 0:
                            for ci in range(len(C)):
                                for cj in range(len(C[ci])):
                                    if node_id[nv] == C[ci][cj]:
                                        add = 1
                                        Sm += [C[ci]]
                                        buff += C[ci]
                                        continue
                                if add == 1:
                                    continue
                    Sm.remove(va_id[va])
                else:
                    Sm = [[i] for i in node_id]
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
                    lm = 0
                    lm_s = 0
                    lm_si = 0
                    for i in range(data.shape[0]):
                        a_it = 0
                        b_it = 0
                        for s in range(m):
                            if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] > leaf_part[s, 0])):
                                a_it += 1

                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] > leaf_part[S[s], 0])):
                                b_it +=1

                        if a_it == m:
                            lm += 1
                        if b_it == S_size:
                            lm_s += 1

                            nv_bool = 0
                            for nv in range(len(va_id[va])):
                                if ((data[i, va_id[va][nv]] > leaf_part[va_id[va][nv], 1]) or (data[i, va_id[va][nv]] <= leaf_part[va_id[va][nv], 0])):
                                    nv_bool += 1
                                    continue

                            if nv_bool == 0:
                                lm_si += 1
                    print('Node = {} - va -{} - Sm = {} - S = {}'.format(node_id, va_id[va], Sm, S[:S_size]))
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
                            coef += binomialC(m - len(Sm) - 1, l) * binomialC(m - 1, l + va_size)**(-1)

                        if S_size == 0:
                            p_s = lm/data.shape[0]
                        else:
                            p_s = (cs * lm)/lm_s

                        p_si = (csi * lm)/lm_si

                        for nv in range(len(va_id[va])):
                            for i2 in range(d):
                                phi[i, va_id[va][nv], i2] += (binomialC(m-1, va_size)**(-1) + coef) * (p_si - p_s) * value[i2]



    return phi / m

cpdef shap_values_leaves(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling,
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
    cdef list va_id, node_id, Sm, remove_va
    cdef dtype_t coef, p_s, p_si, lm, lm_s, lm_si


    if C[0] != []:
        coal_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
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

            if C[0] != []:
                for nv in range(len(node_id)):
                    if node_id[nv] in node_id_v2:
                        continue
                    add = 0
                    for ns in range(len(remove_va)):
                        if node_id[nv] == remove_va[ns]:
                            add = 1
                            node_id_v2 += [node_id[nv]]
                            continue
                    if add == 0:
                        for ci in range(len(C)):
                            for cj in range(len(C[ci])):
                                if C[ci][cj] == node_id[nv]:
                                    add = 1
                                    node_id_v2 += C[ci]
                                    continue
                            if add == 1:
                                continue
                node_id = list(set(node_id_v2))

            for va in range(len(va_id)):
                if va_id[va][0] not in node_id:
                    continue

                buff = []
                if C[0] != []:
                    Sm = []
                    for nv in range(len(node_id)):
                        add = 0
                        if node_id[nv] in buff:
                            continue
                        for ns in range(len(remove_va)):
                            if node_id[nv] == remove_va[ns]:
                                add = 1
                                Sm += [[node_id[nv]]]
                                continue
                        if add == 0:
                            for ci in range(len(C)):
                                for cj in range(len(C[ci])):
                                    if node_id[nv] == C[ci][cj]:
                                        add = 1
                                        Sm += [C[ci]]
                                        buff += C[ci]
                                        continue
                                if add == 1:
                                    continue
                    Sm.remove(va_id[va])
                else:
                    Sm = [[i] for i in node_id]
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
                    lm = 0
                    lm_s = 0
                    lm_si = 0
                    for i in range(data.shape[0]):
                        a_it = 0
                        b_it = 0
                        for s in range(data.shape[1]):
                            if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] > leaf_part[s, 0])):
                                a_it += 1

                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] > leaf_part[S[s], 0])):
                                b_it +=1

                        if a_it == data.shape[1]:
                            lm += 1
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
                            coef += binomialC(m - len(Sm) - 1, l) * binomialC(m - 1, l + va_size)**(-1)

                        if S_size == 0:
                            p_s = lm/data.shape[0]
                        else:
                            p_s = (cs * lm)/lm_s

                        p_si = (csi * lm)/lm_si
                        for nv in range(len(va_id[va])):
                            for i2 in range(d):
                                phi[i, va_id[va][nv], i2] += (binomialC(m-1, va_size)**(-1) + coef) * (p_si - p_s) * value[i2]

    return phi / m


cpdef shap_values_leaves(np.ndarray[dtype_t, ndim=2] X, np.ndarray[dtype_t, ndim=2] data,
            np.ndarray[dtype_t, ndim=3] values, np.ndarray[dtype_t, ndim=4] partition_leaves_trees,
            np.ndarray[long, ndim=2] leaf_idx_trees, np.ndarray[long, ndim=1] leaves_nb, float scaling,
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
    cdef list va_id, node_id, Sm, remove_va, C_b
    cdef dtype_t coef, p_s, p_si, lm, lm_s, lm_si


    if C[0] != []:
        coal_va = [C[ci][cj] for cj in range(len(C[ci])) for ci in range(len(C))]
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

                if va_id[va] not in node_id:
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
                    lm = 0
                    lm_s = 0
                    lm_si = 0
                    for i in range(data.shape[0]):
                        a_it = 0
                        b_it = 0
                        for s in range(data.shape[1]):
                            if ((data[i, s] <= leaf_part[s, 1]) and (data[i, s] > leaf_part[s, 0])):
                                a_it += 1

                        for s in range(S_size):
                            if ((data[i, S[s]] <= leaf_part[S[s], 1]) and (data[i, S[s]] > leaf_part[S[s], 0])):
                                b_it +=1

                        if a_it == data.shape[1]:
                            lm += 1
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
                            coef += binomialC(m - len(Sm) - 1, l) * binomialC(m - 1, l + va_size)**(-1)

                        if S_size == 0:
                            p_s = lm/data.shape[0]
                        else:
                            p_s = (cs * lm)/lm_s

                        p_si = (csi * lm)/lm_si
                        for nv in range(len(va_id[va])):
                            for i2 in range(d):
                                phi[i, va_id[va][nv], i2] += (binomialC(m-1, va_size)**(-1) + coef) * (p_si - p_s) * value[i2]

    return phi / m
