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


%%cython -a -f --compile-args=-DCYTHON_TRACE=1 --compile-args=-fopenmp --link-args=-fopenmp
#We need to define the macro CYTHON_TRACE=1 (cf. http://docs.cython.org/src/reference/compilation.html)

import numpy as np
cimport numpy as np
ctypedef np.float64_t dtype_t
cimport cython
from scipy.special import comb
import itertools
from cython.parallel cimport prange, parallel
cdef extern from "limits.h":
    unsigned long ULONG_MAX

cdef unsigned long binomialC(unsigned long N, unsigned long k) nogil:
    # Fast path with machine integers
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
def shap_values_leaves_prof():

    cdef const double[:, :] X
    cdef const double[:, :] data
    cdef const double[:, :, :] values
    cdef const double[:, :, :, :] partition_leaves_trees
    cdef const long[:, :] leaf_idx_trees
    cdef const long[::1] leaves_nb
    cdef float scaling
    cdef list node_idx_trees
    cdef list C

    from acv_explainers import ACVTree
    from sklearn.ensemble import RandomForestClassifier

    import pandas as pd
    import random
    import pstats, cProfile
    random.seed(2021)

    np.random.seed(2021)
    data_frame = pd.read_csv('/home/samoukou/Documents/ACV/data/lucas0_train.csv')

    y = data_frame.Lung_cancer.values
    data_frame.drop(['Lung_cancer'], axis=1, inplace=True)

    forest = RandomForestClassifier(n_estimators=1, min_samples_leaf=2, random_state=212, max_depth=5)
    forest.fit(data_frame, y)
    acvtree = ACVTree(forest, data_frame.values)


    X = np.array(data_frame.values[:3], dtype=np.float)
    data = np.array(data_frame.values, dtype=np.float)
    values = acvtree.values
    partition_leaves_trees = np.array(acvtree.partition_leaves_trees)
    leaf_idx_trees = np.array(acvtree.leaf_idx_trees)
    leaves_nb = acvtree.leaves_nb
    scaling = acvtree.trees[0].scaling
    node_idx_trees = acvtree.node_idx_trees
    C=[[]]

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
    cdef unsigned int b, leaf_numb, i, s, j, i1, i2, l, na_bool
    cdef list va_id, node_id, Sm, coal_va, remove_va, C_b, node_id_v2
    cdef double lm

    cdef long[:] va_c
    cdef int len_va_c
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

    cdef long[:] b_it, nv_bool
    cdef long[:] S_size, va_size
    cdef  long[:] o_all, csi, cs
    cdef double[::1] p_s, p_si, coef, coef_0, lm_s, lm_si

    b_it = np.zeros((data.shape[0]), dtype=np.int)
    nv_bool = np.zeros((data.shape[0]), dtype=np.int)

    va_size = np.zeros((2**m), dtype=np.int)
    S_size = np.zeros((2**m), dtype=np.int)

    lm_s = np.zeros((2**m))
    lm_si = np.zeros((2**m))

    p_s = np.zeros((N))
    p_si = np.zeros((N))
    o_all = np.zeros((N), dtype=np.int)
    cs = np.zeros((N), dtype=np.int)
    csi = np.zeros((N), dtype=np.int)
    coef = np.zeros((N))
    coef_0 = np.zeros((N))

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

                for counter in prange(0, pow_set_size, nogil=True):
                    va_size[counter] = 0
                    S_size[counter] = 0
                    for ci in range(set_size):
                        if((counter & (1 << ci)) > 0):
                            for cj in range(sm_size[ci]):
                                S[S_size[counter]] = sm_buf[ci, cj]
                                S_size[counter] += 1
                            va_size[counter] += 1

                    lm_s[counter] = 0
                    lm_si[counter] = 0

                    for i in range(data.shape[0]):
                        b_it[i] = 0
                        for s in range(S_size[counter]):
                            if ((data[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (data[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                b_it[i] +=1

                        if b_it[i] == S_size[counter]:
                            lm_s[counter] += 1

                            nv_bool[i] = 0
                            for nv in range(len_va_c):
                                if ((data[i, va_c[nv]] > partition_leaves_trees[b, leaf_numb, va_c[nv], 1]) or (data[i, va_c[nv]] <= partition_leaves_trees[b, leaf_numb, va_c[nv], 0])):
                                    nv_bool[i] += 1
                                    continue

                            if nv_bool[i] == 0:
                                lm_si[counter] += 1

                    for i in range(N):

                        p_s[i] = 0
                        p_si[i] = 0

                        o_all[i] = 0
                        csi[i] = 0
                        cs[i] = 0

                        for s in range(S_size[counter]):
                            if ((X[i, S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (X[i, S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                                o_all[i] +=1

                        if o_all[i] == S_size[counter]:
                            cs[i] = 1
                            nv_bool[i] = 0
                            for nv in range(len_va_c):
                                if ((X[i, va_c[nv]] > partition_leaves_trees[b, leaf_numb, va_c[nv], 1]) or (X[i, va_c[nv]] <= partition_leaves_trees[b, leaf_numb, va_c[nv], 0])):
                                    nv_bool[i] += 1
                                    continue

                            if nv_bool[i] == 0:
                                csi[i] = 1

                        coef[i] = 0
                        for l in range(1, m - set_size):
                            coef[i] += binomialC(m - set_size - 1, l)/binomialC(m - 1, l + va_size[counter]) if binomialC(m - 1, l + va_size[counter]) !=0 else 0

                        coef_0[i] = 1./binomialC(m-1, va_size[counter]) if binomialC(m-1, va_size[counter]) !=0 else 0

                        if S_size[counter] == 0:
                            p_s[i] = lm/data.shape[0]
                        else:
                            p_s[i] = (cs[i] * lm)/lm_s[counter]

                        p_si[i] = (csi[i] * lm)/lm_si[counter]

                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[i, va_c[nv], i2] += (coef_0[i] + coef[i]) * (p_si[i] - p_s[i]) * values[b, leaf_idx_trees[b, leaf_numb], i2]

    return np.asarray(phi)/m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef single_shap_values_acv_leaves(const double[:] X,
    const double[:, :] data,
    const double[:, :, :] values,
    const double[:, :, :, :] partition_leaves_trees,
    const long[:, :] leaf_idx_trees,
    const long[::1] leaves_nb,
    double[:] scaling,
    list node_idx_trees, list S_star, list N_star,
    list C, int num_threads):



    cdef unsigned int d = values.shape[2]
    # cdef unsigned int N = X.shape[0]
    cdef unsigned int m = X.shape[0]

    cdef double[ :, :] phi
    phi = np.zeros((m, d))

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
                        print(node_id[nv], remove_va[ns], node_id[nv] == remove_va[ns])
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
            print(node_id, N_star)
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
                    p_si = (csi * lm)/lm_star
                    for nv in range(len_va_c):
                        for i2 in range(d):
                            phi[va_c[nv], i2] += (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]
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

                    p_s = 0
                    p_si = 0


                    csi = 0
                    cs = 0
                    o_all = 0
                    for s in range(S_size + len_n_star):
                        if ((X[S[s]] <= partition_leaves_trees[b, leaf_numb, S[s], 1]) * (X[S[s]] > partition_leaves_trees[b, leaf_numb, S[s], 0])):
                            o_all = o_all + 1

                    if o_all == S_size + len_n_star:
                        cs = 1
                        nv_bool = 0
                        for nv in range(len_va_c):
                            if ((X[va_c[nv]] > partition_leaves_trees[b, leaf_numb, va_c[nv], 1]) or (X[va_c[nv]] <= partition_leaves_trees[b, leaf_numb, va_c[nv], 0])):
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
                    print(np.array(S[:S_size]), p_s, p_si)
                    if S_size != 0:
                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[va_c[nv], i2] += (coef_0 + coef) * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

                    else:
                        p_off = lm/data.shape[0]
                        for nv in range(len_va_c):
                            for i2 in range(d):
                                phi[va_c[nv], i2] += (p_si-p_off)*values[b, leaf_idx_trees[b, leaf_numb], i2] + coef * (p_si - p_s) * values[b, leaf_idx_trees[b, leaf_numb], i2]

    return np.asarray(phi)/m


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef shap_values_acv_leaves_data(const double[:, :] X,
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

    cdef double[:, :] phi_x
    cdef int i, j , k

    for i in range(N):
        phi_x = single_shap_values_acv_leaves(X[i], data, values, partition_leaves_trees, leaf_idx_trees,
                        leaves_nb, scaling, node_idx_trees, S_star[i], N_star[i], C, num_threads)
        for j in range(m):
            for k in range(d):
                phi[i, j, k] =  phi_x[j, k]
    return np.array(phi)

