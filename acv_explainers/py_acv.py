from .utils import *
import numpy as np
from tqdm import tqdm
from scipy.special import comb


def shap_values_leaves(x, partition_leaves, data_leaves, node_idx, leaf_idx, weight_samples, v, C,
                       num_outputs):
    """
    Compute SV using multi-game algorithm
    Args:
        x (np.array): Input of shape = (# samples, # features)
        partition_leaves (np.array): It contains the hype-rectangle of each leaves, shape = # leaves x D x 2
        data_leaves (np.array): Boolean matrix of shape = # leaves x D * N, if data_leaves[i, j, z ] = 1 means that obs
        z fall in rectangle j of leaf i.
        node_idx (list): node_idx[i] is the indexes of the nodes that it is in the path of leaf i
        leaf_idx (list): Index of the leaves nodes
        weight_samples (array): weight_samples[i] is the number of samples that fall in leaf node i
        v (array): v[i] is the value of node i, shape = # nodes x num_outputs
        C (list): Indexes of the variables group together (coalition variables)
        num_outputs (int): size of the output

    Returns:
            (array): SV with multi-game algorithm, Shape = N x D x num_outputs
    """
    phi = np.zeros((x.shape[0], x.shape[1], num_outputs)).astype(np.float64)

    for leaf_numb, leaf_id in enumerate(leaf_idx):

        partition_leaf = partition_leaves[leaf_numb]
        data_leaf = data_leaves[leaf_numb]
        node_id = node_idx[leaf_numb]
        Lm = weight_samples[leaf_id]

        d = x.shape[1]
        va_id = list(range(d))

        # start handle coalition
        if C[0] != []:
            for c in C:
                d -= len(c)
                va_id = list(set(va_id) - set(c))
            for c in C:
                va_id += [c]
                present = np.sum([c_i in node_id for c_i in c])
                if present != 0:
                    node_id = node_id + c
            d += len(C)

        for i in va_id:
            if not set(convert_list(i)).issubset(node_id):
                continue

            Sm = list(set(node_id) - set(convert_list(i)))

            if C[0] != []:
                buffer_Sm = Sm.copy()
                for c in C:
                    if set(c).issubset(buffer_Sm):
                        Sm = list(set(Sm) - set(c))
                for c in C:
                    if set(c).issubset(buffer_Sm):
                        Sm += [c]
            # end handle coalition

            for S in powerset(Sm):

                comp_si = np.prod([(x[:, s] <= partition_leaf[s, 1]) * (x[:, s] >= partition_leaf[s, 0])
                                   for s in chain_l(S) + convert_list(i)], axis=0)
                comp_s = np.prod([(x[:, s] <= partition_leaf[s, 1]) * (x[:, s] >= partition_leaf[s, 0])
                                  for s in chain_l(S)], axis=0)

                coef = 0
                for l in range(1, d - len(Sm)):
                    coef += comb(d - len(Sm) - 1, l) * comb(d - 1, l + len(S)) ** (-1)

                Lsi = np.sum(np.prod(data_leaf[chain_l(S) + convert_list(i), :], axis=0))
                Ls = np.sum(np.prod(data_leaf[chain_l(S), :], axis=0))

                P_si = (comp_si * Lm) / Lsi

                if len(S) == 0:
                    P_s = Lm / np.sum(weight_samples[leaf_idx])
                else:
                    P_s = (comp_s * Lm) / Ls

                for a in convert_list(i):
                    phi[:, a, :] += (comb(d - 1, len(S), exact=True) ** (-1) + coef) * (P_si - P_s)[:, None] * v[
                                                                                                                   leaf_id][
                                                                                                               None, :]

    return phi / d


def shap_values_acv_leaves(x, partition_leaves, data_leaves, node_idx, leaf_idx, weight_samples, v, C, S_star, N_star,
                           num_outputs):
    """
        Compute ACV-SV using multi-game algorithm
        Args:
            x (np.array): Input of shape = (# samples, # features)
            partition_leaves (np.array): It contains the hype-rectangle of each leaves, shape = # leaves x D x 2
            data_leaves (np.array): Boolean matrix of shape = # leaves x D * N, if data_leaves[i, j, z ] = 1 means that obs
            z fall in rectangle j of leaf i.
            node_idx (list): node_idx[i] is the indexes of the nodes that it is in the path of leaf i
            leaf_idx (list): Index of the leaves nodes
            weight_samples (array): weight_samples[i] is the number of samples that fall in leaf node i
            v (array): v[i] is the value of node i, shape = # nodes x num_outputs
            C (list): Indexes of the variables group together (coalition variables)
            num_outputs (int): size of the output
            S_star (list): Indexes of the Active variables, see section ACV of the papers
            N_star (list): Indexes of the Null variables, see section ACV of papers

        Returns:
                (array): ACV-SV with multi-game algorithm, Shape = N x D x num_outputs
        """
    phi = np.zeros((x.shape[0], x.shape[1], num_outputs)).astype(np.float64)

    for leaf_numb, leaf_id in enumerate(leaf_idx):

        partition_leaf = partition_leaves[leaf_numb]
        data_leaf = data_leaves[leaf_numb]
        node_id = node_idx[leaf_numb]
        Lm = weight_samples[leaf_id]

        va_id = S_star
        d = len(va_id)
        node_id = list(set(node_id) - set(N_star))

        # start handle coalition
        if C[0] != []:
            for c in C:
                if set(c).issubset(S_star):
                    d -= len(c)
                    va_id = list(set(va_id) - set(c))
            for c in C:
                if set(c).issubset(S_star):
                    va_id += [c]
                    present = np.sum([c_i in node_id for c_i in c])
                    if present != 0:
                        node_id = node_id + c
                    d += 1

        for i in va_id:

            if not set(convert_list(i)).issubset(node_id):
                comp_si = np.prod([(x[:, s] <= partition_leaf[s, 1]) * (x[:, s] >= partition_leaf[s, 0])
                                   for s in N_star], axis=0)

                Lsi = np.sum(np.prod(data_leaf[N_star, :], axis=0))
                P_si = comp_si * Lm / Lsi
                P_s = Lm / np.sum(weight_samples[leaf_idx])

                for a in convert_list(i):
                    phi[:, a] += ((P_si - P_s)[:, None] * v[leaf_id][None, :])
                continue

            Sm = list(set(node_id) - set(convert_list(i)))
            buffer_Sm = Sm.copy()

            if C[0] != []:
                for c in C:
                    if set(c).issubset(S_star) and set(c).issubset(buffer_Sm):
                        Sm = list(set(Sm) - set(c))
                for c in C:
                    if set(c).issubset(S_star) and set(c).issubset(buffer_Sm):
                        Sm += [c]

            for S in powerset(Sm):

                comp_si = np.prod([(x[:, s] <= partition_leaf[s, 1]) * (x[:, s] >= partition_leaf[s, 0])
                                   for s in chain_l(S) + convert_list(i) + N_star], axis=0)
                comp_s = np.prod([(x[:, s] <= partition_leaf[s, 1]) * (x[:, s] >= partition_leaf[s, 0])
                                  for s in chain_l(S) + N_star], axis=0)

                coef = 0
                for l in range(1, d - len(Sm)):
                    coef += comb(d - len(Sm) - 1, l) * comb(d - 1, l + len(S)) ** (-1)

                Lsi = np.sum(np.prod(data_leaf[chain_l(S) + convert_list(i) + N_star, :], axis=0))
                Ls = np.sum(np.prod(data_leaf[chain_l(S) + N_star, :], axis=0))

                P_si = comp_si * Lm / Lsi
                P_s = comp_s * Lm / Ls

                if len(S) != 0:
                    for a in convert_list(i):
                        phi[:, a] += (
                                (comb(d - 1, len(S)) ** (-1) + coef) * (P_si - P_s)[:, None] * v[leaf_id][None, :])
                else:
                    P_soff = Lm / np.sum(weight_samples[leaf_idx])
                    for a in convert_list(i):
                        phi[:, a] += ((P_si - P_soff)[:, None] * v[leaf_id][None, :] + coef * (P_si - P_s)[:, None] * v[
                                                                                                                          leaf_id][
                                                                                                                      None,
                                                                                                                      :])

    return phi / d


def cond_sdp_forest_clf(x, fx, tx, forest, S, data):
    """
    Compute SDP(x, forest_classifier) of variables in S
    Args:
        x (array): observation
        fx (float): forest(x)
        tx (float): threshold of the classifier
        forest (All TreeBased models): model
        S (list): index of variables on which we want to compute the SDP
        data (array): data used to compute the SDP

    Returns:
        float: SDP(x, forest_classifier)
    """
    if len(S) == len(x):
        return 1
    elif S == []:
        return 0

    n_trees = len(forest)
    names = ['all', 'up', 'down']
    part_forest = [get_tree_partition(x, fx, tx, forest[i], S, data=data, is_reg=False) for i in
                   range(n_trees)]
    sdp = 0
    for i in range(n_trees):
        mean_forest = {'all': 0, 'up': 0, 'down': 0}
        value = part_forest[i][1]
        for name in names:
            p = part_forest[i][0][name].sum(axis=1).astype(np.float64)
            p_s = part_forest[i][0]['s_{}'.format(name)].sum(axis=1).astype(np.float64)
            prob = np.true_divide(p, p_s, out=np.zeros_like(p), where=p_s != 0)
            # if np.sum(prob) != 0:
            #     prob = prob / np.sum(prob)

            mean_forest[name] += np.sum(prob[:, None] * value, axis=0)

        s = (mean_forest['all'] - mean_forest['down']) / (mean_forest['up'] - mean_forest['down'])
        sdp += 0 * (s[int(fx)] < 0) + 1 * (s[int(fx)] > 1) + s[int(fx)] * (0 <= s[int(fx)] <= 1)
    # sdp = 0 * (sdp[int(fx)] < 0) + 1 * (sdp[int(fx)] > 1) + sdp[int(fx)] * (0 <= sdp[int(fx)] <= 1)
    return sdp / n_trees


def cond_sdp_forest(x, fx, tx, forest, S, data):
    """
        Compute SDP_S(x, treeBased models) of variables in S

        Args:
            x (array): observation
            fx (float): forest(x)
            tx (float): threshold of the classifier
            forest (All TreeBased models): model
            S (list): index of variables on which we want to compute the SDP
            data (array): data used to compute the SDP

        Returns:
            float: SDP_S(x, treeBased models)
        """
    if len(S) == len(x):
        return 1
    elif S == []:
        return 0
    n_trees = len(forest)
    names = ['all', 'up', 'down']
    part_forest = [get_tree_partition(x, fx, tx, forest[i], S, data=data) for i in
                   range(n_trees)]
    mean_forest = {'all': 0, 'up': 0, 'down': 0}
    for i in range(n_trees):
        for j in range(n_trees):
            if i == j:
                value = part_forest[i][1]
                for name in names:
                    p = part_forest[i][0][name].sum(axis=1).astype(np.float64)
                    p_s = part_forest[i][0]['s_{}'.format(name)].sum(axis=1).astype(np.float64)
                    prob = np.true_divide(p, p_s, out=np.zeros_like(p), where=p_s != 0)
                    # if np.sum(prob) != 0:
                    #     prob = prob / np.sum(prob)

                    mean_forest[name] += (np.sum(prob * value ** 2)) / (n_trees ** 2) - (
                            2 * fx * np.sum(prob * value)) / n_trees

            else:

                value_i = part_forest[i][1].reshape(-1, 1)
                value_j = part_forest[j][1].reshape(-1, 1)
                value_ij = np.matmul(value_i, value_j.T)

                for name in names:
                    p_i = part_forest[i][0][name]
                    p_j = part_forest[j][0][name]
                    p_ij = np.matmul(p_i, p_j.T).astype(np.float64)
                    s_ij = np.matmul(part_forest[i][0]['s_{}'.format(name)],
                                     part_forest[j][0]['s_{}'.format(name)].T).astype(np.float64)

                    prob_ij = np.true_divide(p_ij, s_ij, out=np.zeros_like(p_ij), where=s_ij != 0)
                    # if np.sum(prob_ij) != 0:
                    #     prob_ij = prob_ij / np.sum(prob_ij)

                    mean_forest[name] += np.sum(np.multiply(prob_ij, value_ij)) / n_trees ** 2

    sdp = (mean_forest['up'] - mean_forest['all']) / (mean_forest['up'] - mean_forest['down'])
    sdp = 0 * (sdp < 0) + 1 * (sdp > 1) + sdp * (0 <= sdp <= 1)
    return sdp


def brute_force_tree_shap(X, num_output, C, value_function, kwargs, swing=False):
    N = X.shape[0]
    m = X.shape[1]
    va_id = list(range(m))
    va_buffer = va_id.copy()
    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(X.shape[0], X.shape[1], num_output))
    if swing:
        swings = {va: np.zeros((N, 2)) for va in va_id}
        swings_prop = {va: np.zeros((N, 3)) for va in va_id}

    for i in tqdm(va_id):
        Sm = list(set(va_buffer) - set(convert_list(i)))

        if C[0] != []:
            buffer_Sm = Sm.copy()
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm = list(set(Sm) - set(c))
            for c in C:
                if set(c).issubset(buffer_Sm):
                    Sm += [c]

        for S in powerset(Sm):
            weight = comb(m - 1, len(S), exact=True) ** (-1)
            v_plus = value_function(**kwargs, S=chain_l(S) + convert_list(i))
            v_minus = value_function(**kwargs, S=chain_l(S))
            phi[:, chain_l(i)] += weight * (v_plus - v_minus)

            if swing:
                dif_pos = (v_plus - v_minus) > 0
                dif_neg = (v_plus - v_minus) < 0
                dif_null = (v_plus - v_minus) == 0
                value = ((v_plus - v_minus) * weight) / m

                swings[i][:, 0] += dif_pos * value
                swings[i][:, 1] += dif_neg * value

                swings_prop[i][:, 0] += dif_pos
                swings_prop[i][:, 1] += dif_neg
                swings_prop[i][:, 2] += dif_null

                # if v_plus - v_minus == 1:
                #     swings[i][0] += ((v_plus - v_minus) * weight) / m
                #     swings_prop[i][0] += 1
                # elif v_plus - v_minus == -1:
                #     swings[i][1] += ((v_plus - v_minus) * weight) / m
                #     swings_prop[i][1] += 1
                # else:
                #     swings_prop[i][2] += 1
    if swing:
        return phi / m, swings, swings_prop
    else:
        return phi / m


def local_sdp(x, threshold, proba, index, data, final_coal, decay, C, verbose, cond_func):
    """
    Find the Sufficient coalition S* at level "proba", then recompute recursively S* by decreasing the "proba" with
    "decay" for all tree-based models

    Args:
        x (np.array): observation
        f (float): forest(x)
        forest (All TreeBased models): model
        threshold (float): threshold of the classification in (0, 1)
        proba (float): the level of the Sufficient Coalition \pi
        index (list): index of the variables of x
        data (array): data used for the estimation
        final_coal (list): the list that will contain the cluster find by SDP, empty [] at initialization
        decay (float): the probability decay used in the recursion step
        C (list[list]): list of the coalition of variable by their index, default value when no coalition is [[]]

    Returns:
        (list[list]): list of the cluster find, the first is the Sufficient Coalition S* of level "proba". The remaining
        is the Sufficient Coalition when we remove previous S*, set proba = decay * proba and recursively apply the
        function.
    """
    C_off = [()]
    va_id = index.copy()
    m = len(index)

    if C[0] != []:
        for c in C:
            if set(c).issubset(va_id):
                va_id = list(set(va_id) - set(c))
                m -= len(c)
        for c in C:
            if set(c).issubset(index):
                va_id += [c]
                m += 1

    c_value = {i: {} for i in range(m + 1)}
    i_best, c_best = {i: -np.inf for i in range(m + 1)}, {i: -np.inf for i in range(m + 1)}
    find = False

    if len(va_id) == 0:
        if verbose > 0:
            print('End')
        return final_coal
    else:
        for size in range(len(va_id) + 1):
            for c in itertools.combinations(va_id, size):
                if c not in C_off:

                    value = cond_func(x, threshold, S=chain_l(c), data=data)[0]
                    c_value[size][str(c)] = value

                    # if c_value[size][str(c)] < proba:
                    #     C_off.append(c)

                    if c_value[size][str(c)] >= proba and c_value[size][str(c)] >= i_best[size]:
                        i_best[size] = c_value[size][str(c)]
                        c_best[size] = chain_l(c)

            if c_best[size] != -np.inf:
                find = True
                final_coal.append(c_best[size])
                threshold = threshold * decay
                break

            elif len(chain_l(c)) == len(va_id):
                find = True
                final_coal.append(chain_l(tuple(va_id)))

        if find == True:
            for c_i in final_coal[-1]:
                index.remove(c_i)
            if verbose > 0:
                print('REMAINING VARIABLE = {}'.format(index))

            return local_sdp(x, threshold, proba, index, data, final_coal, decay, C, verbose, cond_func)


def global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, cond_func):
    """
    Compute the Global SDP across "data" at level "global_proba" for forest_regressor.

    Args:
        forest (RandomForestRegressor): model
        data (array): data used to compute the Global SDP
        data_bground (array): data used in the estimations of the SDP
        columns_names (list): names of the variables
        global_proba (float): proba used for the selection criterion. We count each time for a variable if it is on a
                              set with SDP >= global proba
        decay (float): decay value used when recursively apply the local_sdp function .
        threshold (float): the radius t of the SDP regressor (see paper: SDP for regression)
        proba (float): the  level of the Sufficient Coalition
        algo (string): name of the estimator, recommended 'plugin'
        C (list[list]): list of the coalition of variable by their index, default value when no coalition is [[]]

    Returns:
        (array): _
        (array): _
        (array): _
        (array): Global SDP for coalition
        (array): GLobal SDP of each variable
    """
    n_size = len(data)
    sdp_importance = {str(i): [] for i in range(data.shape[1])}
    sdp_coal_proba = {}
    sdp_importance_coal_count = {}
    sdp_importance_variable_count = {str(i): 0 for i in range(data.shape[1])}

    for i in tqdm(range(n_size)):
        final_coal = []
        index = list(range(data.shape[1]))
        ind = np.expand_dims(data[i], 0)

        local_sdp(ind, threshold, proba, index, data_bground, final_coal, decay,
                  C=C, verbose=verbose, cond_func=cond_func)

        for c in final_coal:
            value = cond_func(ind, threshold, S=chain_l(c), data=data)[0]
            #             sdp_imp_name = []

            if len(c) > 1:
                sdp_imp_name = [columns_names[i] for i in c]
            else:
                sdp_imp_name = [columns_names[c[0]]]

            if value >= global_proba:
                if str(sdp_imp_name) in sdp_coal_proba.keys():
                    sdp_coal_proba[str(sdp_imp_name)].append(value)
                    sdp_importance_coal_count[str(sdp_imp_name)] += 1 / n_size
                else:
                    sdp_coal_proba[str(sdp_imp_name)] = [value]
                    sdp_importance_coal_count[str(sdp_imp_name)] = 1 / n_size

                for c_i in c:
                    sdp_importance[str(c_i)].append(value)
                    sdp_importance_variable_count[str(c_i)] += 1 / n_size

    #             for c_i in c:
    #                 sdp_importance[str(c_i)].append(value)
    #                 if value >= global_t:
    #                     sdp_importance_variable_count[str(c_i)] += 1/n_size

    sdp_importance_m = [np.mean(sdp_importance[key]) for key in sdp_importance.keys()]
    sdp_coal_proba = {key: np.mean(sdp_coal_proba[key]) for key in sdp_coal_proba.keys()}

    return sdp_importance_m, sdp_importance, sdp_coal_proba, sdp_importance_coal_count, sdp_importance_variable_count

