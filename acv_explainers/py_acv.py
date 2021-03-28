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

    for leaf_id in tqdm(leaf_idx):

        partition_leaf = partition_leaves[leaf_id]
        data_leaf = data_leaves[leaf_id]
        node_id = node_idx[leaf_id]
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

    for leaf_id in tqdm(leaf_idx):

        partition_leaf = partition_leaves[leaf_id]
        data_leaf = data_leaves[leaf_id]
        node_id = node_idx[leaf_id]
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
            if np.sum(prob) != 0:
                prob = prob / np.sum(prob)

            mean_forest[name] += np.sum(prob[:, None] * value, axis=0)

        s = (mean_forest['all'] - mean_forest['down']) / (mean_forest['up'] - mean_forest['down'])
        sdp += 0 * (s[int(fx)] < 0) + 1 * (s[int(fx)] > 1) + s[int(fx)] * (0 <= s[int(fx)] <= 1)
    # sdp = 0 * (sdp[int(fx)] < 0) + 1 * (sdp[int(fx)] > 1) + sdp[int(fx)] * (0 <= sdp[int(fx)] <= 1)
    return sdp/n_trees


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


def brute_force_tree_shap(x, num_output, C, value_function, kwargs, swing=False):
    m = len(x)
    va_id = list(range(m))
    va_buffer = va_id.copy()
    if C[0] != []:
        for c in C:
            m -= len(c)
            va_id = list(set(va_id) - set(c))
        m += len(C)
        for c in C:
            va_id += [c]

    phi = np.zeros(shape=(len(x), num_output))
    if swing:
        swings = {va: [0, 0] for va in va_id}
        swings_prop = {va: [0, 0, 0] for va in va_id}

    for i in va_id:
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
            phi[chain_l(i)] += weight * (v_plus - v_minus)

            if swing:
                if v_plus - v_minus == 1:
                    swings[i][0] += ((v_plus - v_minus) * weight) / m
                    swings_prop[i][0] += 1
                elif v_plus - v_minus == -1:
                    swings[i][1] += ((v_plus - v_minus) * weight) / m
                    swings_prop[i][1] += 1
                else:
                    swings_prop[i][2] += 1
    if swing:
        return phi / m, swings, swings_prop
    else:
        return phi / m


def local_sdp(x, f, threshold, proba, index, data, final_coal, decay, C, verbose, cond_func):
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

                    value = cond_func(x, f, threshold, S=chain_l(c), data=data)
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

            return local_sdp(x, f, threshold, proba, index, data, final_coal, decay, C, verbose, cond_func)


def global_sdp_importance(data, data_bground, columns_names, global_proba, decay, threshold,
                          proba, C, verbose, cond_func, predict):
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
        ind = data[i]
        predx = predict(np.expand_dims(ind, 0))
        if len(predx.shape) != 1:
            fx = np.argmax(predx[0])
        else:
            fx = predict(np.expand_dims(ind, 0))[0]

        local_sdp(ind, fx, threshold, proba, index, data_bground, final_coal, decay,
                  C=C, verbose=verbose, cond_func=cond_func)

        for c in final_coal:
            value = cond_func(ind, fx, threshold, S=chain_l(c), data=data)
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


def explore_partition(i, x, children_left, children_right, features, thresholds, values,
                      compatible_leaves, partition_leaves, partition_global, prob_global, s_global, S, S_bar, data,
                      down_tx, up_tx, intv=False):
    """

    Args:
        i (int):
        x (array):
        children_left (array):
        children_right (array):
        features (array):
        thresholds (array):
        values (array):
        compatible_leaves (list):
        partition_leaves (array):
        partition_global (array):
        prob_global (dict):
        s_global (dict):
        S (list):
        S_bar (list):
        data (array):
        down_tx (array):
        up_tx (array):
        intv (bool):

    Returns:
            Explore the tree, and compute the hyper-rectangle of each leaf ...... à compléter
    """

    if children_left[i] < 0:
        #         tab[i] = 1
        compatible_leaves.append(i)
        partition_global[i] = partition_leaves
        partition_leaves = np.squeeze(np.array(partition_leaves))

        section_x = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0)

        section_x_bar = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S_bar], axis=0)

        section_up = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0) * up_tx

        section_down = np.prod(
            [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
             S], axis=0) * down_tx

        prob_all = section_x * section_x_bar
        prob_up = section_up * section_x_bar
        prob_down = section_down * section_x_bar

        s_all = section_x
        s_up = section_up
        s_down = section_down

        prob_global['all'].append(prob_all.reshape(1, -1))
        prob_global['up'].append(prob_up.reshape(1, -1))
        prob_global['down'].append(prob_down.reshape(1, -1))

        s_global['all'].append(s_all.reshape(1, -1))
        s_global['up'].append(s_up.reshape(1, -1))
        s_global['down'].append(s_down.reshape(1, -1))

    else:
        if features[i] in S:
            if x[features[i]] <= thresholds[i]:
                part_left = partition_leaves.copy()
                part_left[features[i]] = np.concatenate((part_left[features[i]], np.array([[-np.inf, thresholds[i]]])))
                part_left[features[i]] = np.array(
                    [[np.max(part_left[features[i]][:, 0]), np.min(part_left[features[i]][:, 1])]])
                explore_partition(children_left[i], x, children_left, children_right, features, thresholds, values,
                                  compatible_leaves, part_left, partition_global, prob_global, s_global, S, S_bar, data,
                                  down_tx, up_tx, intv)
            else:
                part_right = partition_leaves.copy()
                part_right[features[i]] = np.concatenate((part_right[features[i]], np.array([[thresholds[i], np.inf]])))
                part_right[features[i]] = np.array(
                    [[np.max(part_right[features[i]][:, 0]), np.min(part_right[features[i]][:, 1])]])
                explore_partition(children_right[i], x, children_left, children_right, features, thresholds, values,
                                  compatible_leaves, part_right, partition_global, prob_global, s_global, S, S_bar,
                                  data, down_tx, up_tx, intv)
        else:
            part_left = partition_leaves.copy()
            part_left[features[i]] = np.concatenate((part_left[features[i]], np.array([[-np.inf, thresholds[i]]])))
            part_left[features[i]] = np.array(
                [[np.max(part_left[features[i]][:, 0]), np.min(part_left[features[i]][:, 1])]])

            part_right = partition_leaves.copy()
            part_right[features[i]] = np.concatenate((part_right[features[i]], np.array([[thresholds[i], np.inf]])))
            part_right[features[i]] = np.array(
                [[np.max(part_right[features[i]][:, 0]), np.min(part_right[features[i]][:, 1])]])

            explore_partition(children_left[i], x, children_left, children_right, features, thresholds, values,
                              compatible_leaves, part_left, partition_global, prob_global, s_global, S, S_bar, data
                              ,down_tx, up_tx, intv)
            explore_partition(children_right[i], x, children_left, children_right, features, thresholds, values,
                              compatible_leaves, part_right, partition_global, prob_global, s_global, S, S_bar, data
                              ,down_tx, up_tx, intv)


def get_tree_partition(x, fx, tx, tree, S, data=None, is_reg=True):
    """
    A compléter ....

    Compute the partition (L_m) of each compatible leaf of the condition X_s = x_S, then check for each
    observations in data in which leaves it falls.

    Args:
        x (array): observation
        fx (float): tree(x)
        tx (float): threshold of the classifier
        forest (All treeBased models): model
        S (list): index of variables on which we want to compute the SDP
        data (array): data used to compute the partion

    Returns:
        (array, array): binary array of shape (data_size, compatible_leaves), if [i, j] = 1 then observation i fall in
        leaf j.
        (array): return number of observations that fall in each leaf
    """

    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.features
    thresholds = tree.thresholds
    # r_w = tree.node_samples_weight
    index = range(x.shape[0])
    if is_reg:
        values = tree.values.reshape(-1) / tree.scaling
        y_pred = tree.predict(data)
        dist = (y_pred - fx) ** 2

        up_tx = np.array(dist > tx).reshape(-1)
        down_tx = np.array(dist <= tx).reshape(-1)
    else:
        values = tree.values / tree.scaling
        y_pred = tree.predict(data)

        if len(y_pred.shape) == 1:
            y_pred = np.array([1 - y_pred, y_pred]).T

        argmax_y_pred = np.argmax(y_pred, axis=1)

        up_tx = np.array(argmax_y_pred == int(fx)).reshape(-1)
        down_tx = np.array(argmax_y_pred != int(fx)).reshape(-1)

    S_bar = [i for i in index if i not in S]
    partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
    partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                        for i in range(len(tree.features))}

    prob_global = {'all': [], 'up': [], 'down': []}
    s_global = {'all': [], 'up': [], 'down': []}

    part_final = {}
    compatible_leaves = []
    explore_partition(0, x, children_left, children_right, features, thresholds, values,
                      compatible_leaves, partition_leaves, partition_global, prob_global, s_global, S, S_bar, data,
                      down_tx, up_tx, intv=False)

    part_final['all'] = np.concatenate(prob_global['all'], axis=0)
    part_final['up'] = np.concatenate(prob_global['up'], axis=0)
    part_final['down'] = np.concatenate(prob_global['down'], axis=0)

    part_final['s_all'] = np.concatenate(s_global['all'], axis=0)
    part_final['s_up'] = np.concatenate(s_global['up'], axis=0)
    part_final['s_down'] = np.concatenate(s_global['down'], axis=0)

    return part_final, values[compatible_leaves]


#
# def cond_sdp_tree_clf_v1(x, fx, tx, tree, S, data, ntrees):
#     """
#     Compute SDP(x, tree_classifier) of variables in S
#
#     Args:
#         x (array): observation
#         fx (float): tree(x)
#         tx (float): threshold of the classifier
#         tree (DecisionTreeClassifier.tree_): model
#         S (list): index of variables on which we want to compute the SDP
#         algo (string): name of the estimators, recommended 'pluging'
#         data (array): data used to compute the SDP
#
#     Returns:
#         float: SDP(x, tree_classifier)
#     """
#     if len(S) == len(x):
#         return 1
#     elif S == []:
#         return 0
#
#     index = range(len(x))
#     a = tree.children_left
#     b = tree.children_right
#     f = tree.features
#     t = tree.thresholds
#     # r_w = tree.node_sample_weight
#     v = tree.values / tree.scaling
#
#     # y_pred = pred_func(data, i=idx)
#     # y_pred = y_pred[:, int(fx)]
#     #
#     # up_tx = np.array(y_pred > tx).reshape(-1)
#     # down_tx = np.array(y_pred <= tx).reshape(-1)
#
#     # y_pred = pred_func(data, i=idx)
#     y_pred = tree.predict(data)
#
#     if len(y_pred.shape) == 1:
#         y_pred = np.array([1 - y_pred, y_pred]).T
#
#     argmax_y_pred = np.argmax(y_pred, axis=1)
#
#     up_tx = np.array(argmax_y_pred == int(fx)).reshape(-1)
#     down_tx = np.array(argmax_y_pred != int(fx)).reshape(-1)
#
#     # up_tx = np.array(y_pred > tx).reshape(-1)
#     # down_tx = np.array(y_pred <= tx).reshape(-1)
#
#     def explore_partition(i, tab, partition_leaves, partition_global, prob_global, S, S_bar, data, intv=False):
#
#         if a[i] < 0:
#             #         tab[i] = 1
#             compatible_leaves.append(i)
#             partition_global[i] = partition_leaves
#             partition_leaves = np.squeeze(np.array(partition_leaves))
#
#             section_x = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0)
#
#             section_x_bar = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S_bar], axis=0)
#
#             section_up = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0) * up_tx
#
#             section_down = np.prod(
#                 [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
#                  S], axis=0) * down_tx
#
#             prob_all = np.sum(section_x * section_x_bar) / np.sum(section_x)
#             prob_up = np.sum(section_up * section_x_bar) / np.sum(section_up)
#             prob_down = np.sum(section_down * section_x_bar) / np.sum(section_down)
#
#             if np.isnan(prob_up):
#                 prob_up = 0
#             if np.isnan(prob_down):
#                 prob_down = 0
#             if np.isnan(prob_all):
#                 prob_all = 0
#
#             prob_global[i] = [prob_all, prob_up, prob_down]
#
#         else:
#             if f[i] in S:
#                 if x[f[i]] <= t[i]:
#                     part = partition_leaves.copy()
#                     part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
#                     part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#                     explore_partition(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
#                 else:
#                     part = partition_leaves.copy()
#                     part[f[i]] = np.concatenate((part[f[i]], np.array([[t[i], np.inf]])))
#                     part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#                     explore_partition(b[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
#             else:
#                 part = partition_leaves.copy()
#                 part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
#                 part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
#
#                 part_2 = partition_leaves.copy()
#                 part_2[f[i]] = np.concatenate((part_2[f[i]], np.array([[t[i], np.inf]])))
#                 part_2[f[i]] = np.array([[np.max(part_2[f[i]][:, 0]), np.min(part_2[f[i]][:, 1])]])
#
#                 explore_partition(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
#                 explore_partition(b[i], tab, part_2, partition_global, prob_global, S, S_bar, data, intv)
#
#     S_bar = [i for i in index if i not in S]
#     partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
#     partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
#                         for i in range(len(tree.features))}
#     prob_global = {}
#     compatible_leaves = []
#     explore_partition(0, compatible_leaves, partition_leaves, partition_global, prob_global, S, S_bar, data,
#                       False)
#
#     p_all = np.array([prob_global[key][0] for key in compatible_leaves])
#     if np.sum(p_all) != 0:
#         p_all = p_all / np.sum(p_all)
#
#     p_up = np.array([prob_global[key][1] for key in compatible_leaves])
#     if np.sum(p_up) != 0:
#         p_up = p_up / np.sum(p_up)
#
#     p_down = np.array([prob_global[key][2] for key in compatible_leaves])
#     if np.sum(p_down) != 0:
#         p_down = p_down / np.sum(p_down)
#     # print(p_all, p_up, p_down, 'p value')
#
#     value = v[compatible_leaves]
#     mean_up = np.sum(p_up[:, None] * value, axis=0)
#     mean_down = np.sum(p_down[:, None] * value, axis=0)
#     mean_all = np.sum(p_all[:, None] * value, axis=0)
#
#     sdp = (mean_all - mean_down) / (mean_up - mean_down)
#
#     sdp = 0 * (sdp[int(fx)] < 0) + 1 * (sdp[int(fx)] > 1) + sdp[int(fx)] * (0 <= sdp[int(fx)] <= 1)
#     # sdp[1] = 0 * (sdp[1] < 0) + 1 * (sdp[1] > 1) + sdp[1] * (0 <= sdp[1] <= 1)
#
#     return sdp

