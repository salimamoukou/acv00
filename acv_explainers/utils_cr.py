import numpy as np
from tqdm import tqdm
# from .utils import *
from .utils import find_union, rand, pickle, get_active_null_coalition_list, IsolationForest
import acv_explainers

def return_xy_cnt(x, y_target, x_train, y_train, w):
    """
    return the observations with y=y_target that fall in the projected **leaf** of x
    when we condition given S=S_bar of x.
    """

    x_train_cnt = []
    for i, wi in enumerate(w):
        if wi != 0 and y_train[i] == y_target:
            x_train_cnt.append(x_train[i].copy())

    if len(x_train_cnt) == 0:
        x_train_cnt = x_train[y_train == y_target][:10]
        y_train_cnt = y_train[y_train == y_target][:10]
    else:
        x_train_cnt = np.array(x_train_cnt)
        y_train_cnt = np.array(x_train_cnt.shape[0] * [y_target])

    return x_train_cnt, y_train_cnt


def return_leaf_cnt(ac_explainer, S_star, x_train_cnt, y_train_cnt, x_train, y_train, pi=0.9, p_best=0.85, n_try=100,
                       batch=100, max_iter=100, temp=1, max_iter_convergence=100):
    """
    return the original leaves of the observations with y=y_target that fall in the projected **leaf**
    when we condition given S=S_bar of x.
    """
    sdp, rules = ac_explainer.compute_sdp_rule(x_train_cnt, y_train_cnt,
                                               x_train, y_train,
                                               x_train_cnt.shape[0] * [list(range(x_train.shape[1]))]
                                               )
    if np.sum(sdp >= pi) != 0:
        rules_unique = np.unique(rules[sdp >= pi], axis=0)
    else:
        argsort = np.argsort(-sdp)
        rules_unique = rules[argsort[:5]]

    r_buf = rules_unique.copy()
    # TODO: check if 5 is sufficient
    for i in range(rules_unique.shape[0]):
        list_ric = [r.copy() for r in r_buf if not np.allclose(r, rules_unique[i])]
        # find_union(rules_unique[i], list_ric, S=S_star)
        sr, _ = rules_by_annealing(volume_rectangles, np.expand_dims(rules_unique[i], axis=0),
                                                np.expand_dims(list_ric, axis=0), x_train,
                                   p_best=p_best, n_try=n_try,
                                   batch=batch, max_iter=max_iter, temp=temp, max_iter_convergence=max_iter_convergence)
        rules_unique[i] = np.squeeze(sr[-1])
    return rules_unique


def remove_in(ra):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                ra[i] = ra[j]
    return np.unique(ra, axis=0)


def get_compatible_leaf(acvtree, x, y_target, S_star, S_bar_set, w, x_train, y_train, pi=0.9, acc_level=0.9,
                        p_best=0.85, n_try=100,
                        batch=100, max_iter=100, temp=1, max_iter_convergence=100
                        ):
    """
    Retrieve the compatible leaves and order them given their accuracy
    """
    x_train_cnt, y_train_cnt = return_xy_cnt(x, y_target, x_train, y_train, w)
    compatible_leaves = return_leaf_cnt(acvtree, S_star, x_train_cnt, y_train_cnt, x_train, y_train, pi,
                                        p_best=p_best, n_try=n_try,
                                        batch=batch, max_iter=max_iter, temp=temp,
                                        max_iter_convergence=max_iter_convergence
                                        )

    compatible_leaves = np.unique(compatible_leaves, axis=0)
    compatible_leaves = remove_in(compatible_leaves)
    #     compatible_leaves = np.rint(compatible_leaves)
    compatible_leaves = np.round(compatible_leaves, 2)

    partition_leaf = compatible_leaves.copy()
    d = partition_leaf.shape[1]
    nb_leaf = partition_leaf.shape[0]
    leaves_acc = []
    suf_leaf = []

    for i in range(nb_leaf):
        x_in = np.prod([(x_train[:, s] <= partition_leaf[i, s, 1]) * (x_train[:, s] >= partition_leaf[i, s, 0])
                        for s in range(d)], axis=0).astype(bool)
        y_in = y_train[x_in]
        acc = np.mean(y_in == y_target)
        leaves_acc.append(acc)

        if acc >= acc_level:
            suf_leaf.append(partition_leaf[i])
    # TODO: check if it's good idea in general
    best_id = np.argmax(leaves_acc)
    if len(suf_leaf) == 0:
        suf_leaf.append(partition_leaf[best_id])

    return suf_leaf, partition_leaf, leaves_acc, partition_leaf[best_id], leaves_acc[best_id]


def return_counterfactuals(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, x_train, y_train, pi_level):
    """
    Compute the SDP of each C_S and return the ones that has sdp >= pi_level
    """
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []
    sdp_buf = []
    for leaf in suf_leaf:

        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = leaf[S_star]
        # Condition on S_bar and otherwise used the rules of C_S
        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [0.]), np.array(size * [0.]),
                                                       x_train, y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        sdp_buf.append(sdp)

        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    if len(counterfactuals) == 0:
        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10
        best_id = np.argmax(sdp_buf)

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = suf_leaf[best_id][S_star]
        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [0.]), np.array(size * [0.]),
                                                       x_train, y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        counterfactuals.append(cond)
        counterfactuals_sdp.append(sdp)
        counterfactuals_w.append(w)

    return np.array(counterfactuals).reshape(-1, x_train.shape[1], 2), np.array(counterfactuals_sdp).reshape(-1), np.array(counterfactuals_w).reshape(-1, x_train.shape[0])


def return_global_counterfactuals(ac_explainer, data, y_data, s_star, n_star, x_train, y_train, w, acc_level, pi_level,
                                  p_best=0.85, n_try=100,
                                  batch=100, max_iter=100, temp=1, max_iter_convergence=100
                                  ):
    """
    stack all the observations to compute the C_S for each observations: return the local CR
    """
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf(ac_explainer, data[i], y_data[i], s_star[i], n_star[i], w[i],
                                                   x_train, y_train, pi=pi_level, acc_level=acc_level,
                                                   p_best=p_best, n_try=n_try,
                                                   batch=batch, max_iter=max_iter, temp=temp,
                                                   max_iter_convergence=max_iter_convergence
                                                   )
        suf_leaves.append(suf_leaf)
        counterfactuals, counterfactuals_sdp, w_cond = \
            return_counterfactuals(ac_explainer, suf_leaf, s_star[i], n_star[i], data[i].reshape(1, -1),
                                   y_data[i].reshape(1, -1), x_train, y_train, pi_level)

        counterfactuals, counterfactuals_sdp, w_cond = remove_in_wsdp(counterfactuals, counterfactuals_sdp, w_cond)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)

    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


# Fonction global explanations
def return_ge_counterfactuals(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, x_train, y_train, cond_s, pi_level):
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []
    sdp_buf = []

    for leaf in suf_leaf:

        cond = cond_s.copy()
        cond[:, S_star] = leaf[S_star]
        # condition only based on the rules cond
        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [0.]), np.array(size * [0.]),
                                                       x_train, y_train, S=[[-6]], cond=cond)
        sdp_buf.append(sdp)

        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    if len(counterfactuals) == 0:
        best_id = np.argmax(sdp_buf)

        cond = cond_s.copy()
        cond[:, S_star] = suf_leaf[best_id][S_star]

        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [0.]), np.array(size * [0.]),
                                                       x_train, y_train, S=[[-6]], cond=cond)
        counterfactuals.append(cond)
        counterfactuals_sdp.append(sdp)
        counterfactuals_w.append(w)

    return np.array(counterfactuals).reshape(-1, x_train.shape[1], 2), np.array(counterfactuals_sdp).reshape(-1), np.array(counterfactuals_w).reshape(-1, x_train.shape[0])


def return_ge_global_counterfactuals(ac_explainer, data, y_data, s_star, n_star, x_train, y_train, w, acc_level, cond,
                                     pi_level, p_best=0.85, n_try=100,
                                  batch=100, max_iter=100, temp=1, max_iter_convergence=100):
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf(ac_explainer, data[i], y_data[i], s_star[i], n_star[i], w[i],
                                                   x_train, y_train, pi=pi_level, acc_level=acc_level,
                                                   p_best=p_best, n_try=n_try,
                                                   batch=batch, max_iter=max_iter, temp=temp,
                                                   max_iter_convergence=max_iter_convergence
                                                   )
        suf_leaves.append(suf_leaf)

        counterfactuals, counterfactuals_sdp, w_cond = return_ge_counterfactuals(ac_explainer, suf_leaf, s_star[i],
                                                                                 n_star[i],
                                                                                 data[i].reshape(1, -1),
                                                                                 y_data[i].reshape(1, -1), x_train,
                                                                                 y_train,
                                                                                 np.expand_dims(cond[i], 0),
                                                                                 pi_level)

        counterfactuals, counterfactuals_sdp, w_cond = remove_in_wsdp(counterfactuals, counterfactuals_sdp, w_cond)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)

    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


def print_rule(col_name, r, decision=None, sdp=True, output=None):
    for i, col in enumerate(col_name):
        if not ((r[i, 0] <= -1e+10 and r[i, 1] >= 1e+10)):
            print('If {} in [{}, {}] and '.format(col, r[i, 0], r[i, 1]))
            print(' ')

    if sdp == True:
        print('Then the output is = {}'.format(output))
        print('SDP Probability = {}'.format(decision))
    else:
        print('Then the output is  {}'.format(output))
        print('Counterfactual CDP Probability = {}'.format(decision))


def generate_candidate(x, S, x_train, cond, n_iterations):
    x_poss = [x_train[(cond[i, 0] <= x_train[:, i]) * (x_train[:, i] <= cond[i, 1]), i] for i in S]

    x_cand = np.repeat(x.reshape(1, -1), repeats=n_iterations, axis=0)
    for i in range(len(S)):
        # TODO: check if it's a good idea
        if x_poss[i].shape[0] == 0:
            x_cand[:, S[i]] = cond[S[i], 0]
        else:
            rdm_id = np.random.randint(0, x_poss[i].shape[0], n_iterations)
            x_cand[:, S[i]] = x_poss[i][rdm_id]

    return x_cand


def simulated_annealing(outlier_score, x, S, x_train, cond, batch, max_iter, temp,
                        max_iter_convergence=10):
    """
    Generate sample s.t. (X | X_S \in cond) using simulated annealing and outlier score.
    Args:
        outlier_score (lambda functon): outlier_score(X) return a outlier score. If the value are negative, then the observation is an outlier.
        x (numpy.ndarray)): 1-D array, an observation
        S (list): contains the indices of the variables on which to condition
        x_train (numpy.ndarray)): 2-D array represent the training samples
        cond (numpy.ndarray)): 3-D (#variables x 2 x 1) representing the hyper-rectangle on which to condition
        batch (int): number of sample by iteration
        max_iter (int): number of iteration of the algorithm
        temp (double): the temperature of the simulated annealing algorithm
        max_iter_convergence (double): minimun number of iteration to stop the algorithm if it find an in-distribution observation

    Returns:
        The generated sample, and its outlier score
    """
    best = generate_candidate(x, S, x_train, cond, 1)
    best_eval = outlier_score(best)[0]
    curr, curr_eval = best, best_eval

    move = 0
    for i in range(max_iter):

        x_cand = generate_candidate(curr, S, x_train, cond, batch)
        score_candidates = outlier_score(x_cand)

        candidate_eval = np.max(score_candidates)
        candidate = x_cand[np.argmax(score_candidates)]

        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval
            move = 0
        else:
            move += 1

        # check convergence
        if best_eval > 0 and move > max_iter_convergence:
            break

        diff = candidate_eval - curr_eval
        t = temp / np.log(float(i + 1))
        metropolis = np.exp(-diff / t)

        if diff > 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval

def save_model(model, name='{}'.format('dataset')):
    with open('{}.pickle'.format(name), 'wb') as f:
        pickle.dump(model, f)


def load_model(name='{}'.format('dataset')):
    with open('{}.pickle'.format(name), 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj

#------------------------------REGRESSION
def return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w):
    """
    return the observations with y in (down, up) that fall in the projected leaf of x
    when we condition given S=S_bar of x.
    """

    x_train_cnt = []
    y_train_cnt = []
    for i, wi in enumerate(w):
        if wi != 0 and down <= y_train[i] <= up:
            x_train_cnt.append(x_train[i].copy())
            y_train_cnt.append(y_train[i].copy())

    if len(x_train_cnt) == 0:
        x_train_cnt = x_train[(down <= y_train) * (y_train <= up)][:10]
        y_train_cnt = y_train[(down <= y_train) * (y_train <= up)][:10]

    x_train_cnt = np.array(x_train_cnt)
    y_train_cnt = np.array(y_train_cnt)
    return x_train_cnt, y_train_cnt


def return_leaf_cnt_reg(ac_explainer, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi,
                        p_best=0.85, n_try=100,
                        batch=100, max_iter=100, temp=1, max_iter_convergence=100
                        ):
    """
    return the original leaves with sdp >= pi of the observations with y in (down, up) that fall in the projected leaf
    when we condition given S=S_bar of x.
    """
    size = x_train_cnt.shape[0]
    sdp, rules = ac_explainer.compute_cdp_rule(x_train_cnt, y_train_cnt,
                                               np.array(size * [down]), np.array(size * [up]),
                                               x_train, y_train,
                                               size * [list(range(x_train.shape[1]))]
                                               )
    if np.sum(sdp >= pi) != 0:
        rules_unique = np.unique(rules[sdp >= pi], axis=0)
    else:
        argsort = np.argsort(-sdp)
        rules_unique = rules[argsort[:5]]

    # TODO: check if 5 is sufficient
    r_buf = rules_unique.copy()
    for i in range(rules_unique.shape[0]):
        list_ric = [r.copy() for r in r_buf if not np.allclose(r, rules_unique[i])]
        # find_union(rules_unique[i], list_ric, S=S_star)
        sr, _ = rules_by_annealing(volume_rectangles,
                                   np.expand_dims(rules_unique[i], axis=0),
                                   np.expand_dims(list_ric, axis=0), x_train,
                                   p_best=p_best, n_try=n_try,
                                   batch=batch, max_iter=max_iter, temp=temp,
                                   max_iter_convergence=max_iter_convergence
                                   )
        rules_unique[i] = np.squeeze(sr[-1])
    return rules_unique


def get_compatible_leaf_reg(acvtree, x, y_target, down, up, S_star, S_bar_set, w, x_train, y_train, pi, acc_level,
                            p_best=0.85, n_try=100,
                            batch=100, max_iter=100, temp=1, max_iter_convergence=100
                            ):
    """
    Compute the compatible leaves and order given their accuracy
    """
    x_train_cnt, y_train_cnt = return_xy_cnt_reg(x, y_target, down, up, S_bar_set, x_train, y_train, w)
    compatible_leaves = return_leaf_cnt_reg(acvtree, S_star, x_train_cnt, y_train_cnt, down, up, x_train, y_train, pi,
                                            p_best=p_best, n_try=n_try,
                                            batch=batch, max_iter=max_iter, temp=temp,
                                            max_iter_convergence=max_iter_convergence
                                            )
    compatible_leaves = np.unique(compatible_leaves, axis=0)
    compatible_leaves = remove_in(compatible_leaves)
    compatible_leaves = np.round(compatible_leaves, 2)

    partition_leaf = compatible_leaves.copy()
    d = partition_leaf.shape[1]
    nb_leaf = partition_leaf.shape[0]
    leaves_acc = []
    suf_leaf = []

    for i in range(nb_leaf):
        x_in = np.prod([(x_train[:, s] <= partition_leaf[i, s, 1]) * (x_train[:, s] >= partition_leaf[i, s, 0])
                        for s in range(d)], axis=0).astype(bool)

        y_in = y_train[x_in]
        acc = np.mean((down <= y_in) * (y_in <= up))
        leaves_acc.append(acc)

        if acc >= acc_level:
            suf_leaf.append(partition_leaf[i])

    best_id = np.argmax(leaves_acc)
    if len(suf_leaf) == 0:
        suf_leaf.append(partition_leaf[best_id])

    return suf_leaf, partition_leaf, leaves_acc, partition_leaf[best_id], leaves_acc[best_id]


def return_counterfactuals_reg(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, down, up, x_train, y_train, pi_level):
    """
    Compute the SDP of each C_S and return the ones that has sdp >= pi_level
    """
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []
    sdp_buf = []
    for leaf in suf_leaf:

        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = leaf[S_star]
        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [down]), np.array(size * [up]),
                                                       x_train, y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        sdp_buf.append(sdp)

        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    if len(counterfactuals) == 0:

        best_id = np.argmax(sdp_buf)
        cond = np.ones(shape=(1, x_train.shape[1], 2))
        cond[:, :, 0] = -1e+10
        cond[:, :, 1] = 1e+10

        for s in S_bar_set:
            cond[:, s, 0] = x[:, s]
            cond[:, s, 1] = x[:, s]

        cond[:, S_star] = suf_leaf[best_id][S_star]
        size = x.shape[0]
        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, np.array(size * [down]), np.array(size * [up]),
                                                       x_train, y_train, S=[S_bar_set], cond=cond,
                                                       pi_level=pi_level)
        counterfactuals.append(cond)
        counterfactuals_sdp.append(sdp)
        counterfactuals_w.append(w)

    return np.array(counterfactuals).reshape(-1, x_train.shape[1], 2), np.array(counterfactuals_sdp).reshape(-1), np.array(counterfactuals_w).reshape(-1, x_train.shape[0])


def return_global_counterfactuals_reg(ac_explainer, data, y_data, down, up, s_star, n_star, x_train, y_train, w,
                                      acc_level, pi_level, p_best=0.85, n_try=100,
                            batch=100, max_iter=100, temp=1, max_iter_convergence=100):
    """
    stack all to compute the C_S for each observations
    """
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf_reg(ac_explainer, data[i], y_data[i], down[i], up[i],
                                                       s_star[i], n_star[i], w[i], x_train, y_train,
                                                       pi=pi_level, acc_level=acc_level,
                                                       p_best=p_best, n_try=n_try,
                                                       batch=batch, max_iter=max_iter, temp=temp,
                                                       max_iter_convergence=max_iter_convergence
                                                       )
        suf_leaves.append(suf_leaf)

        counterfactuals, counterfactuals_sdp, w_cond = \
            return_counterfactuals_reg(ac_explainer, suf_leaf, s_star[i], n_star[i], data[i].reshape(1, -1),
                                       y_data[i].reshape(1, -1), down[i], up[i], x_train, y_train, pi_level)

        counterfactuals, counterfactuals_sdp, w_cond = remove_in_wsdp(counterfactuals, counterfactuals_sdp, w_cond)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)
    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


def return_ge_counterfactuals_reg(ac_explainer, suf_leaf, S_star, S_bar_set, x, y, down, up, x_train, y_train, cond_s,
                                  pi_level):
    counterfactuals = []
    counterfactuals_sdp = []
    counterfactuals_w = []
    sdp_buf = []
    for leaf in suf_leaf:

        cond = cond_s.copy()
        cond[:, S_star] = leaf[S_star]

        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, down, up, x_train, y_train, S=[[-6]], cond=cond)
        sdp_buf.append(sdp)

        if sdp >= pi_level:
            counterfactuals.append(cond)
            counterfactuals_sdp.append(sdp)
            counterfactuals_w.append(w)

    if len(counterfactuals) == 0:
        best_id = np.argmax(sdp_buf)

        cond = cond_s.copy()
        cond[:, S_star] = suf_leaf[best_id][S_star]

        sdp, w = ac_explainer.compute_cdp_cond_weights(x, y, down, up, x_train, y_train, S=[[-6]], cond=cond)

        counterfactuals.append(cond)
        counterfactuals_sdp.append(sdp)
        counterfactuals_w.append(w)

    return np.array(counterfactuals).reshape(-1, x_train.shape[1], 2), np.array(counterfactuals_sdp).reshape(-1), np.array(counterfactuals_w).reshape(-1, x_train.shape[0])


def return_ge_global_counterfactuals_reg(ac_explainer, data, y_data, down, up, s_star, n_star, x_train, y_train, w,
                                         acc_level, cond,
                                         pi_level, p_best=0.85, n_try=100,
                            batch=100, max_iter=100, temp=1, max_iter_convergence=100):
    N = data.shape[0]
    suf_leaves = []
    counterfactuals_samples = []
    counterfactuals_samples_sdp = []
    counterfactuals_samples_w = []

    for i in tqdm(range(N)):
        suf_leaf, _, _, _, _ = get_compatible_leaf_reg(ac_explainer, data[i], y_data[i], down[i], up[i],
                                                       s_star[i], n_star[i], w[i], x_train, y_train,
                                                       pi=pi_level, acc_level=acc_level,
                                                       p_best=p_best, n_try=n_try,
                                                       batch=batch, max_iter=max_iter, temp=temp,
                                                       max_iter_convergence=max_iter_convergence
                                                       )
        suf_leaves.append(suf_leaf)

        counterfactuals, counterfactuals_sdp, w_cond = return_ge_counterfactuals_reg(ac_explainer, suf_leaf, s_star[i],
                                                                                     n_star[i],
                                                                                     data[i].reshape(1, -1),
                                                                                     y_data[i].reshape(1, -1),
                                                                                     down[i].reshape(-1),
                                                                                     up[i].reshape(-1),
                                                                                     x_train, y_train,
                                                                                     np.expand_dims(cond[i], 0),
                                                                                     pi_level)

        counterfactuals, counterfactuals_sdp, w_cond = remove_in_wsdp(counterfactuals, counterfactuals_sdp, w_cond)

        counterfactuals_samples.append(counterfactuals)
        counterfactuals_samples_sdp.append(counterfactuals_sdp)
        counterfactuals_samples_w.append(w_cond)

    return counterfactuals_samples, counterfactuals_samples_sdp, counterfactuals_samples_w


def return_possible_values(rule, rule_data):
    d = rule.shape[0]
    adj_rules = rule_data.copy()

    left_by_var = []
    right_by_var = []
    for var in range(d):
        left = set([adj_rules[i, var, 0] for i in range(adj_rules.shape[0]) if adj_rules[i, var, 0] < rule[var, 0]])
        right = set([adj_rules[i, var, 1] for i in range(adj_rules.shape[0]) if adj_rules[i, var, 1] > rule[var, 1]])

        left.add(rule[var, 0])
        right.add(rule[var, 1])

        left_by_var.append(list(left))
        right_by_var.append(list(right))
    return left_by_var, right_by_var


def rdm_sample_fromlist(list_values):
    return list_values[np.random.randint(0, len(list_values))]


def generate_newrule(rule, left_by_var, right_by_var, p_best=0.6):
    d = rule.shape[0]
    rule_c = rule.copy()
    if np.random.rand() > p_best:
        for i in range(d):
            rule_c[i, 0] = rdm_sample_fromlist(left_by_var[i])
            rule_c[i, 1] = rdm_sample_fromlist(right_by_var[i])
    else:
        for i in range(d):
            left = [left_by_var[i][j] for j in range(len(left_by_var[i])) if left_by_var[i][j] <= rule_c[i, 0]]
            right = [right_by_var[i][j] for j in range(len(right_by_var[i])) if right_by_var[i][j] >= rule_c[i, 1]]

            rule_c[i, 0] = rdm_sample_fromlist(left)
            rule_c[i, 1] = rdm_sample_fromlist(right)

    return rule_c


def check_rule(rule, x_train, bad_sample):
    x_in = np.prod([(x_train[:, s] <= rule[s, 1]) * (x_train[:, s] >= rule[s, 0])
                    for s in range(x_train.shape[1])], axis=0)
    return np.sum(x_in * bad_sample) == 0


def generate_valid_rule(rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.9, n_try=100):
    valid = False
    it = 0
    while valid == False and it <= n_try:
        gen_rule = generate_newrule(rule, left_by_var, right_by_var, p_best)
        valid = check_rule(gen_rule, x_train, bad_sample)
        it += 1
    if valid:
        return gen_rule
    else:
        return rule


def generate_rules(rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.6, n_try=100, n_gen=100):
    rules = [generate_valid_rule(rule, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try) for i in range(n_gen)]
    return rules


def convert_boundvalues(rules, max_values, min_values):
    for i in range(rules.shape[0]):
        if rules[i, 0] <= -1e+10:
            rules[i, 0] = min_values[i]
        if rules[i, 1] <= -1e+10:
            rules[i, 1] = min_values[i]
        if rules[i, 0] >= 1e+10:
            rules[i, 0] = max_values[i]
        if rules[i, 1] >= 1e+10:
            rules[i, 1] = max_values[i]
    return rules


def volume_rectangle(rec, max_values, min_values):
    d = rec.shape[0]
    rec_c = rec.copy()
    rec_c = convert_boundvalues(rec_c, max_values, min_values)
    v = 1
    for i in range(d):
        v *= (rec_c[i, 1] - rec_c[i, 0])
    return v


def volume_rectangles(recs, max_values, min_values):
    return [volume_rectangle(recs[i], max_values, min_values) for i in range(len(recs))]


def max_dens(recs, x_train, bad_sample):
    d = x_train.shape[1]
    x_in = np.prod([(x_train[:, s] <= recs[s, 1]) * (x_train[:, s] >= recs[s, 0])
                    for s in range(d)], axis=0)
    return np.sum(x_in * (1 - bad_sample)) / np.sum(1 - bad_sample)


def max_denss(recs, x_train, bad_sample):
    return [max_dens(recs[i], x_train, bad_sample) for i in range(len(recs))]


def rules_simulated_annealing(value_function, rule, left_by_var, right_by_var, x_train, bad_sample, p_best=0.6,
                              n_try=100,
                              batch=100, max_iter=100, temp=1,
                              max_iter_convergence=50):
    max_values = [np.max(x_train[:, i]) for i in range(x_train.shape[1])]
    min_values = [np.min(x_train[:, i]) for i in range(x_train.shape[1])]

    best = generate_rules(rule, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try, n_gen=1)
    best_eval = value_function(best, max_values, min_values)[0]
    curr, curr_eval = np.squeeze(best[0]), best_eval

    move = 0
    for i in range(max_iter):

        x_cand = generate_rules(curr, left_by_var, right_by_var, x_train, bad_sample, p_best, n_try, n_gen=batch)
        score_candidates = value_function(x_cand, max_values, min_values)

        candidate_eval = np.max(score_candidates)
        candidate = x_cand[np.argmax(score_candidates)]

        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval
            move = 0
        else:
            move += 1

        # check convergence
        if move > max_iter_convergence:
            break

        diff = candidate_eval - curr_eval
        t = temp / np.log(float(i + 1))
        metropolis = np.exp(-diff / t)

        if diff > 0 or rand() < metropolis:
            curr, curr_eval = candidate, candidate_eval

    return best, best_eval


def rules_by_annealing(value_function, rules, rules_data, x_train, p_best=0.85, n_try=100,
                       batch=100, max_iter=100, temp=1, max_iter_convergence=100):
    sr, sr_eval = [], []
    d = x_train.shape[1]

    for idx in range(rules.shape[0]):
        rule = rules[idx]
        rule_data = rules_data[idx]

        x_in = np.prod([(x_train[:, s] <= rule[s, 1]) * (x_train[:, s] >= rule[s, 0])
                for s in range(d)], axis=0)

        for r in rule_data:
            x_in += np.prod([(x_train[:, s] <= r[s, 1]) * (x_train[:, s] >= r[s, 0])
                for s in range(d)], axis=0)

        bad_sample = 1. - (x_in > 0)
        # print('bad', bad_sample)
        left_by_var, right_by_var = return_possible_values(rule, rule_data)
        best, best_eval = rules_simulated_annealing(value_function, rule, left_by_var, right_by_var, x_train,
                                                    bad_sample,
                                                    p_best, n_try,
                                                    batch, max_iter, temp, max_iter_convergence)
        sr.append(best)
        sr_eval.append(best_eval)
    return np.array(sr), sr_eval


def remove_in_wsdp(ra, sa, wa):
    """
    remove A if A subset of B in the list of compatible leaves
    """
    ra_b = ra.copy()
    sa_b = sa.copy()
    wa_b = wa.copy()
    remove_list = []
    for i in range(ra.shape[0]):
        for j in range(ra.shape[0]):
            if i != j and np.prod([(ra[i, s, 1] <= ra[j, s, 1]) * (ra[i, s, 0] >= ra[j, s, 0])
                                   for s in range(ra.shape[1])], axis=0).astype(bool):
                # ra[i] = ra[j]
                # sa[i] = sa[j]
                # wa[i] = wa[j]
                remove_list.append(j)
    ra_b = np.delete(ra_b, remove_list, axis=0)
    sa_b = np.delete(sa_b, remove_list, axis=0)
    wa_b = np.delete(wa_b, remove_list, axis=0)
    return ra_b, sa_b, wa_b
