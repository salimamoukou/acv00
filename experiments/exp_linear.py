# Python packages
from .utils import *

# R packages
from rpy2.robjects import FloatVector, IntVector, pandas2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

pandas2ri.activate()
cond = importr('condMVNorm')
mvn = importr('mvtnorm')


class ExperimentsLinear:

    def __init__(self, mean=np.array([0, 0, 0]), cov=0.8 * np.ones(shape=(3, 3)) - (0.8 - 1) * np.eye(3),
                 coefs=np.array([1, 1, 1]), n=1000, C=[[]]):
        """
        Load toy experiments where data D={(X,Y), i={1,...n}} are built with linear model f, such that Y = f(X) = B^tX and
        X follow a multivariate gaussian distribution N(mu, sigma).

        Args:
            mean (array): mean of the Gaussian
            cov (array): covariance matrix
            coefs (array): coefficient of  regression
            n (int): size of data generated
            C (list[list]): list of variables group together
        """

        self.n = n
        self.d = len(mean)
        self.index = range(len(mean))

        # Python init
        self.mean = np.array(mean)
        self.cov = cov
        self.coefs = coefs
        self.C = C

        # R init
        self.r_mean = FloatVector(self.mean)
        sig = robjects.Vector(self.cov)
        sig = robjects.Matrix(sig)
        self.r_cov = sig

        # data generation
        self.data_gen = st.multivariate_normal(self.mean, self.cov)
        self.data = self.data_gen.rvs(self.n)
        self.y_train = linear_regression(self.coefs, self.data)

        self.data_test = self.data_gen.rvs(self.n)
        self.y_test = linear_regression(self.coefs, self.data_test)

    def cond_exp_tree(self, x, tree, S, algo="shap", N=50000, data=None):
        """
        Compute conditional expectation of a tree_regressor given S

        Args:
            x (int): observation
            tree (DecisionTreeRegressor.tree_): model
            S (list): index of the variable on which we are conditioning
            algo (string): name of the estimator. there are 5 estimators: 'pluging', 'shap', 'exact', 'monte_carlo',
                            'monte_carlo_interventional', 'plugin_interventional'
            data (array): data use to compute the expectation
            N (int): size of observations sample when using Monte carlo estimators.

        Returns:
            float: E[tree | X_S = x_S]
        """

        if type(data) != type(self.data):
            data = self.data

        a = tree.children_left
        b = tree.children_right
        f = tree.feature
        t = tree.threshold
        r_w = tree.weighted_n_node_samples
        v = np.array([tree.value[i][0][0] for i in range(tree.node_count)])
        index = range(x.shape[0])

        def G(i, w):

            if f[i] == -2:
                return float(w * v[i])
            else:
                if (f[i] in S):
                    if x[f[i]] <= t[i]:
                        return G(a[i], w)
                    else:
                        return G(b[i], w)
                else:
                    return G(a[i], w * r_w[a[i]] / r_w[i]) + G(b[i], w * r_w[b[i]] / r_w[i])

        def explore(i, tab):

            if f[i] == -2:
                tab[i] = 1
            else:
                if f[i] in S:
                    if x[f[i]] <= t[i]:
                        explore(a[i], tab)
                    else:
                        explore(b[i], tab)
                else:
                    explore(a[i], tab)
                    explore(b[i], tab)

        def get_partition(leaf_id, part):

            left = np.where(tree.children_left == leaf_id)[0]
            right = np.where(tree.children_right == leaf_id)[0]

            if (len(left) == 0) * (len(right) == 0):
                return (part)

            else:
                if len(right) != 0:
                    right = int(right[0])

                    part[f[right]] = np.concatenate((part[f[right]], np.array([[t[right], np.inf]])))
                    return get_partition(right, part)
                else:
                    left = int(left[0])
                    part[f[left]] = np.concatenate((part[f[left]], np.array([[-np.inf, t[left]]])))
                    return get_partition(left, part)

        def get_final_partition(part):
            final_partition = {}
            for i, var_part in enumerate(part):
                final_partition[i] = [np.max(var_part[:, 0]), np.min(var_part[:, 1])]
            return final_partition

        def explore_part(i, tab, partition_leaves, partition_global, prob_global, S, S_bar, data, intv=False):
            if f[i] == -2:
                #         tab[i] = 1
                compatible_leaves.append(i)
                partition_global[i] = partition_leaves
                partition_leaves = np.squeeze(np.array(partition_leaves))
                # partition_leaves = np.reshape(np.array(partition_leaves), (self.d + 2, 2)).T
                # partition_leaves = pd.DataFrame(partition_leaves)[S]
                low = FloatVector(partition_leaves[:, 0][S_bar])
                up = FloatVector(partition_leaves[:, 1][S_bar])

                prob_global[i] = cond.pcmvnorm(low, up, self.r_mean, self.r_cov, c_index, given, FloatVector(x[S]))[0]
                # print(prob_global[i])

            else:
                if f[i] in S:
                    if x[f[i]] <= t[i]:
                        part = partition_leaves.copy()
                        part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
                        part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
                        explore_part(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                    else:
                        part = partition_leaves.copy()
                        part[f[i]] = np.concatenate((part[f[i]], np.array([[t[i], np.inf]])))
                        part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
                        explore_part(b[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                else:
                    part = partition_leaves.copy()
                    part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
                    part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])

                    part_2 = partition_leaves.copy()
                    part_2[f[i]] = np.concatenate((part_2[f[i]], np.array([[t[i], np.inf]])))
                    part_2[f[i]] = np.array([[np.max(part_2[f[i]][:, 0]), np.min(part_2[f[i]][:, 1])]])

                    explore_part(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                    explore_part(b[i], tab, part_2, partition_global, prob_global, S, S_bar, data, intv)

        def explore_partition(i, tab, partition_leaves, partition_global, prob_global, S, S_bar, data, intv=False):

            if f[i] == -2:
                #         tab[i] = 1
                compatible_leaves.append(i)
                partition_global[i] = partition_leaves
                partition_leaves = np.squeeze(np.array(partition_leaves))

                if not intv:
                    section_x = np.prod(
                        [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in S],
                        axis=0)
                else:
                    section_x = np.prod(
                        [(data[:, s] <= partition_leaves[s, 1]) * (data[:, s] >= partition_leaves[s, 0]) for s in
                         S_bar], axis=0)

                prob_global[i] = np.sum(section_x)

            else:
                if f[i] in S:
                    if x[f[i]] <= t[i]:
                        part = partition_leaves.copy()
                        part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
                        part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
                        explore_partition(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                    else:
                        part = partition_leaves.copy()
                        part[f[i]] = np.concatenate((part[f[i]], np.array([[t[i], np.inf]])))
                        part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])
                        explore_partition(b[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                else:
                    part = partition_leaves.copy()
                    part[f[i]] = np.concatenate((part[f[i]], np.array([[-np.inf, t[i]]])))
                    part[f[i]] = np.array([[np.max(part[f[i]][:, 0]), np.min(part[f[i]][:, 1])]])

                    part_2 = partition_leaves.copy()
                    part_2[f[i]] = np.concatenate((part_2[f[i]], np.array([[t[i], np.inf]])))
                    part_2[f[i]] = np.array([[np.max(part_2[f[i]][:, 0]), np.min(part_2[f[i]][:, 1])]])

                    explore_partition(a[i], tab, part, partition_global, prob_global, S, S_bar, data, intv)
                    explore_partition(b[i], tab, part_2, partition_global, prob_global, S, S_bar, data, intv)

        if algo == 'shap':
            return G(0, 1)

        elif algo == 'plugin':
            if S == []:
                compatible_leaves = np.zeros(tree.node_count)
                explore(0, compatible_leaves)
                compatible_leaves = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
                p = r_w[compatible_leaves] / np.sum(r_w[compatible_leaves])

            elif len(S) == len(x):
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
                return np.mean(y_pred)

            else:
                S_bar = [i for i in self.index if i not in S]
                partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                                    for i in range(tree.node_count)}
                prob_global = {}
                compatible_leaves = []
                explore_partition(0, compatible_leaves, partition_leaves, partition_global, prob_global, S, S_bar, data,
                                  False)

                nbs_leaf = np.array([prob_global[key] for key in compatible_leaves])
                p = r_w[compatible_leaves] / nbs_leaf

            return np.sum(p * v[compatible_leaves])


        elif algo == 'plugin_interventionalv2':
            return self.cond_exp_tree_intv(tree, x, S, v, explore, get_partition, get_final_partition, data)

        elif algo == 'plugin_interventional':
            S_bar = [i for i in self.index if i not in S]
            if S == []:
                compatible_leaves = np.zeros(tree.node_count)
                explore(0, compatible_leaves)
                compatible_leaves = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
                p = r_w[compatible_leaves] / np.sum(r_w[compatible_leaves])

            elif len(S) == len(x):
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
                return np.mean(y_pred)

            else:
                partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                                    for i in range(tree.node_count)}
                prob_global = {}
                compatible_leaves = []
                explore_partition(0, compatible_leaves, partition_leaves, partition_global, prob_global, S, S_bar, data,
                                  True)
                p = np.array([prob_global[key] / len(data) for key in compatible_leaves])

            return np.sum(p * v[compatible_leaves])

        elif algo == 'monte_carlo':
            if len(S) != len(index):
                S_bar = [i for i in index if i not in S]
                rg_data = np.zeros(shape=(N, len(index)))
                rg_data[:, S] = x[S]
                rg = sampleMVN(N, self.mean, self.cov, S_bar, S, x[S])
                rg_data[:, S_bar] = rg

                y_pred = tree.predict(np.array(rg_data, dtype=np.float32))

            else:
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

            return np.mean(y_pred)

        elif algo == 'monte_carlo_interventional':
            if len(S) != len(index):
                rg = sampleMarMVN(N, self.mean, self.cov, self.index, [])
                rg_data = pd.DataFrame(rg, columns=[str(i) for i in self.index])

                def get_given_data(idx):
                    val = np.array([x[idx]])
                    val = np.tile(val, N)
                    return val

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]
                y_pred = tree.predict(np.array(rg_data.values, dtype=np.float32))

            else:
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

            return np.mean(y_pred)

        elif algo == 'exact':
            S_bar = [i for i in self.index if i not in S]
            given = IntVector(np.array([i + 1 for i in S]))
            c_index = np.array([i + 1 for i in index if i not in S])
            # S_bar = [i for i in index if i not in S]
            # if S == []:
            #     compatible_leaves = np.zeros(tree.node_count)
            #     explore(0, compatible_leaves)
            #     compatible_leaves = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
            #     p = r_w[compatible_leaves] / np.sum(r_w[compatible_leaves])

            if len(S) == len(x):
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
                return np.mean(y_pred)

            else:

                # S_bar = [i for i in self.index if i not in S]
                partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                partition_global = {i: [np.array([[-np.inf, np.inf]]) for i in range(data.shape[1])]
                                    for i in range(tree.node_count)}
                prob_global = {}
                compatible_leaves = []
                explore_part(0, compatible_leaves, partition_leaves, partition_global, prob_global, S, S_bar, data,
                             False)

                p = np.array([prob_global[key] for key in compatible_leaves])
                # print(p)
                # p = r_w[compatible_leaves] / nbs_leaf

            return np.sum(p * v[compatible_leaves])

        elif algo == 'exact_v1':

            if len(S) != len(index):
                given = IntVector(np.array([i + 1 for i in S]))
                c_index = np.array([i + 1 for i in index if i not in S])
                S_bar = [i for i in index if i not in S]

                compatible_leaves = np.zeros(tree.node_count)
                explore(0, compatible_leaves)

                leaf_idx = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
                leaf_proba = []
                for leaf_id in leaf_idx:
                    partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(x.shape[0])]
                    partition_leaves = get_partition(leaf_id, partition_leaves)
                    partition_leaves = pd.DataFrame(get_final_partition(partition_leaves))[S_bar]

                    low = FloatVector(partition_leaves.iloc[0])
                    up = FloatVector(partition_leaves.iloc[1])

                    leaf_proba.append(
                        cond.pcmvnorm(low, up, self.r_mean, self.r_cov, c_index, given, FloatVector(x[S]))[0])

                leaf_proba = np.array(leaf_proba)
                return np.sum(leaf_proba * v[compatible_leaves == 1])

            else:
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
                return np.mean(y_pred)
        else:
            raise ValueError("This algo is not implemented. Available estimators are: plugin, exact, monte_carlo,\
                             monte_carlo_interventional, plugin_interventional")

    def cond_exp_linear_model(self, x, S):
        """
        Compute exact conditional expectation given variable in S of the linear model

        Args:
            x (array): observation
            S (list): index of the variables on which we are conditioning

        Returns:
            (array): E[f(x)[X_S=x_S]
        """

        x_in = x.copy()
        index = range(x.shape[0])

        S_bar = [i for i in index if i not in S]

        cond_dist = condMVN(self.mean, self.cov, S_bar, S, x_in[S])

        if len(S) != len(index):
            x_in[S_bar] = cond_dist[0]

        out = np.sum(self.coefs * x_in)
        return out

    def cond_exp_tree_intv(self, model, x, S, v, explore, get_partition, get_final_partition, data):
        """ Compute marginal expectation of a tree_regressor """

        S_bar = np.array([i for i in self.index if i not in S and type(i) != str])
        x_in = x.copy()

        if len(S) != len(x_in):
            cat_va = [ca for ca in S if type(ca) == str]
            S = [cu for cu in S if type(cu) != str]

            compatible_leaves = np.zeros(model.node_count)
            explore(0, compatible_leaves)

            leaf_idx = [i for i in range(model.node_count) if compatible_leaves[i] == 1]
            leaf_proba = []
            for leaf_id in leaf_idx:
                partition_leaves = [np.array([[-np.inf, np.inf]]) for i in range(x.shape[0])]
                partition_leaves = get_partition(leaf_id, partition_leaves)

                partition_leaves = pd.DataFrame(get_final_partition(partition_leaves))[S_bar]
                low = partition_leaves.iloc[0]
                up = partition_leaves.iloc[1]

                section_x = np.sum(
                    np.prod([(data[:, s_bar] <= up[s_bar]) * (data[:, s_bar] >= low[s_bar]) for s_bar in S_bar],
                            axis=0))
                # for i in range(data.shape[0]):
                #    section_x += np.prod([(data[i][s_bar] <= up[s_bar])*(data[i][s_bar] >= low[s_bar]) for s_bar in S_bar])

                prob_leaf = section_x / len(data)
                leaf_proba.append(prob_leaf)

            leaf_proba = np.array(leaf_proba)
            return np.sum(leaf_proba * v[compatible_leaves == 1])

        else:
            y_pred = model.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

        return np.mean(y_pred)

    def mar_exp_linear_model(self, x, S):
        """
        Compute marginal expectation given variable in S of the linear model

        Args:
            x (array): observation
            S (list): index of variables in S

        Returns:
            (array): E[f(x_S, X_\bar{S})]
        """

        x_in = x.copy()
        index = range(x.shape[0])

        S_bar = [i for i in index if i not in S]

        if len(S) != len(index):
            x_in[S_bar] = self.mean[S_bar]

        out = np.sum(self.coefs * x_in)
        return out

    def linear_shap(self, x):
        """
        Compute observational shapley value of x with the linear model

        Args:
            x (array): observation

        Returns:
            (array): Shapley values of x with the linear model
        """
        m = len(x)
        va_id = list(range(m))
        if self.C[0] != []:
            for c in self.C:
                m -= len(c)
                va_id = list(set(va_id) - set(c))
            m += len(self.C)
            for c in self.C:
                va_id += [c]

        phi = np.zeros(len(x))
        for p in itertools.permutations(va_id):
            for i in range(m):
                phi[chain_l(p[i])] += \
                    self.cond_exp_linear_model(x, chain_l(p[:i + 1])) - self.cond_exp_linear_model(x, chain_l(p[:i]))
        return phi / math.factorial(m)

    def linear_mar_shap(self, x):
        """
        Compute interventional shapley value of x with the linear model

        Args:
            x (array): observation

        Returns:
            (array): Shapley values of x with the linear model
        """

        m = len(x)
        va_id = list(range(m))
        if self.C[0] != []:
            for c in self.C:
                m -= len(c)
                va_id = list(set(va_id) - set(c))
            m += len(self.C)
            for c in self.C:
                va_id += [c]

        phi = np.zeros(len(x))
        for p in itertools.permutations(va_id):
            for i in range(m):
                phi[chain_l(p[i])] += \
                    self.mar_exp_linear_model(x, chain_l(p[:i + 1])) - self.mar_exp_linear_model(x, chain_l(p[:i]))
        return phi / math.factorial(m)

    def tree_shap(self, tree, x, algo, N=50000, data=None):
        """
        Compute Classic Shapley values of a regressor of x

        Args:
            tree (DecisionTreeClassifier):
            x (array): observation
            algo (string): name of the estimator. there are 5 estimators: 'pluging', 'shap', 'exact', 'monte_carlo',
                            'monte_carlo_interventional', 'plugin_interventional', 'monte_carlo'
            data (array): data used to compute the Shapley values
            C (list[list]): list of the different coalition of variables by their index
            N (int): size of observations sample when using monte_carlo estimators

        Returns:
            array: Shapley values of x
        """
        if type(data) != type(self.data):
            data = self.data
        if tree == None:
            tree_model = DecisionTreeRegressor()
            tree_model = self.fit_model(tree_model).tree_
        else:
            tree_model = tree.tree_

        m = len(x)
        va_id = list(range(m))

        if self.C[0] != []:
            for c in self.C:
                m -= len(c)
                va_id = list(set(va_id) - set(c))
            m += len(self.C)
            for c in self.C:
                va_id += [c]

        phi = np.zeros(len(x))
        for p in itertools.permutations(va_id):
            for i in range(m):
                phi[chain_l(p[i])] += \
                    self.cond_exp_tree(x=x, tree=tree_model, S=chain_l(p[:i + 1]), algo=algo, N=N, data=data) - \
                    self.cond_exp_tree(x=x, tree=tree_model, S=chain_l(p[:i]), algo=algo, N=N, data=data)
        return phi / math.factorial(m)

    def shap_fit_metric(self, model, data, nb, plot=True):
        """
        Compute Shapley values of observations in data given the different estimators, and return the errors between
        the exact computation and the different estimators.

        Args:
            model (DecisionTreeRegressor):
            data (array): data used to compute the Shapley values
            nb (int): number of observations used
            plot (Bool): If True, Show a boxplot of the errors

        Returns:
            (dataFrame): data that contains the SV of each observation given the different estimators and their erros.
        """

        shap_true, shap_plugin, shap_shap, shap_monte_carlo, shap_plugin, shap_sal = [], [], [], [], [], []

        if model is None:
            tree_model = DecisionTreeRegressor()
            tree_model = self.fit_model(tree_model)
        else:
            tree_model = model

        for i in tqdm(range(nb)):
            shap_true.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='exact'), axis=0))
            # shap_plugin.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='plugin'), axis=0))
            # shap_shap.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='shap'), axis=0))
            shap_monte_carlo.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='monte_carlo'), axis=0))
            shap_plugin.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='plugin'), axis=0))
            # shap_sal.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='v_2'), axis=0))

        shap_true, shap_monte_carlo, shap_plugin = np.concatenate(shap_true, axis=0), \
                                                       np.concatenate(shap_monte_carlo, axis=0),\
                                                       np.concatenate(shap_plugin, axis=0)

        explainer_observational = shap.TreeExplainer(tree_model, feature_perturbation='observational')
        # explainer_interventional = shap.TreeExplainer(tree_model, feature_perturbation='interventional', data=self.data)

        shap_observational = explainer_observational.shap_values(data[:nb])
        # shap_interventional = explainer_interventional.shap_values(data[:nb])

        shap_data = np.concatenate([shap_true, shap_monte_carlo, shap_plugin,
                                    shap_observational], axis=0)

        shap_dataFra = pd.DataFrame(data=shap_data, columns=[str(i) for i in self.index])
        shap_dataFra['label'] = nb * ['exact'] + nb * ['mc'] + \
                                nb * ['Plug-In'] + nb * ['SHAP']

        err_true = l1_norm(shap_true - shap_true)
        # err_plugin = l1_norm(shap_plugin-shap_true)
        # err_shap = l1_norm(shap_shap-shap_true)
        err_monte_carlo = l1_norm(shap_monte_carlo - shap_true)
        err_plugin_sal = l1_norm(shap_plugin - shap_true)
        err_shap_observational = l1_norm(shap_observational - shap_true)
        # err_shap_interventional = l1_norm(shap_interventional - shap_true)
        # err_shap_sal = l1_norm(shap_sal - shap_true)

        err_concat = np.concatenate([err_true, err_monte_carlo,
                                     err_plugin_sal, err_shap_observational], axis=0)

        shap_dataFra['error_l1'] = err_concat

        if plot:
            fig, ax = plt.subplots(dpi=150)
            sns.boxplot(data=shap_dataFra, x='label', y='error_l1')

        return shap_dataFra

    def fit_model(self, model):
        """ Fit a model and show the errors. """
        model.fit(self.data, self.y_train)

        error_test = mean_squared_error(model.predict(self.data_test), self.y_test)
        error_train = mean_squared_error(model.predict(self.data), self.y_train)

        print('fitted model train_error = {}'.format(error_train))
        print('fitted model test_error = {}'.format(error_test))
        return model
