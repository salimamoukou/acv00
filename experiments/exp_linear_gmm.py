from .utils import *


class ExperimentsLinearGMM:

    def __init__(self, mean={'a': [0., 0., 0.], 'b': [0., 0., 0.], 'c': [0., 0., 0.]},
                 cov={key: 0.8 * np.ones(shape=(3, 3)) - (0.8 - 1) * np.eye(3) for key in ['a', 'b', 'c']},
                 coefs={'a': [1, 1, 1], 'b': [1, 1, 1], 'c': [1, 1, 1]},
                 pi={'a': 0.25, 'b': 0.25, 'c': 0.5}, n=1000, C=[[]], cat_index=[]):
        """
        Load toy experiments where data D={(X, Y, Z), i={1,...n}} are built with linear model f, such that Z = f(X, Y) = B_Y^t X,
        X follow a multivariate gaussian distribution N(mu, Sigma) and Y are in {a, b, c}.
        Categorical variable are encode with dummy variables.

        Args:
            mean (dict[array]): a dict that contains the means for each class Y in {a, b, c}
            cov (dict[array]): a dict that contains the covariance matrix for each class Y in {a, b, c}
            coefs (dict[array]): a dict that contains coefficients of the linear regression for each class Y in {a, b, c}
            pi (dict[float]): a dict that contains the probabilities P(Y=y) for each class Y in {a, b, c}
            n (int): the size of training and test set
            C (list[list]): list of variables group together
            cat_index (list): index of the categorical variables
        """

        self.n = n
        self.d = len(mean['a'])
        self.index = list(np.arange(self.d))
        self.cat_index = cat_index
        self.index_cat = {cat_index[0]: 'a', cat_index[1]: 'b'}
        self.coefs = coefs

        self.cov = cov
        self.mean = {key: np.array(mean[key]) for key in mean.keys()}
        self.C = C
        self.pi = pi

        self.dummy = {'a': np.array([1., 0.]), 'b': np.array([0., 1.]), 'c': np.array([0., 0.])}
        self.revers_dummy = {str([1., 0.]): 'a', str([0., 1.]): 'b', str([0., 0.]): 'c'}

        self.gaus = {key: st.multivariate_normal(self.mean[key], self.cov[key]) for key in self.mean.keys()}

        # Data init
        self.data, self.y_train = self.gen_data(n)
        self.data_test, self.y_test = self.gen_data(n)

    def gen_data_y(self, y='a', n=10000):
        """
        Generate data D={(X, Y, Z), i={1,...n}} of class Y
        Args:
            y (string): label of the class
            n (int): size of the sample

        Returns:
            array: X
            array: Z
        """
        data_gen = st.multivariate_normal(self.mean[y], self.cov[y])
        data = data_gen.rvs(n)
        y_train = linear_regression(self.coefs[y], data)
        return data, y_train

    def gen_data(self, n):
        """
        Generate data D={(X, Y, Z), i={1,...n}}
        Args:
            n (int): size of data

        Returns:
            array: X
            array: Z
        """

        mixture_idx = np.random.choice(list(self.pi.keys()), size=n, replace=True, p=list(self.pi.values()))
        data_a = self.gaus['a'].rvs(np.sum(mixture_idx == 'a'))
        y_a = linear_regression(self.coefs['a'], data_a)
        data_a = np.concatenate([data_a, np.tile([1., 0.], (np.sum(mixture_idx == 'a'), 1))], axis=1)

        data_b = self.gaus['b'].rvs(np.sum(mixture_idx == 'b'))
        y_b = linear_regression(self.coefs['b'], data_b)
        data_b = np.concatenate([data_b, np.tile([0., 1.], (np.sum(mixture_idx == 'b'), 1))], axis=1)

        data_c = self.gaus['c'].rvs(np.sum(mixture_idx == 'c'))
        y_c = linear_regression(self.coefs['c'], data_c)
        data_c = np.concatenate([data_c, np.tile([0., 0.], (np.sum(mixture_idx == 'c'), 1))], axis=1)

        data, y_train = np.concatenate([data_a, data_b, data_c], axis=0), np.concatenate([y_a, y_b, y_c], axis=0)
        return data, y_train

    def linear_gmm(self, x):
        """
        Compute f([X,Y]) for an observation x = [X, Y] where f is the linear model

        Args:
            x (array): [X, Y] observation where variable Y are encode with dummy variables

        Returns:
            float: f([X,Y])
        """
        a_cond = (x[self.cat_index[0]] == 1) * (x[self.cat_index[1]] == 0)
        b_cond = (x[self.cat_index[0]] == 0) * (x[self.cat_index[1]] == 1)
        c_cond = (x[self.cat_index[0]] == 0) * (x[self.cat_index[1]] == 0)

        return linear_regression_0(self.coefs['a'], x[:self.d]) * a_cond \
               + linear_regression_0(self.coefs['b'], x[:self.d]) * b_cond \
               + linear_regression_0(self.coefs['c'], x[:self.d]) * c_cond

    def linear_gmm_batch(self, x):
        """
        Compute f([X,Y]) for a batch of observation x = [X, Y] where f is the linear model

        Args:
            x (array): batch àf observations where variable Y are encode with dummy variables

        Returns:
            float: f([X,Y])
        """
        a_cond = (x[:, self.cat_index[0]] == 1) * (x[:, self.cat_index[1]] == 0)
        b_cond = (x[:, self.cat_index[0]] == 0) * (x[:, self.cat_index[1]] == 1)
        c_cond = (x[:, self.cat_index[0]] == 0) * (x[:, self.cat_index[1]] == 0)

        return linear_regression(self.coefs['a'], x[:, :self.d]) * \
               a_cond + linear_regression(self.coefs['b'], x[:, :self.d]) * b_cond \
               + linear_regression(self.coefs['c'], x[:, :self.d]) * c_cond

    def density_x_y_set(self, x, y, S):
        """
        Compute marginal density given S evaluate on x

        Args:
            x (array): observation
            y (string): label of the class Y in {a, b, c}
            S (list): index of the continuous variable on which we condition

        Returns:
            float: marginal density given S evaluate on x
        """

        # S doesn't take str (y)
        x_in = x.copy()[:self.d]
        if S != []:  # if condition on numeric variable S
            x_in = x_in[S]
            mean = self.mean[y][S]
            cov = self.cov[y][S][:, S]
        else:  # this loops is useless ?
            mean = self.mean[y]
            cov = self.cov[y]

        gaus_S = st.multivariate_normal(mean, cov)
        cond_dist_x = gaus_S.pdf(x_in)

        return cond_dist_x

    def mar_exp_linear_y_set(self, x, y, S):
        """
        Compute exact marginal expectation given S and class y of the linear model E[f(x, X_{\bar{S}})]

        Args:
            x (array): observation
            y (string): label of the class {a, b, c}
            S (list): index of the continuous variable on which on condition

        Returns:
            float: marginal expectation given S and class y of the linear model E[f(x, X_{\bar{S}})]
        """

        # S doesn't take str (y)
        x_in = x.copy()[:self.d]

        S_bar = [i for i in self.index if i not in S]

        cond_dist_param = marMVN(self.mean[y], self.cov[y], S_bar, S, x_in[S])

        mu_bar = cond_dist_param[0]
        if len(S) != self.d:
            x_in[S_bar] = mu_bar

        out = np.sum(self.coefs[y] * x_in)
        return out

    def cond_exp_linear_y_set(self, x, y, S):
        """
        Compute exact conditional expectation given S and class y of the linear model E[f(x)| X_S=x_S, Y=y]

        Args:
            x (array): observation
            y (string): label of the class {a, b, c}
            S (list): index of the continuous variable on which on condition

        Returns:
            float: exact conditional expectation given S and class y of the linear model E[f(x)| X_S=x_S, Y=y]
        """

        # S doesn't take str (y)
        x_in = x.copy()[:self.d]

        S_bar = [i for i in self.index if i not in S]

        cond_dist_param = condMVN(self.mean[y], self.cov[y], S_bar, S, x_in[S])

        mu_bar = cond_dist_param[0]
        if len(S) != self.d:
            x_in[S_bar] = mu_bar

        out = np.sum(self.coefs[y] * x_in)
        return out

    def cond_exp_linear_gmm(self, x, S):
        """
        Compute exact conditional expectation given S of the linear model E[f(x)| X_S=x_S]. Here, S can contains the
        index of the categorical variable.

        Args:
            x (array): observation
            S (list): index of the variable (continuous and categorical) on which on condition

        Returns:
            float: conditional expectation given S and class y of the linear model E[f(x)| X_S=x_S]
        """


        x_in = x.copy()
        if len(S) != len(x_in):  # if not condition on all variable

            cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
            S = [cu for cu in S if cu not in self.cat_index]  # get numerical index

            cond_exp = 0
            if cat_va == []:  # if not condition on categorical variable
                p_x = 0
                p_x_y = 0
                if S != []:  # if condition on numeric variable
                    for key in self.mean.keys():
                        p_x += self.pi[key] * self.density_x_y_set(x_in, key, S)
                        p_x_y += self.pi[key] * self.density_x_y_set(x_in, key, S) * self.cond_exp_linear_y_set(x_in,
                                                                                                                key, S)
                    cond_exp = p_x_y / p_x
                else:
                    for key in self.mean.keys():
                        cond_exp += self.pi[key] * self.cond_exp_linear_y_set(x_in, key, S)
            else:  # if condition on categorical variable
                if len(cat_va) > 1:  # ==> condition on a group of dummy variable [y1, y2] ==> on cat variable
                    cond_exp = self.cond_exp_linear_y_set(x, self.revers_dummy[str(list(x[cat_va]))], S)

                else:  # ==> conditioning on 1 dummy
                    if x_in[cat_va] == 1:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        cond_exp = self.cond_exp_linear_y_set(x, self.index_cat[cat_va[0]], S)

                    else:  # else y1 or y2 =0 ==> uses the complementary
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]
                        p_total = 0

                        if S != []:
                            for key in cat_sample:
                                p_total += self.pi[key] * self.density_x_y_set(x_in, key, S)
                                cond_exp += self.pi[key] * self.density_x_y_set(x_in, key,
                                                                                S) * self.cond_exp_linear_y_set(x,
                                                                                                                key,
                                                                                                                S)
                            cond_exp = cond_exp / p_total
                        else:
                            for key in cat_sample:
                                p_total += self.pi[key]
                                cond_exp += self.pi[key] * self.cond_exp_linear_y_set(x_in, key, S)
                            cond_exp = cond_exp / p_total
            return cond_exp

        else:
            return self.linear_gmm(x_in)

    def cond_exp_monte_carlo_linear(self, x, S, N):
        """
        Compute conditional expectation given S of the linear model with monte carlo estimator E[f(x)| X_S=x_S]. Here, S can contains the
        index of the categorical variable.

        Args:
            x (array): observation
            S (list): index of the variable (continuous and categorical) on which on condition
            N (int): size of the sampling of monte carlo estimator

        Returns:
            float: Conditional expectation given S of the linear model with monte carlo estimator E[f(x)| X_S=x_S]
        """

        x_in = x.copy()
        cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
        S_cu = [cu for cu in S if cu not in self.cat_index]  # get numerical index

        def get_given_data(idx):
            val = np.array([x[idx]])
            val = np.tile(val, N)
            return val

        def gen_data_by_cat(cat_va, prob, N):
            # to do: make it real function of S, c_index, etc

            mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)

            rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key],
                                 self.cov[key], S_bar, S, x_in[S]) for key in cat_va
                  if np.sum(mixture_idx == key) != 0}

            rg = np.concatenate([np.concatenate([rg[key], np.tile(self.dummy[key],
                                                                  (np.sum(mixture_idx == key), 1))], axis=1) for key in
                                 rg.keys()])

            rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

            for val_id in S:
                rg_data[str(val_id)] = get_given_data(val_id)

            rg_data = rg_data[sorted(rg_data.columns)]
            return rg_data

        if len(S_cu) != self.d:
            cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
            S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
            S_bar = [i for i in self.index if i not in S]

            if cat_va == []:
                if S != []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x = np.sum(prob)
                    prob = [p / p_x for p in prob]
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                else:
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=list(self.pi.values()))

                rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                      for key in self.pi.keys() if np.sum(mixture_idx == key) != 0}

                rg = np.concatenate([np.concatenate(
                    [rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))], axis=1)
                    for key in rg.keys()])

                rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]

            else:
                if len(cat_va) > 1:  # this two condition ==> grouping categorical variable
                    p = [1]
                    rg_data = gen_data_by_cat([self.revers_dummy[str(list(x[cat_va]))]], p, N)

                else:  # ==> number of categorical variable is 1

                    if x_in[cat_va] == 1:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        p = [1]
                        rg_data = gen_data_by_cat([self.index_cat[cat_va[0]]], p, N)

                    else:  # if y1 or y2 =0 ==> uses the complementary
                        # il faut faire le cas ou S == [] et le cas contraire !
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]
                        if S != []:
                            prob = [self.pi[key] * self.density_x_y_set(x, key, S) for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)
                        else:
                            prob = [self.pi[key] for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)

            y_pred = self.linear_gmm_batch(np.array(rg_data.values, dtype=np.float32))

        else:
            if len(S) == len(x_in):
                y_pred = self.linear_gmm_batch(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
            else:
                S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
                if cat_va == []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x = np.sum(prob)
                    prob = [p / p_x for p in prob]

                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                    data_a = np.tile(np.concatenate([x_in[S], np.array([1., 0.])], axis=0),
                                     (np.sum(mixture_idx == 'a'), 1))
                    data_b = np.tile(np.concatenate([x_in[S], np.array([0., 1.])], axis=0),
                                     (np.sum(mixture_idx == 'b'), 1))
                    data_c = np.tile(np.concatenate([x_in[S], np.array([0., 0.])], axis=0),
                                     (np.sum(mixture_idx == 'c'), 1))
                    rg_data = np.concatenate([data_a, data_b, data_c], axis=0)

                    y_pred = self.linear_gmm_batch(np.array(rg_data, dtype=np.float32))
                else:
                    if len(cat_va) == 1 and x_in[cat_va] == 0:  # if dummy, y1 or y2 = 0 ==> condition on complementary
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]

                        prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in cat_sample]
                        p_total = np.sum(prob)
                        prob = [p / p_total for p in prob]

                        mixture_idx = np.random.choice(cat_sample, size=N, replace=True, p=prob)

                        data_1 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[0]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[0]), 1))
                        data_2 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[1]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[1]), 1))

                        rg_data = np.concatenate([data_1, data_2], axis=0)

                        y_pred = self.linear_gmm_batch(np.array(rg_data, dtype=np.float32))
                    else:  # on peut l'enlever grace à la première condition
                        y_pred = self.linear_gmm_batch(np.expand_dims(np.array(x_in, dtype=np.float32), axis=0))

        return np.mean(y_pred)

    def cond_exp_monte_carlo_gen(self, model, x, S, N):
        """
        Compute conditional expectation given S of the any "model" with monte carlo estimator E[f(x) | X_S = x_S]. Here, S can contains the
        index of the categorical variable.
        Args:
            model (Sklearn Ensemble): model
            x (array): observation
            S (list): index of the variable (continuous and categorical) on which on condition
            N (int): size of the monte carlo sampling

        Returns:
            float: Conditional expectation given S of any "model" with monte carlo estimator E[f(x) | X_S = x_S]
        """

        x_in = x.copy()
        cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
        S_cu = [cu for cu in S if cu not in self.cat_index]  # get numerical index

        def get_given_data(idx):
            val = np.array([x[idx]])
            val = np.tile(val, N)
            return val

        def gen_data_by_cat(cat_va, prob, N):

            mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)

            rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                  for key in cat_va if np.sum(mixture_idx == key) != 0}
            rg = np.concatenate([np.concatenate(
                [rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))], axis=1)
                for key in rg.keys()], axis=0)

            rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

            for val_id in S:
                rg_data[str(val_id)] = get_given_data(val_id)

            rg_data = rg_data[sorted(rg_data.columns)]
            return rg_data

        if len(S_cu) != self.d:
            cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
            S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
            S_bar = [i for i in self.index if i not in S]

            if cat_va == []:
                if S != []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                else:
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=list(self.pi.values()))

                rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                      for key in self.pi.keys() if np.sum(mixture_idx == key) != 0}

                # print('rg', [rg[key].shape for key in rg.keys()])
                # print('dummy', [np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1)).shape for key in rg.keys()])

                rg = np.concatenate(
                    [np.concatenate([rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))],
                                    axis=1) for key in rg.keys()], axis=0)

                rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]

            else:
                if len(cat_va) > 1:  # this two condition ==> grouping categorical variable
                    p = [1]
                    rg_data = gen_data_by_cat([self.revers_dummy[str(list(x[cat_va]))]], p, N)

                else:  # ==> number of categorical variable is 1

                    if x_in[cat_va] == 1:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        p = [1]
                        rg_data = gen_data_by_cat([self.index_cat[cat_va[0]]], p, N)


                    else:  # if y1 or y2 =0 ==> uses the complementary
                        # il faut faire le cas ou S == [] et le cas contraire !
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]
                        if S != []:
                            prob = [self.pi[key] * self.density_x_y_set(x, key, S) for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)
                        else:
                            prob = [self.pi[key] for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)

            y_pred = model.predict(np.array(rg_data.values, dtype=np.float32))

        else:
            if len(S) == len(x_in):
                y_pred = model.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))
            else:
                S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
                if cat_va == []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]

                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                    data_a = np.tile(np.concatenate([x_in[S], np.array([1, 0])], axis=0),
                                     (np.sum(mixture_idx == 'a'), 1))
                    data_b = np.tile(np.concatenate([x_in[S], np.array([0, 1])], axis=0),
                                     (np.sum(mixture_idx == 'b'), 1))
                    data_c = np.tile(np.concatenate([x_in[S], np.array([0, 0])], axis=0),
                                     (np.sum(mixture_idx == 'c'), 1))
                    rg_data = np.concatenate([data_a, data_b, data_c], axis=0)

                    y_pred = model.predict(np.array(rg_data, dtype=np.float32))
                else:
                    if len(cat_va) == 1 and x_in[cat_va] == 0:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]

                        prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in cat_sample]
                        p_total = np.sum(prob)
                        prob = [p / p_total for p in prob]

                        mixture_idx = np.random.choice(cat_sample, size=N, replace=True, p=prob)

                        data_1 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[0]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[0]), 1))
                        data_2 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[1]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[1]), 1))

                        rg_data = np.concatenate([data_1, data_2], axis=0)

                        y_pred = model.predict(np.array(rg_data, dtype=np.float32))
                    else:
                        y_pred = model.predict(np.expand_dims(np.array(x_in, dtype=np.float32), axis=0))

        return np.mean(y_pred)

    def cond_dist_monte_carlo_gen(self, model, x, fx, S, N):
        """
        Compute conditional expectation of Euclidean distance of a tree_regressor given S and fx:
        E[d(model(x), fx) | X_S = x_S] with monte carlo

        Args:
            model (Sklearn ensemble)
            x (int): observation
            fx (float): a scalar on which the distance will be compute
            S (list): index of the variable on which we are conditioning
            N (int): size of observations sample when using Monte carlo estimators.

        Returns:
            float: E[d(model(x), fx) | X_S = x_S]
        """
        x_in = x.copy()
        cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
        S_cu = [cu for cu in S if cu not in self.cat_index]  # get numerical index

        def get_given_data(idx):
            val = np.array([x[idx]])
            val = np.tile(val, N)
            return val

        def gen_data_by_cat(cat_va, prob, N):

            mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)

            rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                  for key in cat_va if np.sum(mixture_idx == key) != 0}
            rg = np.concatenate([np.concatenate(
                [rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))], axis=1)
                for key in rg.keys()], axis=0)

            rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

            for val_id in S:
                rg_data[str(val_id)] = get_given_data(val_id)

            rg_data = rg_data[sorted(rg_data.columns)]
            return rg_data

        if len(S_cu) != self.d:
            cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
            S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
            S_bar = [i for i in self.index if i not in S]

            if cat_va == []:
                if S != []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                else:
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=list(self.pi.values()))

                rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                      for key in self.pi.keys() if np.sum(mixture_idx == key) != 0}

                # print('rg', [rg[key].shape for key in rg.keys()])
                # print('dummy', [np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1)).shape for key in rg.keys()])

                rg = np.concatenate(
                    [np.concatenate([rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))],
                                    axis=1) for key in rg.keys()], axis=0)

                rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]

            else:
                if len(cat_va) > 1:  # this two condition ==> grouping categorical variable
                    p = [1]
                    rg_data = gen_data_by_cat([self.revers_dummy[str(list(x[cat_va]))]], p, N)

                else:  # ==> number of categorical variable is 1

                    if x_in[cat_va] == 1:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        p = [1]
                        rg_data = gen_data_by_cat([self.index_cat[cat_va[0]]], p, N)


                    else:  # if y1 or y2 =0 ==> uses the complementary
                        # il faut faire le cas ou S == [] et le cas contraire !
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]
                        if S != []:
                            prob = [self.pi[key] * self.density_x_y_set(x, key, S) for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)
                        else:
                            prob = [self.pi[key] for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)

            y_pred = (model.predict(np.array(rg_data.values, dtype=np.float32)) - fx) ** 2

        else:
            if len(S) == len(x_in):
                y_pred = (model.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0)) - fx) ** 2
            else:
                S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
                if cat_va == []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]

                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                    data_a = np.tile(np.concatenate([x_in[S], np.array([1, 0])], axis=0),
                                     (np.sum(mixture_idx == 'a'), 1))
                    data_b = np.tile(np.concatenate([x_in[S], np.array([0, 1])], axis=0),
                                     (np.sum(mixture_idx == 'b'), 1))
                    data_c = np.tile(np.concatenate([x_in[S], np.array([0, 0])], axis=0),
                                     (np.sum(mixture_idx == 'c'), 1))
                    rg_data = np.concatenate([data_a, data_b, data_c], axis=0)

                    y_pred = (model.predict(np.array(rg_data, dtype=np.float32)) - fx) ** 2
                else:
                    if len(cat_va) == 1 and x_in[cat_va] == 0:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]

                        prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in cat_sample]
                        p_total = np.sum(prob)
                        prob = [p / p_total for p in prob]

                        mixture_idx = np.random.choice(cat_sample, size=N, replace=True, p=prob)

                        data_1 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[0]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[0]), 1))
                        data_2 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[1]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[1]), 1))

                        rg_data = np.concatenate([data_1, data_2], axis=0)

                        y_pred = (model.predict(np.array(rg_data, dtype=np.float32)) - fx) ** 2
                    else:
                        y_pred = (model.predict(np.expand_dims(np.array(x_in, dtype=np.float32), axis=0)) - fx) ** 2

        return np.mean(y_pred)

    def cond_data_monte_carlo_gen(self, model, x, S, N):
        """
        Sample data that follow condition law given S
        Args:
            x (array): observation
            S (list): index of the variable (continuous and categorical) on which on condition
            N (int): size of the sampling of monte carlo estimator

        Returns:
            dataFrame: data
        """

        x_in = x.copy()
        cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
        S_cu = [cu for cu in S if cu not in self.cat_index]  # get numerical index

        def get_given_data(idx):
            val = np.array([x[idx]])
            val = np.tile(val, N)
            return val

        def gen_data_by_cat(cat_va, prob, N):

            mixture_idx = np.random.choice(cat_va, size=N, replace=True, p=prob)

            rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                  for key in cat_va if np.sum(mixture_idx == key) != 0}
            rg = np.concatenate([np.concatenate(
                [rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))], axis=1)
                for key in rg.keys()], axis=0)

            rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

            for val_id in S:
                rg_data[str(val_id)] = get_given_data(val_id)

            rg_data = rg_data[sorted(rg_data.columns)]
            return rg_data

        if len(S_cu) != self.d:
            cat_va = [ca for ca in S if ca in self.cat_index]  # get categorical index
            S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
            S_bar = [i for i in self.index if i not in S]

            if cat_va == []:
                if S != []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                else:
                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=list(self.pi.values()))

                rg = {key: sampleMVN(np.sum(mixture_idx == key), self.mean[key], self.cov[key], S_bar, S, x_in[S])
                      for key in self.pi.keys() if np.sum(mixture_idx == key) != 0}

                # print('rg', [rg[key].shape for key in rg.keys()])
                # print('dummy', [np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1)).shape for key in rg.keys()])

                rg = np.concatenate(
                    [np.concatenate([rg[key], np.tile(self.dummy[key], (np.sum(mixture_idx == key), 1))],
                                    axis=1) for key in rg.keys()], axis=0)

                rg_data = pd.DataFrame(rg, columns=[str(s) for s in S_bar] + [str(ca) for ca in self.cat_index])

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]

            else:
                if len(cat_va) > 1:  # this two condition ==> grouping categorical variable
                    p = [1]
                    rg_data = gen_data_by_cat([self.revers_dummy[str(list(x[cat_va]))]], p, N)

                else:  # ==> number of categorical variable is 1

                    if x_in[cat_va] == 1:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        p = [1]
                        rg_data = gen_data_by_cat([self.index_cat[cat_va[0]]], p, N)


                    else:  # if y1 or y2 =0 ==> uses the complementary
                        # il faut faire le cas ou S == [] et le cas contraire !
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]
                        if S != []:
                            prob = [self.pi[key] * self.density_x_y_set(x, key, S) for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)
                        else:
                            prob = [self.pi[key] for key in cat_sample]
                            p_total = np.sum(prob)
                            prob = [p / p_total for p in prob]
                            rg_data = gen_data_by_cat(cat_sample, prob, N)

        else:
            if len(S) == len(x_in):
                rg_data = np.expand_dims(np.array(x, dtype=np.float32), axis=0)
                rg_data = pd.DataFrame(rg_data,
                                       columns=[str(s) for s in self.index] + [str(ca) for ca in self.cat_index])
            else:
                S = [cu for cu in S if cu not in self.cat_index]  # get numerical index
                if cat_va == []:
                    prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in self.pi.keys()]
                    p_x_y = np.sum(prob)
                    prob = [p / p_x_y for p in prob]

                    mixture_idx = np.random.choice(list(self.pi.keys()), size=N, replace=True, p=prob)
                    data_a = np.tile(np.concatenate([x_in[S], np.array([1, 0])], axis=0),
                                     (np.sum(mixture_idx == 'a'), 1))
                    data_b = np.tile(np.concatenate([x_in[S], np.array([0, 1])], axis=0),
                                     (np.sum(mixture_idx == 'b'), 1))
                    data_c = np.tile(np.concatenate([x_in[S], np.array([0, 0])], axis=0),
                                     (np.sum(mixture_idx == 'c'), 1))
                    rg_data = np.concatenate([data_a, data_b, data_c], axis=0)
                    rg_data = pd.DataFrame(rg_data,
                                           columns=[str(s) for s in self.index] + [str(ca) for ca in self.cat_index])

                else:
                    if len(cat_va) == 1 and x_in[cat_va] == 0:  # if dummy, y1 or y2 = 1 ==> condition on 'a' or 'b'
                        cat_sample = [key for key in self.pi.keys() if key != self.index_cat[cat_va[0]]]

                        prob = [self.pi[key] * self.density_x_y_set(x_in, key, S) for key in cat_sample]
                        p_total = np.sum(prob)
                        prob = [p / p_total for p in prob]

                        mixture_idx = np.random.choice(cat_sample, size=N, replace=True, p=prob)

                        data_1 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[0]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[0]), 1))
                        data_2 = np.tile(np.concatenate([x_in[S], np.array(self.dummy[cat_sample[1]])], axis=0),
                                         (np.sum(mixture_idx == cat_sample[1]), 1))

                        rg_data = np.concatenate([data_1, data_2], axis=0)
                        rg_data = pd.DataFrame(rg_data, columns=[str(s) for s in self.index] + [str(ca) for ca in
                                                                                                self.cat_index])
        return rg_data

    def mar_exp_linear_mc(self, x, S, N):
        """
        Compute marginal expectation given S of the linear model with monte carlo estimator E[f(x_S, x_{\bar{S})]
        Args:
            x (array): observation
            S (list): index of the variable (continuous and categorical) on which on condition
            N (int): size of the sampling of monte carlo estimator

        Returns:
            float: Compute marginal expectation given S of the linear model with monte carlo estimator E[f(x_S, x_{\bar{S})]
        """
        def get_given_data(idx):
            val = np.array([x[idx]])
            val = np.tile(val, N)
            return val

        if len(S) != len(x):

            rg, _ = self.gen_data(N)
            rg_data = pd.DataFrame(rg, columns=[str(n) for n in self.index] + [str(ca) for ca in self.cat_index])

            for val_id in S:
                rg_data[str(val_id)] = get_given_data(val_id)

            rg_data = rg_data[sorted(rg_data.columns)]
            y_pred = self.linear_gmm_batch(np.array(rg_data.values, dtype=np.float32))

        else:
            y_pred = self.linear_gmm_batch(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

        return np.mean(y_pred)

    def cond_exp_tree(self, x, tree, S, algo="shap", data=None, N=50000):
        """
        Compute conditional expectation of a tree_regressor given S: E[tree(x) | X_S = x_S]

        Args:
            x (int): observation
            tree (DecisionTreeRegressor.tree_): model
            S (list): index of the variable on which we are conditioning
            algo (string): name of the estimator. there are 5 estimators: 'pluging', 'shap', 'monte_carlo',
                            'monte_carlo_interventional', 'plugin_interventional'
            data (array): data use to compute the expectation
            N (int): size of observations sample when using Monte carlo estimators.

        Returns:
            float: E[tree(x) | X_S = x_S]
        """
        if type(data) != type(self.data):
            data = self.data

        a = tree.children_left.copy()
        b = tree.children_right.copy()
        f = tree.feature.copy()
        t = tree.threshold.copy()
        r_w = tree.weighted_n_node_samples.copy()
        r = tree.n_node_samples.copy()
        v = np.array([tree.value[i][0][0] for i in range(tree.node_count)])

        def G(i, w):

            if f[i] == -2:
                return (float(w * v[i]))
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

        # Algorithms implemented !

        if algo == 'shap':
            return G(0, 1)

        elif algo == 'plugin_v1':
            compatible_leaves = np.zeros(tree.node_count)
            # partition_leaves = [np.array([-np.inf, np.inf]) for i in range(tree.node_count)]
            explore(0, compatible_leaves)

            compatible_leaves = (compatible_leaves == 1)
            D = np.sum(r_w[compatible_leaves])
            N = np.sum(r_w[compatible_leaves] * v[compatible_leaves])

            return N / D

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

        elif algo == 'monte_carlo':
            return self.cond_exp_monte_carlo_gen(tree, x, S, N)

        elif algo == 'monte_carlo_interventional':

            def get_given_data(idx):
                val = np.array([x[idx]])
                val = np.tile(val, N)
                return val

            if len(S) != len(x):

                rg, _ = self.gen_data(N)
                rg_data = pd.DataFrame(rg, columns=[str(n) for n in self.index] + [str(ca) for ca in self.cat_index])

                for val_id in S:
                    rg_data[str(val_id)] = get_given_data(val_id)

                rg_data = rg_data[sorted(rg_data.columns)]
                y_pred = tree.predict(np.array(rg_data.values, dtype=np.float32))

            else:
                y_pred = tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0))

            return np.mean(y_pred)

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


        else:
            raise NotImplementedError(
                "Algorithms available are: monte_carlo, monte_carlo_interventional, shap, plugin, plugin_interventional")

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

    def linear_gmm_shap(self, x, monte_carlo=False, N=50000):
        """
        Compute Observational Shapley values of the linear model
        Args:
            x (array): observation
            monte_carlo (Bool): If True use monte carlo estimator otherwise exact computation
            N (int): size of the monte carlo sampling

        Returns:
            array: Observational Shapley values
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
                if monte_carlo:
                    phi[chain_l(p[i])] += \
                        self.cond_exp_monte_carlo_linear(x, S=chain_l(p[:i + 1]), N=N) \
                        - self.cond_exp_monte_carlo_linear(x, S=chain_l(p[:i]), N=N)
                else:
                    phi[chain_l(p[i])] += \
                        self.cond_exp_linear_gmm(x, S=chain_l(p[:i + 1])) \
                        - self.cond_exp_linear_gmm(x, S=chain_l(p[:i]))
        return phi / math.factorial(m)

    def linear_gmm_mar_shap(self, x, monte_carlo=False, N=50000):
        """
        Compute marginal Shapley values of the linear model
        Args:
            x (array): observation
            monte_carlo (Bool): If True use monte carlo estimator otherwise exact computation
            N (int): Size of the monte carlo sampling

        Returns:
            array: marginal shapley values
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
                    self.mar_exp_linear_mc(x, S=chain_l(p[:i + 1]), N=N) \
                    - self.mar_exp_linear_mc(x, S=chain_l(p[:i]), N=N)
        return phi / math.factorial(m)

    def tree_shap(self, tree, x, algo, data=None, N=50000):
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

        if tree is None:
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
                phi[chain_l(p[i])] += self.cond_exp_tree(x, tree_model, chain_l(p[:i + 1]), algo, data,
                                                         N=N) - self.cond_exp_tree(
                    x, tree_model, chain_l(p[:i]), algo, data, N=N)
        return phi / math.factorial(m)

    def cond_dist_tree(self, x, fx, tree, S, algo="shap", N=50000, data=None):
        """
        Compute conditional expectation of Euclidean distance of a tree_regressor given S and fx:
        E[d(tree(x), fx) | X_S = x_S]

        Args:
            x (int): observation
            fx (float): a scalar on which the distance will be compute
            tree (DecisionTreeRegressor.tree_): model
            S (list): index of the variable on which we are conditioning
            algo (string): name of the estimator. there are 5 estimators: 'pluging', 'shap', 'monte_carlo',
            data (array): data use to compute the expectation
            N (int): size of observations sample when using Monte carlo estimators.

        Returns:
            float: E[d(tree(x), fx) | X_S = x_S]
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

        def explore_partition(i, tab, partition_leaves, partition_global, prob_global, S, S_bar, data, intv=False):
            # a = tree.children_left.copy()
            # b = tree.children_right.copy()
            # f = tree.feature.copy()
            # t = tree.threshold.copy()
            # r_w = tree.weighted_n_node_samples.copy()
            # r = tree.n_node_samples.copy()
            # v = np.array([tree.value[i][0][0] for i in range(tree.node_count)])
            if f[i] == -2:
                #         tab[i] = 1
                compatible_leaves.append(i)
                partition_global[i] = partition_leaves
                partition_leaves = np.squeeze(np.array(partition_leaves))
                # partition_leaves = np.reshape(np.array(partition_leaves), (self.d + 2, 2)).T
                # partition_leaves = pd.DataFrame(partition_leaves)[S]
                # low = partition_leaves[:, 0]
                # up = partition_leaves[:, 1]

                # pool = mp.Pool(mp.cpu_count())
                # section_x = np.array(pool.starmap_async(nb_row, [(row, up, low, S) for row in data], chunksize=500))
                # pool.close()
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
                y_pred = (tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0)) - fx) ** 2
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

            p = p / np.sum(p)
            value = (v[compatible_leaves] - fx) ** 2
            dist = np.sum(p * value)
            return dist

        elif algo == 'plugin2':
            if S == []:
                compatible_leaves = np.zeros(tree.node_count)
                explore(0, compatible_leaves)
                compatible_leaves = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
                p = r_w[compatible_leaves] / np.sum(r_w[compatible_leaves])

            elif len(S) == len(x):
                y_pred = (tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0)) - fx) ** 2
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

            value = v[compatible_leaves]

            p = p / np.sum(p)
            dist = np.sum(p * value ** 2) + fx ** 2 - 2 * fx * np.sum(p * value)
            # dist = np.sum(p*value)
            return dist

        elif algo == 'monte_carlo':
            return self.cond_dist_monte_carlo_gen(tree, x, fx, S, N)

    def cond_sdp_tree(self, x, fx, t_x, tree, S, algo="shap", N=50000, data=None):
        """
        Compute SDP_S(x, tree_regressor) of variables in S

        Args:
            x (array): observation
            fx (float): tree(x)
            t_x (float): threshold of the regressor
            tree (DecisionTreeRegressor): model
            S (list): index of variables on which we want to compute the SDP
            algo (string): name of the estimators pluging or monte_carlo
            data (array): data used to compute the SDP
            N (int): Size of monte carlo sample

        Returns:
            float: SDP_S(x, forest_regressor)
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

        y_pred = tree.predict(np.array(data, dtype=np.float32))
        dist = (y_pred - fx) ** 2

        up_tx = np.array(dist > t_x).reshape(-1)
        down_tx = np.array(dist <= t_x).reshape(-1)

        # print(down_tx.shape)
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

        def explore_partition(i, tab, partition_leaves, partition_global, prob_global, S, S_bar, data, intv=False):
            # a = tree.children_left.copy()
            # b = tree.children_right.copy()
            # f = tree.feature.copy()
            # t = tree.threshold.copy()
            # r_w = tree.weighted_n_node_samples.copy()
            # r = tree.n_node_samples.copy()
            # v = np.array([tree.value[i][0][0] for i in range(tree.node_count)])
            if f[i] == -2:
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

                prob_all = np.sum(section_x * section_x_bar) / np.sum(section_x)
                prob_up = np.sum(section_up * section_x_bar) / np.sum(section_up)
                prob_down = np.sum(section_down * section_x_bar) / np.sum(section_down)

                if np.isnan(prob_up):
                    prob_up = 0
                if np.isnan(prob_down):
                    prob_down = 0
                if np.isnan(prob_all):
                    prob_all = 0

                # print(prob_all, prob_up, prob_down, 'prob value')

                prob_global[i] = [prob_all, prob_up, prob_down]

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

        if algo == 'plugin':
            if S == []:
                compatible_leaves = np.zeros(tree.node_count)
                explore(0, compatible_leaves)
                compatible_leaves = [i for i in range(tree.node_count) if compatible_leaves[i] == 1]
                p = r_w[compatible_leaves] / np.sum(r_w[compatible_leaves])
                return np.sum(p * v[compatible_leaves])

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

                p_all = np.array([prob_global[key][0] for key in compatible_leaves])
                # p_all = p_all/np.sum(p_all)

                p_up = np.array([prob_global[key][1] for key in compatible_leaves])
                # if np.sum(p_up) != 0:
                #     p_up = p_up/np.sum(p_up)

                p_down = np.array([prob_global[key][2] for key in compatible_leaves])
                # if np.sum(p_down) != 0:
                #     p_down = p_down/np.sum(p_down)
                # print(p_all, p_up, p_down, 'p value')

                value = v[compatible_leaves]
                mean_up = np.sum(p_up * value ** 2) + fx ** 2 - 2 * fx * np.sum(p_up * value)
                mean_down = np.sum(p_down * value ** 2) + fx ** 2 - 2 * fx * np.sum(p_down * value)
                mean_all = np.sum(p_all * value ** 2) + fx ** 2 - 2 * fx * np.sum(p_all * value)

                return (mean_up - mean_all) / (mean_up - mean_down)

        elif algo == 'monte_carlo':

            if len(S) != len(index):
                rg_data = self.cond_data_monte_carlo_gen(tree, x, S, N)
                y_pred = tree.predict(np.array(rg_data.values, dtype=np.float32))
                y_pred_dist = (y_pred - fx) ** 2

                y_pred_up = np.mean(
                    (tree.predict(np.array(rg_data[y_pred_dist > t_x].values, dtype=np.float32)) - fx) ** 2)
                y_pred_down = np.mean(
                    (tree.predict(np.array(rg_data[y_pred_dist <= t_x].values, dtype=np.float32)) - fx) ** 2)

                if np.isnan(y_pred_up):
                    y_pred_up = 0
                if np.isnan(y_pred_down):
                    y_pred_down = 0

                sdp = (y_pred_up - np.mean(y_pred_dist)) / (y_pred_up - y_pred_down)
                if y_pred_up + np.mean(y_pred_dist) + y_pred_down == 0:
                    sdp = 1

                return sdp
            else:
                y_pred = (tree.predict(np.expand_dims(np.array(x, dtype=np.float32), axis=0)) - fx) ** 2
                return np.mean(y_pred)

        else:
            raise ValueError("This algo is not implemented. Estimators implemented are: plugin, monte_carlo")

    def fit_model(self, model):
        """ Fit a model and show the errors. """
        model.fit(self.data, self.y_train)

        error_test = mean_squared_error(model.predict(self.data_test), self.y_test)
        error_train = mean_squared_error(model.predict(self.data), self.y_train)

        print('fitted model train_error = {}'.format(error_train))
        print('fitted model test_error = {}'.format(error_test))
        return model

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
        shap_true, shap_plugin, shap_shap, shap_monte_carlo, shap_plugin = [], [], [], [], []

        if model is None:
            tree_model = DecisionTreeRegressor()
            tree_model = self.fit_model(tree_model)
        else:
            tree_model = model

        for i in tqdm(range(nb)):
            shap_true.append(np.expand_dims(self.linear_gmm_shap(data[i]), axis=0))
            shap_plugin.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='plugin'), axis=0))
            shap_shap.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='shap'), axis=0))
            shap_monte_carlo.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='monte_carlo'), axis=0))
            shap_plugin.append(np.expand_dims(self.tree_shap(tree_model, x=data[i], algo='plugin_v_2'), axis=0))

        shap_true, shap_plugin, shap_shap, shap_monte_carlo, shap_plugin = np.concatenate(shap_true,
                                                                                          axis=0), np.concatenate(
            shap_plugin, axis=0), np.concatenate(shap_shap, axis=0), np.concatenate(shap_monte_carlo,
                                                                                    axis=0), np.concatenate(shap_plugin,
                                                                                                            axis=0)

        explainer_observational = shap.TreeExplainer(tree_model, feature_perturbation='observational')
        explainer_interventional = shap.TreeExplainer(tree_model, feature_perturbation='interventional', data=self.data)

        shap_observational = explainer_observational.shap_values(data[:nb])
        shap_interventional = explainer_interventional.shap_values(data[:nb])

        shap_data = np.concatenate([shap_true, shap_plugin, shap_shap, shap_monte_carlo, shap_plugin,
                                    shap_observational, shap_interventional], axis=0)

        shap_dataFra = pd.DataFrame(data=shap_data,
                                    columns=[str(i) for i in self.index] + [str(i) for i in self.cat_index])
        shap_dataFra['label'] = nb * ['true'] + nb * ['plugin'] + nb * ['shap'] + nb * ['mc'] + \
                                nb * ['plugin_sal'] + nb * ['lund_obs'] + nb * ['lund_int']

        err_true = l1_norm(shap_true - shap_true)
        err_plugin = l1_norm(shap_plugin - shap_true)
        err_shap = l1_norm(shap_shap - shap_true)
        err_monte_carlo = l1_norm(shap_monte_carlo - shap_true)
        err_plugin_sal = l1_norm(shap_plugin - shap_true)
        err_shap_observational = l1_norm(shap_observational - shap_true)
        err_shap_interventional = l1_norm(shap_interventional - shap_true)

        err_concat = np.concatenate(
            [err_true, err_plugin, err_shap, err_monte_carlo, err_plugin_sal, err_shap_observational,
             err_shap_interventional], axis=0)

        shap_dataFra['error_l1'] = err_concat

        if plot:
            fig, ax = plt.subplots(dpi=150)
            sns.boxplot(data=shap_dataFra, x='label', y='error_l1')

        return shap_dataFra

    def cond_exp_tree_plot(self, model, data, S, nb, plot):
        """
        Compute conditional expectation of observations in data given the different estimators, and return the errors between
        the exact computation and the different estimators.

        Args:
            model (DecisionTreeRegressor):
            data (array): data used to compute the Shapley values
            S (list): index of the variable on which we condition
            nb (int): number of observations used
            plot (Bool): If True, Show a boxplot of the errors

        Returns:
            (dataFrame): data that contains the SV of each observation given the different estimators and their erros.
        """

        if model is None:
            tree_model = DecisionTreeRegressor()
            tree_model = self.fit_model(tree_model)
        else:
            tree_model = model

        e_monte_carlo, e_exact, e_shap, e_plugin, e_plugin_sal = [], [], [], [], []

        for i in tqdm(range(data[:nb].shape[0])):
            # e_plugin.append(self.cond_exp_tree(x=data[i], tree=tree_model.tree_, S=S, algo="plugin"))
            e_shap.append(self.cond_exp_tree(x=data[i], tree=tree_model.tree_, S=S, algo="shap"))
            e_monte_carlo.append(self.cond_exp_tree(x=data[i], tree=tree_model.tree_, S=S, algo="monte_carlo"))
            # e_exact.append(self.cond_exp_linear_gmm(x=data[i], S=S))
            e_plugin_sal.append(self.cond_exp_tree(x=data[i], tree=tree_model.tree_, S=S, algo="plugin"))

        e_monte_carlo, e_exact, e_shap, e_plugin, e_plugin_sal = \
            np.array(e_monte_carlo), np.array(e_exact), np.array(e_shap), np.array(e_plugin), np.array(e_plugin_sal)

        # error_plugin = np.abs(e_plugin-e_exact)
        error_shap = np.abs(e_shap - e_monte_carlo)
        # error_monte_carlo = np.abs(e_monte_carlo-e_exact)
        error_plugin_sal = np.abs(e_plugin_sal - e_monte_carlo)

        errors = np.concatenate([error_shap, error_plugin_sal], axis=0)
        data_errors = pd.DataFrame(errors, columns=['errors'])
        data_errors['label'] = nb * ['lundb_obs'] + nb * ['ours']

        if plot:
            fig, ax = plt.subplots(dpi=200)
            sns.boxplot(data=data_errors, x='label', y='errors')

        return data_errors
