from .utils_cr import *
import acv_explainers

class CR:

    def __init__(self, acv_explainer, x_train, x_test, y_train, y_test, columns_name, model=None):
        self.model = None
        self.columns_name = columns_name
        self.acv_explainer = acv_explainer
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.ddp_importance_local, self.ddp_index_local, self.size_local, self.ddp_local = None, None, None, None
        self.S_star_local, self.S_bar_set_local = None, None
        self.ddp_local, self.w_local = None, None
        self.counterfactuals_samples_local, self.counterfactuals_samples_sdp_local, \
        self.counterfactuals_samples_w_local = None, None, None
        self.isolation = None, None
        self.dist_local = None
        self.score_local = None
        self.errs_local = None
        self.errs_local_original = None
        self.accuracy_local = None
        self.accuracy_local_original = None
        self.coverage_local = None
        self.sdp_importance_se, self.sdp_index_se, self.size_se, self.sdp_se = None, None, None, None
        self.S_star_se, self.N_star_se = None, None
        self.sdp_rules, self.rules, self.sdp_all, self.rules_data, self.w_rules = None, None, None, None, None

        self.ddp_importance_regional, self.ddp_index_regional, self.size_regional, self.ddp_regional = None, None, None, None
        self.S_star_regional, self.S_bar_set_regional = None, None
        self.counterfactuals_samples_regional, self.counterfactuals_samples_sdp_regional, \
        self.counterfactuals_samples_w_regional = None, None, None
        self.dist_regional = None
        self.score_regional = None
        self.errs_regional = None
        self.errs_regional_original = None
        self.accuracy_regional = None
        self.accuracy_regional_original = None
        self.coverage_regional = None

    def run_local_divergent_set(self, x, y_target=None, down=None, up=None, t=None, stop=True, pi_level=0.8):
        if y_target is None:
            y_target = np.zeros(shape=(x.shape[0]))
        if down is None:
            down = np.zeros(shape=(x.shape[0]))
        if up is None:
            up = np.zeros(shape=(x.shape[0]))

        print('### Computing the local divergent set of (x, y)')
        self.ddp_importance_local, self.ddp_index_local, self.size_local, self.ddp_local = \
            self.acv_explainer.importance_cdp_rf(x, y_target, down, up, self.x_train, self.y_train, t=t,
                                                 stop=stop, pi_level=pi_level)

        self.S_star_local, self.S_bar_set_local = \
            acv_explainers.utils.get_active_null_coalition_list(self.ddp_index_local, self.size_local)

        self.ddp_local, self.w_local = self.acv_explainer.compute_cdp_weights(x, y_target, down, up, self.x_train,
                                                                              self.y_train,
                                                                              S=self.S_bar_set_local)

    def run_local_counterfactual_rules(self, x, y_target=None, down=None, up=None,
                                       acc_level=0.8, pi_level=0.8, p_best=0.85, n_try=100,
                                  batch=100, max_iter=100, temp=1, max_iter_convergence=100):
        if y_target is None:
            y_target = np.zeros(shape=(x.shape[0]))
        if down is None:
            down = np.zeros(shape=(x.shape[0]))
        if up is None:
            up = np.zeros(shape=(x.shape[0]))

        print('### Computing the local counterfactual rules of (x, y)')
        if self.acv_explainer.classifier:
            self.counterfactuals_samples_local, self.counterfactuals_samples_sdp_local, \
            self.counterfactuals_samples_w_local = return_global_counterfactuals(self.acv_explainer, x, y_target,
                                                                                 self.S_star_local,
                                                                                 self.S_bar_set_local,
                                                                                 self.x_train, self.y_train,
                                                                                 self.w_local,
                                                                                 acc_level=acc_level,
                                                                                 pi_level=pi_level,
                                                                                 p_best=p_best, n_try=n_try,
                                                                                 batch=batch, max_iter=max_iter,
                                                                                 temp=temp,
                                                                                 max_iter_convergence=max_iter_convergence
                                                                                 )
        else:
            self.counterfactuals_samples_local, self.counterfactuals_samples_sdp_local, \
            self.counterfactuals_samples_w_local = return_global_counterfactuals_reg(self.acv_explainer, x, y_target,
                                                                                     down, up,
                                                                                     self.S_star_local,
                                                                                     self.S_bar_set_local,
                                                                                     self.x_train, self.y_train,
                                                                                     self.w_local,
                                                                                     acc_level=acc_level,
                                                                                     pi_level=acc_level)

    def run_sampling_local_counterfactuals(self, x, y_target=None, down=None, up=None, batch=1000, max_iter=1000, temp=0.5):
        if y_target is None:
            y_target = np.zeros(shape=(x.shape[0]))
        if down is None:
            down = np.zeros(shape=(x.shape[0]))
        if up is None:
            up = np.zeros(shape=(x.shape[0]))

        print('### Sampling using the local counterfactual rules of (x, y)')
        self.isolation = IsolationForest()
        self.isolation.fit(self.x_train)
        outlier_score = lambda x: self.isolation.decision_function(x)

        self.dist_local = []
        self.score_local = []
        self.errs_local = []
        self.errs_local_original = []
        for i in tqdm(range(x.shape[0])):
            if len(self.counterfactuals_samples_local[i]) != 0:
                a, sco = simulated_annealing(outlier_score, x[i], self.S_star_local[i], self.x_train,
                                             self.counterfactuals_samples_local[i][
                                                 np.argmax(self.counterfactuals_samples_sdp_local[i])],
                                             batch, max_iter, temp)
                self.dist_local.append(np.squeeze(a))
                self.score_local.append(sco)

                if self.acv_explainer.classifier:
                    self.errs_local.append(self.acv_explainer.predict(self.dist_local[-1].reshape(1, -1)) != self.acv_explainer.predict(
                            x[i].reshape(1, -1)))
                else:
                    self.errs_local.append(
                        down[i] <= self.acv_explainer.predict(self.dist_local[-1].reshape(1, -1)) <= up[i])
                if self.model != None:
                    self.errs_local_original.append(
                        self.model.predict(self.dist_local[-1].reshape(1, -1)) != self.model.predict(
                            x[i].reshape(1, -1)))

        self.accuracy_local = np.mean(self.errs_local)
        self.accuracy_local_original = np.mean(self.errs_local_original)
        self.coverage_local = len(self.errs_local) / x.shape[0]

    def run_sufficient_rules(self, x_rule, y_rule, t=None, pi_level=0.9, algo2=False, p_best=0.6, n_try=50,
                       batch=100, max_iter=200, temp=1, max_iter_convergence=50):
        print('### Computing the Sufficient Explanations and the Sufficient Rules')
        self.x_rules, self.y_rules, self.t = x_rule, y_rule, t
        self.sdp_importance_se, self.sdp_index_se, self.size_se, self.sdp_se = \
            self.acv_explainer.importance_sdp_rf(x_rule, y_rule,
                                                 self.x_train, self.y_train, t=t,
                                                 stop=False,
                                                 pi_level=pi_level)

        self.S_star_se, self.N_star_se = get_active_null_coalition_list(self.sdp_index_se, self.size_se)
        self.sdp_rules, self.rules, self.sdp_all, self.rules_data, self.w_rules =\
            self.acv_explainer.compute_sdp_maxrules(x_rule, y_rule, self.x_train, self.y_train, self.S_star_se,
                                                    t=t, verbose=True, algo2=algo2, pi_level=pi_level,
                                                    p_best=p_best, n_try=n_try, batch=batch, max_iter=max_iter,
                                                    temp=temp, max_iter_convergence=max_iter_convergence)

    def run_regional_divergent_set(self, y_rules_target=None, down_up=None, stop=True, pi_level=0.8):
        if y_rules_target is None:
            y_rules_target = np.zeros(shape=(self.x_rules.shape[0]))
        if down_up is None:
            down_up = np.zeros(shape=(self.x_rules.shape[0], 2))
        print('### Computing the regional divergent set of (x, y)')
        self.y_rules_target, self.down_up = y_rules_target, down_up

        self.ddp_importance_regional, self.ddp_index_regional, self.size_regional, self.ddp_regional = \
            self.acv_explainer.importance_cdp_intv(self.x_rules, self.y_rules_target,
                                                   self.x_train, self.y_train,
                                                   self.rules,
                                                   t=self.down_up,
                                                   stop=stop,
                                                   pi_level=pi_level)

        self.S_star_regional, self.S_bar_set_regional = \
            acv_explainers.utils.get_active_null_coalition_list(self.ddp_index_regional, self.size_regional)

        self.ddp_regional, self.w_regional = self.acv_explainer.compute_cdp_intv_weights(self.x_rules, self.y_rules_target,
                                                                                         self.x_train, self.y_train,
                                                                                         cond=self.rules,
                                                                                         t=self.down_up,
                                                                                         S=self.S_bar_set_regional)

    def run_regional_counterfactual_rules(self, acc_level=0.8, pi_level=0.8,  p_best=0.85, n_try=100,
                                  batch=100, max_iter=100, temp=1, max_iter_convergence=100):
        if self.acv_explainer.classifier:
            self.counterfactuals_samples_regional, self.counterfactuals_samples_sdp_regional, \
            self.counterfactuals_samples_w_regional = \
                return_ge_global_counterfactuals(self.acv_explainer, self.x_rules, self.y_rules_target,
                                                 self.S_star_regional, self.S_bar_set_regional,
                                                 self.x_train, self.y_train, self.w_regional, acc_level, self.rules,
                                                 pi_level,  p_best=p_best, n_try=n_try,
                                                   batch=batch, max_iter=max_iter, temp=temp,
                                                   max_iter_convergence=max_iter_convergence)
        else:
            down = self.down_up[:, 0]
            up = self.down_up[:, 1]
            print('### Computing the regional counterfactual rules of (x, y)')
            self.counterfactuals_samples_regional, self.counterfactuals_samples_sdp_regional, \
            self.counterfactuals_samples_w_regional = \
                return_ge_global_counterfactuals_reg(self.acv_explainer, self.x_rules, self.y_rules_target, down, up,
                                                     self.S_star_regional, self.S_bar_set_regional,
                                                     self.x_train, self.y_train, self.w_regional, acc_level, self.rules,
                                                     pi_level, p_best=p_best, n_try=n_try,
                                                     batch=batch, max_iter=max_iter, temp=temp,
                                                     max_iter_convergence=max_iter_convergence
                                                     )

    def run_sampling_regional_counterfactuals(self, max_obs=2, batch=1000, max_iter=1000, temp=0.5):
        down = self.down_up[:, 0]
        up = self.down_up[:, 1]
        print('### Sampling using the regional counterfactual rules')
        outlier_score = lambda x: self.isolation.decision_function(x)
        self.dist_regional = []
        self.score_regional = []
        self.errs_regional = []
        self.errs_regional_original = []
        nb = 0
        for i in range(self.x_rules.shape[0]):
            if len(self.counterfactuals_samples_regional[i]) != 0:
                x_in = np.prod([(self.x_test[:, s] <= self.rules[i, s, 1]) * (self.x_test[:, s] >= self.rules[i, s, 0])
                                for s in range(self.x_train.shape[1])], axis=0).astype(bool)
                nb += np.sum(x_in) if np.sum(x_in) <= max_obs else max_obs
                print('observations in rule = {}'.format(np.sum(x_in)))
                if np.sum(x_in) > 0:
                    for xi in tqdm(self.x_test[x_in][:max_obs]):
                        a, sco = simulated_annealing(outlier_score, xi, self.S_star_regional[i], self.x_train,
                                                     self.counterfactuals_samples_regional[i][
                                                         np.argmax(self.counterfactuals_samples_sdp_regional[i])],
                                                     batch, max_iter, temp)
                        self.dist_regional.append(np.squeeze(a))
                        self.score_regional.append(sco)
                        if self.acv_explainer.classifier:
                            self.errs_regional.append(self.acv_explainer.predict(
                                self.dist_regional[-1].reshape(1, -1)) != self.acv_explainer.predict(xi.reshape(1, -1)))
                        else:
                            self.errs_regional.append(
                                down[i] <= self.acv_explainer.predict(self.dist_regional[-1].reshape(1, -1)) <= up[i])
                        if self.model != None:
                            self.errs_regional_original.append(self.model.predict(
                                self.dist_regional[-1].reshape(1, -1)) != self.model.predict(xi.reshape(1, -1)))

        self.accuracy_regional = np.mean(self.errs_regional)
        self.accuracy_regional_original = np.mean(self.errs_regional_original)
        self.coverage_regional = len(self.errs_regional) / nb

    def run_sampling_regional_counterfactuals_alltests(self, max_obs=2, batch=1000, max_iter=1000, temp=0.5):
        down = self.down_up[:, 0]
        up = self.down_up[:, 1]

        print('### Sampling using the regional counterfactual rules')
        outlier_score = lambda x: self.isolation.decision_function(x)

        self.dist_regional = []
        self.score_regional = []
        self.errs_regional = []
        self.errs_regional_original = []
        x_test_pb = []

        for i in range(self.x_rules.shape[0]):
            x_in = np.prod([(self.x_test[:, s] <= self.rules[i, s, 1]) * (self.x_test[:, s] > self.rules[i, s, 0])
                            for s in range(self.x_train.shape[1])], axis=0)
            if len(self.counterfactuals_samples_regional[i]) != 0:
                x_in = np.max(self.counterfactuals_samples_sdp_regional[i]) * x_in
            else:
                x_in = 0 * x_in
            x_test_pb.append(x_in)

        x_test_pb = np.array(x_test_pb)
        best_counterfactuals = np.argmax(x_test_pb, axis=0)

        for i in tqdm(range(self.x_test.shape[0])):
            xi = self.x_test[i]
            best_id = best_counterfactuals[i]
            if len(self.counterfactuals_samples_regional[best_id]) != 0:
                a, sco = simulated_annealing(outlier_score, xi, self.S_star_regional[best_id], self.x_train,
                                             self.counterfactuals_samples_regional[best_id][
                                                 np.argmax(self.counterfactuals_samples_sdp_regional[best_id])][0],
                                             batch, max_iter, temp)
                self.dist_regional.append(np.squeeze(a))
                self.score_regional.append(sco)
                if self.acv_explainer.classifier:
                    self.errs_regional.append(self.acv_explainer.predict(
                        self.dist_regional[-1].reshape(1, -1)) != self.acv_explainer.predict(xi.reshape(1, -1)))
                else:
                    self.errs_regional.append(
                        down[i] <= self.acv_explainer.predict(self.dist_regional[-1].reshape(1, -1)) <= up[i])

                if self.model != None:
                    self.errs_regional_original.append(self.model.predict(
                        self.dist_regional[-1].reshape(1, -1)) != self.model.predict(xi.reshape(1, -1)))

        self.accuracy_regional = np.mean(self.errs_regional)
        self.accuracy_regional_original = np.mean(self.errs_regional_original)
        self.coverage_regional = len(self.errs_regional) / self.x_test.shape[0]

    def show_global_counterfactuals(self):
        if not self.acv_explainer.classifier:
            for idt in range(self.rules.shape[0]):
                print('Example {}'.format(idt))
                r = self.rules[idt]

                print_rule(self.columns_name, r, self.sdp_rules[idt], True, self.y_rules[idt])

                print('  ')
                print('  ')
                print('  ')
                for l in range(len(self.counterfactuals_samples_sdp_regional[idt])):
                    print('Example {} - Counterfactual {}'.format(idt, l))
                    w = self.counterfactuals_samples_w_regional[idt][l]
                    print_rule(self.columns_name, self.counterfactuals_samples_regional[idt][l],
                               self.counterfactuals_samples_sdp_regional[idt][l], False, np.mean(self.y_train[w != 0]))
                    print('  ')
                print(' ')
        else:
            for idt in range(self.rules.shape[0]):
                print('Example {}'.format(idt))
                r = self.rules[idt]

                print_rule(self.columns_name, r, self.sdp_rules[idt], True, self.y_rules[idt])

                print('  ')
                print('  ')
                print('  ')
                for l in range(len(self.counterfactuals_samples_sdp_regional[idt])):
                    print('Example {} - Counterfactual {}'.format(idt, l))
                    print_rule(self.columns_name, self.counterfactuals_samples_regional[idt][l],
                               self.counterfactuals_samples_sdp_regional[idt][l], False, self.y_rules_target[idt])
                    print('  ')
                print(' ')

    def show_local_counterfactuals(self, x, y_target):
        if not self.acv_explainer.classifier:
            for idt in range(x.shape[0]):
                print(self.columns_name)
                print('Example {} = {}'.format(idt, x[idt]))

                print('  ')
                print('  ')
                print('  ')
                for l in range(len(self.counterfactuals_samples_sdp_local[idt])):
                    w = self.counterfactuals_samples_w_local[idt][l]
                    print('Example {} - Counterfactual {}'.format(idt, l))
                    print_rule(self.columns_name, self.counterfactuals_samples_local[idt][l],
                               self.counterfactuals_samples_sdp_local[idt][l], False, np.mean(self.y_train[w != 0]))
                    print('  ')
                print(' ')
        else:
            for idt in range(x.shape[0]):
                print(self.columns_name)
                print('Example {} = {}'.format(idt, x[idt]))

                print('  ')
                print('  ')
                print('  ')
                for l in range(len(self.counterfactuals_samples_sdp_local[idt])):
                    print('Example {} - Counterfactual {}'.format(idt, l))
                    print_rule(self.columns_name, self.counterfactuals_samples_local[idt][l],
                               self.counterfactuals_samples_sdp_local[idt][l], False, y_target[idt])
                    print('  ')
                print(' ')
