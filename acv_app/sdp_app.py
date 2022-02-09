from acv_app.colors import _colors as colors
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from acv_explainers.utils import extend_partition

labels = {
    'MAIN_EFFECT': "SHAP main effect value for\n%s",
    'INTERACTION_VALUE': "SHAP interaction value",
    'INTERACTION_EFFECT': "SHAP interaction value for\n%s and %s",
    'VALUE': "SHAP value (impact on model output)",
    'GLOBAL_VALUE': "mean(|SHAP value|) (average impact on model output magnitude)",
    'VALUE_FOR': "SHAP value for\n%s",
    'PLOT_FOR': "SHAP plot for %s",
    'FEATURE': "Feature %s",
    'FEATURE_VALUE': "Feature value",
    'FEATURE_VALUE_LOW': "Low",
    'FEATURE_VALUE_HIGH': "High",
    'JOINT_VALUE': "Joint SHAP value",
    'MODEL_OUTPUT': "Model output value"
}


def write_pg(x_train, x_test, y_train, y_test, acvtree):
    st.sidebar.title("Parameters")
    if y_test.dtype == int or y_test.dtype == bool:
        CLASSIFIER = st.sidebar.checkbox('Classifier', value=True)
    else:
        CLASSIFIER = st.sidebar.checkbox('Classifier', value=False)

    col1, col2, col3 = st.columns(3)

    with col1:
        nb = st.sidebar.number_input(label='SAMPLE SIZE', value=10, min_value=5, max_value=500)
    with col2:
        pi_level = st.sidebar.number_input(label='SDP MIN (\pi)', value=0.9, min_value=0.7, max_value=1.)
    with col3:
        t = st.sidebar.number_input(label='SDP THRESHOLD FOR REGRESSOR', value=10., min_value=1., max_value=500.)

    idx = st.selectbox(
        'Choose the observation you want to explain',
        list(range(nb))
    )

    @st.cache(allow_output_mutation=True)
    def compute_sdp(nb, x_train, y_train, x_test, y_test, pi_level, t):
        sufficient_coal, sdp_coal, sdp_global = acvtree.sufficient_expl_rf(x_test[:nb], y_test[:nb], x_train, y_train,
                                                                           stop=False, pi_level=pi_level,
                                                                           t=t)
        for i in range(len(sufficient_coal)):
            sufficient_coal[i].pop(0)
            sdp_coal[i].pop(0)
        return sufficient_coal, sdp_coal, sdp_global

    @st.cache(allow_output_mutation=True)
    def compute_sdp_rule(obs, x_train_np, y_train_np, x_test_np, y_test_np, t, S):
        sdp, rules = acvtree.compute_sdp_rule(x_test_np[obs:obs+1], y_test_np[obs:obs+1],
                                              x_train_np, y_train_np, S=[S], t=t)
        rule = rules[0]
        columns = [x_train.columns[i] for i in range(x_train.shape[1])]
        rule_string = ['{} <= {} <= {}'.format(rule[i, 0] if rule[i, 0] > -1e+10 else -np.inf, columns[i],
                                               rule[i, 1] if rule[i, 1] < 1e+10 else +np.inf) for i in S]
        rule_string = ' and '.join(rule_string)
        return rule_string

    @st.cache(allow_output_mutation=True)
    def compute_sdp_maxrule(obs, x_train_np, y_train_np, x_test_np, y_test_np, t, S, pi):
        sdp, rules, sdp_all, rules_data, w = acvtree.compute_sdp_maxrules(x_test_np[obs:obs + 1], y_test_np[obs:obs + 1],
                                              x_train_np, y_train_np, S=[S], t=t, pi_level=pi)

        acvtree.fit_global_rules(x_train_np, y_train_np, rules, [S])

        # extend_partition(rules, rules_data, sdp_all, pi=pi, S=[S])

        rule = rules[0]
        columns = [x_train.columns[i] for i in range(x_train.shape[1])]
        rule_string = ['{} <= {} <= {}'.format(rule[i, 0] if rule[i, 0] > -1e+10 else -np.inf, columns[i],
                                               rule[i, 1] if rule[i, 1] < 1e+10 else +np.inf) for i in S]
        rule_string = ' and '.join(rule_string)
        return rule_string

    @st.cache(allow_output_mutation=True)
    def transform_scoal_to_col(sufficient_coal, columns_names):
        col_byobs = []
        for obs in sufficient_coal:
            col = []
            for S in obs:
                name = ''
                for i in range(len(S)):
                    if i != len(S) - 1:
                        name += columns_names[S[i]] + ' - '
                    else:
                        name += columns_names[S[i]]
                col.append(name)
            col_byobs.append(col)
        return col_byobs

    @st.cache(allow_output_mutation=True)
    def compute_local_sdp(idx, sufficient_coal):
        flat = [item for sublist in sufficient_coal[idx] for item in sublist]
        flat = pd.Series(flat)
        flat = dict(flat.value_counts() / len(sufficient_coal[idx]))
        local_sdp = np.zeros(x_train.shape[1])
        for key in flat.keys():
            local_sdp[key] = flat[key]
        return local_sdp

    @st.cache(allow_output_mutation=True)
    def color_max(data, sdp_index):
        color = []
        for i in range(x_train.shape[1]+1):
            if i in sdp_index:
                color.append('background-color: #3e82fc')
            else:
                color.append('')
        color.append('background-color: #ff073a')
        return color

    @st.cache(allow_output_mutation=True)
    def bar_legacy(shap_values, features=None, feature_names=None, max_display=None, show=True):
        # unwrap pandas series

        fig = plt.figure()

        if str(type(features)) == "<class 'pandas.core.series.Series'>":
            if feature_names is None:
                feature_names = list(features.index)
            features = features.values

        if feature_names is None:
            feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])

        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        feature_order = np.argsort(-np.abs(shap_values))

        #
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds), 0, -1)
        plt.barh(
            y_pos, shap_values[feature_inds],
            0.7, align='center',
            color=[colors.red_rgb if shap_values[feature_inds[i]] < 0 else colors.blue_rgb for i in range(len(y_pos))]
        )
        for y in range(len(y_pos)):
            plt.text(shap_values[feature_inds][y] + 0.001, y_pos[y] - 0.07, round(shap_values[feature_inds][y], 3))

        plt.yticks(y_pos, fontsize=13)
        if features is not None:
            features = list(features)

            # try and round off any trailing zeros after the decimal point in the feature values
            for i in range(len(features)):
                try:
                    if round(features[i]) == features[i]:
                        features[i] = int(features[i])
                except TypeError:
                    pass  # features[i] must not be a number
        yticklabels = []
        for i in feature_inds:
            if features is not None:
                yticklabels.append(feature_names[i] + " = " + str(features[i]))
            else:
                yticklabels.append(feature_names[i])
        plt.gca().set_yticklabels(yticklabels)
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)

        plt.xlabel("Frequency of apparition in the Sufficient Coalitions")
        return fig

    # explantions_load_state = st.text('Computing SDP explanations...')
    sufficient_coal, sdp_coal, sdp_global = compute_sdp(nb, x_train.values.astype(np.double),
                                                        y_train.astype(np.double), x_test.values.astype(np.double),
                                                        y_test.astype(np.double), pi_level=pi_level, t=t)

    sufficient_coal_names = transform_scoal_to_col(sufficient_coal, x_train.columns)
    # explantions_load_state.text("SDP explanation Done!")

    # st.subheader('All sufficient coalitions')

    if len(sufficient_coal[idx]) == 0:
        st.text('No explanation was found for this observation')
    else:

        col1, col2 = st.columns(2)
        with col1:
            st.header('All sufficient explanations')
            sufficient_coal_df = {'Sufficient explanations': sufficient_coal_names[idx],
                                  'SDP': sdp_coal[idx]}

            sufficient_coal_df = pd.DataFrame(sufficient_coal_df)
            # print(sufficient_coal_df.head())
            st.dataframe(sufficient_coal_df, 6000, 6000)

        with col2:
            st.header('Local Explanatory Importance')
            local_sdp = compute_local_sdp(idx, sufficient_coal)
            # data = {'feature_names': [x_train.columns[i] for i in range(x_train.shape[1])],
            #         'feature_importance': local_sdp}
            # fi_df = pd.DataFrame(data)

            # Sort the DataFrame in order decreasing feature importance
            # fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)
            # sns.set_theme(font='sans-serif')
            fig = bar_legacy(local_sdp, x_test.values[idx], x_test.columns)
            # sns.set(font_scale=1.5)
            # sns.set_theme(font='sans-serif')
            # sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'], color='#3e82fc')
            # plt.xlabel('Frequency of apparition in the Sufficient Coalitions')
            # for y in range(len(fi_df['feature_importance'].values)):
            #     plt.text(fi_df['feature_importance'].values[y], y, round(fi_df['feature_importance'].values[y], 3))
            #
            # plt.ylabel(' ')
            st.pyplot(fig)

        st.header('Feature values highlight by SDP')
        st.text('This observation has {} different explanations, below to observe their values'.format(
            len(sufficient_coal[idx])))

        exp_idx = st.selectbox(
            'Change the explanations',
            list(range(len(sufficient_coal[idx])))
        )

        x_group = pd.DataFrame(x_test.values[idx:idx + 1], columns=x_test.columns)
        x_group['Output'] = y_test[idx]
        x_group['SDP'] = sdp_coal[idx][exp_idx]
        st.dataframe(x_group.iloc[:1].style.apply(color_max, sdp_index=sufficient_coal[idx][exp_idx], axis=1))

        st.header('Local rule explanation')
        rule_string = compute_sdp_rule(idx, x_train.values.astype(np.double), y_train.astype(np.double),
                                       x_test.values.astype(np.double), y_test.astype(np.double), t, sufficient_coal[idx][exp_idx])
        # st.markdown(rule_string)

        st.markdown("<" + 'h3' + " style='text-align: " + \
                    "; color:" + 'black' + "; '>" + rule_string + "</" + 'h3' + ">",
                    unsafe_allow_html=True)

        st.header('Sufficient local rule explanation (Maximal rule)')
        maxrule = st.checkbox('Compute', value=False)

        if maxrule:
            rule_string = compute_sdp_maxrule(idx, x_train.values.astype(np.double), y_train.astype(np.double),
                                           x_test.values.astype(np.double), y_test.astype(np.double), t,
                                           sufficient_coal[idx][exp_idx], pi_level)

            st.markdown("<" + 'h3' + " style='text-align: " + \
                        "; color:" + 'black' + "; '>" + rule_string + "</" + 'h3' + ">",
                        unsafe_allow_html=True)
            # st.markdown(rule_string)

            rule_info = {'Rule coverage': acvtree.rules_coverage,
                          'Rule accuracy/mse': acvtree.rules_acc}

            rule_info = pd.DataFrame(rule_info)
            st.dataframe(rule_info, 6000, 6000)
