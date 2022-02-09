from acv_app.plots import *
import numpy as np
import streamlit as st
# from acv_explainers import ACVTree

def write_pg(x_train, x_test, y_train, y_test, acvtree):
    @st.cache(allow_output_mutation=True)
    def compute_sv():
        # treemodel = ACVTree(acvtree.model, x_train.values.astype(np.double))
        # return treemodel.shap_values(x_test.values.astype(np.double)[:100], C=[[]])
        acvtree.check_is_explainer()
        return acvtree.ACXplainer.shap_values(x_test.values.astype(np.double)[:100], x_train.values.astype(np.double))
    idx = st.selectbox(
        'Choose the observation you want to explain',
        list(range(x_test.shape[0]))
    )

    col1, col2 = st.columns(2)
    with col1:
        sv = compute_sv()
        if acvtree.classifier != True:
            pred = acvtree.predict(x_test.values.astype(np.double))
        else:
            pred = acvtree.predict_proba(x_test.values.astype(np.double))
        if sv.shape[2] != 1:
            base = np.mean(pred[:, 1])
            fig = waterfall(base, sv[idx, :, 1], x_test.values[idx], x_test.columns)
        else:
            base = np.mean(pred)
            fig = waterfall(base, sv[idx, :, 0], x_test.values[idx], x_test.columns)
        st.pyplot(fig)