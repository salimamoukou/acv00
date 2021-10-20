from acv_app.plots import *
import numpy as np
import streamlit as st

def write_pg(x_train, x_test, y_train, y_test, acvtree):
    @st.cache(allow_output_mutation=True)
    def compute_sv():
        return acvtree.shap_values(x_test.values.astype(np.double)[:100], C=[[]])

    idx = st.selectbox(
        'Choose the observation you want to explain',
        list(range(x_test.shape[0]))
    )

    col1, col2 = st.columns(2)
    with col1:
        sv = compute_sv()
        pred = acvtree.predict(x_test.values.astype(np.double))
        if sv.shape[2] != 1:
            base = np.mean(pred[:, 1])
            fig = waterfall(base, sv[idx, :, 1], x_test.values[idx], x_test.columns)
        else:
            base = np.mean(pred)
            fig = waterfall(base, sv[idx, :, 0], x_test.values[idx], x_test.columns)
        st.pyplot(fig)