import os
import pandas as pd
import streamlit as st
from acv_explainers.acv_tree import ACVTreeAgnostic, ACVTree
from acv_app import sdp_app
from acv_app import sv_app
from joblib import dump, load


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data')
LOAD_PATH = '/home/samoukou/Documents/INVASE-master'

# Set page title
# st.set_page_config(page_title='ACPR challenge', layout="wide")

# define all required cache data for the app

def compile_acv(model, x_train, y_train, x_test, y_test, path):
    acvtree = ACVTree(model, x_train.values)
    acvtree_agnostic = ACVTreeAgnostic(model, x_train.values[:5])

    dump(acvtree, os.path.join(path, 'acvtree.joblib'))
    dump(acvtree_agnostic, os.path.join(path, 'acvtree_agnostic.joblib'))
    x_train.to_csv(os.path.join(path, 'x_train.csv'), index=False)
    y_train.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    x_test.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)


def main():
    """Loads data, preprocess it, creates a simple model then wrapps the data and the model in a Xplainer
    Once data and the model are wrapped, it creates the streamlit front PAGES
    """

    st.set_page_config(page_title='ACV demo', layout="wide")

    PAGES = {
        "SDP based explanations": sdp_app,
        "Shapley based Explanations": sv_app
    }

    st.sidebar.title("Choose your explanations")
    persona = st.sidebar.radio("", list(PAGES.keys()))


    @st.cache
    def load_data():
        x_train = pd.read_csv(os.path.join(LOAD_PATH, 'x_train.csv'))
        y_train = pd.read_csv(os.path.join(LOAD_PATH, 'y_train.csv')).values.squeeze()
        x_test = pd.read_csv(os.path.join(LOAD_PATH, 'x_test.csv'))
        y_test = pd.read_csv(os.path.join(LOAD_PATH, 'y_test.csv')).values.squeeze()
        return x_train, x_test, y_train, y_test

    # data_load_state = st.sidebar.text('Loading data...')
    x_train, x_test, y_train, y_test = load_data()
    # data_load_state.text("Load data Done !")

    @st.cache(allow_output_mutation=True)
    def load_model():
        acvtree = load(os.path.join(LOAD_PATH, 'acvtree_agnostic.joblib'))
        return acvtree

    # model_load_state = st.sidebar.text('Loading model...')
    acvtree = load_model()
    # model_load_state.text("Load model Done!")

    if persona == 'Shapley based Explanations':
        st.title('Local Explanation based on Shapley values ')
        acvtree = load(os.path.join(LOAD_PATH, 'acvtree.joblib'))
    else:
        st.title('Local Explanation based on SDP ')

    page = PAGES[persona]
    page.write_pg(x_train, x_test, y_train, y_test, acvtree)


if __name__ == "__main__":
    main()
