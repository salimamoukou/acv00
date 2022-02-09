import os
import pandas as pd
import streamlit as st
from acv_explainers.acv_tree import ACVTreeAgnostic, ACVTree
from acv_app import sdp_app
from acv_app import sv_app
from joblib import dump, load
import argparse
import subprocess

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'data')
APP_PATH = __file__


# Set page title
# st.set_page_config(page_title='ACPR challenge', layout="wide")

# define all required cache data for the app

def compile_acv(model, x_train, y_train, x_test, y_test, path):
    if type(x_train) != pd.core.frame.DataFrame:
        x_train = pd.DataFrame(x_train, columns=['X{}'.format(i) for i in range(x_train.shape[1])])
        x_test = pd.DataFrame(x_test, columns=['X{}'.format(i) for i in range(x_test.shape[1])])

    if type(y_train) != pd.core.frame.DataFrame:
        y_test = pd.DataFrame(y_test)
        y_train = pd.DataFrame(y_train)

    acvtree = ACVTree(model, x_train.values)
    acvtree_agnostic = ACVTreeAgnostic(model, x_train.values[:5])

    dump(acvtree, os.path.join(path, 'acvtree.joblib'))
    dump(acvtree_agnostic, os.path.join(path, 'acvtree_agnostic.joblib'))
    x_train.to_csv(os.path.join(path, 'x_train.csv'), index=False)
    y_train.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    x_test.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)


def compile_ACXplainers(ACXplainers, x_train, y_train, x_test, y_test, path):
    acvtree = ACXplainers
    acvtree_agnostic = ACXplainers

    if type(x_train) != pd.core.frame.DataFrame:
        x_train = pd.DataFrame(x_train, columns=['X{}'.format(i) for i in range(x_train.shape[1])])
        x_test = pd.DataFrame(x_test, columns=['X{}'.format(i) for i in range(x_test.shape[1])])

    if type(y_train) != pd.core.frame.DataFrame:
        y_test = pd.DataFrame(y_test)
        y_train = pd.DataFrame(y_train)

    dump(acvtree, os.path.join(path, 'acvtree.joblib'))
    dump(acvtree_agnostic, os.path.join(path, 'acvtree_agnostic.joblib'))
    x_train.to_csv(os.path.join(path, 'x_train.csv'), index=False)
    y_train.to_csv(os.path.join(path, 'y_train.csv'), index=False)
    x_test.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    y_test.to_csv(os.path.join(path, 'y_test.csv'), index=False)


def run_webapp(pickle_path):
    run_file = subprocess.run(["streamlit", "run", APP_PATH, "--", "--path", pickle_path])
    return run_file


def main(LOAD_PATH):
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
        st.title('Local Explanations based on Shapley values ')
        acvtree = load(os.path.join(LOAD_PATH, 'acvtree.joblib'))
    else:
        st.title('Consistent Local Explanations based on SDP ')

    page = PAGES[persona]
    page.write_pg(x_train, x_test, y_train, y_test, acvtree)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params of ACV Explanations Web App")
    parser.add_argument(
        '--path',
        default='./',
        type=str)

    args = parser.parse_args()

    main(args.path)
