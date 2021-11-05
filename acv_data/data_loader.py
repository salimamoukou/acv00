"""
Data loader module
"""
import os.path as op
import json
import pandas as pd


def data_loading(dataset):
    """
    data_loading allows shapash user to try the library with small but clear datasets.
    Titanic's or house_prices' reworked data loader
    from 'titanicdata.csv' and 'house_prices_dataset.csv'
    with well labels in a dictionnary.

    Parameters
    ----------
    dataset : String
        Dataset's name to return.
         - 'titanic'
         - 'house_prices'
    Returns
    -------
    data : pandas.DataFrame
        Dataset required
    """
    current_path = op.dirname(op.abspath(__file__))
    if dataset == 'telco':
        data_house_prices_path = op.join(current_path, "telco_churn.csv")
        data = pd.read_csv(data_telco)

    else:
        raise ValueError("Dataset not found. Check the docstring for available values")

    return data