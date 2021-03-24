# Active Coalition of Variables (ACV):

ACV is a python library that aims to explain any machine learning model. It provides explanations based on two approaches:
* Same Decision Probability (SDP)
* A coalitional version of Shapley values

In addition, we use the coalitional version of SV to properly handle categorical variables in the computation of SV.

See paper "The Shapley Value of coalition of variables provides better explanations" for details.

## Requirements
Python 3.6+ 

Install the required packages:
```
$ pip install -r requirements.txt
```

## Installation

Install the acv package:
```
$ python3 setup.py install 
```

## How does ACV work?
ACV works for XGBoost, LightGBM, CatBoost, scikit-learn and pyspark tree models. 
To use it, we need to transform our model into ACVTree. 

```python
from acv_explainers import ACVTree

forest = RandomForestClassifier(), any ML models
#...trained the model

acvtree = ACVTree(forest, data)
```

### Same Decision Probability
Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we do not observe the variables <img src="https://latex.codecogs.com/gif.latex?x_{\bar{S}}" />.
* **How to compute <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
forest = RandomForestClassifier()
#...trained the model

sdp = acvtree.compute_sdp_clf(x=x, fx=fx, tx=threshold, S=id_var, data=data)

"""
Description of the arguments    
   
x (array): observation        
fx (float): tree(x)
threshold (float): threshold of the classifier
forest (RandomForestClassifier): model
S (list): index of variables on which we want to compute the SDP
data (array): data used to compute the SDP
"""
```
* **How to compute the Sufficient Coalition <img src="https://latex.codecogs.com/gif.latex?S^\star" />** ?
```python 
forest = RandomForestClassifier()
#...trained the model
sdp_cluster = []
acvtree.compute_local_sdp_clf(x, fx, threshold, proba, index, data, sdp_cluster, decay, verbose=0, C=C)

"""
Description of the arguments

x (array): observation
f (float): forest(x)
threshold (float): the radius t of the SDP regressor (see paper: SDP for regression)
proba (float): the level of the Sufficient Coalition \pi
index (list): index of the variables of x
data (array): data used for the estimation
sdp_cluster (list): holder of the SDP
decay (float): the probability decay used in the recursion step
C (list[list]): list of the index of variables group together
"""
```

*  **How to compute the Global SDP importance ?**
```python
forest = RandomForestClassifier()
#...trained the model

acvtree.global_sdp_importance_clf(data, data_bground, columns_names, 
                            global_proba, decay, threshold, proba, C=C, verbose=0)

"""
Description of the arguments

data (array): data used to compute the Global SDP
data_bground (array): data used in the estimations of the SDP
columns_names (list): names of the variables
global_proba (float): proba used for the selection criterion. We count each time for a variable if it is on a set with SDP >= global proba
decay (float): decay value used when recursively apply the local_sdp function .
threshold (float): the radius t of the SDP regressor (see paper: SDP for regression)
proba (float): the  level of the Sufficient Coalition
C (list[list]): list of index of the variables group together
"""
```
### Active Shapley values

The Active Shapley values is a SV based on a new game defined in the Paper ("The Shapley Value of coalition of variables provides better explanations") such that null (non-important) has zero SV and the "payout" is fairly distribute among active variables.

* **How to compute Active Shapley values ?**

```python
forest = RandomForestClassifier()
#...trained the model

# First, we need to compute the Sufficient coalition 
sdp_cluster = []
acvtree.compute_local_sdp_clf(x, fx, threshold, proba, index, data, sdp_cluster, decay, verbose=0, C=C)

# Then, we used the active coalition found to compute the Active Shapley values.
S_star = sdp_cluster[0]
N_star = sdp_cluster[-1]

forest_asv = acvtree.shap_values_acv(x, C, N_star, S_star)

"""
Description of the arguments

x (array): observation
data (array): data used to compute the Shapley values
C (list[list]): list of the different coalition of variables by their index
S_star (list): index of variables in the Sufficient Coalition
N_star (list): index of the remaining variables
"""
```

### Shapley values of categorical variables
Let assume we have a categorical variable Y with k modalities that we encoded by introducing the dummy variables <img src="https://latex.codecogs.com/gif.latex?Y_1%2C%5Cdots%2C%20Y_%7Bk-1%7D" />. As show in the paper, we must take the coalition of the dummy variables to correctly calculate the Shapley values.

```python
from acv_tools import *

cat_index = [[i for i in data.columns if data.dtypes[i] =='category']] # get the index of categorical variables
forest_sv = forest_shap_clf(forest, x, algo, data, C=cat_index)
```
In addition, we can compute the SV given any coalitions. For example, if we want the following coalition <img src="https://latex.codecogs.com/gif.latex?C_0%20%3D%20%28X_0%2C%20X_1%2C%20X_2%29%2C%20C_1%3D%28X_3%2C%20X_4%29%2C%20C_2%3D%28X_5%2C%20X_6%29" />

```python

coalition = [[0, 1, 2], [3, 4], [5, 6]]
forest_sv = acvtree.shap_values(x, C=coalition)
```
*Remarks:* The computation for a regressor is similar, we just need to remove "_clf" in each function. 

## Examples and tutorials (a lot more to come...)
We can find a tutorial of the usages of ACV in [demo_acv](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/demo_acv_tools.ipynb) and 
the notebooks below demonstrate different use cases for ACV. Look inside the notebook directory of the repository if you want to try playing with the original notebooks yourself.
* [SDP on toy regression model](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/sdp_on_regression.ipynb)
* [SDP on lung cancer classification](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/sdp_on_lucas_data.ipynb)

## Experiments of the papers
* [Comparisons of the different estimators](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/comparisons_of_the_different_estimators.ipynb)
* [Comparisons of SV on toy model: Coalition or SUM ?](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/coalition_or_sum_toy_model.ipynb)
* [Comparisons of SV on Census: Coalition or SUM ?](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/coalition_or_sum_adult.ipynb)
* [Active Shapley + SDP on toy model](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/sdp_on_regression.ipynb)
* [SDP and global SDP on Lucas](https://github.com/salimamoukou/shap-explainer/blob/master/notebook/sdp_on_lucas_data.ipynb)


