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
**OSX users**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)
## How does ACV work?
ACV works for XGBoost, LightGBM, CatBoostClassifier, scikit-learn and pyspark tree models. 
To use it, we need to transform our model into ACVTree. 

```python
from acv_explainers import ACVTree

forest = RandomForestClassifier() # or any Tree Based models
#...trained the model

acvtree = ACVTree(forest, data) # data should be np.ndarray with dtype=double
```

### Same Decision Probability
Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we do not observe the variables <img src="https://latex.codecogs.com/gif.latex?x_{\bar{S}}" />.
* **How to compute <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
sdp = acvtree.compute_sdp_clf(X, S, data, num_threads=5)

"""
Description of the arguments    
   
X (np.ndarray[2]): observations        
S (np.ndarray[1]): index of variables on which we want to compute the SDP
data (np.ndarray[2]): data used to compute the SDP
num_threads (int): how many threads to use for parallelism 
"""
```
* **How to compute the Sufficient Coalition <img src="https://latex.codecogs.com/gif.latex?S^\star" />** ?
```python 
forest = RandomForestClassifier()
#...trained the model
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C=[[]], global_proba=0.9, num_threads=5)

"""
Description of the arguments

X (np.ndarray[2]): observations
data (np.ndarray[2]): data used for the estimation
C (list[list]): list of the index of variables group together
global_proba (double): the level of the SDP, default value = 0.9

sdp_index[i, :size[i]] corresponds to the index of the variables in $S^\star$ of observation i  
sdp[i] corresponds to the SDP value of the $S^\star$ of observation i
"""
```

*  **How to compute the Global SDP importance ?**
```python
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C=[[]], global_proba=0.9, num_threads=5)

"""
Description of the arguments

X (np.ndarray[2]): observations
data (np.ndarray[2]): data used for the estimation
C (list[list]): list of the index of variables group together
global_proba(double): the level of the SDP, default value = 0.9
num_threads (int): how many threads to use for parallelism 

sdp_importance:= corresponds to the global sdp of each variables 
"""
```
### Active Shapley values

The Active Shapley values is a SV based on a new game defined in the Paper ("The Shapley Value of coalition of variables provides better explanations") such that null (non-important) has zero SV and the "payout" is fairly distribute among active variables.

* **How to compute Active Shapley values ?**

```python
import acv_explainers

# First, we need to compute the Active and Null coalition
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C, global_proba, num_threads=5)

# Then, we used the active coalition found to compute the Active Shapley values.
S_star, N_star = acv_explainers.utils.get_null_coalition(sdp_index, size)

forest_acv_adap = acvtree.shap_values_acv_adap(X, C, S_star, N_star)

"""
Description of the arguments

X (np.ndarray[2]): observations
C (list[list]): list of the different coalition of variables by their index
S_star (np.ndarray[2]): index of variables in the Sufficient Coalition
N_star (np.ndarray[2]): index of the remaining variables
num_threads (int): how many threads to use for parallelism 
"""
# We can also compute ACV with the same active and null coalition for each observations.
# This is much faster than the previous method.

forest_acv = acvtree.shap_values_acv(X, C, S_star, N_star)

"""
Description of the arguments

X (np.ndarray[2]): observations
C (list[list]): list of the different coalition of variables by their index
S_star (list): index of variables in the Sufficient Coalition
N_star (list): index of the remaining variables
num_threads (int): how many threads to use for parallelism 
"""
```

### Shapley values of categorical variables
Let assume we have a categorical variable Y with k modalities that we encoded by introducing the dummy variables <img src="https://latex.codecogs.com/gif.latex?Y_1%2C%5Cdots%2C%20Y_%7Bk-1%7D" />. As show in the paper, we must take the coalition of the dummy variables to correctly calculate the Shapley values.

```python

# cat_index := list(list) that contains the index of the dummies or one-hot variables grouped 
# together for each variable. For example, we have only 2 categorical variables Y, Z 
# transformed into [Y_0, Y_1, Y_2] and [Z_0, Z_1, Z_2]

cat_index = [[0, 1, 2], [3, 4, 5]]
forest_sv = acvtree.shap_values(X, C=cat_index, num_threads=5)
```
In addition, we can compute the SV given any coalitions. For example, if we want the following coalition <img src="https://latex.codecogs.com/gif.latex?C_0%20%3D%20%28X_0%2C%20X_1%2C%20X_2%29%2C%20C_1%3D%28X_3%2C%20X_4%29%2C%20C_2%3D%28X_5%2C%20X_6%29" />

```python

coalition = [[0, 1, 2], [3, 4], [5, 6]]
forest_sv = acvtree.shap_values(X, C=coalition, num_threads=5)
```
*Remarks:* The computation for a regressor is similar, we just need to replace "_clf" in each function with "_reg".

## Examples and tutorials (a lot more to come...)
We can find a tutorial of the usages of ACV in [demo_acv](https://github.com/salimamoukou/acv00/blob/main/notebooks/demo_acv_explainer/demo_acv_explainers.ipynb) and 
the notebooks below demonstrate different use cases for ACV. Look inside the notebook directory of the repository if you want to try playing with the original notebooks yourself.

## Experiments of the papers
* [Comparisons of the different estimators](https://github.com/salimamoukou/acv00/blob/main/notebooks/experiments_paper/comparisons_of_the_different_estimators.ipynb)
* [Comparisons of SV on toy model: Coalition or SUM ?](https://github.com/salimamoukou/acv00/blob/main/notebooks/experiments_paper/coalition_or_sum_toy_model.ipynb)
* [Comparisons of SV on Census: Coalition or SUM ?](https://github.com/salimamoukou/acv00/blob/main/notebooks/experiments_paper/coalition_or_sum_adult.ipynb)
