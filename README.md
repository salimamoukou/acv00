# Active Coalition of Variables (ACV):

ACV is a python library that aims to explain any machine learning model. It provides **a better estimation of Shapley values for tree-based model** (>= dependent TreeSHAP) and explanations based on three approaches:
* Same Decision Probability (SDP)
* Active Shapley values (Local and Sparse by design)
* Swing Shapley Values (The Shapley values are interpretable by design)

In addition, we use the coalitional version of SV to properly handle categorical variables in the computation of SV.

See the papers [here](https://github.com/salimamoukou/acv00/blob/main/papers/)
## Requirements
Python 3.6+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C

## Installation

Install the acv package:
```
$ pip install acv-exp
```

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
sdp = acvtree.compute_sdp_clf(X, S, data)

"""
Description of the arguments    
   
X (np.ndarray[2]): observations        
S (np.ndarray[1]): index of variables on which we want to compute the SDP
data (np.ndarray[2]): data used to compute the SDP
"""
```
* **How to compute the Sufficient Coalition <img src="https://latex.codecogs.com/gif.latex?S^\star" />** ?
```python 
forest = RandomForestClassifier()
#...trained the model
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C=[[]], global_proba=0.9)

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
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C=[[]], global_proba=0.9)

"""
Description of the arguments

X (np.ndarray[2]): observations
data (np.ndarray[2]): data used for the estimation
C (list[list]): list of the index of variables group together
global_proba(double): the level of the SDP, default value = 0.9

sdp_importance:= corresponds to the global sdp of each variable
"""
```
### Active Shapley values

The Active Shapley values is a SV based on a new game defined in the Paper ([Accurate and robust Shapley Values for explaining predictions and focusing on local important variables](https://github.com/salimamoukou/acv00/blob/main/papers/) such that null (non-important) has zero SV and the "payout" is fairly distribute among active variables.

* **How to compute Active Shapley values ?**

```python
import acv_explainers

# First, we need to compute the Active and Null coalition
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data, C, global_proba)

# Then, we used the active coalition found to compute the Active Shapley values.
S_star, N_star = acv_explainers.utils.get_null_coalition(sdp_index, size)

forest_acv_adap = acvtree.shap_values_acv_adap(X, C, S_star, N_star)

"""
Description of the arguments

X (np.ndarray[2]): observations
C (list[list]): list of the different coalition of variables by their index
S_star (np.ndarray[2]): index of variables in the Sufficient Coalition
N_star (np.ndarray[2]): index of the remaining variables
"""
```

We can also compute ACV with the same active and null coalition for all observations.
This is much faster than the previous method.
```python
forest_acv = acvtree.shap_values_acv(X, C, S_star, N_star)

"""
Description of the arguments

X (np.ndarray[2]): observations
C (list[list]): list of the different coalition of variables by their index
S_star (list): index of variables in the Sufficient Coalition
N_star (list): index of the remaining variables
"""
```

### Shapley values of categorical variables
Let assume we have a categorical variable Y with k modalities that we encoded by introducing the dummy variables <img src="https://latex.codecogs.com/gif.latex?Y_1%2C%5Cdots%2C%20Y_%7Bk-1%7D" />. As show in the paper, we must take the coalition of the dummy variables to correctly calculate the Shapley values.

```python

# cat_index := list(list) that contains the index of the dummies or one-hot variables grouped 
# together for each variable. For example, if we have only 2 categorical variables Y, Z 
# transformed into [Y_0, Y_1, Y_2] and [Z_0, Z_1, Z_2]

cat_index = [[0, 1, 2], [3, 4, 5]]
forest_sv = acvtree.shap_values(X, C=cat_index)
```
In addition, we can compute the SV given any coalitions. For example, if we want the following coalition <img src="https://latex.codecogs.com/gif.latex?C_0%20%3D%20%28X_0%2C%20X_1%2C%20X_2%29%2C%20C_1%3D%28X_3%2C%20X_4%29%2C%20C_2%3D%28X_5%2C%20X_6%29" />

```python

coalition = [[0, 1, 2], [3, 4], [5, 6]]
forest_sv = acvtree.shap_values(X, C=coalition)
```

### Remarks
The computation for a regressor is similar, you have to replace "_clf" in each function with "_reg". If you don't want to use
multi-threaded (due to scaling or memory problem), you have to add "_nopa" to each function (e.g. compute_sdp_clf ==> compute_sdp_clf_nopa).
You can also compute the different values needed in cache by setting cache=True 
in ACVTree initialization e.g. ACVTree(model, data, cache=True).
## Examples and tutorials (a lot more to come...)
We can find a tutorial of the usages of ACV in [demo_acv](https://github.com/salimamoukou/acv00/blob/main/notebooks/demo_acv_explainer) and 
the notebooks below demonstrate different use cases for ACV. Look inside the notebook directory of the repository if you want to try playing with the original notebooks yourself.


* [Experiments of the papers](https://github.com/salimamoukou/acv00/blob/main/notebooks/experiments_paper)
