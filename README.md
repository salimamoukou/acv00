 # Active Coalition of Variables (ACV):

ACV is a python library that aims to explain **any machine learning models** or **data**. 

* It gives **local rule-based** explanations for any model or data.
* It provides **a better estimation of Shapley Values for tree-based model** (more accurate than [path-dependent TreeSHAP](https://github.com/slundberg/shap])). 
 It also proposes new Shapley Values that have better local fidelity.

We can regroup the different explanations in two groups: **Agnostic Explanations** and **Tree-based Explanations**.

See the papers [here](https://github.com/salimamoukou/acv00/tree/main/papers).
 
### Installation

##### Requirements
Python 3.6+ 

**OSX**: ACV uses Cython extensions that need to be compiled with multi-threading support enabled. 
The default Apple Clang compiler does not support OpenMP.
To solve this issue, obtain the lastest gcc version with Homebrew that has multi-threading enabled: 
see for example [pysteps installation for OSX.](https://pypi.org/project/pysteps/1.0.0/)

**Windows**: Install MinGW (a Windows distribution of gcc) or Microsoft’s Visual C


Install the acv package:
```
$ pip install acv-exp
```

 
## A. Agnostic explanations
The Agnostic approaches explain any data (**X**, **Y**) or model (**X**, **f(X)**) using the following 
explanation methods:

* Same Decision Probability (SDP) and **Sufficient Explanations**
* **Sufficient Rules**

See the paper [Consistent Sufficient Explanations and Minimal Local Rules for explaining regression and classification models](https://arxiv.org/abs/2111.04658) for more details.

**I. First, we need to fit our explainer (ACXplainers) to input-output of the data **(X, Y)** or model
**(X, f(X))** if we want to explain the data or the model respectively.**

```python
from acv_explainers import ACXplainer

# It has the same params as a Random Forest, and it should be tuned to maximize the performance.  
acv_xplainer = ACXplainer(classifier=True, n_estimators=50, max_depth=5)
acv_xplainer.fit(X_train, y_train)

roc = roc_auc_score(acv_xplainer.predict(X_test), y_test)
```

**II. Then, we can load all the explanations in a webApp as follow:**

```python 
import acv_app
import os

# compile the ACXplainer
acv_app.compile_ACXplainers(acv_xplainer, X_train, y_train, X_test, y_test, path=os.getcwd())

# Launch the webApp
acv_app.run_webapp(pickle_path=os.getcwd())
```
![Capture d’écran de 2021-11-03 19-50-12](https://user-images.githubusercontent.com/40361886/140174581-4c5bf018-05ad-49e0-b005-2a65453626e1.png)



**III. Or we can compute each explanation separately as follow:**

#### Same Decision Probability (SDP)
The main tool of our explanations is the Same Decision Probability (SDP). Given <img src="https://latex.codecogs.com/gif.latex?x%20%3D%20%28x_S%2C%20x_%7B%5Cbar%7BS%7D%7D%29" />, the same decision probability <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> of variables <img src="https://latex.codecogs.com/gif.latex?x_S" />  is the probabilty that the prediction remains the same when we fixed variables 
<img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> or when the variables <img src="https://latex.codecogs.com/gif.latex?X_{\bar{S}}" /> are missing.
* **How to compute <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  ?**

```python
sdp = acv_xplainer.compute_sdp_rf(X, S, data_bground) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```
#### Minimal Sufficient Explanations
The Sufficient Explanations is the Minimal Subset S such that fixing the values <img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> 
permit to maintain the prediction with high probability <img src="https://latex.codecogs.com/gif.latex?\pi" />.
See the paper [here](https://github.com/salimamoukou/acv00/tree/main/papers/Suffient%20Explanations%20and%20Sufficient%20Rules) for more details. 

* **How to compute the Minimal Sufficient Explanation <img src="https://latex.codecogs.com/gif.latex?S^\star" /> ?**
    
    The following code return the Sufficient Explanation with minimal cardinality. 
```python
sdp_importance, min_sufficient_expl, size, sdp = acv_xplainer.importance_sdp_rf(X, y, X_train, y_train, pi_level=0.9)
```

* **How to compute all the Sufficient Explanations  ?**

    Since the Minimal Sufficient Explanation may not be unique for a given instance, we can compute all of them.
```python
sufficient_expl, sdp_expl, sdp_global = acv_xplainer.sufficient_expl_rf(X, y, X_train, y_train, pi_level=0.9)
```

#### Local Explanatory Importance
For a given instance, the local explanatory importance of each variable corresponds to the frequency of 
apparition of the given variable in the Sufficient Explanations. See the paper [here](https://github.com/salimamoukou/acv00/tree/main/papers/Suffient%20Explanations%20and%20Sufficient%20Rules) for more details. 

* **How to compute the Local Explanatory Importance ?**

```python
lximp = acv_xplainer.compute_local_sdp(d=X_train.shape[1], sufficient_expl)
```

#### Local rule-based explanations
For a given instance **(x, y)** and its Sufficient Explanation S such that <img src="https://latex.codecogs.com/gif.latex?SDP_S(\boldsymbol{x};&space;y)&space;\geq&space;\pi" title="SDP_S(\boldsymbol{x}; y) \geq \pi" />, we compute a local minimal rule which contains **x** such 
that every observation **z** that satisfies this rule has <img src="https://latex.codecogs.com/gif.latex?SDP_S(\boldsymbol{z};&space;y)&space;\geq&space;\pi" title="SDP_S(\boldsymbol{z}; y) \geq \pi" />. See the paper [here](https://github.com/salimamoukou/acv00/tree/main/papers/Suffient%20Explanations%20and%20Sufficient%20Rules) for more details

* **How to compute the local rule explanations ?**

```python
sdp, rules, _, _, _ = acv_xplainer.compute_sdp_maxrules(X, y, data_bground, y_bground, S) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```

## B. Tree-based explanations
ACV gives Shapley Values explanations for XGBoost, LightGBM, CatBoostClassifier, scikit-learn and pyspark tree models. It
provides the following  Shapley Values: 

* Classic local Shapley Values (The value function is the conditional expectation <img src="https://latex.codecogs.com/gif.latex?E[f(x)&space;|&space;\boldsymbol{X}_S&space;=&space;\boldsymbol{x}_S]" title="E[f(x) | \boldsymbol{X}_S = \boldsymbol{x}_S]" />)
* Active Shapley values (Local fidelity and Sparse by design)
* Swing Shapley Values (The Shapley values are interpretable by design) *(Coming soon)*

In addition, we use the coalitional version of SV **to properly handle categorical variables in the computation of SV**.

See the papers [here](https://github.com/salimamoukou/acv00/blob/main/papers/)

To explain the tree-based models above, we need to transform our model into ACVTree. 
```python
from acv_explainers import ACVTree

forest = XGBClassifier() # or any Tree Based models
#...trained the model

acvtree = ACVTree(forest, data_bground) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```
#### Accurate Shapley Values

```python
sv = acvtree.shap_values(X)
```
Note that it provides a better estimation of the [tree-path dependent of TreeSHAP](https://github.com/slundberg/shap]) when the variables are dependent.

#### Accurate Shapley Values with encoded categorical variables

Let assume we have a categorical variable Y with k modalities that we encoded by introducing the dummy variables <img src="https://latex.codecogs.com/gif.latex?Y_1%2C%5Cdots%2C%20Y_%7Bk-1%7D" />. As shown in the paper, we must take the coalition of the dummy variables to correctly compute the Shapley values.

```python

# cat_index := list[list[int]] that contains the column indices of the dummies or one-hot variables grouped 
# together for each variable. For example, if we have only 2 categorical variables Y, Z 
# transformed into [Y_0, Y_1, Y_2] and [Z_0, Z_1, Z_2]

cat_index = [[0, 1, 2], [3, 4, 5]]
forest_sv = acvtree.shap_values(X, C=cat_index)
```
In addition, we can compute the SV given any coalitions. For example, let assume we have 10 variables 
<img src="https://latex.codecogs.com/gif.latex?(\boldsymbol{X}_0,&space;\boldsymbol{X}_1,&space;\dots,&space;\boldsymbol{X}_{10})" title="(\boldsymbol{X}_0, \boldsymbol{X}_1, \dots, \boldsymbol{X}_{10})" /> and we want the following coalition <img src="https://latex.codecogs.com/gif.latex?C_0%20%3D%20%28X_0%2C%20X_1%2C%20X_2%29%2C%20C_1%3D%28X_3%2C%20X_4%29%2C%20C_2%3D%28X_5%2C%20X_6%29" />

```python

coalition = [[0, 1, 2], [3, 4], [5, 6]]
forest_sv = acvtree.shap_values(X, C=coalition)
```
#### **How to compute <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" />  for tree-based classifier ?**
Recall that the <img src="https://latex.codecogs.com/gif.latex?SDP_S%28x%2C%20f%29" /> is the probability that the prediction remains the same when we fixed variables 
<img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> given the subset S.
```python
sdp = acvtree.compute_sdp_clf(X, S, data_bground) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```
#### **How to compute the Sufficient Coalition <img src="https://latex.codecogs.com/gif.latex?S^\star" /> and the Global SDP importance for tree-based classifier ?**
Recall that the Minimal Sufficient Explanations is the Minimal Subset S such that fixing the values <img src="https://latex.codecogs.com/gif.latex?X_S=x_S" /> 
permit to maintain the prediction with high probability <img src="https://latex.codecogs.com/gif.latex?\pi" />.

```python 
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data_bground) # data_bground is the background dataset that is used for the estimation. It should be the training samples.
```

#### Active Shapley values

The Active Shapley values is a SV based on a new game defined in the Paper ([Accurate and robust Shapley Values for explaining predictions and focusing on local important variables](https://arxiv.org/abs/2106.03820) such that null (non-important) variables has zero SV and the "payout" is fairly distribute among active variables.

* **How to compute Active Shapley values ?**

```python
import acv_explainers

# First, we need to compute the Active and Null coalition
sdp_importance, sdp_index, size, sdp = acvtree.importance_sdp_clf(X, data_bground)
S_star, N_star = acv_explainers.utils.get_active_null_coalition_list(sdp_index, size)

# Then, we used the active coalition found to compute the Active Shapley values.
forest_asv_adap = acvtree.shap_values_acv_adap(X, C, S_star, N_star, size)
```

##### Remarks for tree-based explanations: 
If you don't want to use multi-threaded (due to scaling or memory problem), you have to add "_nopa" to each function (e.g. compute_sdp_clf ==> compute_sdp_clf_nopa).
You can also compute the different values needed in cache by setting cache=True 
in ACVTree initialization e.g. ACVTree(model, data_bground, cache=True).

## Examples and tutorials (a lot more to come...)
We can find a tutorial of the usages of ACV in [demo_acv](https://github.com/salimamoukou/acv00/blob/main/notebooks/demo_acv_explainer) and 
the notebooks below demonstrate different use cases for ACV. Look inside the notebook directory of the repository if you want to try playing with the original notebooks yourself.
