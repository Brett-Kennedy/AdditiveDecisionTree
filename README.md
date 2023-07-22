# AdditiveDecisionTree

## Summary
This tool provides an implementation of a decision tree, similar to a standard decision tree such as in sklearn, but utilizing an additive approach to fitting the data. Both AdditiveDecitionTreeClassifier and AdditiveDecisionTreeRegressor classes are provided. 

This tool provides, on the whole, comparable accuracy to standard decision trees, but in many cases provides greater accuracy and/or improved interpretability. As such, it can be a useful tool for generating interpretable models and may be considered a useful XAI tool. It is not intended to be competitive with approaches such as boosting or neural networks in terms of accuracy, but is simply a tool to generate interpretable models. It can often produce models comparable in accuracy to deeper standard decision trees, while having a lower overall complexity compared to these. 

#### Limitations of Decision Trees
This tool addresses some well-known limitations of decision trees, in particular their limited stability, their necessity to split based on fewer and fewer samples lower in the trees, repeated sub-trees, and their tendency to overfit if not restricted or pruned. These limitations are typically addressed by ensembling decision trees, either through bagging or boosting, which results in highly uninterpretable, though generally more accurate, models. Constructing oblivious trees (this is done, for example, within CatBoost) and oblique decision trees (see: [RotationFeatures](https://github.com/Brett-Kennedy/RotationFeatures)) is another approach to mitigate some of these limitations, and also often produces more stable trees. 

Decision trees are considered to be among the more interpretable models, but only where it is possible to construct them to a shallow depth, perhaps to 4 or 5 levels at most. However, decision trees often need to be fairly deep to acheive higher levels of accuracy, which can greatly undermine their interpretability. 

#### Intuition Behind Additive Decision Trees
The intuition behind AdditiveDecisionTrees in that often the true function *(f(x))*, mapping the input X to the target y, is based on logical conditions and in other cases it is simply a probabalistic function where each input feature may be considered somewhat independently (as with the Naive Bayes assumption). For example, the true f(x) may include something to the effect: 
```
If A > 10 Then: y = class Y 
Else if B < 19 Then: y = class X
Else if C * D > 44 Then: y = class Y
Else y = class Z. 
```
In this case, the true f(x) is composed of logical conditions and may be accurately and in a simple manner represented as a series of rules, such as in a 
Decision Tree, Rule List, or Rule Set. Note that conditions may be viewed as interactions, where how one feature predicts the target is depedent on the value of another columns. Here, one rule is based explicitely on the interaction C * D, but all rules entail interactions, as they may fire only if previous rules do not, and therefore the relationships between the features used in these rules is effected by other features.

As well, the true f(x) may be a set of patterns related to probabilities, more of the form: 
```
The higher A is, the more likely y is to be class X, regardless of B, C and D
The higher B is, up to 100.0, the more likely y is class Y, regardless of A, C and D 
The higher B is, where B is 100.0 or more, the more likely y is to be class Z, regardless of A, C and D
The higher C * D is, the more likely y is class X, regardless of A and B.
```
Some of these patterns may involve two or more features and some a single feature. In this form of function, for each instance, the feature values, or combinations of feature values, each contribute some probability to the target value (to each class in the case of classification), and these probabilities are summed to determine the overall probability distribution. Here feature interactions may exist within the probabalistic patterns, as in the case of C * D. 

While there are other means to taxonify functions, this system is quite useful, and many true functions may be viewed as some combination of these two broad classes of function. 

Standard decision trees do not explicitely assume the true function is conditional, and can accurately (often through the use of very large trees) capture non-conditional relationships such as those based on probabilities. They do, however, model the functions as conditions, which can limit their expressive power and lower their interpretability.  

Though f(x) is ultimately a set of rules for any classifiction problem, the rules may be largely independent of each other, each simply a probability distribution based on one or a small number of features. 

AdditiveDecisionTrees remove the assumption in standard decision trees that f(x) may be best modeled as a set of conditions, but does support conditions where the data suggests they exist. The central idea is that the true f(x) may be based on logical conditions, probabilities, or some combination of these.

The case where f(x) is based on the features independently (each feature's relationship to the target is not based on any other feature) may be modelled better by linear or logistic regression, Naive Bayes models, or GAM (Generalized Additive Model), among other models, which simply predict based on a weighted sum of each independent feature. That is, each relevant feature contributes to the final prediction without consideration of the other features (though interaction features may be created). f(x), in this case, is simply a probability distribution associated with each input feature. In these cases, linear regression, logistic regression, Naive Bayes, and GAMs can be quite interpretable and may be suitable choices for XAI. 

Conversely, linear and logistic regressions do not capture well where there are strong conditions in the function f(x), while decision trees can model these, at least potentially, quite closely. It is usually not know a priori if the true f(x) contains strong conditions and, as such, if it is desirable model the function as a decsion tree does: to repeatedly split the data into subsets and develop a different prediction for each leaf node based entirely on the datapoints within it. 

#### Splitting Policy
We describe here how Additive Decision Trees are constructed, and particularly their splitting policy. Note, the process is simpler to present for classification problems, and so most examples relate to classification, but the ideas apply equally to regression.

The approach taken by AdditiveDecisionTrees is to split the dataspace where appropriate and to make an aggregate decision based on numerous potential splits (all standard axis-parallel splits over different input parameters) where this appears most appropriate. This is done such that the splits appear higher in the tree, where there are larger numbers of samples to base the splits on and they may be found in a more reliable manner, while lower in the tree, where there are less samples to rely on, the decisions are based on a collection of splits, each using the full set of samples in that subset. 

This provides for straight-forward explanations for each row (known as *local* explanations) and for the models as a whole (known as *global* explanations). Though the final trees may be somewhat more complex than an standard decision tree of equal depth, as some nodes may be based on multiple splits, Additive Decision Trees are more accurate than standard decision trees of equal depth, and simpler than standard decision trees of equal accuracy. The explanations for individual rows  may be presented simply through the corresponding decision paths, as with standard decision trees, but the final nodes may be based on averaging over multiple splits. The maximum number of splits aggregated together is configurable, but 4 or 5 is typically sufficient. In most cases, as well, all splits agree, and only one needs to be presented to the user. And in fact, even where the splits disagree, the majority prediction may be presented as a single split. Therefore, the explanations are usually similar as those for standard decision trees, but with shorter decision paths. 

This, then, produces a model where there are a small number of splits, ideally representing the true conditions, if any, in the model, followed by *additive nodes*, which are leaf nodes that average the predictions of multiple splits, providing more robust predictions. This reduces the need to split the data into progressively smaller subsets, each with less statistical significance. 

AdditiveDecisionTrees, therefore, provide a simple form of ensembling, but one that still allows a single, interpretable model, easily supporting both global and local explanations. As it still follows a simple tree structure, contrapositive explanations may be easily generated as well. 

## Intallation

`
pip install AdditiveDecisionTrees
`

## Pruning Algorithm
The pruning algorihm executes after a tree, similar to a standard decision tree is constructed. The prunnig algorithm seeks to reduce significant sub-trees within the tree into single additive nodes, based on a small set of simple rules (comparable to the rule used in standard decsision trees, but such that the addititive nodes use multiple such rules). 

The algorithm behaves similarly to most pruning algorithms, starting at the bottom, at the leaves, and working towards the root node. At each node, a decision is made to either leave the node as is, or convert it to an additive node, that is, a node combining multiple data splits. 

At each node, the accuracy of the tree is evaluated on the training data given the current split, then again treating this node as an additive node. If the accuracy is higher with this node set as an additive node, it is set as such, and all nodes below it removed. This node itself may be later removed, if node above it is converted to an additive node. Testing indicates a very significant proportion of sub-trees benefit from being aggregated in this way. 

## Inference at Additive Nodes
An additive node is a terminal node where the predictions are based on multiple splits. To make a prediction, when reaching an additive node, a prediction based on each split is made, then these are aggregated to create a single prediction. Multiple aggregation schemes are available for classification and regression. 

Standard, non-additive leaf nodes behave as in any other decision tree, producing an classification estimate based on the majority class in the node's subspace, or a regression estimate based on the the average value in the node's subspace. 

## Evaluation Metrics
To evaluate the effectiveness of the tool we consdidered both accuracy (macro f1-score for classification and normalized root mean squared error (NRMSE) for regression) and interpretability, measured by the size of the tree. Details regarding the complexity metric are included below. 

To evaluate, we compared to standard decision trees, both comparing where both models used default hyperparameters, and where both models used a grid search to estimate the best parameters. [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) was used to collect the datasests and run the tests. This allowed evaluating on a large number of datasets (100 were used) without bias, as the datasets were randomly selected from OpenML. 

Results for classification on 100 datasets:

| Model	|  Avg f1_macro	| Avg. Train-Test Gap |	Avg. Fit Time	Avg. | Complexity | 
| ----	|  ----	| ----	| ---- | ---- | 
| DT	|  0.634	| 0.359	| 0.0172	| 251.893 | 
| ADT	|  0.617	| 0.156	| 3.991	| 39.907 | 

Here 'ADT' refers to Additive Decision Trees. The Train-Test Gap was found subtracting the F1 macro score on test set from that on the train set, and is used to estimate overfitting. ADT models suffered considerably less from over-fitting. 

AdditiveTrees did very similar to standard decision trees with respect to accuracy, though often do better. The complextity is, however, considerably lower. This allows users to understand the models considering fewer overall rules. 

Results over 100 Classification sets: 
![Result Plot](https://github.com/Brett-Kennedy/AdditiveDecisionTree/blob/main/Results/results_18_08_2021_14_26_34_plot.png)

The first plot tracks the 100 datasets on the x-axis, with F1 score (macro) on y-axis. The second tracks the same 100 datasets on the x-axis, and model complexity on the y-axis. 

This shows in, the first plot, model accuracy (higher is better) and, in the second plot, model complexity (lower is better). It can be seen here that, compared to standard decision trees, at least for the 100 random files tested, AdditiveDecisionTrees are competitive in terms of accuracy, and consistently better in terms of complexity (and thus interpretability), though altenative measures of model complexity could be used. 

## Examples
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from AdditiveDecisionTree import AdditiveDecisionTreeClasssifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

adt = AdditiveDecisionTreeClasssifier()
adt.fit(X_train, y_train)
y_pred_test = adt.predict(X_test)
```

## Example Files
Two example files are provided.

[**Simple_Example_Additive_Decision_Tree**](https://github.com/Brett-Kennedy/AdditiveDecisionTree/blob/main/examples/Simple_Example_Additive_Decision_Tree.ipynb)

is a notebook providing some simple examples using the model.

[**Accuracy_Test_Additive_Tree.py**](https://github.com/Brett-Kennedy/AdditiveDecisionTree/blob/main/examples/Accuracy_Test_Additive_Decision_Tree.py) 

is a python file indended to test the accuracy and model complexity of the AdditiveDecisionTrees compared to sklearn Decision Trees, evaluated over 100 datasets, for both classification and regression problems. To provide a fair comparison, tests are performed where both models use default parameters and where both use CV grid search to estimate the optinal parameters. Results for an execution of this file are included in the Results folder.


## Interpretability Metric
The evaluation uses a straightforward approach to measuring the global complexity of models, that is the overall-description of the model (as opposed to local complexity which measures the complexity of explanations for individual rows). For standard decision trees, it simply uses the number of nodes (a common metric, though others are commonly used, for example number of leaf nodes). For additive trees, we do this as well, but for each additive node, count it as many times as there are splits aggregated together at this node. We, therefore, measure the total number of comparisons of feature values to thresholds (the number of splits) regardless if the results are aggregated or not. Future work will consider additional metrics. 
