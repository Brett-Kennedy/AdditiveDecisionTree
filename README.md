# AdditiveDecisionTree

## Summary
This tool provides an implementation of a decision tree, similar to a standard decision tree, but supporting an additive approach to fitting the data. Both AdditiveDecitionTreeClassifier and AdditiveDecitionTreeRegressor classes are provided. 

This tool provides, on the whole, comparable accuracy to standard decision trees, but in many cases provides greater accuracy and/or improved interpretability. As such, it can be a useful tool for generating interpretable models and may be considered an XAI tool. It is not intended to be competitive with approaches such as boosting or neural networks in terms of accuracy, but is simply a tool to generate interpretable models. It can often produce models comparable in accuracy to deeper standard decision trees, while having a lower overall complexity compared to these. 

#### Limitations of Decision Trees
This tool addresses some well-known limitations of decision trees, in particular their limited stability, their necessity to split based on fewer and fewer samples lower in the trees, and their tendency to overfit if not restricted or pruned. These limitations are typically addressed by ensembling decision trees, either through bagging or boosting, but these result highly uninterpretable, though generally more accurate, models. Constructing oblivous decision trees is another approach to mitigate some of these limitations, and also often produces more stable trees. 

Decision trees are considered to be among the more interpretable models, but only where it is possible to construct them to a shallow depth, perhaps to 4 or 5 levels at most. However, decision trees often need to be fairly deep to acheive higher levels of accuracy, which can greatly undermine their interpretability. 

#### Intuition Behind Additive Decision Trees
The intuition behind AdditiveDecisionTrees in that often the true function *(f(x))*, mapping the input X to the target y, is based on logical conditions and in other cases it is simply a probabalistic funtion where each input feature may be considered independently (as with the Naive Bayes assumption). For example the true f(x) may include something to the effect: If a>10 Then: If B<19 Then: y = class X. Or the true f(x) may be of the form: If a>10 Then class X is 10% more likely, regardless of B. And, if b<19 Then class X is 8% more likely, regardless of A. 

Conditions may be viewed as interactions, where how one feature predicts the target is depedent on the value of another column or set of columns. AdditiveDecisionTrees remove the assumption in standard decision trees that f(x) may be best modeled as a set of conditions, but support conditions where the data suggests they exist. 

The case where f(x) is based on the features independently (each feature's relationship to the target is not based on any other feature) may be modelled better by linear or logistic regression or Naive Bayes models, among others, which simply predict based on a weighted sum of each independent feature. That is, each relevant feature contributes to the final prediction without consideration of the other features (though interaction features may be created). f(x), in this case, is simply a probability distribution associated with each input feature. 

Conversely, linear and logistic regressions do not capture well where there are conditions in the function f(x), while decision trees can model these, at least potentially, quite closely. It is usually not know apriori if the true f(x) contains conditions and, as such, if it is desirable to repeatedly split the data into subsets and develop a different prediction for each leaf node based entirely on the datapoints within it. 

Note, this is true for regression problems, but often also for classification. Though f(x) is ultimately a set of rules for any classifiction problem, the rules may be independent of each other, each simply a probability distribution based on one or a small number of features. 

#### Splitting Policy
The approach taken by AdditiveDecisionTrees is to split the dataspace where appropriate and to make an aggregate decisions based on numerous potential splits (all standard axis-parallel splits over different input parameters) where this appears most appropriate. This is done such that the splits appear higher in the tree, where there are larger numbers of samples to base the splits on and they may be found in a more reliable manner, while lower in the tree, where there are less samples to rely on, the decisions are based on a collection of splits, each use the full set of samples in that subset. 

This provides for straight-forward explanations for each row and for the models as a whole, though somewhat more complex than an equally-deep standard decision tree. The explanations for individual rows (known as local explanations) may be presented simply through the corresponding decision paths, as with standard decision trees, but the final nodes may be based on averaging over multiple splits. The maximum number of splits aggregated together is configurable, but 4 or 5 is typically sufficient. In most cases, as well, all splits agree, and only one needs to be presented to the user. And in fact, even where the splits disagree, the majority prediction may be presented as a single split. Therefore, the explanations are usually the same as for standard decision trees, but with shorter decision paths. 

This, then, produces a model where there are a small number of splits, ideally representing the true conditions, if any, in the model, followed by *additive nodes*, which are leaf nodes that average the predictions of multiple splits, providing more robust predictions. This reduces the need to split the data into progressively smaller subsets, each with less statistical significance. 

AdditiveDecisionTrees, therefore, provide a simple form of ensembling, but one that still allows a single, interpretable model, easily supporting both global and local explanations. As it still follows a simple tree structure, contrapositive explanations may be easily generated as well. 

## Algorithm
The algorithm behaves similar to most pruning algorithms, starting at the bottom, at the leaves, and working towards the root node. 

At each node, the accuracy of the tree on the training data is measured given the current split, then again treating this node as an additive node. If the accuracy is higher with this node as an additive node, it is set as such, and all nodes below it removed. This node itself may be removed later, if an ancestor node is converted to an additive node. 

An additive node is a terminal node where the predictions are based on multiple splits. To make a prediction, when reaching an additive node, the prediction based on each split is made, then aggregated to create a single prediction. Multiple aggregation schemes are available for classification and regression. 

## Stats
To evaluate the effectiveness of the tool we consdidered both accuracy (macro f1-score for classificatino and normalized root mean squared error (NRMSE) for regression) and interpretability, measured by the size of the tree. Details regarding the complexity metric are included below. 

To evaluate, we compared to standard decision trees, both comparing where both models used default hyperparameters, and where both models used a grid search to estimate the best parameters. DatasetsEvaluator was used to collect the datasests and run the tests. 100 randomly-selected files were used to compare default hyperparameters. Using the DatasetsEvaluator tool fascilitated testing on a large number of datasets without bias. 

Results for classification on 100 datasets:

| Model	| Feature Engineering Description	| Avg f1_macro	| Avg. Train-Test Gap |	Avg. Fit Time	Avg. | Complexity | 
| ----	| ----	| ----	| ----	| ----	| ---- | 
| DT	| Original Features	| 0.634449504	| 0.359755376	| 0.017208587	| 251.8933333 | 
| DT	| Rotation-based Features	| 0.637339482	| 0.356865398	| 3.18966572	| 187.8866667 | 

AdditiveTrees did very similar to standard decision trees with respect to accuracy, though often do better. The complextity is, however, considerably lower. This allows users to understand the models considering fewer overall rules. 


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

- give quick examples and point to example notebook & .py accuracy test

## Example Files
Two example files are provided.

Simple_Example_Additive_Decision_Tree is a notebook providing some simple examples using the model.

Accuracy_Test_Additive_Tree.py is a python file indended to test the accuracy and model complexity of the AdditiveDecisionTrees compared to sklearn Decision Trees, evaluated over 100 datasets, for both classification and regression problems. To provide a fair comparison, tests are performed where both models use default parameters and where both use CV grid search to estimate the optinal parameters. 


## Interpretability Metric
The evaluation uses a straightforward approach to measuring the global complexity of models, that is the overall-description of the model (as opposed to local complexity which measures the complexity of explanations for individual rows). For standard decision trees, it simply uses the number of nodes (a common metric, though others are commonly used, for example number of leaf nodes). For additive trees, we do this as well, but add a penalty for each additive node, counting it as many times as there are splits aggregated together at this node. We, therefore, measure the total number of comparisons of feature values to thresholds (the number of splits) regardless if the results are aggregated or not. Future work will consider additional metrics. 
