# AdditiveDecisionTree

## Summary
This tool provides an implementation of a decision tree, similar to a standard decision tree, but supporting an additive approach to fitting the data. Both AdditiveDecitionTreeClassifier and AdditiveDecitionTreeRegressor classes are provided. 

This tool provides, on the whole, comparable accuracy to standard decision trees, but in many cases provides greater accuracy and/or improved interpretability. As such, it can be a useful tool for generating interpretable models and may be considered an XAI tool. It is not intended to be competitive with approaches such as boosting or neural networks in terms of accuracy, but is simply a tool to generate interpretable models. It can often produce models comparable in accuracy to deeper standard decision trees, while having a lower overall complexity compared to these. 

#### Limitations of Decision Trees
This tool addresses some well-known limitations of decision trees, in particular their limited stability, their necessity to split based on fewer and fewer samples lower in the trees, and their tendency to overfit if not restricted or pruned. These limitations are typically addressed by ensembling decision trees, either through bagging or boosting, but these result highly uninterpretable, though generally more accurate, models. Constructing oblivous decision trees is another approach to mitigate some of these limitations, and also often produces more stable trees. 

Decision trees are considered to be among the more interpretable models, but only where it is possible to construct them to a shallow depth, perhaps to 4 or 5 levels at most. However, decision trees often need to be quite deep to acheive higher levels of accuracy, which can greatly undermine their interpretability. 

#### Intuition Behind Additive Decision Trees
The intuition behind AdditiveDecisionTrees in that often the true function *(f(x))*, mapping the input X to the target y, is based on conditions and in other cases it is simply a probabalistic funtion where each input feature may be considered independently (as with the Naive Bayes assumption). AdditiveDecisionTrees remove the assumption in standard decision trees that f(x) may be best modeled as a set of conditions. Note that conditions may be viewed as interactions, where how one feature predicts the target is depedent on the value of another column or set of columns.

The case where f(x) is based on the features independently (each feature's relationship to the target is not based on any other feature) may be modelled better by linear or logistic regression models, which simply predict based on a weighted sum of each independent feature. That is, each relevant feature contributes to the final prediction without consideration of the other features (though interaction features may be created). f(x), in this case, is simply a probability distribution associated with each input feature. 

Conversely, linear and logistic regressions do not capture well where there are conditions in the function f(x), while decision trees can model these, at least potentially, quite closely. It is usually not know apriori if the true f(x) contains conditions and, as such, if it is desirable to repeatedly split the data into subsets and develop a different prediction for each leaf node based entirely on the datapoints within it. 

Note, this is true for regression problems, but often also for classification. Though f(x) is ultimately a set of rules for any classifiction problem, the rules may be independent of each other, each simply a probability distribution based on one or a small number of features. 

#### Splitting Policy
The approach taken by AdditiveDecisionTrees is to split the dataspace where appropriate and to make an aggregate decisions based on numerous potential splits (all standard axis-parallel splits over different input parameters) where this appears most appropriate. This is done such that the splits appear higher in the tree, where there are larger numbers of samples to base the splits on and they may be found in a more reliable manner, while lower in the tree, where there are less samples to rely on, the decisions are based on a collection of splits, each use the full set of samples in that subset. 

This provides for straight-forward explanations for each row and for the models as a whole, though somewhat more complex than an equally-deep standard decision tree. The explanations for individual rows (known as local explanations) may be presented simply through the corresponding decision paths, as with standard decision trees, but the final nodes may be based on averaging over multiple splits. The maximum number of splits aggregated together is configurable, but 4 or 5 is typically sufficient. In most cases, as well, all splits agree, and only one needs to be presented to the user. And in fact, even where the splits disagree, the majority prediction may be presented as a single split. Therefore, the explanations are usually the same as for standard decision trees, but with shorter decision paths. 

This, then, produces a model where there are a small number of splits, ideally representing the true conditions, if any, in the model, followed by *additive nodes*, which are leaf nodes that average the predictions of multiple splits, providing more robust predictions. This reduces the need to split the data into progressively smaller subsets, each with less statistical significance. 

AdditiveDecisionTrees, therefore, provide a simple form of ensembling, but one that still allows a single, interpretable model, easily supporting both global and local explanations. As it still follows a simple tree structure, contrapositive explanations may be easily generated as well. 

## Algorithm
The algorithm behaves similar to most pruning algorithms, starting at the bottom, at the leaves, and working towards the root node. 

start at bottom, check if both children are either already considered or a leaf. 
check accuracy of tree
try replacing this node with a single additive node & check accuracy that way
keep if higher, restore otherwise

to predict: when get to additive node, split on all ways, combine the results. Explain the averaging.

## Stats
- explain the DatasetsEvaluator project -- large numbers, no bias, no cherry picking
- give stats about accuracy, overall complexity -- need a metric. 
- give stats about how many regular split nodes & how many additive nodes tend to end up with doing this


## Examples
- give quick examples and point to example notebook & .py accuracy test

## Example Files
Two example files are provided.

Simple_Example_Additive_Decision_Tree is a notebook providing some simple examples using the model

Accuracy_Test_Additive_Tree.py is a python file indended to test the accuracy of the AdditiveDecisionTrees compared to sklearn Decision Trees, evaluated over 100 datasets, for both classification and regression problems. To provide a fair comparison, tests are performed where both models use default parameters and where both use CV grid search to estimate the optinal parameters. 


## Methods
- describe the methods, parameters, and return values

## Interpretability Metric
-- describe
