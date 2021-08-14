# AdditiveDecisionTree

## Summary
An implementation of a decision tree, similar to a standard decision tree, but supporting an additive approach to prediction. Both AdditiveDecitionTreeClassifier and AdditiveDecitionTreeRegressor classes are provided. This tool provides, on the whole, comparable accuracy to standard decision trees, but in many cases provides greater accuracy and/or improved interpretability. As such, it can be a useful tool for generating interpretable models and may be considered an XAI tool. It is not intended to be competitive with approaches such as boosting in terms of accuracy, but simply a tool to generate interpretable models. It can often produce models comparable in accuracy to considerably deeper standard decision trees, so may have a lower overall complexity compared to these. 

This tool addresses some well-known limitations of decision trees, in particular their limited stability and their necessity to split based on fewer and fewer samples lower in the trees. These limitations are typically addressed by ensembling decision trees, either through bagging or boosting, but these result highly uninterpretable, though generally more accurate, models. 

The intuition behind AdditiveDecisionTrees in that often the true function (f(x)), mapping the input X to the target y, is based on conditions and in other cases it is simply a probabalistic funtion where each input feature may be considered independently. The latter case may be modelled better by linear or logistic regressions, which simply predict based on a weighted sum of each independent feature. That is, each relevant feature contributes to the final prediction without consideration of the other features, though interaction features may be created. f(x) may simply be based on a probability distribution associated with each input feature. 

Conversely, linear and logistic regressions do not capture well where there are conditions in the function f(x), while decision trees can model these potentially quite well. It is often not know apriori, if the true f(x) contains conditions, and as such, if it is desirable to repeatedly split the data into subsets and develop a different prediction for each leaf node based entirely on the datapoints within it. 

Note, this is true for regression problems, but often also for classification. Though f(x) is ultimately a set of rules for any classifiction problem, the rules may be independent of each other. 

The approach taken by AdditiveDecisionTrees is to split the dataspace where appropriate and to make an aggregate decision based on numerous potential splits (all standard axis-parallel splits over different input parameters) where this appears most appropriate. 

Don't need to visualize, just to explain. Global explanations with additive trees more complex than standard decision trees, but still manageable, and the accuracy is often higher. As well, typically shorter.

For local explanations, can generally just give 1 or 2 of the splits that contributed to the result. Users can get more if they wish, but can get the gist easily enough.

A form of ensembling, but simple enough it's still interpretable. It's still a single tree. 

the true f(x) may contain conditions or not. It could be that each predictive column is independently predictive (as in the naive Bayes assumption). Or that there are some conditions and some additive properties to the true f(x). Conditions can be viewed as interactions, where how one feature predicts the target is depedent on the value of another column or set of columns. Can cite the papers. 

DTs likely among the most interpretable of models. They do suffer from low stability. And from overfitting unless regularized. one of the major issues with DTs is that as we go to lower nodes, they're splitting on less data, so the decsisions are less statistically sound. As well, the child nodes have even fewer rows so splitting further becomes even more dicy, or if a leaf node, the majority class may be harder to determine. Some approaches such as oblivious trees can also address this. Here, we put the splits near the top where there is typically enough data to make good splits. 

## Algorithm
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

## Methods
- describe the methods, parameters, and return values
