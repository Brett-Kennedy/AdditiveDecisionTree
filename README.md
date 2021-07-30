# AdditiveDecisionTree

## Summary
An implementation of a classifier, similar to a standard decision tree, but allowing as well an additive system.

- if we keep, explain rotataion features
- if we keep, explain arithmetic features

Are few classes of models considered interpretable. Mostly: decision trees, decision tables, rule lists, rule sets, and GAMs. 

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
