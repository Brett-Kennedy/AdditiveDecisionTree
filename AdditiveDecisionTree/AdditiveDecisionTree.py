import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, mean_squared_error
from info_gain import info_gain

# Node types
NOT_SPLIT = -1
CAN_NOT_SPLIT = -2
ADDITIVE_NODE = -100


# Colours used in visualizations, with each class represented by a consistent colour.
tableau_palette_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink",
                        "tab:gray", "tab:olive", "tab:cyan"]


def clean_data(df):
    df = df.fillna(0.0)  # todo: fill as in datasets evaluator. including one-hot encoding cat cols
    df = df.replace([np.inf, -np.inf], 0.0)                
    return df


class InternalTree:
    def __init__(self):
        pass 


class AdditiveDecisionTree(BaseEstimator):
    def __init__(   self, 
                    min_samples_split=8,
                    min_samples_leaf=6, 
                    max_depth=np.inf,
                    verbose_level=0,
                    allow_additive_nodes=True,
                    max_added_splits_per_node=5):
        """
        :param min_samples_split: As with standard decision trees. The minimum number of samples in a node to allow
            further splitting.
        :param min_samples_leaf: As with standard decision trees. The minimum number of samples in that would result
            in a leaf node to allow further splitting.
        :param max_depth: As with standard decision trees. The maximum path length from the root to any leaf node.
        :param verbose_level: Controls the display of output during fitting and predicting.
        :param allow_additive_nodes: If False, behaves similar to standard decision tree.
        :param max_added_splits_per_node: The maximum number of splits that may be included in any additive node.
        """

        # Variables related to the fitting process
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.allow_additive_nodes = allow_additive_nodes
        self.max_added_splits_per_node = max_added_splits_per_node
        
        # Dataframe holding the original and generated features and arrays describing the features
        self.X = None               
        self.y = None               

        # Parallel arrays related to the tree that is generated
        self.tree_ = InternalTree()
        self.tree_.children_left = [CAN_NOT_SPLIT]  # Index of the left child. -2 for leaves.
        self.tree_.children_right = [CAN_NOT_SPLIT] # Index of the right child. -2 for leaves.
        self.tree_.feature = [NOT_SPLIT]            # The feature used for this node. -1 for nodes not yet split. -2 for leaves.
        self.tree_.threshold = [CAN_NOT_SPLIT]      # The threshold used for this node. -2 for leaves.
        self.tree_.indexes = [[]]                   # The row indexes out of the full self.X covered by each node
        self.tree_.depths = [0]                     # The depth of each node
        self.tree_.can_split = [True]               # Indication if the igr can be calculated for this node. Generally True.
        self.tree_.leaf_explanation = [""]          # Indicates the reason the leaf was not split further. Blank for internal nodes and leaves with full purity.
        self.tree_.node_best_threshold_arr = [[]]        
        self.tree_.node_used_feature_arr = [[]]
        self.tree_.node_used_threshold_arr = [[]]
        self.tree_.node_measurement = [CAN_NOT_SPLIT]  # The information gain ratio (IGR) of the split at each node. -2 for leaves.
        self.tree_.node_best_measurement_arr = [[]]
                
        # Logging information
        self.verbose_level = verbose_level

    def __str__(self):
        return (f"min_samples_split: {self.min_samples_split}, min_samples_leaf: {self.min_samples_leaf}, "
                f"max_depth: {self.max_depth}, allow_additive_nodes: {self.allow_additive_nodes}, "
                f"max_added_splits_per_node: {self.max_added_splits_per_node}")

    def check_nodes_arrays(self):
        """
        Test for internal consistency.
        """
        assert len(self.tree_.children_left) == len(self.tree_.children_right)
        assert len(self.tree_.children_left) == len(self.tree_.feature)
        assert len(self.tree_.children_left) == len(self.tree_.threshold)
        assert len(self.tree_.children_left) == len(self.tree_.indexes)
        assert len(self.tree_.children_left) == len(self.tree_.depths)
        assert len(self.tree_.children_left) == len(self.tree_.can_split) 
        assert len(self.tree_.children_left) == len(self.tree_.node_measurement) 
        assert len(self.tree_.children_left) == len(self.tree_.node_best_measurement_arr) 
        assert len(self.tree_.children_left) == len(self.tree_.node_best_threshold_arr) 
        assert len(self.tree_.children_left) == len(self.tree_.node_used_feature_arr) 
        assert len(self.tree_.children_left) == len(self.tree_.node_used_threshold_arr) 

    def fit(self, X, y):
        """
        Fit a model.
        :param X:  matrix of values.
        :param y: target column. May be categorical or numeric.
        :return: self
        """
        # If possible, split a single node, creating two child nodes.
        def split_node(node_idx):
            if (self.tree_.feature[node_idx] != NOT_SPLIT):
                self.log(3, "Node already split. (Features[node_idx] is not -1.). Feature: " + str(self.tree_.feature[node_idx]))
                return False

            self.log(2, "\n\n####################################") 
            self.log(1, "Calling split_node for:", node_idx)                
            
            if len(self.tree_.indexes[node_idx]) <= self.min_samples_split:
                self.log(2, "Too few rows to split further. #rows: ", len(self.tree_.indexes[node_idx]))
                self.tree_.feature[node_idx] = CAN_NOT_SPLIT
                self.tree_.leaf_explanation[node_idx] = "Too few rows to split further."
                return False
            
            if self.tree_.depths[node_idx] >= self.max_depth:
                self.log(2, "Maximum depth reached. Depth: ", self.tree_.depths[node_idx])
                self.tree_.feature[node_idx] = CAN_NOT_SPLIT
                self.tree_.leaf_explanation[node_idx] = "Maximum depth reached."
                return False

            if not self.tree_.can_split[node_idx]:
                self.log(2, "Cannot split this node. (Cannot calculate the igr.)")
                self.tree_.feature[node_idx] = CAN_NOT_SPLIT
                self.tree_.leaf_explanation[node_idx] = "Cannot calculate splitting criteria."
                return False
            
            if self.check_node_complete(node_idx):
                return True

            # Get the set of rows at this node
            X_local = self.X.loc[self.tree_.indexes[node_idx]]
            y_local = self.y.loc[self.tree_.indexes[node_idx]]
            self.log(5, self.tree_.indexes[node_idx])
            self.log(3, "# rows this node: ", len(X_local))
            assert len(X_local) == len(y_local), "Lengths wrong: " + str(len(X_local)) + ", " + str(len(y_local))

            # todo: this should only be necessary for very large datasets, but necessary on my slow computer
            # todo: should be a hyperparameter
            sample_size = 1000
            if len(X_local)>sample_size:
                X_local = X_local.sample(n=sample_size, random_state=0)
                y_local = y_local.loc[X_local.index]
                assert len(X_local) == len(y_local), "Lengths wrong after taking sample: " + str(len(X_local)) + ", " + str(len(y_local))

            self.log(4, "X_local:")
            if self.verbose_level >= 4:
                print("X:")
                print(X_local.head())
                print("y:")
                print(y.head())

            # If the verbose_level is sufficiently high, render plots of each pair of features at this node.
            # todo: move to subclasses
            if self.verbose_level >= 6:
                # todo: use the general method to produce scatter plots
                for c1_idx in range(len(X.columns)-1):
                    for c2_idx in range(c1_idx+1, len(X.columns)):
                        for class_idx in range(len(self.classes_)):
                            class_name = self.classes_[class_idx]
                            idx_arr = y_local.loc[y_local==class_name].index
                            X_curr_class = X_local.loc[idx_arr]                           
                            plt.scatter(X_curr_class[X_curr_class.columns[c1_idx]], 
                                        X_curr_class[X_curr_class.columns[c2_idx]], 
                                        alpha=0.1, 
                                        c=tableau_palette_list[class_idx],
                                        label=self.classes_[class_idx])                            
                        plt.title("Columns "+str(c1_idx)+"-"+str(c2_idx))
                        plt.legend()
                        plt.show()

            # Loop through each column and determine the information gain ratio using that column.
            best_col_idx, best_col_threshold, best_col_measurement, measurement_arr, threshold_arr, good_split_found = \
                self.find_best_col(X_local, y_local, node_idx)

            if not good_split_found:
                return False                 
                
            # Check if this split would result in too few rows in either child
            X_local = self.X.loc[self.tree_.indexes[node_idx]]
            y_local = self.y.loc[self.tree_.indexes[node_idx]]
            attribute_arr = np.where(X_local[self.X.columns[best_col_idx]]<=best_col_threshold, 0, 1)            
            if (attribute_arr.tolist().count(0) < self.min_samples_leaf) or (attribute_arr.tolist().count(1) < self.min_samples_leaf):
                self.tree_.feature[node_idx] = CAN_NOT_SPLIT
                self.log(2, "Split would result in too small child nodes.")
                self.tree_.leaf_explanation[node_idx] = "Split would result in too small child nodes."                
                return True

            # Update this node
            new_left_idx = len(self.tree_.feature)
            new_right_idx = len(self.tree_.feature) + 1
            self.tree_.children_left[node_idx] = new_left_idx
            self.tree_.children_right[node_idx] = new_right_idx
            self.tree_.feature[node_idx] = best_col_idx  
            self.tree_.threshold[node_idx] = best_col_threshold
            self.tree_.node_measurement[node_idx] = best_col_measurement 
            self.tree_.node_best_measurement_arr[node_idx] = measurement_arr 
            self.tree_.node_best_threshold_arr[node_idx] = threshold_arr
            
            # Create nodes for the 2 children
            self.tree_.children_left.extend([-2,-2])
            self.tree_.children_right.extend([-2,-2])
            self.tree_.feature.extend([-1,-1])
            self.tree_.threshold.extend([-2,-2])
            self.tree_.indexes.extend([[],[]])
            new_depth = self.tree_.depths[node_idx]+1
            self.tree_.depths.extend([new_depth, new_depth])
            self.tree_.can_split.extend([True,True])
            self.tree_.leaf_explanation.extend(["",""])
            self.tree_.node_measurement.extend([-2,-2]) 
            self.tree_.node_best_measurement_arr.extend([[],[]]) 
            self.tree_.node_best_threshold_arr.extend([[],[]])
            self.tree_.node_used_feature_arr.extend([[],[]])
            self.tree_.node_used_threshold_arr.extend([[],[]])

            # Set the indexes of the two child nodes
            self.tree_.indexes[new_left_idx] = X_local.iloc[np.where(attribute_arr<=0)[0]].index
            self.tree_.indexes[new_right_idx] = X_local.iloc[np.where(attribute_arr>0)[0]].index

            # Update the subclass-specific stats about the new nodes
            self.update_node_stats(new_left_idx, new_right_idx)

            # Log messages
            self.log(5, "where arr 1:", np.where(attribute_arr<=0)[0]) 
            self.log(5, "where arr 2:", np.where(attribute_arr>0)[0]) 
            self.log(3, "# rows in left child: ", len(np.where(attribute_arr<=0)[0]))
            self.log(3, "# rows in right child: ", len(np.where(attribute_arr>0)[0]))
            self.log(5, "new_left_idx indexes", self.tree_.indexes[new_left_idx])
            self.log(5, "new_right_idx indexes", self.tree_.indexes[new_right_idx])
            
            return True
        
        # Initialize the variables related to the data
        self.X = X 
        self.y = y
        self.init_variables(y)

        # Initialize the variables related to the root node of the tree
        self.tree_.indexes[0] = self.X.index  # The first node contains all rows
        self.init_root(y)

        # Build the tree. Loop through each node until no more nodes can be split. 
        num_nodes_split = 1
        while num_nodes_split > 0:
            num_nodes_split = 0
            for node_idx in range(len(self.tree_.feature)):
                num_nodes_split += split_node(node_idx)                
            self.check_nodes_arrays()

        # Create additive nodes
        if self.allow_additive_nodes:
            self.create_additive_nodes()
            self.remove_stranded_nodes()

        return self

    def output_tree(self):
        print("\n********************************************************")
        print("Generated Tree")
        print("********************************************************")
        print(f"\n# Nodes: {self.get_num_nodes()}")
        print(f"\nLeft Chidren:\n{self.tree_.children_left}")
        print(f"\nRight Chidren:\n{self.tree_.children_right}")
        print(f"\n# Rows: \n{[len(x) for x in self.tree_.indexes]}")
        print(f"\nFeatures:\n{self.tree_.feature}")
        print(f"\nFeatures in additive nodes:\n{self.tree_.node_used_feature_arr}")
        print(f"\nThresholds:\n{self.tree_.threshold}")
        print(f"\nDepths:\n{self.tree_.depths}")
        print(f"\nCan split: \n{self.tree_.can_split}")
        self.subclass_output_tree() # todo: probably have subclasses call super.outputtree()
        print("********************************************************\n")

    def remove_stranded_nodes(self):
        node_reachable_arr = [0]*len(self.tree_.feature)
        node_reachable_arr[0] = 1
        for i in range(len(self.tree_.children_left)):
            node_idx = self.tree_.children_left[i]
            if (node_idx>=0 and node_reachable_arr[i]==1):
                node_reachable_arr[node_idx]=1 
            node_idx = self.tree_.children_right[i]
            if (node_idx>=0 and node_reachable_arr[i]==1):
                node_reachable_arr[node_idx]=1 

        for e in range(len(node_reachable_arr)-1, -1, -1):
            if node_reachable_arr[e] == 0:
                self.tree_.children_left.pop(e)
                self.tree_.children_right.pop(e)
                self.tree_.feature.pop(e)
                self.tree_.threshold.pop(e)
                self.tree_.indexes.pop(e)
                self.tree_.depths.pop(e)
                self.tree_.can_split.pop(e)
                self.tree_.node_measurement.pop(e)
                self.tree_.node_best_measurement_arr.pop(e)
                self.tree_.node_best_threshold_arr.pop(e) 
                self.tree_.node_used_feature_arr.pop(e)
                self.tree_.node_used_threshold_arr.pop(e)
                self.pop_unused_array_elements(e)

        self.check_nodes_arrays()

        # Adjust the indexes of the child nodes
        num_popped_prev = [0]*len(node_reachable_arr)
        count_false = 0 
        for i in range(len(node_reachable_arr)):
            num_popped_prev[i] = count_false
            if node_reachable_arr[i] == 0:
                count_false += 1
        for i in range(len(self.tree_.children_left)):
            left_child = self.tree_.children_left[i]
            if left_child >= 0:
                self.tree_.children_left[i] -= num_popped_prev[left_child]
            right_child = self.tree_.children_right[i]
            if right_child >= 0:
                self.tree_.children_right[i] -= num_popped_prev[right_child]

    def get_num_nodes(self):
        return len(self.tree_.feature)

    def get_model_complexity(self):
        """
        returns global complexity. Count each node as 1, except additive nodes, which count based on the number
        of aggregations.
        todo: also measure avg. local complexity
        """
        complexity = self.get_num_nodes()
        for i, feature_idx in enumerate(self.tree_.feature):
            if feature_idx == ADDITIVE_NODE:
                complexity += len(self.tree_.node_used_feature_arr[i])
        return complexity

    def _predict(self, predict_X, testing_additive=True):  
        header_log_level = 2 if testing_additive==False else 5
        self.log(header_log_level, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        self.log(2, "PREDICT")
        self.log(header_log_level, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        def find_leaf(row):
            """

            :param row: pandas series
            :return:
            """
            self.log(5, "row:", row)
            curr_node = 0
            path = [0]
            while self.tree_.feature[curr_node] >= 0:
                curr_feat = self.tree_.feature[curr_node]
                curr_feat_name = self.X.columns[curr_feat]
                curr_threshold = self.tree_.threshold[curr_node]
                self.log(5, "curr_node: ", curr_node, ", curr_feat: ", curr_feat)
                if row[curr_feat_name] >= curr_threshold:
                    curr_node = self.tree_.children_right[curr_node]
                else:
                    curr_node = self.tree_.children_left[curr_node]
                path.append(curr_node)
            return curr_node, path

        pred_arr = [] 
        decision_path_arr = []
        self.additive_votes = [""]*len(predict_X)
        row_num = 0               
        for row_idx in range(len(predict_X)):
            row = predict_X.iloc[row_idx]
            leaf_idx, path = find_leaf(row)
            pred_arr.append(self.get_prediction(leaf_idx, row, row_idx))
            decision_path_arr.append(path)
            row_num += 1
        return pred_arr, decision_path_arr

    def predict(self, predict_X, testing_additive=False):
        """
        """
        pred, _ = self._predict(predict_X, testing_additive)
        return pred

    # todo: do for regression
    def get_explanations(self, predict_X, y):
        y = pd.Series(y)
        pred, paths = self._predict(predict_X)

        preamble_str = f"Initial distribution of classes: {self.classes_}: {self.class_counts[0]}"
        explanations = [preamble_str]

        for path_idx, path in enumerate(paths):
            if y.iloc[path_idx] == pred[path_idx]:
                correct_indicator = "Correct"
            else:
                correct_indicator = f"Wrong. Correct target value: {y.iloc[path_idx]}"
            expl = "..............................................................."
            expl += f"\nPrediction for row {path_idx}: {pred[path_idx]} -- {correct_indicator}\n"
            expl += "..............................................................."
            expl += f"\nPath: {path}"
            for path_element_idx, node_idx in enumerate(path):
                col_idx = self.tree_.feature[node_idx]
                if col_idx == ADDITIVE_NODE:
                    and_indicator = "\n"
                    if node_idx > 0:
                        and_indicator = "\n\nAND "
                    expl += f"{and_indicator}vote based on: {self.additive_votes[path_idx]}"
                elif col_idx == -2:
                    expl += f"\nwhere the majority class is: {self.get_majority_class(node_idx)}"
                else:
                    col_name = self.X.columns[col_idx]
                    and_indicator = "\n"
                    if node_idx > 0:
                        and_indicator = "\nAND "
                    sign_indicator = "greater than"
                    if (path[path_element_idx+1] == self.tree_.children_left[node_idx]):
                        sign_indicator = "less than"
                    expl += f"\n{and_indicator}{self.X.columns[col_idx]} is {sign_indicator} "
                    expl += f"{self.tree_.threshold[node_idx]} \n    (has value: {predict_X.iloc[path_idx][col_name]})"
                    next_node_idx = path[path_element_idx+1]                    
                    expl += f" --> (Class distribution: {self.class_counts[next_node_idx]}" 
            explanations.append(expl.strip())                

        return explanations

    def log(self, min_verbose_level, *log_str):
        if self.verbose_level >= min_verbose_level:
            m = ""
            for s in log_str:
                m += str(s)
            print(m)        


class AdditiveDecisionTreeClasssifier(AdditiveDecisionTree):
    def __init__(   self, 
                    min_samples_split=8,
                    min_samples_leaf=6, 
                    max_depth=np.inf,
                    verbose_level=0,
                    allow_additive_nodes=True,
                    max_added_splits_per_node=5):
        super().__init__(min_samples_split, min_samples_leaf, max_depth, verbose_level, allow_additive_nodes, max_added_splits_per_node)

        # Summary information about the class distributions at each node
        self.classes_ = []      # The set of unique target classes in y. Used to maintain a consistent order. 
        self.class_counts = []  # Each node contains an array giving the count of each class, in the order defined by self.classes_
        self.class_counts_arr = [[]]

    def init_variables(self, y):
        # Get the unique set of classes in y. Used to maintain a consistent order.
        self.classes_ = list(set(y))  

    def init_root(self, y):
        self.class_counts.append(self.get_class_counts_for_node(y)) 

    def get_class_counts_for_node(self, local_y):
        counts_arr = pd.Series(local_y).value_counts()
        sorted_counts_arr = []
        for c in self.classes_:
            if c in counts_arr:
                sorted_counts_arr.append(counts_arr[c])
            else:
                sorted_counts_arr.append(0)
        return sorted_counts_arr

    def get_majority_class(self, leaf_idx):
        class_arr = self.class_counts[leaf_idx]
        class_idx = class_arr.index(max(class_arr))
        return self.classes_[class_idx]      

    def get_multiple_spits_majority_class(self, node_idx, row, row_idx):  
        display_row = -1
        if row_idx == display_row: print("node_idx: ", node_idx)
        if row_idx == display_row: print("row: ", row)
        if row_idx == display_row: print("Features used: here: ", self.tree_.node_used_feature_arr[node_idx])
        if row_idx == display_row: print("Thresholds used: here: ", self.tree_.node_used_threshold_arr[node_idx])
        if row_idx == display_row: print("self.class_counts_arr: ", self.class_counts_arr)
        class_votes = [0]*len(self.classes_)
        additive_votes_str = ""
        for i in range(len(self.tree_.node_used_feature_arr[node_idx])):
            col_idx = self.tree_.node_used_feature_arr[node_idx][i]
            threshold = self.tree_.node_used_threshold_arr[node_idx][i]
            curr_feat_name = self.X.columns[col_idx]
            if row_idx == display_row: print("i:", i, ", curr_feat_name: ", curr_feat_name)
            if row_idx == display_row: print("self.class_counts_arr[node_idx]: ", self.class_counts_arr[node_idx])
            if row[curr_feat_name] >= threshold:
                class_arr = self.class_counts_arr[node_idx][i][1]
                class_idx = class_arr.index(max(class_arr))
                class_votes[class_idx] += 1     
                if row_idx == display_row: print(curr_feat_name, " Over. class_idx: ", class_idx, " of: ", class_arr)
                additive_votes_str += f"\n  {i+1}: {curr_feat_name} is greater than {threshold} \n     (has value {row[curr_feat_name]}) "
                additive_votes_str += f" --> (class distribution: {self.class_counts_arr[node_idx][i][1]})"
            else:
                class_arr = self.class_counts_arr[node_idx][i][0]
                class_idx = class_arr.index(max(class_arr))
                class_votes[class_idx] += 1 
                if row_idx == display_row: print(curr_feat_name, " Under. class_idx: ", class_idx, " of: ", class_arr)
                additive_votes_str += f"\n  {i+1}: {curr_feat_name} is less than {threshold}\n     (has value {row[curr_feat_name]}) "
                additive_votes_str += f" --> (class distribution: {self.class_counts_arr[node_idx][i][0]})"
        if row_idx == display_row: print("class_votes: ", class_votes)
        highest_vote = np.argmax(class_votes)
        additive_votes_str += f"\nThe class with the most votes is {self.classes_[highest_vote]}"
        if row_idx == display_row:
            print("highest_vote: ", highest_vote)
        self.additive_votes[row_idx] = additive_votes_str
        return self.classes_[highest_vote]

    def check_node_complete(self, node_idx):
        if (0 in self.class_counts[node_idx]) and (self.class_counts[node_idx].count(0) == (len(self.classes_)-1)):
            self.tree_.feature[node_idx] = -2
            self.log(2, "Node has full purity. Making a leaf node.")
            return True

    def update_node_stats(self, new_left_idx, new_right_idx):
        # Set the class counts in the two child nodes # todo put in subclass
        y_left = self.y.loc[self.tree_.indexes[new_left_idx]]
        self.class_counts.append(self.get_class_counts_for_node(y_left))
        y_right = self.y.loc[self.tree_.indexes[new_right_idx]]
        self.class_counts.append(self.get_class_counts_for_node(y_right))

    def find_best_col(self, X_local, y_local, node_idx):
        best_col_idx = -1
        best_col_threshold = -1
        best_col_igr = -1
        igr_arr = []
        threshold_arr = []

        for col_idx in range(len(self.X.columns)):
            X_local = clean_data(X_local)
            stump = tree.DecisionTreeClassifier(random_state=0, max_depth=1) # Used to get thresholds
            stump.fit(X_local[[X_local.columns[col_idx]]].values.reshape(-1,1), y_local)
            threshold = stump.tree_.threshold[0]            
            attribute_arr = np.where(X_local[X_local.columns[col_idx]]<=threshold, 0, 1)                
            igr = info_gain.info_gain_ratio(attribute_arr, y_local)
            self.log(4, "node_idx:", node_idx, ", col_idx:", col_idx, ", igr: ", round(igr,2), ", threshold:", round(threshold,2))                                        
            igr_arr.append(igr)                
            threshold_arr.append(threshold)
            if igr > best_col_igr:
                best_col_idx = col_idx
                best_col_threshold = threshold
                best_col_igr = igr 

        good_split_found = True
        if (len(X_local) < 50) and (best_col_igr < 0.1):
            self.log(2, "Cannot split this node. (igr is too low given the number of rows)  ")
            self.tree_.feature[node_idx] = -2
            self.tree_.leaf_explanation[node_idx] = "igr is too low given the number of rows."
            good_split_found = False

        if best_col_idx == -1:
            self.tree_.can_split[node_idx] = False
            good_split_found = False

        return best_col_idx, best_col_threshold, best_col_igr, igr_arr, threshold_arr, good_split_found

    def create_additive_nodes(self):

        # Potentially replace any non-leaf nodes with additive nodes. Doing this, we do not change the size
        # of the tree or the parallel arrays, though may leave some nodes unreachable. 
        def check_node(node_index):
            used_col = self.tree_.feature[node_index]
            used_igr = self.tree_.node_measurement[node_index]
            good_cols = []
            good_thresholds = []
            for col_idx, igr_val in enumerate(self.tree_.node_best_measurement_arr[node_index]):
                if igr_val > 0.9 * used_igr or igr_val > 0.4: # todo: make hyperparameters
                    good_cols.append(col_idx)
                    good_thresholds.append(self.tree_.node_best_threshold_arr[node_index][col_idx])
            if (len(good_cols)>1):
                # Get the training score given the current tree
                y_pred = self.predict(self.X, testing_additive=True) 
                curr_train_score = f1_score(self.y, y_pred, average='macro')

                # Temporarily replace this node with an additive node
                self.tree_.feature[node_index] = ADDITIVE_NODE
                self.tree_.node_used_feature_arr[node_index] = good_cols 
                self.tree_.node_used_threshold_arr[node_index] = good_thresholds 

                # Set the class counts in the multiple splits of the data here
                X_local = self.X.loc[self.tree_.indexes[node_index]]
                y_local = self.y.loc[self.tree_.indexes[node_index]]
                assert len(X_local) == len(y_local)
                class_counts_arr = []
                for i in range(len(good_cols)):
                    col_idx = good_cols[i]
                    threshold = good_thresholds[i]

                    attribute_arr = np.where(X_local[self.X.columns[col_idx]]>=threshold, 1, 0)            
                    left_indexes = X_local.iloc[np.where(attribute_arr<=0)[0]].index
                    right_indexes = X_local.iloc[np.where(attribute_arr>0)[0]].index

                    y_left = self.y.loc[left_indexes]
                    y_right = self.y.loc[right_indexes]
                    class_counts_arr.append([self.get_class_counts_for_node(y_left), self.get_class_counts_for_node(y_right)])

                self.class_counts_arr[node_index] = class_counts_arr

                # Determine the training score given an additive node here
                y_pred = self.predict(self.X, testing_additive=True)
                updated_train_score = f1_score(self.y, y_pred, average='macro')

                # If the additive node did not improve the accuracy, return the tree
                if updated_train_score < curr_train_score:
                    self.tree_.feature[node_index] = used_col
                    self.tree_.node_used_feature_arr[node_index] = []
                    self.tree_.node_used_threshold_arr[node_index] = []
                else:
                    self.tree_.children_left[node_index] = -2
                    self.tree_.children_right[node_index] = -2

                # Remove once working. check put back right.
                #y_pred = self.predict(self.X) 
                #curr_train_score = f1_score(self.y, y_pred, average='macro')
                #print("  curr_train_score restored: ", curr_train_score)

        self.class_counts_arr = [[]]*len(self.tree_.feature)

        # todo: recode to just go through backwards
        checked_nodes = [False]*len(self.tree_.children_left)
        for i in range(len(self.tree_.feature)):
            if self.tree_.feature[i] < 0:
                checked_nodes[i]=True 
        count_unchecked = checked_nodes.count(False)
        while count_unchecked > 0:
            for i in range(len(checked_nodes)):
                left_child_idx = self.tree_.children_left[i]
                right_child_idx = self.tree_.children_right[i]
                if (checked_nodes[i]==False and checked_nodes[left_child_idx]==True and checked_nodes[right_child_idx]==True):
                    check_node(i)
                    checked_nodes[i]=True
            count_unchecked = checked_nodes.count(False)

    def get_prediction(self, leaf_idx, row, row_idx):
        if self.tree_.feature[leaf_idx] == ADDITIVE_NODE:
            return self.get_multiple_spits_majority_class(leaf_idx, row, row_idx)   
        else:
            return self.get_majority_class(leaf_idx)

    # todo: don't call these 'subclass'
    def subclass_output_tree(self):
        print(f"\nClass counts:\n{self.class_counts}")
        print(f"\nLeaf Class Counts:\n{[bx for ax,bx in zip(self.tree_.feature,self.class_counts) if ax <0]}")
        print(f"\nNode igr: \n{self.tree_.node_measurement}")

    def pop_unused_array_elements(self, e):
        self.class_counts.pop(e)
        self.class_counts_arr.pop(e)


class AdditiveDecisionTreeRegressor(AdditiveDecisionTree):
    def __init__(   self, 
                    min_samples_split=8,
                    min_samples_leaf=6, 
                    max_depth=np.inf,
                    verbose_level=0,
                    allow_additive_nodes=True,
                    max_added_splits_per_node=5):
        super().__init__(min_samples_split, min_samples_leaf, max_depth, verbose_level, allow_additive_nodes, max_added_splits_per_node)
        self.average_y_value = []      # The average y value at each node

    def init_variables(self, y):
        pass

    def init_root(self, y):
        self.average_y_value.append(self.y.mean())

    # Applies only to classification.
    def check_node_complete(self, node_idx):
        return False  # todo: in principle all rows may have identical y value, should check

    def update_node_stats(self, new_left_idx, new_right_idx):
        y_left = self.y.loc[self.tree_.indexes[new_left_idx]]
        self.average_y_value.append(y_left.mean())
        y_right = self.y.loc[self.tree_.indexes[new_right_idx]]
        self.average_y_value.append(y_right.mean())

    def find_best_col(self, X_local, y_local, node_idx):
        best_col_idx = -1
        best_col_threshold = -1
        best_col_mse_gain = 0.0
        mse_gain_arr = []
        threshold_arr = []

        mse_before = mean_squared_error(y_local, [y_local.mean()]*len(y_local))
        for col_idx in range(len(self.X.columns)):
            X_local = clean_data(X_local)
            stump = tree.DecisionTreeRegressor(random_state=0, max_depth=1) # Used to get thresholds
            stump.fit(X_local[[X_local.columns[col_idx]]].values.reshape(-1,1), y_local)
            threshold = stump.tree_.threshold[0]
            over_threshold_bool_arr = X_local[X_local.columns[col_idx]]>threshold

            left_arr = [y_val for y_val,over_bool in zip(y_local, over_threshold_bool_arr) if over_bool == False]
            right_arr = [y_val for y_val,over_bool in zip(y_local, over_threshold_bool_arr) if over_bool == True]
            if len(left_arr)==0 or len(right_arr)==0:
                continue
            
            mean_left = np.array(left_arr).mean()
            mean_right = np.array(right_arr).mean()
            new_pred_y = [mean_left if over_bool==False else mean_right for over_bool in over_threshold_bool_arr]
            mse_after = mean_squared_error(y_local, new_pred_y)
            mse_gain = mse_before - mse_after
            mse_gain_arr.append(mse_gain)                
            threshold_arr.append(threshold)
            if (mse_gain > best_col_mse_gain and len(left_arr) >= self.min_samples_leaf and len(right_arr) >= self.min_samples_leaf):
                best_col_idx = col_idx
                best_col_threshold = threshold
                best_col_mse_gain = mse_gain 

        good_split_found = True
        if (len(X_local) < 50) and ((best_col_mse_gain/mse_before)<0.01):
            # print("case 1")
            self.log(2, "Cannot split this node. (Gain in MSE is too low given the number of rows)  ")
            self.tree_.feature[node_idx] = -2
            self.tree_.leaf_explanation[node_idx] = "Gain in MSE is too low given the number of rows."
            good_split_found = False

        if best_col_idx == -1:
            self.tree_.can_split[node_idx] = False
            good_split_found = False

        return best_col_idx, best_col_threshold, best_col_mse_gain, mse_gain_arr, threshold_arr, good_split_found

    def create_additive_nodes(self):
        # Potentially replace any non-leaf nodes with additive nodes. Doing this, we do not change the size
        # of the tree or the parallel arrays, though may leave some nodes unreachable. 
        def check_node(node_index):
            used_col = self.tree_.feature[node_index]
            used_mse_gain = self.tree_.node_measurement[node_index]
            good_cols = []
            good_thresholds = []
            good_mse_gains = []
            for col_idx, mse_gain_val in enumerate(self.tree_.node_best_measurement_arr[node_index]):
                if mse_gain_val > 0.0 and mse_gain_val > (0.9 * used_mse_gain):
                    good_cols.append(col_idx)
                    good_thresholds.append(self.tree_.node_best_threshold_arr[node_index][col_idx])
                    good_mse_gains.append(mse_gain_val)
            if len(good_cols) > self.max_added_splits_per_node:
                ind = np.argpartition(good_mse_gains, -self.max_added_splits_per_node)[-self.max_added_splits_per_node:]
                good_cols = np.array(good_cols)[ind].tolist()
                good_thresholds = np.array(good_thresholds)[ind].tolist()
                if not used_col in good_cols:
                    good_cols.append(used_col)
                    good_thresholds.append(self.tree_.threshold[node_index])

            if (len(good_cols)>1):
                # Get the training score given the current tree. todo: just get for this part of the tree -- the rest of the tree is the same
                y_pred = self.predict(self.X, testing_additive=True) 
                curr_train_mse = mean_squared_error(self.y, y_pred)

                # Set the average y value in the multiple splits of the data here
                X_local = self.X.loc[self.tree_.indexes[node_index]]
                y_local = self.y.loc[self.tree_.indexes[node_index]]
                assert len(X_local) == len(y_local)
                new_pred_y_arr = []
                new_average_y_arr = []                
                for i in range(len(good_cols)):
                    col_idx = good_cols[i]
                    threshold = good_thresholds[i]
                    over_threshold_bool_arr = X_local[X_local.columns[col_idx]]>threshold
                    left_arr = [y_val for y_val,over_bool in zip(y_local, over_threshold_bool_arr) if over_bool == False]
                    right_arr = [y_val for y_val,over_bool in zip(y_local, over_threshold_bool_arr) if over_bool == True]
                    if len(left_arr) == 0 or len(right_arr)==0:
                        new_average_y_arr.append((np.NaN, np.NaN))
                    else:
                        mean_left = np.array(left_arr).mean()
                        mean_right = np.array(right_arr).mean()
                        new_pred_y = [mean_left if over_bool==False else mean_right for over_bool in over_threshold_bool_arr]
                        new_pred_y_arr.append(new_pred_y)                    
                        new_average_y_arr.append((mean_left, mean_right))

                # Remove any elements with nan's
                for i in range(len(good_cols)-1,-1,-1):
                    if new_average_y_arr[i][0] is np.NaN:
                        good_cols.pop(i)
                        good_thresholds.pop(i)
                        new_average_y_arr.pop(i)

                if len(new_average_y_arr) == 0:
                    return

                # Temporarily replace this node with an additive node
                self.tree_.feature[node_index] = ADDITIVE_NODE
                self.tree_.node_used_feature_arr[node_index] = good_cols 
                self.tree_.node_used_threshold_arr[node_index] = good_thresholds 
                self.average_y_value_arr[node_index] = new_average_y_arr

                final_pred = []
                new_pred_y_arr_np = np.array(new_pred_y_arr)
                new_pred_y_arr_means = new_pred_y_arr_np.mean(axis=1)

                # Determine the training score given an additive node here
                y_pred = self.predict(self.X, testing_additive=True)
                updated_train_mse = mean_squared_error(self.y, y_pred)

                # If the additive node did not improve the accuracy, return the tree
                if updated_train_mse > curr_train_mse:
                    self.tree_.feature[node_index] = used_col
                    self.tree_.node_used_feature_arr[node_index] = []
                    self.tree_.node_used_threshold_arr[node_index] = []
                    self.average_y_value_arr[node_index] = []
                else:
                    self.tree_.children_left[node_index] = -2
                    self.tree_.children_right[node_index] = -2

        self.average_y_value_arr = [[]]*len(self.tree_.feature)

        # todo: recode to just go through backwards
        checked_nodes = [False]*len(self.tree_.children_left)
        for i in range(len(self.tree_.feature)):
            if self.tree_.feature[i] < 0:
                checked_nodes[i]=True 
        count_unchecked = checked_nodes.count(False)
        while count_unchecked > 0:
            for i in range(len(checked_nodes)):
                left_child_idx = self.tree_.children_left[i]
                right_child_idx = self.tree_.children_right[i]
                if (checked_nodes[i]==False and checked_nodes[left_child_idx]==True and checked_nodes[right_child_idx]==True):
                    check_node(i)
                    checked_nodes[i] = True
            count_unchecked = checked_nodes.count(False)

    def get_multiple_spits_average_prediction(self, leaf_idx, row):
        num_features_in_node = len(self.tree_.node_used_feature_arr[leaf_idx])
        average_prediction = 0.0
        num_features_used = 0
        for i in range(num_features_in_node):
            if row[self.tree_.node_used_feature_arr[leaf_idx][i]] <= self.tree_.node_used_threshold_arr[leaf_idx][i]:
                split_prediction = self.average_y_value_arr[leaf_idx][i][0]
            else:
                split_prediction = self.average_y_value_arr[leaf_idx][i][1]
            if not (split_prediction is np.NaN):
                average_prediction += split_prediction
                num_features_used += 1

        return average_prediction / num_features_used

    def get_prediction(self, leaf_idx, row, row_idx):
        if self.tree_.feature[leaf_idx] == ADDITIVE_NODE:
            return self.get_multiple_spits_average_prediction(leaf_idx, row)
        else:
            return self.average_y_value[leaf_idx]

    def subclass_output_tree(self):
        print(f"\nAverage target values:\n{self.average_y_value}")
        if hasattr(self, "average_y_value_arr"):    
            print(f"\nAverage target values in additive nodes:{self.average_y_value_arr}")

    def pop_unused_array_elements(self, e):
        self.average_y_value.pop(e)
        self.average_y_value_arr.pop(e)
