import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from warnings import filterwarnings

# Todo: remove once have pip install
import sys  
sys.path.insert(0, 'C:\python_projects\AdditiveDecisionTree_project\AdditiveDecisionTree') # todo: rename to _project
from AdditiveDecisionTree import AdditiveDecisionTreeClasssifier

sys.path.insert(0, 'C:\python_projects\DatasetsEvaluator_project\DatasetsEvaluator')
import DatasetsEvaluator as de

filterwarnings('ignore')
np.random.seed(0)


# These specify how many datasets are used in the tests below. Ideally about 50 to 100 datasets would be used,
# but these may be set lower. Set to 0 to skip tests. 
NUM_DATASETS_CLASSIFICATION_DEFAULT = 100
NUM_DATASETS_REGRESSION_DEFAULT = 0
NUM_DATASETS_CLASSIFICATION_GRID_SEARCH = 0
NUM_DATASETS_REGRESSION_GRID_SEARCH = 0


def print_header(test_name):
	stars = "*****************************************************"
	print(f"\n\n{stars}\n{test_name}\n{stars}")


def test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder):
	print_header("Classification with default parameters")

	dt = tree.DecisionTreeClassifier(random_state=0)
	edt1 = AdditiveDecisionTreeClasssifier(allow_additive_nodes=False)
	edt2 = AdditiveDecisionTreeClasssifier(allow_additive_nodes=True)

	summary_df, saved_file_name = datasets_tester.run_tests(
		estimators_arr = [
			("DT", "", "Default", dt),
			("EDT1", "", "without additive", edt1),
			("EDT2", "", "with additive", edt2),
			],
		num_cv_folds=3,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=True)

	datasets_tester.summarize_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)


def test_classification_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder):
	# As this takes much longer than testing with the default parameters, we test with fewer datasets. Note though,
	# run_tests_grid_search() uses CV to evaluate the grid search for the best hyperparameters, it does a train-test 
	# split on the data for evaluation, so evaluates the predictions quickly, though with more variability than if
	# using CV to evaluate as well. 

	print_header("Classification with grid search for best parameters")
	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets(
		problem_type="classification",
		min_num_numeric_features=2,
		max_num_numeric_features=10) # Set lower to be faster

	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_GRID_SEARCH,
		exclude_list=exclude_list,
		preview_data=False,
		save_local_cache=True,
		check_local_cache=True,
		path_local_cache=cache_folder)

	orig_parameters = {
		'dt__max_depth': (3,4,5,6)
	}
	rota_parameters = {
		'rota__degree_increment': (3,4,10,15,30),
		'dt__max_depth': (3,4,5,6,100)
	}
	edt_1_parameters = {
		'edt__max_depth': (3,4,5,6)
	}
	edt_2_parameters = {
		'edt__max_depth': (3,4,5,6),
		'edt__degree_increment':(3,4,10,15,30),
		'edt__regularization_constant':(1.5, 2.0, 2.25, 2.5)
	}
	edt_3_parameters = {
		'edt__max_depth': (3,4,5,6),
		'edt__degree_increment':(3,4,10,15,30),
		'edt__fine_tuning_increment':(1,2,3,4,5),		
		'edt__regularization_constant':(1.5, 2.0, 2.25, 2.5)
	}

	orig_pipe = Pipeline([('dt', tree.DecisionTreeClassifier())])
	rota_pipe = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeClassifier())])
	edt1_pipe  = Pipeline([('edt', ExtendedDecisionTreeClasssifier(allow_rotation_features=False))])
	edt2_pipe  = Pipeline([('edt', ExtendedDecisionTreeClasssifier(allow_rotation_features=True, allow_fine_tuning=False))])
	edt3_pipe  = Pipeline([('edt', ExtendedDecisionTreeClasssifier(allow_rotation_features=True, allow_fine_tuning=True))])

	# This provides an example using some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests_grid_search(
		estimators_arr = [
			("DT", "Original Features", "", orig_pipe),
			("DT", "Rotation-based Features", "", rota_pipe),
			("EDT", "Original Features", "no rotation", edt1_pipe),
			("EDT", "Original Features", "with rotation", edt2_pipe),
			("EDT", "Original Features", "with rotation and fine-tuning", edt3_pipe)
			],
		parameters_arr=[orig_parameters, rota_parameters, edt_1_parameters, edt_2_parameters, edt_3_parameters],
		num_cv_folds=3,
		show_warnings=False,
		results_folder=results_folder,
		partial_result_folder=partial_result_folder,
		run_parallel=True)

	datasets_tester.summarize_results(summary_df, 'f1_macro', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'f1_macro', saved_file_name, results_folder)


def test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder):            
	print_header("Regression with default parameters")

	pipe1 = Pipeline([('dt', tree.DecisionTreeRegressor(random_state=0))])
	pipe2 = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeRegressor(random_state=0))])

	summary_df, saved_file_name = datasets_tester.run_tests(
		estimators_arr = [
			("DT", "Original Features", "Default", pipe1),
			("DT", "Rotation-based Features", "Default", pipe2)],
		num_cv_folds=3,
		show_warnings=True,
		results_folder=results_folder,
		partial_result_folder=partial_result_folder,
		run_parallel=True)
	
	datasets_tester.summarize_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)


def test_regression_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder):
	# As this takes much longer than testing with the default parameters, we test with fewer datasets. Note though,
	# run_tests_grid_search() uses CV to evaluate the grid search for the best hyperparameters, it does a train-test 
	# split on the data for evaluation, so evaluates the predictions quickly, though with more variability than if
	# using CV to evaluate as well. 

	print_header("Regression with grid search for best parameters")
	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets(
		problem_type = "regression",
		min_num_numeric_features=2,
		max_num_numeric_features=10)

	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_REGRESSION_GRID_SEARCH,
		exclude_list=exclude_list,
		preview_data=False,
		save_local_cache=True,
		check_local_cache=True,
		path_local_cache=cache_folder)

	orig_parameters = {
		'dt__max_depth': (3,4,5,6,100)
	}

	rota_parameters = {
		'rota__degree_increment': (3,4,10,15,30),
		'dt__max_depth': (3,4,5,6,100)
	}

	orig_pipe = Pipeline([('dt', tree.DecisionTreeRegressor())])
	rota_pipe = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeRegressor())])

	# This provides an example using some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests_grid_search(
		estimators_arr = [
			("DT", "Original Features", "", orig_pipe),
			("DT", "Rotation-based Features", "", rota_pipe)],
		parameters_arr=[orig_parameters, rota_parameters],
		num_cv_folds=3,
		show_warnings=False,
		partial_result_folder=partial_result_folder,
		results_folder=results_folder,
		run_parallel=True)

	datasets_tester.summarize_results(summary_df, 'NRMSE', saved_file_name, results_folder)
	datasets_tester.plot_results(summary_df, 'NRMSE', saved_file_name, results_folder)


def main():
	cache_folder = "c:\\dataset_cache"
	partial_result_folder = "c:\\intermediate_results"
	results_folder = "c:\\results"

	# These are a bit slower, so excluded from some tests
	exclude_list = ["oil_spill", "fri_c4_1000_50", "fri_c3_1000_50", "fri_c1_1000_50", "fri_c2_1000_50", "waveform-5000", 
				"mfeat-zernikemfeat-zernike", "auml_eml_1_b"]

	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets( 
		problem_type="classification",
		min_num_classes=2,
		max_num_classes=20,
		min_num_minority_class=5,
		max_num_minority_class=np.inf,
		min_num_features=0,
		max_num_features=np.inf,
		min_num_instances=500,
		max_num_instances=5_000,
		min_num_numeric_features=2,
		max_num_numeric_features=50,
		min_num_categorical_features=0,
		max_num_categorical_features=50)

	print("Number matching datasets found: ", len(matching_datasets))

	# Note: some datasets may have errors loading or testing. 
	datasets_tester.collect_data(
		max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_DEFAULT,
		exclude_list=exclude_list,
		save_local_cache=True,
		check_local_cache=True,
		path_local_cache=cache_folder)

	test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder)            
	# test_classification_default_parameters_max_four(datasets_tester, partial_result_folder, results_folder)            
	#test_classification_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder)

	# # Collect & test with the regression datasets
	# matching_datasets = datasets_tester.find_datasets( 
	#     problem_type = "regression",
	#     min_num_features = 0,
	#     max_num_features = np.inf,
	#     min_num_instances = 500,
	#     max_num_instances = 5_000,
	#     min_num_numeric_features = 2,
	#     max_num_numeric_features = 50,
	#     min_num_categorical_features=0,
	#     max_num_categorical_features=50)

	# datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_REGRESSION_DEFAULT, 
	# 							 exclude_list=exclude_list, 
 	#                             	 preview_data=False,
 	#                             	 save_local_cache=True,
 	#                             	 check_local_cache=True,
 	#                             	 path_local_cache=cache_folder)

	# test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder)            
	# test_regression_default_parameters_max_four(datasets_tester, partial_result_folder, results_folder)            
	# test_regression_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder)


if __name__ == "__main__":
	main()