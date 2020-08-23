from configuration_file import *
import pandas as pd


from src.steps import ingesting_data, pre_processing, splitting_data, modeling, saving_results

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

ingested_data, experiment_config = ingesting_data(file_path)

pre_processed_data, experiment_config, response_cols, features_cols = pre_processing(
    ingested_data,
    filter_date_greater_than,
    param_dict,
    response_period_experiment_list,
    pearson_r_threshold,
    experiment_config
)
X_train, y_train_df, X_test, y_test_df, experiment_config = splitting_data(pre_processed_data, test_period, features_cols, response_cols, experiment_config)

model_results = modeling(X_train, y_train_df, X_test, y_test_df)

saving_results(experiment_name, response_period_experiment_list, model_results, experiment_config,
               param_dict)

