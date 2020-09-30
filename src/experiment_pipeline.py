from configuration_file import *
import os
import pandas as pd
from utils import save_info, StockOperations
from steps import ingesting_data, splitting_data, train_and_evaluate_model, saving_results, joining_lines, \
    selecting_day_for_analysis, analyzing_auto_correlation, building_features, filtering_redundant_features, \
    building_response_vars, creating_master, analyzing_cross_correlation, find_best_model, get_best_predictors, \
    get_predictors_kde_and_scatters


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


save_info(PARAM_DICT, "json", "experiment_config")


ingested_data = ingesting_data(FILE_PATH)

filtered_ingested_data = selecting_day_for_analysis(ingested_data, FILTER_DATE_GREATER_THAN, ONLY_DAYS_WITH_VOLUME)

analyzing_auto_correlation(filtered_ingested_data["Close"], RESPONSE_PERIOD_EXPERIMENT_LIST)

all_features = building_features(filtered_ingested_data, PARAM_DICT)

for col in all_features.columns:
    if col not in ["Close", "Date"]:
        analyzing_cross_correlation(all_features["Close"], all_features[col], RESPONSE_PERIOD_EXPERIMENT_LIST)

# Eliminas as colunas iniciais e deixa só as colunas que usaremos na modelagem
all_features = StockOperations.drop_raw_cols(all_features)

filtered_features = filtering_redundant_features(all_features, PEARSON_R_THRESHOLD)

response_vars_df = building_response_vars(filtered_ingested_data, RESPONSE_PERIOD_EXPERIMENT_LIST)

master = creating_master(response_vars_df, filtered_features)

X_train, y_train_df, X_test, y_test_df = splitting_data(master)

for model_name, predictor in PREDICTORS.items():
    calculate_feature_importance = True if "RANDOM FOREST" in model_name else False
    calculate_coeffs = True if "Dummy" not in model_name else False

    model_results = train_and_evaluate_model(X_train, y_train_df, X_test, y_test_df, predictor,
                                             calculate_feature_importance=calculate_feature_importance,
                                             calculate_coeffs=calculate_coeffs)
    saving_results(RESPONSE_PERIOD_EXPERIMENT_LIST, model_name, model_results)

files_path = os.getcwd() + "/results/{}/dataframe/".format(EXPERIMENT_NAME)
joining_lines(files_path, "SVM", "TRAIN_HOLDOUT_ACCURACY")
joining_lines(files_path, "RANDOM FOREST", "TRAIN_HOLDOUT_ACCURACY")
joining_lines(files_path, "Dummy Classifier", "TRAIN_HOLDOUT_ACCURACY")

best_model, best_response = find_best_model(files_path, "TRAIN_HOLDOUT_ACCURACY", EXPERIMENT_NAME)

best_predictors = get_best_predictors(best_response, best_model, EXPERIMENT_NAME, top=10)

get_predictors_kde_and_scatters(master, best_response, best_predictors)
