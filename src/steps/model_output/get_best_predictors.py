import pandas as pd

def get_best_predictors(response_var, best_model, experiment_name, top=10):
    feat_imp = pd.read_csv(filepath_or_buffer=f"C:/Users/rodri/PredictingStockMovement/results/{experiment_name}/dataframe/{best_model}_feature_importance.csv",
                           header=0, sep=",", infer_datetime_format=True, index_col=0)[response_var]
    return list(feat_imp.sort_values(ascending=False).head(top).index)
