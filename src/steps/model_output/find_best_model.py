import pandas as pd
from utils import save_info
import os


def find_best_model(files_path, metric, experiment_name):
    models = []
    metrics = []
    response_var = []
    for file in os.listdir(files_path.format(experiment_name)):
        if "results_table" in file:
            model_name = file[:-18]
            data = pd.read_csv(filepath_or_buffer=files_path + file, header=0, sep=",",
                               infer_datetime_format=True, index_col=0)[metric]
            models.append(model_name)
            metrics.append(data.max())
            response_var.append(int(data.idxmax()) + 1)
    best_results = pd.DataFrame({"model": models, metric: metrics, "response_days": response_var})
    save_info(best_results, "dataframe", "models_best_results")

    best_id = best_results[metric].idxmax()
    best_model = best_results.iloc[best_id, 0]
    while "RANDOM FOREST" not in best_model:
        best_results = best_results.drop(best_id)
        best_id = best_results[metric].idxmax()
        best_model = best_results.iloc[best_id, 0]
    best_response = "response_" + str(best_results.iloc[best_id, 2])
    return best_model, best_response
