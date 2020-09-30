import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_info
from itertools import combinations
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
        print(best_model)
        best_results = best_results.drop(best_id)
        best_id = best_results[metric].idxmax()
        best_model = best_results.iloc[best_id, 0]
    best_response = "response_" + str(best_results.iloc[best_id, 2])
    return best_model, best_response


def get_best_predictors(response_var, best_model, experiment_name, top=10):
    feat_imp = pd.read_csv(filepath_or_buffer=f"C:/Users/rodri/PredictingStockMovement/results/{experiment_name}/dataframe/{best_model}_feature_importance.csv",
                           header=0, sep=",", infer_datetime_format=True, index_col=0)[response_var]
    return list(feat_imp.sort_values(ascending=False).head(top).index)


def get_predictors_kde_and_scatters(master, response, predictors):
    master = master.drop("Date", axis=1).fillna(0)
    for predictor in predictors:
        print(master[predictor])
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=master, x=response, y=predictor)
        #ax.set(xlim=(master[predictor].min(), master[predictor].max()), ylim=(0, 1))
        save_info(fig, "figure", f"kde_{predictor}")
        plt.close(fig)

    for combination in combinations(predictors, 2):
        print(combination[0], combination[1])

        fig, ax = plt.subplots()
        ax = sns.kdeplot(data=master, x=combination[0], y=combination[1], hue=response)
        save_info(fig, "figure", f"kde_{combination[0]}_{combination[1]}")
        plt.close(fig)

        fig, ax = plt.subplots()
        ax = sns.scatterplot(data=master, x=combination[0], y=combination[1], hue=response)
        save_info(fig, "figure", f"scatter_{combination[0]}_{combination[1]}_{response}")
        plt.close(fig)


files_path = os.getcwd() + "/results/10/dataframe/"

master = pd.read_csv(filepath_or_buffer="C:/Users/rodri/PredictingStockMovement/results/10/dataframe/master.csv")

best_model, best_response = find_best_model("C:/Users/rodri/PredictingStockMovement/results/10/dataframe/", "TRAIN_ACCURACY", 10)

best_predictors = get_best_predictors(best_response, best_model, 10, top=10)

get_predictors_kde_and_scatters(master, best_response, best_predictors)
