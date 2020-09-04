import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import functools
import seaborn as sns

def saving_results(experiment_name, response_period_experiment_list, model_results_dict, experiment_config,
                   param_dict):
    """Salva resultados obtidos durante experimento"""

    # Criando diretorios de resultados
    results_path = os.getcwd() + f"/results/{experiment_name}/"
    for dir_name in ["figures/", "describes/"]:
        results_dir_name = results_path + dir_name
        try:
            os.makedirs(results_dir_name)
        except OSError:
            print("Falha na criação do diretório %s" % results_dir_name)

    results_figure_paths = results_path + "figures/"
    results_data_describe = results_path + "describes/"

    # Salvando Resultados dos experimentos
    f = open(results_data_describe + "experiment_config.txt", "w")
    f.write(
        f"O número de linhas filtrado pelo filtro de filter_date_greater_than={experiment_config['filter_date_greater_than']}, foi de {experiment_config['initial_number_of_lines'] - experiment_config['lines_after_filtering_min_data']}")

    f.write(f"Train Period {experiment_config['filter_date_greater_than']} - {experiment_config['test_period']}\n")
    f.write(f"Train Observations {experiment_config['train_size']}\n")
    f.write(f"Test Period {experiment_config['test_period']} - {experiment_config['max_df_date']}\n")
    f.write(f"Test Observations {experiment_config['test_size']}\n")
    f.write(f"Technical Indicators config\n{str(param_dict)}\n")
    f.close()

    x = dict((key, str(value)) for key, value in model_results_dict.items())
    with open(results_data_describe + "experiment_dict.txt", "w") as outfile:
        json.dump(x, outfile)

    experiment_config['describe_dataframe_inicial'].to_csv(
        path_or_buf=results_data_describe + "describe_dataframe_inicial")
    experiment_config['describe_dataframe_after_calculations'].to_csv(
        path_or_buf=results_data_describe + "describe_dataframe_after_calculations")
    experiment_config['describe_X_train'].to_csv(path_or_buf=results_data_describe + "describe_X_train")
    experiment_config['describe_X_test'].to_csv(path_or_buf=results_data_describe + "describe_X_test")
    experiment_config['describe_y_train'].to_csv(path_or_buf=results_data_describe + "describe_y_train")
    experiment_config['describe_y_test'].to_csv(path_or_buf=results_data_describe + "describe_y_test")

    # Salvando feature importances
    feature_importances = []
    i = 1
    for feat_imp_dict in model_results_dict["RANDOM FOREST"]["feature_importance"]:
        names = []
        values = []
        for name, value in feat_imp_dict.items():
            names.append(name)
            values.append(value)
        feature_importances.append(pd.Series(data=values, index=names, name=f"response_{i}").to_frame())
        i += 1
    feature_importance = pd.concat(feature_importances, axis=1)
    feature_importance.to_csv(path_or_buf=results_data_describe + "feature_importances.csv")
    to_plot = feature_importance.T.reset_index(drop=True)
    print(to_plot)
    fig, ax = plt.subplots()
    to_plot.plot(kind="line", ax=ax)
    ax.legend(loc='upper left')
    fig.savefig(results_figure_paths + "feature_importance.png")

    # Salvando Resultados dos Modelos
    for model_name in model_results_dict.keys():
        try:
            os.makedirs(results_figure_paths + f"/{model_name}/")
        except OSError:
            print("Creation of the directory %s failed" % results_figure_paths + f"/{model_name}/")

        f = open(results_path + "model_config.txt", "w")
        f.write(f"{model_name} Configuration {str(model_results_dict[model_name]['model'])}\n")
        f.close()


        for metric in ["Accuracy", "F1", "Recall", "Precision"]:
            metric_upper = metric.upper()
            fig1, ax1 = plt.subplots()
            ax1.plot(response_period_experiment_list, model_results_dict[model_name][f"TRAIN_{metric_upper}"],
                     label=f"Train {metric}", color='blue')
            ax1.plot(response_period_experiment_list, model_results_dict[model_name][f"TEST_{metric_upper}"],
                     label=f"Test {metric}", color='red')

            ax1.set(
                xlabel='Quantidade de dias Seguintes os quais a Variável resposta foi avaliada',
                ylabel='Valor da Métrica',
                title=f'{model_name} - {metric}'
            )
            ax1.grid()
            ax1.legend()
            fig1.savefig(results_figure_paths + f"{model_name}/{metric}.png")


        model_results_dict[model_name].pop('model', None)
        model_results_dict[model_name].pop('best_model', None)
        model_results_dict[model_name].pop('param_grid', None)
        model_results_dict[model_name].pop('feature_importance', None)
        pd.DataFrame.from_dict(model_results_dict[model_name]).to_csv(
            path_or_buf=results_figure_paths + f"{model_name}/results_table.csv")
