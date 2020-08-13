import matplotlib.pyplot as plt
import os
import pandas as pd


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

    experiment_config['describe_dataframe_inicial'].to_csv(
        path_or_buf=results_data_describe + "describe_dataframe_inicial")
    experiment_config['describe_dataframe_after_calculations'].to_csv(
        path_or_buf=results_data_describe + "describe_dataframe_after_calculations")
    experiment_config['describe_X_train'].to_csv(path_or_buf=results_data_describe + "describe_X_train")
    experiment_config['describe_X_test'].to_csv(path_or_buf=results_data_describe + "describe_X_test")
    experiment_config['describe_y_train'].to_csv(path_or_buf=results_data_describe + "describe_y_train")
    experiment_config['describe_y_test'].to_csv(path_or_buf=results_data_describe + "describe_y_test")

    # Salvando Resultados dos Modelos
    for model_name in model_results_dict.keys():
        try:
            os.makedirs(results_figure_paths + f"/{model_name}/")
        except OSError:
            print("Creation of the directory %s failed" % results_figure_paths + f"/{model_name}/")

        f = open(results_path + "model_config.txt", "w")
        f.write(f"{model_name} Configuration {str(model_results_dict[model_name]['model'])}\n")
        f.close()

        print(model_results_dict)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        if "AUC" in list(model_results_dict[model_name].keys()):
            ax1.plot(model_results_dict, model_results_dict[model_name]["AUC"], label="AUC", color='olive')

        if "ACCURACY" in list(model_results_dict[model_name].values()):
            ax1.plot(model_results_dict, model_results_dict[model_name]["ACCURACY"], label="Accuracy", color='orange')

        ax1.set(
            xlabel='Quantidade de dias Seguintes os quais a Variável resposta foi avaliada',
            ylabel='Valor da Métrica',
            title=f'{model_name} - AUC e Accuracy'
        )
        ax1.grid()
        ax1.legend()
        ax2.plot(response_period_experiment_list, model_results_dict[model_name]["F1"], label="F1", color='purple')
        ax2.plot(response_period_experiment_list, model_results_dict[model_name]["RECALL"], label="Recall",
                 color='blue')
        ax2.plot(response_period_experiment_list, model_results_dict[model_name]["PRECISION"], label="Precision",
                 color='red')
        ax2.set(
            xlabel='Quantidade de dias Seguintes os quais a Variável resposta foi avaliada',
            ylabel='Valor da Métrica',
            title=f'{model_name} - F1, Precision e Recall'
        )
        ax2.grid()
        ax2.legend()
        fig1.savefig(results_figure_paths + f"{model_name}/auc_accuracy.png")
        fig2.savefig(results_figure_paths + f"{model_name}/f1_precision_recall.png")

        model_results_dict[model_name].pop('model', None)

        pd.DataFrame.from_dict(model_results_dict[model_name]).to_csv(
            path_or_buf=results_figure_paths + f"{model_name}/results_table")
