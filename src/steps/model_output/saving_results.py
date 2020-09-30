import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info


def saving_results(response_period_experiment_list, model_name, model_results_dict):
    """Salva resultados obtidos durante experimento"""

    # x = dict((key, str(value)) for key, value in model_results_dict.items())
    # save_info(model_results_dict, "json", f"{model_name}_experiment_dict")

    x = model_results_dict.copy()
    x.pop('feature_importance', None)
    x.pop('coeffs', None)
    results = pd.DataFrame.from_dict(x)
    save_info(results, "dataframe", f"{model_name}_results_table")

    # Salvando feature importances
    if model_results_dict.get("feature_importance"):
        feature_importance = pd.concat(model_results_dict["feature_importance"], axis=1)
        save_info(feature_importance, "dataframe", f"{model_name}_feature_importance")
        to_plot = feature_importance.T.reset_index(drop=True)
        fig, ax = plt.subplots()
        to_plot.plot(kind="line", ax=ax)
        ax.legend(loc='upper left')
        ax.grid()
        save_info(fig, "figure", f"{model_name}_feature_importance")
        plt.close(fig=fig)

    if model_results_dict.get("coeffs"):
        coeffs = pd.concat(model_results_dict["coeffs"], axis=1)
        save_info(coeffs, "dataframe", f"{model_name}_coeffs")


    # Salvando Resultados dos Modelos
    for metric in ["Accuracy", "F1", "Recall", "Precision"]:
        metric_upper = metric.upper()
        fig1, ax1 = plt.subplots()
        ax1.plot(response_period_experiment_list, model_results_dict[f"TRAIN_{metric_upper}"],
                 label=f"Treino", color='blue')
        ax1.plot(response_period_experiment_list, model_results_dict[f"TEST_{metric_upper}"],
                 label=f"Teste", color='red')
        ax1.plot(response_period_experiment_list, model_results_dict[f"TRAIN_HOLDOUT_{metric_upper}"],
                 label=f"Média dos Conjuntos Validação", color='green')

        ax1.set(
            xlabel='Janela de dias da variável resposta',
            ylabel='Valor da Métrica',
            xlim=(0, 105),
            ylim=(-0.05, 1.05)
        )
        ax1.grid()
        ax1.legend(loc="lower left")
        save_info(fig1, "figure", f"{model_name}_{metric}")
        plt.close(fig=fig1)
