import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info


def saving_results(response_period_experiment_list, model_name, model_results_dict):
    """Salva resultados obtidos durante experimento"""

    # x = dict((key, str(value)) for key, value in model_results_dict.items())
    save_info(model_results_dict, "json", "experiment_dict")

    x = model_results_dict.copy()
    x.pop('feature_importance', None)
    results = pd.DataFrame.from_dict(x)
    save_info(results, "dataframe", f"{model_name}_results_table")

    # Salvando feature importances
    if model_results_dict.get("feature_importance"):
        feature_importances = []
        i = 1
        for feat_imp_dict in model_results_dict["feature_importance"]:
            names = []
            values = []
            for name, value in feat_imp_dict.items():
                names.append(name)
                values.append(value)
            feature_importances.append(pd.Series(data=values, index=names, name=f"response_{i}").to_frame())
            i += 1
        feature_importance = pd.concat(feature_importances, axis=1)
        save_info(feature_importance, "dataframe", f"{model_name}_feature_importance")
        to_plot = feature_importance.T.reset_index(drop=True)
        fig, ax = plt.subplots()
        to_plot.plot(kind="line", ax=ax)
        ax.legend(loc='upper left')
        save_info(fig, "figure", f"{model_name}_feature_importance")
        plt.close(fig=fig)

    # Salvando Resultados dos Modelos
    for metric in ["Accuracy", "F1", "Recall", "Precision"]:
        metric_upper = metric.upper()
        fig1, ax1 = plt.subplots()
        ax1.plot(response_period_experiment_list, model_results_dict[f"TRAIN_{metric_upper}"],
                 label=f"Train {metric}", color='blue')
        ax1.plot(response_period_experiment_list, model_results_dict[f"TEST_{metric_upper}"],
                 label=f"Test {metric}", color='red')

        ax1.set(
            xlabel='Quantidade de dias Seguintes os quais a Variável resposta foi avaliada',
            ylabel='Valor da Métrica',
            title=f'{model_name} - {metric}'
        )
        ax1.grid()
        ax1.legend()
        save_info(fig1, "figure", f"{model_name}_{metric}")
        plt.close(fig=fig1)
