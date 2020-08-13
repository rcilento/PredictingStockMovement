from utils import StockOperations
import numpy as np


def pre_processing(ingested_data, filter_date_greater_than, param_dict, response_period_experiment_list,
                   pearson_r_threshold, experiment_config):
    """
    Realiza o pre processamento dos dados crus ingeridos:
    Gera Features
    Dropa exclui dadas atipicas

    """
    # Filtrando data minima para treino
    if filter_date_greater_than:
        ingested_data = ingested_data.loc[ingested_data["Date"] > filter_date_greater_than]
        experiment_config["filter_date_greater_than"] = filter_date_greater_than

    experiment_config["lines_after_filtering_min_data"] = len(ingested_data.index)

    # Operações de geração de variáveis resposta e features
    stock = (
        ingested_data
        .pipe(StockOperations.name_standardizer, close_col="Close", high_col="High", low_col="Low",
              volume_col="Volume",
              date_col="Date")
        .pipe(StockOperations.drop_extra_cols)
        .pipe(StockOperations.na_solver, column_name="Volume", mode="drop")
        .pipe(StockOperations.get_response_variables, list_of_days_forward=response_period_experiment_list)
        .pipe(StockOperations.tech_indicators, param_dict=param_dict)
        .pipe(StockOperations.drop_raw_cols)
        .pipe(StockOperations.na_solver, column_name=f"response_{max(response_period_experiment_list)}", mode="drop")
    )

    # Removendo linhas as quais algum indicador ténico não foi calculado
    features_cols = [col for col in stock.columns if "response" not in col and "Date" not in col]
    for col in features_cols:
        stock = stock.pipe(StockOperations.na_solver, column_name=col, mode="drop")

    stock = stock.reset_index().drop(columns="index")
    experiment_config["max_df_date"] = stock["Date"].max()

    experiment_config["describe_dataframe_after_calculations"] = stock.describe()

    # Filtrando features linearmente correlacionadas (redundantes)
    corr_matrix = stock[features_cols].corr("pearson").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Encontrando o index da variável a qual o valor de r é maior que o threshold
    to_drop = [column for column in upper.columns if any(upper[column] > pearson_r_threshold)]

    response_cols = [col for col in stock.columns if col not in features_cols and "Date" not in col]
    features_cols = [col for col in features_cols if col not in to_drop]

    return [stock, experiment_config, response_cols, features_cols]
