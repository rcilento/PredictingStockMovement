from utils import StockOperations, matrix_correlation_filter
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info


def building_response_vars(ingested_data, response_period_experiment_list):
    stock = (
        ingested_data
        .pipe(StockOperations.name_standardizer, close_col="Close", high_col="High", low_col="Low",
              volume_col="Volume", date_col="Date")
        .pipe(StockOperations.drop_extra_cols)
        .pipe(StockOperations.get_response_variables, list_of_days_forward=response_period_experiment_list)
        .pipe(StockOperations.drop_raw_cols)
        .pipe(StockOperations.na_solver, column_name=f"response_{max(response_period_experiment_list)}", mode="drop")
    )

    response_vars = [col for col in stock.columns if col not in ["Date", "Close", "High", "Low", "Volume"]]

    save_info(stock, "dataframe", "response_vars")
    save_info(stock.describe(), "dataframe", "response_vars_describe")

    stock = stock.set_index("Date")

    return stock[response_vars]
