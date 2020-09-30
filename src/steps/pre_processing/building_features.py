from utils import StockOperations, matrix_correlation_filter
from src.utils import save_info


def building_features(ingested_data, param_dict):
    stock = (
        ingested_data
        .pipe(StockOperations.name_standardizer, close_col="Close", high_col="High", low_col="Low",
              volume_col="Volume", date_col="Date")
        .pipe(StockOperations.drop_extra_cols)
        .pipe(StockOperations.tech_indicators, param_dict=param_dict)
    )

    # Removendo linhas as quais algum indicador ténico não foi calculado
    features_cols = [col for col in stock.columns if col not in ["Date", "Close", "High", "Low", "Volume"]]
    for col in features_cols:
        stock = stock.pipe(StockOperations.na_solver, column_name=col, mode="drop")

    save_info(stock, "dataframe", "features")
    save_info(stock.describe(), "dataframe", "features_describe")

    stock = stock.set_index("Date")

    return stock
