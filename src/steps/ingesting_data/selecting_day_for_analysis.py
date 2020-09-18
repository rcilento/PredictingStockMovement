import pandas as pd
from src.utils import save_info


def selecting_day_for_analysis(ingested_data, filter_date_greater_than, only_days_with_volume):
    """
    Realiza o pre processamento dos dados crus ingeridos:
    Gera Features
    """
    # Filtrando data minima para treino
    if filter_date_greater_than:
        ingested_data = ingested_data.loc[ingested_data["Date"] > filter_date_greater_than]

    if only_days_with_volume:
        ingested_data = ingested_data.drop(ingested_data[ingested_data["Volume"].isnull()].index)
        ingested_data = ingested_data[ingested_data["Volume"] > 0]

    ingested_data = ingested_data.reset_index().drop(columns="index")

    save_info(ingested_data, "dataframe", "ingested_data_filtered_days")
    save_info(ingested_data.describe(), "dataframe", "ingested_data_filtered_days_describe")
    save_info(f"Sobraram {len(ingested_data.index)} linhas com filtro seleção de dias", "text", "general_info")

    return ingested_data