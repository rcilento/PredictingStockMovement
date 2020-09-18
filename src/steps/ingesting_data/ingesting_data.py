import pandas as pd
from src.utils import save_info


def ingesting_data(file_path):
    """Ingere dados de ação"""
    ingested_data = pd.read_csv(filepath_or_buffer=file_path, header=0, sep=",", infer_datetime_format=True)
    save_info(ingested_data, "dataframe", "ingested_data")
    save_info(ingested_data.describe(), "dataframe", "ingested_data_describe")
    save_info(f"Foram ingeridas {len(ingested_data.index)} linhas", "text", "general_info")
    return ingested_data
