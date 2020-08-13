import pandas as pd


def ingesting_data(file_path):
    """Ingere dados de ação"""
    results = {}
    ingested_data = pd.read_csv(filepath_or_buffer=file_path, header=0, sep=",", infer_datetime_format=True)
    results["describe_dataframe_inicial"] = ingested_data.describe()
    results["initial_number_of_lines"] = len(ingested_data.index)
    return ingested_data, results
