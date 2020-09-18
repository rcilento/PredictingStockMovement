from utils import StockOperations, matrix_correlation_filter
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info


def creating_master(respose_df, features_df):
    master = pd.concat([respose_df, features_df], axis=1, join="inner")
    save_info(master, "dataframe", "master")
    save_info(master.describe(), "dataframe", "master_describe")
    save_info(master.corr(), "dataframe", "master_corr")
    return master
