from utils import StockOperations, matrix_correlation_filter
import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info



def filtering_redundant_features(features, pearson_r_threshold):
    features_cols = [col for col in features.columns if col not in ["Date"]]
    filtered_features = matrix_correlation_filter(features[features_cols], y=None, method="pearson", min_period=1,
                                                  threshold=pearson_r_threshold)

    filtered_cols = [col for col in features_cols if col not in features.columns]

    return filtered_features