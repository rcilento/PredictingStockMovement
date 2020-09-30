import matplotlib.pyplot as plt
import pandas as pd
from utils import save_info


def analyzing_cross_correlation(a: pd.Series, b: pd.Series, lag_list):
    fig, ax = plt.subplots()
    ax.xcorr(a, b, usevlines=True, maxlags=max(lag_list), normed=True)
    ax.set(
        ylabel='Índice de Correlação',
        xlabel='Lag'
    )
    save_info(fig, "figure", f"{a.name}_{b.name}_xcorr")
    plt.close(fig)
