import matplotlib.pyplot as plt
import pandas as pd
from src.utils import save_info


def analyzing_auto_correlation(serie: pd.Series, lag_list):
    auto_corr_results = [serie.autocorr(lag=i) for i in lag_list]
    auto_corr_df = pd.DataFrame({"lag": lag_list, "auto_correlation": auto_corr_results})
    save_info(auto_corr_df, "dataframe", f"{serie.name}_autocorrelation_analysis")

    fig, ax = plt.subplots()
    ax.acorr(serie, maxlags=max(lag_list))
    ax.set(
        xlabel=f'Autocorrelation of IBOV {serie.name} price data',
        ylabel='Autocorrelation',
        title='Lag'
    )
    save_info(fig, "figure", f"{serie.name}_auto_corr")
    plt.close(fig)
