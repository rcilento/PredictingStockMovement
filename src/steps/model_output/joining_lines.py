import os
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import save_info


def joining_lines(files_path, prefix, metric):
    all_data = []
    for file in os.listdir(files_path):
        print(files_path + file)
        if prefix in file and "results_table" in file:
            model_name = file[:-18]
            data = pd.read_csv(filepath_or_buffer=files_path+file, header=0, sep=",", infer_datetime_format=True)
            data[model_name] = data[metric]
            all_data = data[model_name] if len(all_data) == 0 else pd.concat([all_data, data[model_name]], axis=1)
    to_plot = all_data
    fig, ax = plt.subplots()
    ax.grid()
    ax.set(
            xlabel='Janela de dias da variável resposta',
            ylabel='Valor da Métrica',
            xlim=(0, 105),
            ylim=(-0.05, 1.05)
        )
    ax.legend(loc="lower left")
    to_plot.plot(kind="line", ax=ax)
    save_info(fig, "figure", f"{prefix}_{metric}_joined_lines")
    plt.close(fig=fig)
