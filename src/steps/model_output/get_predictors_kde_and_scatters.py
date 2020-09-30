import seaborn as sns
import matplotlib.pyplot as plt
from utils import save_info
from itertools import combinations


def get_predictors_kde_and_scatters(master, response, predictors):
    master = master.fillna(0)
    for predictor in predictors:
        fig, ax = plt.subplots()
        ax = sns.violinplot(data=master, x=response, y=predictor)
        save_info(fig, "figure", f"violin_{predictor}")
        plt.close(fig)

    for combination in combinations(predictors, 2):

        #fig, ax = plt.subplots()
        #ax = sns.kdeplot(data=master, x=combination[0], y=combination[1], hue=response)
        #save_info(fig, "figure", f"kde_{combination[0]}_{combination[1]}")
        #plt.close(fig)

        fig, ax = plt.subplots()
        ax = sns.scatterplot(data=master, x=combination[0], y=combination[1], hue=response)
        save_info(fig, "figure", f"scatter_{combination[0]}_{combination[1]}_{response}")
        plt.close(fig)
