import numpy as np
import warnings
from classify_plot import scatter, plot_roc, plot_radar


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    plot_roc("heart.csv", "Logistic")
    # plot_radar("heart.csv", "f1_weighted")
