import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image

from regression_algorithm import scatter_plot_data, index_polyline_data


def plot_regression_curve(file_name, regressor_name):
    """
    :param file_name: 数据集名称
    :param regressor_name: 回归预测器的名称["byes", "Markov", "Linear", "Ridge", "XGB", "polynomial", "cart"]
    :return: 散点图和回归曲线
    """
    y, predicted = scatter_plot_data(file_name, regressor_name)
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(regressor_name)
    plt.savefig('./results/{}-regression_curve.jpg'.format(regressor_name))
    img = Image.open(r"./results/{}-regression_curve.jpg".format(regressor_name))
    #plt.show()
    return img



def plot_index_polyline(file_name, score):
    """
    :param file_name: 数据集名称
    :param score: 所绘制的指标
    :return: 指标图象
    """
    data = index_polyline_data(file_name)
    scores = ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_median_absolute_error", "r2"]
    score_data = data[scores.index(score), :]
    algorithm = ["byes", "Linear", "Ridge", "XGB", "polynomial", "cart"]

    fig = plt.figure(figsize=(15, 10))
    plt.plot(np.arange(len(algorithm)), score_data, color='coral', marker='*', linestyle='-.')
    for i in range(len(algorithm)):
        plt.text(i - 0.2, score_data[i] + (score_data.max() - score_data.min())/35,
                 algorithm[i]+":"+str(round(score_data[i], 4)), fontsize=10)
    plt.xlabel("regression algorithm")
    plt.ylabel(score)
    plt.savefig("./results/{}index_polyline.jpg".format(score))
    img = Image.open(r"./results/{}index_polyline.jpg".format(score))
    #plt.show()
    return img

