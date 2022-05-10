import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.metrics import auc, plot_roc_curve
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from classification_algorithm import scatter_data, select_model, radar_data
from preprocessing import data_processing
from PIL import Image

def scatter(file_name, classifier_name):
    """
    :param file_name: 数据集名称
    :param classifier_name: 分类算法
    :return: 分类散点图
    """
    # 数据准备
    X, y, Z = scatter_data(file_name, classifier_name)

    # Put the result into a color plot
    cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue", "LightPink", "Beige"])
    cmap_bold = ["darkorange", "c", "darkblue", "red", "green"]
    if np.unique(y).size == 2:
        cmap_light = ListedColormap(["orange", "cyan"])
        cmap_bold = cmap_bold[:2]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(classifier_name)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.savefig('./results/{}-scatter.jpg'.format(classifier_name))
    img = Image.open(r"./results/{}-scatter.jpg".format(classifier_name))
    return img
    #plt.show()


def plot_roc(file_name, classifier_name):
    """
    :param file_name: 数据集名称
    :param classifier_name: 进行分类的算法
    :return: 绘制ROC曲线
    """
    data, label = data_processing(file_name)
    print("counter:", Counter(label))
    # 定义n折交叉验证
    KF = KFold(n_splits=5)
    # data为数据集 利用KF.split划分训练集和测试集
    tprs = []
    aucs = []
    i = 1
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for train_index, test_index in KF.split(data):
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        # 训练模型
        model = select_model(classifier_name, X_train, Y_train)
        # 预测分类
        viz = plot_roc_curve(model, X_test, Y_test,
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        i += 1

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=classifier_name+" ROC curve")
    ax.legend(fontsize=8, loc="lower right")
    plt.savefig('./results/{}-roc.jpg'.format(classifier_name))
    img = Image.open(r"./results/{}-roc.jpg".format(classifier_name))
    return img
    #plt.show()


def radar(data, algorithm, thetas,score):
    """
    :param data: 绘制雷达图所需数据
    :param algorithm: 雷达图所需标签
    :param thetas: 雷达图所需角度
    :return: 展示雷达图
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')  # 创建极坐标的Axes对象

    ax.set_thetagrids(thetas * 180 / np.pi, algorithm, verticalalignment='top', fontsize=10)  # 设置网格标签，单位转化成度数
    ax.plot(thetas, data, 'o-', label=score)
    # 显示数据点的值
    for a, b in zip(thetas, data):
        # plt.text(a, b + 0.05, '%.4f' % b, verticalalignment='center', horizontalalignment='center', rotation=ro)
        plt.text(a, b + 0.13, '%.4f' % b, verticalalignment='center', horizontalalignment='center')

    ax.set_theta_zero_location('N')  # 设置极坐标0°位置
    ax.set_rlim(0, 1.15)  # 设置显示的极径范围

    ax.fill(thetas, data, facecolor='orange', alpha=0.2)  # 填充颜色

    ax.legend(loc=(0.9, 0.9))
    ax.set_rlabel_position(40)  # 设置极径标签位置
    ax.tick_params(pad=8, grid_color='k', grid_alpha=0.2, grid_linestyle=(0, (5, 5)))
    plt.savefig('./results/{}-radar.jpg'.format(score))



def plot_radar(file_name, score):
    """
    :param score: 绘制雷达图的指标类型
    :param file_name: 数据集名称
    :return: 无 展示出雷达图
    """
    data = radar_data(file_name)
    scores = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    acc = data[scores.index(score), :]
    algorithm = ["KNN", "byes", "GBDT", "cart", "BP", "AdaBoost", "RF", "Logistic"]
    theta = np.linspace(0, 2 * np.pi, len(acc), endpoint=False)  # 计算区间角度
    thetas = np.concatenate((theta, [theta[0]]))  # 添加第一个数据，实现闭合
    algorithm = np.concatenate((algorithm, [algorithm[0]]))
    data = np.concatenate((acc, [acc[0]]))  # 添加第一个数据，实现闭合
    radar(data, algorithm, thetas,score)
    img = Image.open(r"./results/{}-radar.jpg".format(score))
    return img
