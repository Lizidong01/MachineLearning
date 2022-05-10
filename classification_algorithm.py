from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from preprocessing import data_processing
import sklearn.model_selection as ms
import numpy as np


def select_model(classifier_name, X, y, flag=True):
    """
    :param flag: 是否降选择的模型进行训练
    :param classifier_name: 模型名称
    :param X: 训练集条件属性
    :param y: 训练集决策属性
    :return: 训练好的模型
    """
    classify_list = ["KNN", "byes", "GBDT", "cart", "BP", "AdaBoost", "RF", "Logistic"]
    classifier = [KNeighborsClassifier(n_neighbors=7), GaussianNB(), GradientBoostingClassifier(),
                  DecisionTreeClassifier(),
                  MLPClassifier(hidden_layer_sizes=(10,), random_state=10, learning_rate='constant'),
                  AdaBoostClassifier(n_estimators=10), RandomForestClassifier(n_estimators=10),
                  LogisticRegression(solver='liblinear')]
    # 根据需求选择分类器
    clf = classifier[classify_list.index(classifier_name)]
    if flag:
        clf.fit(X, y)  # 训练模型

    return clf


def scatter_data(file_name, classifier_name):
    """
    :param file_name: 数据名称
    :param classifier_name: 模型名称
    :return: 绘制散点图所需要的数据
    """
    # 数据预处理
    X, y = data_processing(file_name, flag=True)

    clf = select_model(classifier_name, X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    return X, y, Z


def radar_data(file_name):
    """
    :param file_name:数据集名称
    :return: 绘制雷达图所需数据 accuracy precision recall f1
    """
    X, y = data_processing(file_name)
    classify_list = ["KNN", "byes", "GBDT", "cart", "BP", "AdaBoost", "RF", "Logistic"]
    scores = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    all_scores = np.ones((len(scores), len(classify_list)))
    for i in range(len(classify_list)):
        clf = select_model(classify_list[i], X, y, flag=False)
        for j in range(len(scores)):
            all_scores[j, i] = ms.cross_val_score(clf, X, y, cv=5, scoring=scores[j]).mean()

    return all_scores


