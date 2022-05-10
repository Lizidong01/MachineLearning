import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from preprocessing import data_processing
from sklearn.model_selection import cross_val_predict, train_test_split
import sklearn.model_selection as ms


def choice_model(regression_name, X, y, flag=True):
    """
    :param flag: 返回的预测器是否经过训练 默认是经过训练
    :param regression_name: 预测器名称["byes", "Markov", "Linear", "Ridge", "XGB", "polynomial", "cart"]
    :param X: 数据集条件属性
    :param y:数据集决策属性
    :return:训练好的预测器
    """
    regress_list = ["byes", "Linear", "Ridge", "XGB", "polynomial", "cart"]
    regressor = [BayesianRidge(), LinearRegression(), Ridge(alpha=.5), XGBRegressor(n_estimators=100),
                 LinearRegression(), DecisionTreeRegressor()]

    clf = regressor[regress_list.index(regression_name)]
    if flag:
        clf.fit(X, y)

    return clf


def scatter_plot_data(file_name, regressor_name):
    """
    :param file_name: 数据集名称
    :param regressor_name: 分类预测器名称
    :return: 绘制散点图和回归曲线所需数据
    """
    X, y = data_processing(file_name, route_flag=False)  # 数据预处理 读取预测数据 不需要进行降维
    if regressor_name == "Markov":
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # 马尔可夫模型更加适合研究问题是基于序列的，比如时间序列或状态序列；存在两种状态的意义，一种是观测序列，一种是隐藏状态序列。
        clf = GaussianHMM(n_components=3, n_iter=1000)
        clf.fit(x_train)
        predicted = clf.predict(x_test)
        y = y_test
    else:
        if regressor_name == "polynomial":
            poly_reg = PolynomialFeatures(degree=2)
            X = poly_reg.fit_transform(X)
        clf = choice_model(regressor_name, X, y, flag=False)  # 选择模型
        predicted = cross_val_predict(clf, X, y, cv=5)  # 对数据进行交叉验证预测

    return y, predicted


def index_polyline_data(file_name):
    """
    :param file_name: 数据名称
    :return: 预测的指标值 []
    """
    X, y = data_processing(file_name, route_flag=False)
    regress_list = ["byes", "Linear", "Ridge", "XGB", "polynomial", "cart"]
    scores = ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_median_absolute_error", "r2"]

    all_scores = np.ones((len(scores), len(regress_list)))
    for i in range(len(regress_list)):
        if regress_list[i] == "polynomial":
            poly_reg = PolynomialFeatures(degree=2)
            X = poly_reg.fit_transform(X)
        clf = choice_model(regress_list[i], X, y, flag=False)
        for j in range(len(scores)):
            all_scores[j, i] = -ms.cross_val_score(clf, X, y, cv=5, scoring=scores[j]).mean()

    return all_scores



