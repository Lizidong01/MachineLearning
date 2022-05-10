import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def data_processing(file_name, flag=False, route_flag=True):
    """
    :param route_flag: 选择数据路径
    :param file_name: 数据集名称
    :param flag: 是否对标准化后的数据进行降维
    :return: 返回经标准化(PCA降维)后的数据
    """
    if not route_flag:
        data = pd.read_csv("./regression_data/" + file_name, header=None)
    else:
        data = pd.read_csv("./classified_data/" + file_name, header=None)
    datax = np.array(data.iloc[:, :-1])
    Y = np.array(data.iloc[:, -1])
    # 数据标准化
    sc = StandardScaler()  # 初始化一个对象sc去对数据集作变换
    sc.fit(datax)  # 用对象去拟合数据集X_train，并且存下来拟合参数
    X = sc.transform(datax)
    # 数据降维 降成两维的数据 利用在二维平面上展示
    if flag:
        pca = PCA(n_components=2)  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X = pca.transform(X)  # 获取新矩阵

    return X, Y


def cluster_processing(file_name, flag=True):
    """
    :param flag: 默认数据都是需要降维的
    :param file_name: 需要聚类数据集的名称
    :return: 返回标准化和PCA降维之后的数据
    """
    data = np.array(pd.read_csv("./clustering_data/" + file_name, header=None))
    # 数据标准化
    sc = StandardScaler()  # 初始化一个对象sc去对数据集作变换
    sc.fit(data)  # 用对象去拟合数据集X_train，并且存下来拟合参数
    X = sc.transform(data)

    if flag:
        pca = PCA(n_components=2)  # 实例化
        pca = pca.fit(X)  # 拟合模型
        X = pca.transform(X)  # 获取新矩阵

    return X



