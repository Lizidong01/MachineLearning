import numpy as np
from pyclustering.cluster.clique import clique, clique_visualizer
from sklearn.cluster import KMeans, DBSCAN, OPTICS, MeanShift, estimate_bandwidth, Birch
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, silhouette_score, \
    calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture

from preprocessing import cluster_processing, data_processing


def election_model(clustering_name, X, flag=True):
    """
    :param flag: 聚类之后簇中心的个数 默认是2
    :param clustering_name: 聚类算法名称
    :param X: 聚类的数据
    :return: 聚类模型
    """
    if flag:
        n_clusters = 2
        n_components = 2
    else:
        n_clusters = 5
        n_components = 5
    clustering_list = ["k_means", "BIRCH", "DBSCAN", "GMM", "OPTICS", "CLIQUE", "MeanShift"]

    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    cluster = [KMeans(n_clusters=n_clusters), Birch(n_clusters=n_clusters),
               DBSCAN(eps=0.8, min_samples=30),
               GaussianMixture(n_components=n_components), OPTICS(min_samples=7),
               clique(X, amount_intervals=5, density_threshold=0),
               MeanShift(bandwidth=bandwidth, bin_seeding=True)]

    clt = cluster[clustering_list.index(clustering_name)]
    if clustering_name == "CLIQUE":
        clt.process()
    else:
        clt.fit(X)

    return clt


def clustering_scatter(file_name, clustering_name):
    """
    :param file_name: 数据集名称
    :param clustering_name: 聚类算法名称
    :return: 绘制聚类散点图所需要的数据
    """
    X = cluster_processing(file_name)
    clt = election_model(clustering_name, X, flag=file_name!="page-blocks.csv")
    if clustering_name == "CLIQUE":
        # 聚类中心
        clique_cluster = clt.get_clusters()
        # 被认为是异常值的点（噪点）
        noise = clt.get_noise()
        # CLIQUE形成的网格单元
        cells = clt.get_cells()
        return X, clique_cluster, noise, cells
    elif clustering_name == "OPTICS" or clustering_name == "BIRCH" or clustering_name == "DBSCAN":
        labels = clt.labels_
    else:
        labels = clt.predict(X)  # 聚类的结果
    if clustering_name == "k_means" or clustering_name == "MeanShift":
        cluster_centers = clt.cluster_centers_  # 聚类中心
        return X, labels, cluster_centers

    return X, labels


def clustering_performance_index(file_name):
    """
    :param file_name: 数据集名称
    :return: 聚类的各个指标的数值  不可返回CLIQUE聚类指标
    """
    X, labels_true = data_processing(file_name, flag=True)
    if labels_true.min() == 1:
        labels_true -= 1
    clustering_list = ["k_means", "BIRCH", "DBSCAN", "GMM", "OPTICS", "MeanShift"]
    all_scores = []
    for clt in clustering_list:
        labels_pred = clustering_scatter(file_name, clt)[1]
        indicator_scores = [adjusted_rand_score(labels_true, labels_pred),
                            adjusted_mutual_info_score(labels_true, labels_pred),
                            homogeneity_score(labels_true, labels_pred),
                            silhouette_score(X, labels_pred),
                            calinski_harabasz_score(X, labels_pred),
                            davies_bouldin_score(X, labels_pred)]
        all_scores.append([round(score, 4) for score in indicator_scores])

    return np.array(all_scores)
