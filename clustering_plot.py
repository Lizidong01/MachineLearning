import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyclustering.cluster.clique import clique_visualizer

from clustering_algorithm import clustering_scatter, clustering_performance_index


def plot_clustering(file_name, clustering_name):
    """
    :param file_name: 数据名称
    :param clustering_name: 聚类模型名称
    :return: 聚类散点图
    """
    fig = plt.figure(figsize=(12, 10))
    clustering_result = clustering_scatter(file_name, clustering_name)
    colors = ["olive", "pink", "cyan", "darkred", "forestgreen", "beige", "burlywood",
              "hotpink", "lightskyblue", "mediumpurple"]
    if len(clustering_result) <= 3:
        if len(clustering_result) == 3:
            X, labels, cluster_centers = clustering_result
        else:
            X, labels = clustering_result
        n_clusters_ = len(np.unique(labels))
        for k, col in zip(range(n_clusters_), colors[:n_clusters_]):
            my_members = labels == k
            if len(clustering_result) == 3:
                cluster_center = cluster_centers[k]
            # X[my_members, 0]属于是numpy数组布尔型索引
            plt.plot(X[my_members, 0], X[my_members, 1], "o", markerfacecolor=col,
                     markeredgecolor="k", markersize=6)
            if len(clustering_result) == 3:
                plt.plot(cluster_center[0], cluster_center[1], "*", markerfacecolor=col,
                     markeredgecolor="k", markersize=20)
        plt.title(clustering_name)
        plt.savefig('./results/{}-clustering.jpg'.format(clustering_name))
        img = Image.open(r"./results/{}-clustering.jpg".format(clustering_name))

        #plt.show()
        return img
    else:
        pass
        '''
        X, clique_cluster, noise, cells = clustering_result
        # 显示由算法形成的网格
        clique_visualizer.show_grid(cells, X)
        # 显示聚类结果
        clique_visualizer.show_clusters(X, clique_cluster, noise)  # show clustering results
        '''


def draw_histogram(file_name, score):
    """
    :param file_name: 绘制柱状图聚类数据名称
    :param score: 绘制柱状图指标名称
    :return: 指定指标柱状图
    """
    indicator_scores = ["adjusted_rand_score", "adjusted_mutual_info_score", "homogeneity_score",
                        "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score"]
    all_scores = clustering_performance_index(file_name)
    score_data = all_scores[:, indicator_scores.index(score)]
    clustering_list = ["k_means", "BIRCH", "DBSCAN", "GMM", "OPTICS", "MeanShift"]
    fig = plt.figure(figsize=(12, 10))  # 设置画布大小，防止x轴标签拥挤
    plt.bar(clustering_list, score_data, width=0.6)
    x = np.arange(len(score_data))
    for a, b in zip(x, score_data):  # 标记值
        plt.text(a, b, '%.4f' % b, ha='center', va='bottom', fontsize=14, fontdict={'family': 'Times New Roman'})
    plt.xlabel('Algorithm', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.ylabel('Recall Standard Deviation', fontdict={'family': 'Times New Roman', 'size': 15})
    plt.yticks(FontProperties='Times New Roman', size=14)
    plt.xticks(FontProperties='Times New Roman', size=14)
    plt.title(score, fontdict={'family': 'Times New Roman', 'size': 20})
    plt.savefig('./results/{}-histogram.jpg'.format(score))

    img = Image.open(r"./results/{}-histogram.jpg".format(score))

    #plt.show()
    return img
