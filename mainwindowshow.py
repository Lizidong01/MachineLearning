# -*- coding: utf-8 -*-
from clustering_algorithm import clustering_performance_index
from clustering_plot import plot_clustering, draw_histogram
import warnings
from classify_plot import scatter, plot_roc, plot_radar
import sys
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow
from regression_plot import plot_regression_curve, plot_index_polyline
from PyQt5.QtCore import pyqtSlot, Qt

##from PyQt5.QtWidgets import   QLabel

##from PyQt5.QtGui import QColor

import numpy as np
warnings.filterwarnings("ignore")
import matplotlib as mpl

##from matplotlib.ticker import NullFormatter

from MainWindow import Ui_MainWindow


class QmyMainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数，创建窗体
        self.ui = Ui_MainWindow()  # 创建UI对象
        self.ui.setupUi(self)  # 构造UI界面
        self.calssify_data="heart.csv"
        self.cluster_data="heart.csv"
        self.regression_data="dee.csv"
        self.setWindowTitle("机器学习算法系统")
        self.setCentralWidget(self.ui.tabWidget)

        ##  黑体：SimHei 宋体：SimSun 新宋体：NSimSun 仿宋：FangSong  楷体：KaiTi
        mpl.rcParams['font.sans-serif'] = ['SimHei']
        mpl.rcParams['font.size'] = 9  # 显示汉字
        mpl.rcParams['axes.unicode_minus'] = False  # 减号unicode编码



    ##  ==========由connectSlotsByName()自动连接的槽函数============

    ##===========page 1 分类============
    @pyqtSlot(bool)  ##显示工具栏
    def on_gBoxHist_toolbar_clicked(self, checked):
        self.ui.widgetHist.setToolbarVisible(checked)

    @pyqtSlot(bool)  ##显示工具栏
    def on_gBoxHist_toolbar_2_clicked(self, checked):
        self.ui.widgetFill.setToolbarVisible(checked)

    @pyqtSlot(bool)  ##显示工具栏
    def on_gBoxHist_toolbar_3_clicked(self, checked):
        self.ui.widgetPie.setToolbarVisible(checked)

    @pyqtSlot(bool)  ## 显示坐标提示
    def on_chkBoxHist_ShowHint_clicked(self, checked):
        self.ui.widgetHist.setDataHintVisible(checked)

    @pyqtSlot()  ## 紧凑布局1
    def on_btnHist_tightLayout_clicked(self):
        self.ui.widgetHist.figure.tight_layout()  # 对所有子图 进行一次tight_layout
        self.ui.widgetHist.redraw()  # 刷新

    @pyqtSlot()  ## 紧凑布局2
    def on_btnHist_tightLayout_2_clicked(self):
        self.ui.widgetFill.figure.tight_layout()  # 对所有子图 进行一次tight_layout
        self.ui.widgetFill.redraw()  # 刷新
    @pyqtSlot()  ## 紧凑布局3
    def on_btnPie_tightLayout_clicked(self):
        self.ui.widgetPie.figure.tight_layout()
        self.ui.widgetPie.redraw()

    @pyqtSlot(bool)  ##显示图例
    def on_chkBoxHist_Legend_clicked(self, checked):
        axesList = self.ui.widgetHist.figure.axes  # 子图列表
        leg = axesList[1].get_legend()
        leg.set_visible(checked)
        self.ui.widgetHist.redraw()

    # =================data_choice================
    def data_choice(self):
        if self.ui.radioButton_3.isChecked():
            self.calssify_data = "page-blocks.csv"
        if self.ui.radioButton_4.isChecked():
            self.calssify_data = "pima.csv"
        if self.ui.radioButton_5.isChecked():
            self.calssify_data = "heart.csv"
        if self.ui.radioButton_6.isChecked():
            self.regression_data = "abalone.csv"
        if self.ui.radioButton_7.isChecked():
            self.regression_data = "weather.csv"
        if self.ui.radioButton_8.isChecked():
            self.regression_data = "dee.csv"
        if self.ui.radioButton_9.isChecked():
            self.cluster_data = "page-blocks.csv"
        if self.ui.radioButton_10.isChecked():
            self.cluster_data = "titanic.csv"
        if self.ui.radioButton_11.isChecked():
            self.cluster_data = "heart.csv"
    #==========绘图-action=============
    #============分类===============
    @pyqtSlot()  ## 显示坐标提示
    def on_pushButton_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "KNN")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():

            img=scatter(self.calssify_data, "KNN")

            ax=self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)

        self.ui.widgetHist.redraw()

    @pyqtSlot()  ## 显示坐标提示
    def on_pushButton_2_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()


        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "byes")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "byes")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()

    @pyqtSlot()  ## 显示坐标提示
    def on_pushButton_6_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        self.ui.widgetHist.redraw()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "GBDT")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "GBDT")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "cart")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "cart")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_7_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "RF")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "RF")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_5_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "AdaBoost")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "AdaBoost")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "BP")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            img=scatter(self.calssify_data, "BP")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_24_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        if self.ui.radioButton.isChecked():
            img=plot_roc(self.calssify_data, "Logistic")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)
        if self.ui.radioButton_2.isChecked():
            self.data_choice()
            img=scatter(self.calssify_data, "Logistic")
            ax = self.ui.widgetHist.figure.add_subplot(111)
            ax.axis("off")

            ax.imshow(img)

        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_27_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()

        img=plot_radar(self.calssify_data, "accuracy")
        ax = self.ui.widgetHist.figure.add_subplot(111)
        ax.axis("off")

        ax.imshow(img)
        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_28_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()
        img=plot_radar(self.calssify_data, "precision_weighted")
        ax = self.ui.widgetHist.figure.add_subplot(111)
        ax.axis("off")

        ax.imshow(img)

        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_29_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()

        img=plot_radar(self.calssify_data, "recall_weighted")
        ax = self.ui.widgetHist.figure.add_subplot(111)
        ax.axis("off")

        ax.imshow(img)

        self.ui.widgetHist.redraw()
    @pyqtSlot()
    def on_pushButton_30_clicked(self):
        self.data_choice()
        self.ui.widgetHist.figure.clear()

        img=plot_radar(self.calssify_data, "f1_weighted")
        ax = self.ui.widgetHist.figure.add_subplot(111)
        ax.axis("off")

        ax.imshow(img)

        self.ui.widgetHist.redraw()

#==========regression_system====================
    @pyqtSlot()
    def on_pushButton_8_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "byes")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_9_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "Markov")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_10_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "Linear")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_12_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "Ridge")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_11_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "XGB")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_13_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "polynomial")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_14_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_regression_curve(self.regression_data, "cart")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_15_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_index_polyline(self.regression_data, "neg_mean_absolute_error")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_25_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_index_polyline(self.regression_data, "neg_mean_squared_error")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_31_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_index_polyline(self.regression_data, "neg_median_absolute_error")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()

    @pyqtSlot()
    def on_pushButton_32_clicked(self):
        self.data_choice()
        self.ui.widgetFill.figure.clear()
        img = plot_index_polyline(self.regression_data, "r2")
        ax = self.ui.widgetFill.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetFill.redraw()
#===================聚类=======================
    @pyqtSlot()
    def on_pushButton_16_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        print(self.cluster_data)
        img = plot_clustering(self.cluster_data, "k_means")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_17_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        if self.cluster_data=="heart.csv":
            img = Image.open(r"./results/Figure_1.jpg")
        if self.cluster_data=="titanic.csv":
            img = Image.open(r"./results/Figure_3.jpg")
        if self.cluster_data == "page-blocks.csv":
            img = Image.open(r"./results/Figure_4.jpg")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_18_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = plot_clustering(self.cluster_data, "DBSCAN")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_19_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = plot_clustering(self.cluster_data, "GMM")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_20_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()

        img = plot_clustering(self.cluster_data, "OPTICS")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_22_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = Image.open(r"./results/Figure_2.jpg")
        if self.cluster_data=="heart.csv":
            img = Image.open(r"./results/Figure_2.jpg")
        if self.cluster_data=="titanic.csv":
            img = Image.open(r"./results/Figure_6.jpg")
        if self.cluster_data == "page-blocks.csv":
            img = Image.open(r"./results/Figure_5.jpg")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_23_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = plot_clustering(self.cluster_data, "MeanShift")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_21_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "adjusted_rand_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_26_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "adjusted_mutual_info_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_33_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "homogeneity_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_34_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "silhouette_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()

    @pyqtSlot()
    def on_pushButton_35_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "calinski_harabasz_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()
    #image_show
    @pyqtSlot()
    def on_pushButton_36_clicked(self):
        self.data_choice()
        self.ui.widgetPie.figure.clear()
        img = draw_histogram(self.cluster_data, "davies_bouldin_score")
        ax = self.ui.widgetPie.figure.add_subplot(111)
        ax.axis("off")
        ax.imshow(img)

        self.ui.widgetPie.redraw()



##  ============窗体测试程序 ================================
if __name__ == "__main__":  # 用于当前窗体测试
    app = QApplication(sys.argv)  # 创建GUI应用程序
    form = QmyMainWindow()  # 创建窗体
    form.show()
    sys.exit(app.exec_())
