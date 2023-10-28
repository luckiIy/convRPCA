'''
进行数据集中异常值的检测（怎么感觉这个和无监督分类这么像，而且效果大概率没那么好）
'''
import math

import numpy as np
from matplotlib import pyplot as plt

from utils.dataload import mnist_outlier_1_7
from models.outlier_pursuit import lr_outlier_pursuit, convlr_outlier_pursuit

def mnist_outlier_detection(distibution='random', is_small=True):
    # 读取数据集
    x, y = mnist_outlier_1_7(structure='matrix', ratio=0.1, distribution='ordered', data_num=600)
    # 注意算法中一列为一个样本，所以需要转置
    x = x.T
    # 将数据置入outlier pursuit算法
    pursuit_methods = ['lr', 'conv']

    for pursuit_method in pursuit_methods:
        # 将损坏图像置入outlier pursuit算法
        if pursuit_method == 'conv':
            kernel_size = (5, 5)
            lambda1 = 10 * kernel_size[0] * kernel_size[1] / (math.sqrt(x.shape[0]))
            (L, S) = convlr_outlier_pursuit(x, (5, 5), lambda1=lambda1, max_iter=1000, tol=1e-6, display=True)
        elif pursuit_method == 'lr':
            lambda1 = 10 / (math.sqrt(x.shape[0]))
            (L, S) = lr_outlier_pursuit(x, lambda1=lambda1, max_iter=1000, tol=1e-6, display=True)
        else:
            print("No such pursuit method!")
            exit()
        # 得到矩阵的可视化
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(x, cmap='gray')
        plt.title('Original Image')
        plt.subplot(2, 2, 2)
        plt.imshow(L, cmap='gray')
        plt.title('Low Rank Image')
        plt.subplot(2, 2, 3)
        plt.imshow(S, cmap='gray')
        plt.title('Sparse Image')


        # 计算异常值检测的准确率, 1为正常数据，7为异常数据
        # 统计得到的S各列的l2范数，找到异常值7对应列中l2范数的最小值，同时找到正常值1对应列中l2范数的最大值
        S_norm = np.linalg.norm(S, ord=2, axis=0)

        outlier_min = np.max(S_norm[y == 1])
        normal_max = np.min(S_norm[y == 0])
        if outlier_min > normal_max:
            print("Outlier Detection Success!")

        # 可视化，以各列l2范数为纵坐标值，各列对应的标签y为横坐标值，正常值1为黑色，异常值7为红色
        plt.figure(figsize=(10, 10))
        plt.scatter(range(S_norm.shape[0]), S_norm, c=y, cmap='brg')
        plt.title(pursuit_method)
        plt.show()



if __name__ == '__main__':
    mnist_outlier_detection()
    exit()
