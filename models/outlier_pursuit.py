import os
import math
import numpy as np
import scipy
import pprint as pp

from PIL import Image
import matplotlib.pyplot as plt

from models.conv import adj_nd, cconv_nd
from models.svd import SVT

def outlier_pursuit(M, K, max_iter=100, tol=1e-6, display=False):
    """
    :param M: input matrix
    :param K: convolution kernel size (k1, k2, ...)
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :return: convolution low rank matrix L and sparse matrix S
    """
    (m, n) = M.shape
    (k1, k2) = K
    # initialize, 这个初始化第一次Z摆更新了，不过问题不大？，考虑初始化L为M应该也不错
    # L = np.zeros((m, n))
    L = M
    S = np.zeros((m, n))
    Z = np.zeros((m * n, k1 * k2))
    Y1 = np.zeros((m * n, k1 * k2))
    Y2 = np.zeros((m, n))

    k = k1 * k2
    lambda1 = 1 / (math.sqrt(max(m, n)))
    mu = 1.25 / np.linalg.norm(M, ord=2)
    rho = 1.2
    mu_bar = mu * 1e7

    M_norm2 = np.linalg.norm(M, ord=2)
    # iterate
    for i in range(max_iter):
        AkL = cconv_nd(L, K)
        # update Z
        Z = SVT(AkL + Y1 / mu, 1 / mu)
        # update L
        L = (M - S - Y2 / mu + adj_nd(Z - Y1 / mu, M.shape, K)) / (1 + k)
        # update S
        Q = M - L - Y2 / mu
        S = columnwise_shrinkage(Q, lambda1 / mu)
        # update Y1, Y2
        Y1 = Y1 + mu * (AkL - Z)
        Y2 = Y2 + mu * (M - L - S)
        # update mu
        mu = min(mu * rho, mu_bar)
        # check convergence
        diff_Z = np.linalg.norm(AkL - Z, ord=2) / (math.sqrt(k) * M_norm2)
        diff_M = np.linalg.norm(M - L - S, ord=2) / M_norm2
        # 计算||AkL||_*和||S||_2,1，用于展示，后续删除
        AkL_norm = np.linalg.norm(AkL, ord='nuc')
        S_norm = np.sum(np.linalg.norm(S[:, i], ord=2) for i in range(n))
        stopC = max(diff_Z, diff_M)
        if i % 1 == 0 and display:
            # 输出并区分收敛中的误差以及AkL的核范数
            print("iter:", i, "diff_Z:", diff_Z, "diff_M:", diff_M, "AkL_norm:", AkL_norm, "S_norm:", S_norm)
        if stopC < tol:
            print("Converged!")
            print("iter:", i, "diff_Z:", diff_Z, "diff_M:", diff_M, "AkL_norm:", AkL_norm, "S_norm:", S_norm)
            break

    return L, S




def columnwise_shrinkage(M, tau):
    """
    :param M: input matrix
    :param tau: threshold
    :return: columnwise shrinkage
    """
    (m, n) = M.shape
    for i in range(n):
        # ||M[:, i]||_2 > tau时，赋值为(||M[:, i]||_2 - tau) * M[:, i] / ||M[:, i]||_2，否则赋值为0
        norm2 = np.linalg.norm(M[:, i], ord=2)
        M[:, i] = np.where(norm2 > tau, (norm2 - tau) * M[:, i] / norm2, 0)
    return M

if __name__ == '__main__':
    # # 读取一张自然图像
    # # image = scipy.misc.imread(os.path.join(os.path.dirname(__file__), '../datasets/lenna.png'))
    # image = Image.open(os.path.join(os.path.dirname(__file__), '../datasets/lenna.png')).convert('L')
    # image = np.array(image)
    # # 图像以0.1的概率丢失列
    # image_outlier = image.copy()
    # image_outlier[:, np.random.rand(image.shape[1]) < 0.1] = 0
    # # 将损坏图像置入outlier pursuit算法
    # (L, S) = outlier_pursuit(image_outlier, (3, 3), max_iter=1000, tol=1e-6, display=True)
    # # 可视化对比原图像、破损图像、低秩图像、稀疏图像
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.subplot(2, 2, 2)
    # plt.imshow(image_outlier, cmap='gray')
    # plt.title('Corrupted Image')
    # plt.subplot(2, 2, 3)
    # plt.imshow(L, cmap='gray')
    # plt.title('Low Rank Image')
    # plt.subplot(2, 2, 4)
    # plt.imshow(S, cmap='gray')
    # plt.title('Sparse Image')
    # plt.show()
    # 使用自己产生的很小的低秩矩阵测试
    M = np.array([[1, 2, 3, 1, 2, 3],
                    [2, 4, 6, 2, 4, 6],
                    [3, 6, 9, 3, 6, 9]])
    (L, S) = outlier_pursuit(M, (3, 2), max_iter=1000, tol=1e-6, display=True)
    print("L:", L)
    print("S:", S)




