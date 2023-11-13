import os
import math
import numpy as np
import scipy
import pprint as pp

from PIL import Image
import matplotlib.pyplot as plt

from models.conv import adj_nd, cconv_nd
from models.svd import SVT


def convlr_outlier_pursuit(M, K, lambda1=0, max_iter=100, tol=1e-6, display=False):
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
    L = M.copy()
    S = np.zeros((m, n))
    Z = np.zeros((m * n, k1 * k2))
    Y1 = np.zeros((m * n, k1 * k2))
    Y2 = np.zeros((m, n))

    k = k1 * k2
    if lambda1 == 0:
        lambda1 = 1 * k / (math.sqrt(m))
    M_norm2 = np.linalg.norm(M, ord=2)
    # 对两个尺度不同的增广项引入不同的惩罚系数
    mu1 = 1 / (k * M_norm2) * 1e-3
    mu2 = 1 / M_norm2 * 1e-3

    rho = 1.05
    mu1_bar = mu1 * 1e10
    mu2_bar = mu2 * 1e10

    # iterate
    for i in range(max_iter):
        AkL = cconv_nd(L, K)
        # update Z
        Z = SVT(AkL + Y1 / mu1, 1 / mu1)
        # update L
        L = (M - S - Y2 / mu2 + adj_nd(Z - Y1 / mu1, M.shape, K)) / (1 + k)
        # update S
        Q = M - L - Y2 / mu2
        S = columnwise_shrinkage(Q, lambda1 / mu2)
        # update Y1, Y2
        Y1 = Y1 + mu1 * (AkL - Z)
        Y2 = Y2 + mu2 * (L + S - M)
        # update mu
        mu1 = min(mu1 * rho, mu1_bar)
        mu2 = min(mu2 * rho, mu2_bar)
        # check convergence
        diff_Z = np.linalg.norm(AkL - Z, ord=2) / (math.sqrt(k) * M_norm2)
        diff_M = np.linalg.norm(L + S - M, ord=2) / M_norm2
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





def lr_outlier_pursuit(M, lambda1=0, max_iter=100, tol=1e-6, display=False):
    """
    :param M: input matrix
    :param max_iter: maximum number of iterations
    :param tol: tolerance
    :return: low rank matrix L and sparse matrix S
    """
    (m, n) = M.shape
    # initialize, 这个初始化第一次Z摆更新了，不过问题不大？，考虑初始化L为M应该也不错
    L = M.copy()
    # L = np.zeros((m, n))
    S = np.zeros((m, n))

    Y = np.zeros((m, n))
    if lambda1 == 0:
        lambda1 = 1 / (math.sqrt(m))
    M_norm2 = np.linalg.norm(M, ord=2)
    mu = 1 / M_norm2
    rho = 1.3
    mu_bar = mu * 1e10

    # iterate
    for i in range(max_iter):
        # update L
        L = SVT(M - S - Y / mu, 1 / mu)
        # update S
        S = columnwise_shrinkage(M - L - Y / mu, lambda1 / mu)
        # update Y
        Y = Y + mu * (L + S - M)
        # update mu
        mu = min(mu * rho, mu_bar)
        # check convergence
        diff_M = np.linalg.norm(L + S - M, ord=2) / M_norm2
        stopC = diff_M
        # 计算||L||_*和||S||_2,1，用于展示，后续删除
        L_norm = np.linalg.norm(L, ord='nuc')
        S_norm = np.sum(np.linalg.norm(S[:, i], ord=2) for i in range(n))
        if i % 1 == 0 and display:
            print("iter:", i, "diff_M:", diff_M, "L_norm:", L_norm, "S_norm:", S_norm)
        if stopC < tol:
            print("Converged!")
            print("iter:", i, "diff_M:", diff_M, "L_norm:", L_norm, "S_norm:", S_norm)
            break
    return L, S

def columnwise_shrinkage(M, tau):
    """
    :param M: input matrix
    :param tau: threshold
    :return: columnwise shrinkage
    """
    X = M.copy()
    (m, n) = M.shape
    for i in range(n):
        # ||M[:, i]||_2 > tau时，赋值为(||M[:, i]||_2 - tau) * M[:, i] / ||M[:, i]||_2，否则赋值为0
        norm2 = np.linalg.norm(M[:, i], ord=2)
        if norm2 > tau:
            X[:, i] = (norm2 - tau) / norm2 * M[:, i]
        else:
            X[:, i] = 0
    return X



if __name__ == '__main__':

    exit()
    # 使用自己产生的很小的低秩矩阵测试
    # M = np.array([[1, 2, 3, 1, 2, 3],
    #                 [2, 4, 6, 2, 4, 6]]).T
    # (L, S) = outlier_pursuit(M, (3, 2), max_iter=100, tol=1e-6, display=True)
    # print("L:", L)
    # print("S:", S)
