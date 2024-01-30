'''
会用到svd操作的函数，放在一起方便以后更换svd库，加速或者GPU加速等
'''
import numpy as np


def SVT(M, tau):
    """
    :param M: input matrix
    :param tau: threshold
    :return: singular value thresholding
    """
    U, sigma, VT = np.linalg.svd(M, full_matrices=False)
    return U @ np.diag(shrink(sigma, tau)) @ VT


def shrink(X, tau):
    """
    # 对矩阵X进行软阈值化
    :param X: 输入矩阵
    :param tau: 阈值
    :return: 软阈值化后的矩阵
    """
    return np.sign(X) * np.maximum(abs(X) - tau, 0)

