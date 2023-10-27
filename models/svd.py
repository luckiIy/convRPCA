'''
会用到svd操作的函数，放在一起方便以后更换svd库
'''
import numpy as np


def SVT(M, tau):
    """
    :param M: input matrix
    :param tau: threshold
    :return: singular value thresholding
    """
    (U, sigma, V) = np.linalg.svd(M, full_matrices=False)
    sigma = np.where(sigma > tau, sigma - tau, 0)
    return U @ np.diag(sigma) @ V
