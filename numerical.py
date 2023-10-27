import math
import numpy as np
import pprint as pp
size = 10
p = 0.5
# 使用M = AB的矩阵分解方式形成秩为尺度1/10的低秩随机矩阵
A = np.random.rand(size, math.floor(size / 10))
B = np.random.rand(math.floor(size / 10), size)
M = A @ B
# M = np.random.rand(size, size)
(U, sigma, V) = np.linalg.svd(M, full_matrices=False)
I = np.eye(size)
# Bernoulli sampling operator
def bernoulli_sampling_operator(m, n, p):
    """
    :param m: number of rows
    :param n: number of columns
    :param p: probability of 1
    :return: sampling operator
    """
    A = np.random.rand(m, n)
    A = np.where(A < p, 1, 0)
    return A

# diagonal bernoulli sampling operator
D = np.diag(np.where(np.random.rand(10) < 0.5, 1, 0))

Z = U.T @ D @ U
(_, sigmaZ, _) = np.linalg.svd(Z, full_matrices=True)
norm_2 = np.linalg.norm(I - Z, ord=2)
pp.pprint(U.T @ U)
pp.pprint(Z)

