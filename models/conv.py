'''
卷积与反卷操作相应的函数
'''
import numpy as np


def adj_nd(M, isize=None, ksize=None):
    input_dims = M.ndim  # 获取输入数据的维度
    if input_dims != 2:
        raise ValueError("Input data dimensions must be greater than 1.")
    if ksize is None:
        ksize = M.shape[1]
    if isinstance(ksize, int):
        ksize = (ksize,)
    output_dims = len(ksize)

    if output_dims == 1:
        return adj_1D(M)
    elif output_dims == 2:
        return adj_2D(M, isize, ksize)
    elif output_dims == 3:
        return adj_3D(M, isize, ksize)
    else:
        raise ValueError("Unsupported input data dimensions.")


def cconv_nd(X, ksize):
    input_dims = X.ndim  # 获取输入数据的维度
    if isinstance(ksize, int):
        ksize = (ksize,)
    output_dims = len(ksize)
    if input_dims != output_dims:
        raise ValueError("Input data dimensions do not match the specified ksize dimensions.")

    if input_dims == 1:
        return cconv_1D(X, ksize)
    elif input_dims == 2:
        return cconv_2D(X, ksize)
    elif input_dims == 3:
        return cconv_3D(X, ksize)
    else:
        raise ValueError("Unsupported input data dimensions.")


def adj_1D(M):
    m, n = M.shape
    X = np.zeros((m, 1))
    for i in range(n):
        Xi = M[:, i]
        Xi = Xi.reshape(m, 1)
        Xi = np.roll(Xi, shift=- i, axis=0)
        X += Xi
    return X


def cconv_1D(X, ksize):
    m = X.shape[0]
    A = np.zeros((m, np.prod(ksize)))
    Xi = X.copy()
    for i in range(ksize[0]):
        A[:, i] = Xi.ravel()
        # if i < ksize[0] - 1:
        Xi = np.roll(Xi, shift=1, axis=0)
    return A


def adj_2D(M, isize, ksize):
    X = np.zeros(isize)
    for j in range(ksize[1]):
        for i in range(ksize[0]):
            Xi = M[:, j * ksize[0] + i]
            Xi = Xi.reshape(isize, order='F')
            Xi = np.roll(Xi, shift=(- i, - j), axis=(0, 1))
            X += Xi
    return X


def cconv_2D(X, ksize):
    m, n = X.shape
    A = np.zeros((m * n, np.prod(ksize)))
    Xj = X.copy()
    for j in range(ksize[1]):
        Xi = Xj.copy()
        for i in range(ksize[0]):
            # 将Xi展开成一维向量，注意要按列展开
            A[:, j * ksize[0] + i] = Xi.ravel(order='F')
            # if i < ksize[0] - 1:
            Xi = np.roll(Xi, shift=1, axis=0)
        # if j < ksize[1] - 1:
        Xj = np.roll(Xj, shift=1, axis=1)
    return A


def adj_3D(M, isize, ksize):
    X = np.zeros(isize)
    for k in range(ksize[2]):
        for j in range(ksize[1]):
            for i in range(ksize[0]):
                Xi = M[:, (k * ksize[1] + j) * ksize[0] + i]
                Xi = Xi.reshape(isize, order='F')
                Xi = np.roll(Xi, shift=(- i, - j, - k), axis=(0, 1, 2))
                X += Xi
    return X


def cconv_3D(X, ksize):
    m, n, l = X.shape
    A = np.zeros((m * n * l, np.prod(ksize)))
    Xk = X.copy()
    for k in range(ksize[2]):
        Xj = Xk.copy()
        for j in range(ksize[1]):
            Xi = Xj.copy()
            for i in range(ksize[0]):
                A[:, (k * ksize[1] + j) * ksize[0] + i] = Xi.ravel(order='F')
                Xi = np.roll(Xi, shift=1, axis=0)
            Xj = np.roll(Xj, shift=1, axis=1)
        Xk = np.roll(Xk, shift=1, axis=2)
    return A


if __name__ == '__main__':
    # 分别对1D、2D、3D的数据进行测试
    # 1D
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).T
    A = cconv_nd(X, 3)
    X_adj = adj_nd(A)
    # 在各个维度上除去ksize的值
    X_adj = X_adj / 3
    assert np.allclose(X, X_adj.ravel())
    # 2D
    X = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                  [9, 10, 11, 12, 13, 14, 15, 16]]).T
    A = cconv_nd(X, (3, 2))
    X_adj = adj_nd(A, (8, 2), (3, 2))
    X_adj = X_adj / 6
    assert np.allclose(X, X_adj)
    # 3D
    X = np.array([[[1, 2, 3, 4, 5, 6, 7, 8],
                   [9, 10, 11, 12, 13, 14, 15, 16]],
                  [[17, 18, 19, 20, 21, 22, 23, 24],
                   [25, 26, 27, 28, 29, 30, 31, 32]]])
    A = cconv_nd(X, (2, 2, 3))
    X_adj = adj_nd(A, (2, 2, 8), (2, 2, 3))
    X_adj = X_adj / 12
    assert np.allclose(X, X_adj)
    print("All tests passed.")
