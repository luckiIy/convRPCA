'''
读取各式数据集的工具
'''

import os
import numpy as np
from PIL import Image

from models.conv import adj_nd, cconv_nd

def read_data(data_name, data_path):
    """
    :param data_name: 数据集名称
    :param data_path: 数据集路径
    :return: 数据集
    """
    if data_name == 'mnist':
        return read_mnist(data_path)
    else:
        raise ValueError('Unsupported dataset.')

def read_mnist(data_path):
    """
    :param data_path: 数据集路径
    :return: 数据集
    """
    # 读取数据集
    data = np.load(os.path.join(data_path, 'mnist.npz'))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

    # 数据预处理
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 返回数据集
    return x_train, y_train, x_test, y_test

def mnist_outlier_1_7(structure='matrix', ratio=0.1, distribution='random', data_path='../datasets/digit-recognizer', data_num=1000):
    """
    使用mnist中的1和一小部分的7组成的数据集, 1为正常数据，7为异常数据
    如果structure为matrix则每张图片展开，返回矩阵形式的数据集，如果structure为tensor则返回张量形式的数据集
    :param structure: 数据集的结构，matrix为矩阵，tensor为张量
    :param ratio: 异常数据的比例
    :param distribution: 异常数据的分布，random为随机分布，ordered为有序分布
    :param data_path: 数据集路径
    :param data_num: 数据集大小, 不能大于mnist中1和7的数据集大小


    :return: x: 数据集, y: 标签
    """
    # 读取数据集
    data = np.load(os.path.join(os.path.dirname(__file__), data_path, 'mnist.npz'))
    x_train = data['x_train']
    y_train = data['y_train']

    # 数据预处理
    x_train = x_train.astype('float32') / 255.0

    # 选取1和7的数据
    num_1 = int(data_num * 0.9)
    num_7 = int(data_num - num_1)
    x_train_1 = x_train[y_train == 1][:num_1]
    x_train_7 = x_train[y_train == 7][:num_7]

    # 按比例选取7的数据作为异常数据
    num_outlier = int(x_train_1.shape[0] * ratio)
    x_outlier = x_train_7[:num_outlier]

    # 按照数据结构选择是否展开数据为向量
    if structure == 'matrix':
        x = x_train_1.reshape((x_train_1.shape[0], -1))
        x_outlier = x_outlier.reshape((x_outlier.shape[0], -1))
    elif structure == 'tensor':
        x = x_train_1.reshape((x_train_1.shape[0], 28, 28))
        x_outlier = x_outlier.reshape((x_outlier.shape[0], 28, 28))
    else:
        raise ValueError('Unsupported data structure.')
    # 按照分布方式选择是否有序插入异常数据, 记录异常数据的位置
    x = np.concatenate((x, x_outlier), axis=0)
    y = np.concatenate((np.zeros(x_train_1.shape[0]), np.ones(x_outlier.shape[0])), axis=0)
    if distribution == 'ordered':
        return x, y
    elif distribution == 'random':
        # 打乱数据集
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation]
        y = y[permutation]
        return x, y
    else:
        raise ValueError('Unsupported distribution.')


def low_rank_synthetic_data(rank, m=400, n=400, outlier_ratio=0.1, outlier_distribution='random', insert_mode='add',
                            noise_mode='natrual'):
    """
    生成被破坏的低秩矩阵
    :param rank: 矩阵秩
    :param m: 矩阵行数
    :param n: 矩阵列数

    :param outlier_ratio: 异常数据比例
    :param outlier_distribution: 异常数据分布，random为随机分布，ordered为有序分布
    :param insert_mode: 异常数据插入方式，replace为替换，add为相加
    :param noise_mode: 噪声分布，natrual为各列均不相同的正态分布，adversarial为各列均相同的正态分布，zero为全为0

    :return: synthetic_matrix: 破坏后的矩阵, outlier_omega: 异常数据位置, low_rank_matrix: 低秩矩阵, noise_matrix: 噪声矩阵
    """
    # 生成低秩矩阵
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    low_rank_matrix = U @ V

    # 生成异常数据
    outlier_num = int(n * outlier_ratio)
    if outlier_distribution == 'ordered':
        # 有序分布
        outlier_omega = np.zeros(n)
        outlier_omega[:outlier_num] = 1
    elif outlier_distribution == 'random':
        # 随机分布
        outlier_omega = np.zeros(n)
        outlier_omega[np.random.choice(n, outlier_num, replace=False)] = 1

    else:
        raise ValueError('Unsupported outlier distribution.')

    # 生成不同形式噪声
    if noise_mode == 'natrual':
        # natrual噪声方式为各列均不相同的正态分布
        noise_matrix = np.zeros((m, n))
        noise_matrix[:, outlier_omega == 1] = np.random.normal(0, 1, (m, outlier_num))
    elif noise_mode == 'adversarial':
        # adversarial噪声方式为各列均相同的正态分布
        noise_matrix = np.zeros((m, n))
        noise_matrix[:, outlier_omega == 1] = np.random.normal(0, 1, (m, 1))
    elif noise_mode == 'zero':
        noise_matrix = np.zeros((m, n))
    else:
        raise ValueError('Unsupported noise mode.')

    # 生成破坏后的矩阵
    if insert_mode == 'replace':
        # 替换异常值
        synthetic_matrix = low_rank_matrix.copy()
        synthetic_matrix[:, outlier_omega == 1] = noise_matrix[:, outlier_omega == 1]
    elif insert_mode == 'add':
        # 相加异常值
        synthetic_matrix = low_rank_matrix.copy()
        synthetic_matrix[:, outlier_omega == 1] += noise_matrix[:, outlier_omega == 1]
    else:
        raise ValueError('Unsupported insert mode.')

    noise_matrix = synthetic_matrix - low_rank_matrix
    return synthetic_matrix, outlier_omega, low_rank_matrix, noise_matrix

def convlr_synthetic_data(conv_rank, m=400, n=400, outlier_ratio=0.1, outlier_distribution='random', insert_mode='add',
                            noise_mode='natrual'):
    """
    生成被破坏的卷积低秩矩阵
    :param conv_rank: 卷积矩阵秩
    :param m: 矩阵行数
    :param n: 矩阵列数

    :param outlier_ratio: 异常数据比例
    :param outlier_distribution: 异常数据分布，random为随机分布，ordered为有序分布
    :param insert_mode: 异常数据插入方式，replace为替换，add为相加
    :param noise_mode: 噪声分布，natrual为各列均不相同的正态分布，adversarial为各列均相同的正态分布，zero为全为0

    :return:
    """
    # 生成卷积低秩的矩阵
    # 卷积低秩部分的生成方式为M_mn = \sum_{i=1}^{a} sin(2 * i * m * pi / mn) * sin(2 * i * n * pi / mn)
    M = np.zeros((m, n))
    a = conv_rank // 2
    for (mi, ni) in np.ndindex(m, n):
        for i in range(1, a + 1):
            M[mi, ni] += np.sin(2 * i * mi * np.pi / m) * np.sin(2 * i * ni * np.pi / n)
    # 数据归一化
    M = M / np.linalg.norm(M, ord=2)

    # 验证卷积矩阵的秩，后续删除该部分
    # Ak_M = cconv_nd(M, (m//3, n//3))
    # U, S, VT = np.linalg.svd(Ak_M)
    # print(S)
    # print(np.linalg.matrix_rank(Ak_M))



    # 生成异常数据
    outlier_num = int(n * outlier_ratio)
    if outlier_distribution == 'ordered':
        # 有序分布
        outlier_omega = np.zeros(n)
        outlier_omega[:outlier_num] = 1
    elif outlier_distribution == 'random':
        # 随机分布
        outlier_omega = np.zeros(n)
        outlier_omega[np.random.choice(n, outlier_num, replace=False)] = 1

    else:
        raise ValueError('Unsupported outlier distribution.')

    # 生成不同形式噪声
    if noise_mode == 'natrual':
        # natrual噪声方式为各列均不相同的正态分布
        noise_matrix = np.zeros((m, n))
        noise_matrix[:, outlier_omega == 1] = np.random.normal(0, 1, (m, outlier_num))
    elif noise_mode == 'adversarial':
        # adversarial噪声方式为各列均相同的正态分布
        noise_matrix = np.zeros((m, n))
        noise_matrix[:, outlier_omega == 1] = np.random.normal(0, 1, (m, 1))
    elif noise_mode == 'zero':
        noise_matrix = np.zeros((m, n))
    else:
        raise ValueError('Unsupported noise mode.')

    # 生成破坏后的矩阵
    if insert_mode == 'replace':
        # 替换异常值
        synthetic_matrix = M.copy()
        synthetic_matrix[:, outlier_omega == 1] = noise_matrix[:, outlier_omega == 1]
    elif insert_mode == 'add':
        # 相加异常值
        synthetic_matrix = M.copy()
        synthetic_matrix[:, outlier_omega == 1] += noise_matrix[:, outlier_omega == 1]
    else:
        raise ValueError('Unsupported insert mode.')

    noise_matrix = synthetic_matrix - M
    return synthetic_matrix, outlier_omega, M, noise_matrix


# def tensor_highway_outlier(outlier_num=5, outlier_distribution='random', noise_mode='part_zero'):
#     """
#     读取highway张量形式数据集，对其进行异常数据插入，返回异常数据集
#
#     :param outlier_num: 异常数据数量
#     :param outlier_distribution: 异常数据分布，random为随机分布，ordered为有序分布
#     :param noise_mode: 噪声分布，part_zero为对异常数据的部分区域置0，part_random为对异常数据的部分区域置随机值，all_zero为对异常数据置0，all_random为对异常数据置随机值
#     :return: corrupted_tensor: 异常数据集, outlier_omega: 异常数据位置, original_tensor: 原始数据集, outlier_tensor: 异常数据
#     """
#     # 读取数据集
#     data_path = '../datasets/highway'
#     # 数据为按名称顺序排列的一组灰度图像，将其整合为张量
#     # 先查看当前目录下png文件的名称与数量
#     file_list = os.listdir(data_path)
#     file_list = [file for file in file_list if file.endswith('.png')]
#     file_list.sort()
#
#     # 读取第一张图片，获取图片大小
#     image = Image.open(os.path.join(data_path, file_list[0])).convert('L')
#     image = np.array(image).astype('float32') / 255.0
#     # 将图像转换为矩阵
#     original_tensor = np.zeros((image.shape[0], image.shape[1], len(file_list)))
#     original_tensor[:, :, 0] = image
#     for i in range(1, len(file_list)):
#         image = Image.open(os.path.join(data_path, file_list[i])).convert('L')
#         image = np.array(image).astype('float32') / 255.0
#         original_tensor[:, :, i] = image
#
#     # 生成异常数据
#     outlier_num = outlier_num
#
#     outlier_omega = np.zeros(len(file_list))
#     # 选择数据损坏方式
#     if outlier_distribution == 'ordered':
#         # 有序分布
#         outlier_omega[:outlier_num] = 1
#     elif outlier_distribution == 'random':
#         # 随机分布
#         outlier_omega[np.random.choice(len(file_list), outlier_num, replace=False)] = 1
#     else:
#         raise ValueError('Unsupported outlier distribution.')
#     # outlier_tensor = np.zeros((image.shape[0], image.shape[1], len(file_list)))
#
#     # 仅在异常通道上添加异常数据，同时添加的异常数据具有不同的形式
#     if noise_mode == 'part_zero':
#         corrupted_tensor = original_tensor.copy()
#         corrupted_tensor_outlier = corrupted_tensor[:, :, outlier_omega == 1]
#         # 遍历异常通道
#         for i in range(corrupted_tensor_outlier.shape[2]):
#             image_outlier = corrupted_tensor_outlier[:, :, i]
#             (image_x, image_y) = image_outlier.shape
#             # 随机选取要破坏的区域方块位置，破坏其中10*10的区域
#             x = np.random.randint(0, image_x - 10)
#             y = np.random.randint(0, image_y - 10)
#             image_outlier[x:x+10, y:y+10] = 0
#
#         corrupted_tensor[:, :, outlier_omega == 1] = corrupted_tensor_outlier
#     else:
#         raise ValueError('Unsupported noise mode.')
#
#     outlier_tensor = corrupted_tensor - original_tensor
#
#     return corrupted_tensor, outlier_omega, original_tensor, outlier_tensor




if __name__ == '__main__':
    # 读取mnist数据集中的1和7组成的数据集
    # x, y = mnist_outlier_1_7()
    # print(x.shape)
    # print(y.shape)
    # 测试合成低秩数据的生成效果
    # outlier_distributions = ['random', 'ordered']
    # insert_modes = ['replace', 'add']
    # noise_modes = ['natrual', 'adversarial', 'zero']
    # for outlier_distribution in outlier_distributions:
    #     for insert_mode in insert_modes:
    #         for noise_mode in noise_modes:
    #
    #             synthetic_matrix, outlier_omega, low_rank_matrix, noise_matrix = convlr_synthetic_data(10, m=50, n=50, outlier_ratio=0.1, outlier_distribution=outlier_distribution, insert_mode=insert_mode, noise_mode=noise_mode)
    #
    # corrupted_tensor, outlier_omega, original_tensor, outlier_tensor = tensor_highway_outlier()
    exit()
