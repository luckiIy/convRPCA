'''
读取各式数据集的工具
'''

import os
import numpy as np


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


if __name__ == '__main__':
    # 读取mnist数据集中的1和7组成的数据集
    x, y = mnist_outlier_1_7()
    print(x.shape)
    print(y.shape)
    exit()
