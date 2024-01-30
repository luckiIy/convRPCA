"""
实验验证：不同损坏形式下，损坏部分标准化之后的卷积秩与条件数以及谱范数等
默认gamma^2 = 100
这里由于奇异值重尾的问题，可以考虑计算截断后的奇异值的最大值与最小值的比值吗？不然确实没什么意义
先固定以1/4为系数截断，或者其实可以考虑截断某一条件数的情况下，计算秩，感觉这个才是更好的选项
"""
import os
import math
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

from utils.dataload import convlr_synthetic_data, tensor_highway_outlier
from models.conv import adj_nd, cconv_nd

def compute_metrics(conv_matrix, sigma_cond=100):
    """
    计算给定卷积矩阵的秩、条件数、谱范数等指标
    :param tensor: 给定的卷积矩阵
    :return: trunc_rank: 秩, cond: 条件数, norm: 谱范数
    """

    # 这里要计算的实际上是奇异值不为0的部分的最大值与最小值的比值，所以考虑直接计算奇异值
    S = np.linalg.svd(conv_matrix, compute_uv=False, full_matrices=False)
    # rank = np.sum(S > 1e-6)
    norm = np.max(S)
    cond_S = (norm / S)**2
    S = S[cond_S < sigma_cond]
    trunc_rank = len(S)
    return trunc_rank, norm

def metrics_of_outlier():
    """
    计算不同损坏形式下，损坏部分标准化之后的卷积秩与条件数以及谱范数等
    :return:
    """
    # 产生不同形式的数据
    data_types = ['highway', 'lena']
    # 不同形式的损坏

    outlier_ratio = 0.1
    outlier_distributions = ['random', 'ordered']



    outlier_cond = 100
    for data_type in data_types:
        if data_type == 'synthetic':
            # 生成合成的卷积低秩数据集
            m = 80
            n = 60
            # conv_ranks = [2, 4, 6, 8, 10, 12, 14, 16,
            #                 18, 20, 22, 24, 26, 28, 30, 32,
            #                 34, 36, 38, 40, 42, 44, 46, 48,
            #                 50, 52, 54, 56, 58, 60, 62, 64]
            conv_ranks = [16]

            outlier_ratios = [(2*i + 1) / n for i in range(0, 8, 1)]
            insert_mode = 'add'# 'replace'
            noise_mode = 'natrual'
            kernel_ratio = 4
            # 用于储存不同conv_rank，以及不同采样率下的卷积秩、谱范数等，将损坏方式、行列数、kernel_ratio、条件数等信息作为表名记录
            synthetic_metrics = pd.DataFrame(columns=['outlier_distribution', 'conv_rank', 'outlier_ratio', 'trunc_rank', 'norm'])

            for outlier_distribution in outlier_distributions:
                for conv_rank in conv_ranks:
                    for outlier_ratio in outlier_ratios:
                        # 生成合成的卷积低秩数据集
                        synthetic_matrix, outlier_omega, low_rank_matrix, noise_matrix = \
                            convlr_synthetic_data(conv_rank, m=m, n=n, outlier_ratio=outlier_ratio,
                                                  outlier_distribution=outlier_distribution, insert_mode=insert_mode, noise_mode=noise_mode)

                        # 对noise_matrix进行按最后一维（按列或按时间维）标准化
                        nc_noise_matrix = noise_matrix / np.linalg.norm(noise_matrix, ord=2, axis=0, keepdims=True)
                        # 卷积
                        kernel_size = (nc_noise_matrix.shape[0] // kernel_ratio, nc_noise_matrix.shape[1] // kernel_ratio)
                        k = np.prod(kernel_size)
                        conv_nc_noise_matrix = cconv_nd(nc_noise_matrix, kernel_size)
                        # 计算损坏部分的标准化卷积秩、条件数、谱范数等
                        conv_nc_noise_matrix[np.isnan(conv_nc_noise_matrix)] = 0
                        outlier_rank, outlier_norm = compute_metrics(conv_nc_noise_matrix, sigma_cond=outlier_cond)
                        # 先写print，后续跑模拟数据集上不同参数的实验时再写入文件
                        print('data_type: {}, outlier_distribution: {}, outlier_rank: {}, k:{}, outlier_cond: {}, '
                              'outlier_norm: {}'.format(data_type, outlier_distribution, outlier_rank, k, outlier_cond,
                                                        outlier_norm))
                        # 记录数据
                        now_data = {'outlier_distribution': outlier_distribution, 'conv_rank': conv_rank, 'outlier_ratio': outlier_ratio, 'trunc_rank': outlier_rank, 'norm': outlier_norm}
                        # 我吐了pandas你他妈版本更新了中文文档不更新的，append方法更新没了，用concat方法或者_append
                        synthetic_metrics = synthetic_metrics._append(now_data, ignore_index=True)
                        # 保存数据
                        synthetic_metrics.to_csv('./synthetic_metrics.csv')
            # # 记录并保存当前表单的非循环参数
            # info = {'data_type': data_type, 'm': m, 'n': n, 'outlier_cond':outlier_cond, 'kernel_ratio': kernel_ratio, 'k': k,
            #         'insert_mode': insert_mode}
            # info = pd.DataFrame(info, index=[0])
            # info.to_csv('./synthetic_metrics_info.csv')

        elif data_type == 'highway':
            # 使用highway数据集

            noise_mode = 'total_noise'
            outlier_ratios = [(2 * i + 1) / 62 for i in range(0, 8, 1)]
            kernel_ratio = 6
            for outlier_ratio in outlier_ratios:
                outlier_num = math.ceil(62 * outlier_ratio)
                for outlier_distribution in outlier_distributions:
                    corrupted_tensor, original_tensor, outlier_omega, outlier_tensor = \
                        tensor_highway_outlier(outlier_num=outlier_num, outlier_distribution=outlier_distribution, noise_mode=noise_mode)
                    # 对noise_matrix进行按最后一维（按列或按时间维）标准化
                    nc_noise_matrix = outlier_tensor / np.linalg.norm(outlier_tensor, ord=2, axis=(0, 1), keepdims=True)
                    # 卷积 注意这里对于张量，在时间维度上尽量增大卷积核的大小
                    kernel_size = (nc_noise_matrix.shape[0] // kernel_ratio, nc_noise_matrix.shape[1] // kernel_ratio, nc_noise_matrix.shape[2])
                    k = np.prod(kernel_size)
                    conv_nc_noise_matrix = cconv_nd(nc_noise_matrix, kernel_size)
                    # 计算损坏部分的标准化卷积秩、条件数、谱范数等
                    conv_nc_noise_matrix[np.isnan(conv_nc_noise_matrix)] = 0
                    outlier_rank, outlier_norm = compute_metrics(conv_nc_noise_matrix, sigma_cond=outlier_cond)
                    print('data_type: {}, outlier_distribution: {}, outlier_rank: {}, k:{}, outlier_cond: {}, '
                          'outlier_norm: {}'.format(data_type, outlier_distribution, outlier_rank, k, outlier_cond,
                                                    outlier_norm))

        elif data_type == 'lena':
            pass

if __name__ == '__main__':
    metrics_of_outlier()
    exit()
