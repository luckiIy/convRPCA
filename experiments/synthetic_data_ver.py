'''
验证在合成的带噪声的卷积低秩数据集上的复原能力
'''
import math

import numpy as np
from matplotlib import pyplot as plt

from utils.dataload import convlr_synthetic_data, low_rank_synthetic_data
from models.outlier_pursuit import lr_outlier_pursuit, convlr_outlier_pursuit
from models.conv import adj_nd, cconv_nd

def synthetic_data_ver(m=100, n=100):
    '''
    验证在合成的带噪声的卷积低秩数据集上的复原能力
    :param m: 横轴表示数据维度
    :param n: 纵轴表示不同的数据点
    :return:
    '''

    pursuit_methods = ['conv']
    # 设定需要测试的不同的卷积秩conv_rank，以及不同的异常数据比例outlier_ratio
    # 卷积秩从2~40，间隔为2
    conv_rank_list = list(range(2, 40, 2))
    # 异常数据比例从0.01~0.2，间隔为0.01
    outlier_ratio_list = [i / 100 for i in range(1, 21, 1)]

    # 生成矩阵记录不同卷积秩和异常数据比例下是否得到精确恢复
    # 1表示精确恢复，0表示未精确恢复
    success_matrix = np.zeros((len(conv_rank_list), len(outlier_ratio_list)))

    # 生成合成的卷积低秩数据集
    for conv_rank in conv_rank_list:
        for outlier_ratio in outlier_ratio_list:
            # 生成合成的卷积低秩数据集
            synthetic_matrix, outlier_omega, low_rank_matrix, noise_matrix = \
                convlr_synthetic_data(conv_rank, m=m, n=n, outlier_ratio=outlier_ratio,
                                      outlier_distribution='random', insert_mode='add', noise_mode='natrual')
            # 将合成的卷积低秩数据集置入outlier pursuit算法

            for pursuit_method in pursuit_methods:
                # 将损坏图像置入outlier pursuit算法
                if pursuit_method == 'conv':
                    kernel_size = (m//4, n//4)
                    lambda1 = 1 * kernel_size[0] * kernel_size[1] / (math.sqrt(n))
                    (L, S) = convlr_outlier_pursuit(synthetic_matrix, kernel_size, lambda1=lambda1, max_iter=1000,
                                                    tol=1e-6, display=True)
                elif pursuit_method == 'lr':
                    lambda1 = 1 / (math.sqrt(n))
                    (L, S) = lr_outlier_pursuit(synthetic_matrix, lambda1=lambda1, max_iter=1000, tol=1e-6,
                                                display=True)
                else:
                    print("No such pursuit method!")
                    exit()
                # 得到矩阵的可视化
                plt.figure(figsize=(10, 10))
                plt.suptitle(pursuit_method + " conv_rank:" + str(conv_rank) + " outlier_ratio:" + str(outlier_ratio))
                plt.subplot(2, 2, 1)
                plt.imshow(synthetic_matrix.reshape(m, n), cmap='gray')
                plt.title('Original Image')
                plt.subplot(2, 2, 2)
                plt.imshow(L.reshape(m, n), cmap='gray')
                plt.title('Low Rank Image')
                plt.subplot(2, 2, 3)
                plt.imshow(S.reshape(m, n), cmap='gray')
                plt.title('Sparse Image')

                # 对图像列进行l2范数统计，以列位置为横坐标，l2范数为纵坐标, 条形图, OMEGA为红色，非OMEGA为蓝色
                plt.subplot(2, 2, 4)
                Sj_norm = np.linalg.norm(S, ord=2, axis=0) + 0.01
                plt.xlim(0, S.shape[1])
                plt.ylim(0, Sj_norm.max())
                plt.bar(range(S.shape[1]), Sj_norm, color=['r' if outlier_omega[i] else 'b' for i in range(S.shape[1])], width=1)
                plt.title('Column L2 Norm')
                plt.show()

                # 验证是否能够正确分离出异常值
                # 这里有个问题，我们是仅关注L的列空间与S在正常值部分是否为0，还是关注L与S是否得到精确恢复？
                # L_diff = np.zeros((m, n))
                # L_diff[:, outlier_omega == 1] = L[:, outlier_omega == 1] - low_rank_matrix[:, outlier_omega == 1]
                # S_diff = np.zeros((m, n))
                # S_diff[:, outlier_omega == 0] = S[:, outlier_omega == 0] - noise_matrix[:, outlier_omega == 0]
                L_diff_norm = np.linalg.norm(L - low_rank_matrix, ord=2)
                S_diff_norm = np.linalg.norm(S - noise_matrix, ord=2)
                is_success = 1 if L_diff_norm < 1e-6 and S_diff_norm < 1e-6 else 0
                print("pursuit_method:", pursuit_method, "conv_rank:", conv_rank, "outlier_ratio:", outlier_ratio,
                      "L_diff_norm:", L_diff_norm, "S_diff_norm:", S_diff_norm, "is_success:", is_success)
                if pursuit_method == 'conv':
                    # 计算原始矩阵L的卷积核范数与S的2,1范数
                    lr_conv_nuc_norm = np.linalg.norm(cconv_nd(low_rank_matrix, kernel_size), ord='nuc')
                    noise_21_norm = np.sum(np.linalg.norm(noise_matrix[:, i], ord=2) for i in range(noise_matrix.shape[1]))
                    # 计算分解后的矩阵L的卷积核范数与S的2,1范数
                    L_conv_nuc_norm = np.linalg.norm(cconv_nd(L, kernel_size), ord='nuc')
                    S_21_norm = np.sum(np.linalg.norm(S[:, i], ord=2) for i in range(S.shape[1]))
                    print("lr_conv_nuc_norm:", lr_conv_nuc_norm, "noise_21_norm:", noise_21_norm,
                          "L_conv_nuc_norm:", L_conv_nuc_norm, "S_21_norm:", S_21_norm)




                # 记录是否得到精确恢复，矩阵按照conv_rank从小到大排列，按照outlier_ratio从小到大排列
                success_matrix[conv_rank_list.index(conv_rank), outlier_ratio_list.index(outlier_ratio)] = is_success

    # 可视化success_matrix，坐标轴为conv_rank和outlier_ratio，值为0表示未精确恢复，值为1表示精确恢复
    plt.figure(figsize=(10, 10))
    plt.imshow(success_matrix, cmap='gray')
    plt.xlabel('outlier_ratio')
    plt.ylabel('conv_rank')
    plt.xticks(range(len(outlier_ratio_list)), outlier_ratio_list)
    plt.yticks(range(len(conv_rank_list)), conv_rank_list)
    plt.title('Success Matrix')
    plt.show()

if __name__ == '__main__':
    synthetic_data_ver()
    exit()


