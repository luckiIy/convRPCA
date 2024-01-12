"""
进行highway数据集的恢复实验
"""
import os

import math
import time

import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
# 设置最大打开图像数为1000
matplotlib.rcParams['agg.path.chunksize'] = 10000

from models.outlier_pursuit import tensor_convlr_outlier_pursuit
# from utils.dataload import tensor_highway_outlier
from utils.evaluation_indicator import psnr

def tensor_highway_outlier(outlier_num=5, outlier_distribution='random', noise_mode='part_zero'):
    """
    读取highway张量形式数据集，对其进行异常数据插入，返回异常数据集

    :param outlier_num: 异常数据数量
    :param outlier_distribution: 异常数据分布，random为随机分布，ordered为有序分布
    :param noise_mode: 噪声分布，part_zero为对异常数据的部分区域置0，part_random为对异常数据的部分区域置随机值，total_noise为对异常数据加入随机噪声
    :return: corrupted_tensor: 异常数据集, outlier_omega: 异常数据位置, original_tensor: 原始数据集, outlier_tensor: 异常数据
    """
    # 读取数据集
    data_path = '../datasets/highway'
    # 数据为按名称顺序排列的一组灰度图像，将其整合为张量
    # 先查看当前目录下png文件的名称与数量
    file_list = os.listdir(data_path)
    file_list = [file for file in file_list if file.endswith('.png')]
    file_list.sort()

    # 读取第一张图片，获取图片大小
    image = Image.open(os.path.join(data_path, file_list[0])).convert('L')
    image = np.array(image).astype('float32') / 255.0
    # 将图像转换为矩阵
    original_tensor = np.zeros((image.shape[0], image.shape[1], len(file_list)))
    original_tensor[:, :, 0] = image
    for i in range(1, len(file_list)):
        image = Image.open(os.path.join(data_path, file_list[i])).convert('L')
        image = np.array(image).astype('float32') / 255.0
        original_tensor[:, :, i] = image

    # 生成异常数据
    outlier_num = outlier_num

    outlier_omega = np.zeros(len(file_list))
    # 选择数据按通道损坏方式
    if outlier_distribution == 'ordered':
        # 有序分布
        outlier_omega[:outlier_num] = 1
    elif outlier_distribution == 'random':
        # 随机分布
        outlier_omega[np.random.choice(len(file_list), outlier_num, replace=False)] = 1
    else:
        raise ValueError('Unsupported outlier distribution.')
    # outlier_tensor = np.zeros((image.shape[0], image.shape[1], len(file_list)))

    # 仅在异常通道上添加异常数据，同时添加的异常数据具有不同的形式
    if noise_mode == 'part_zero':
        corrupted_tensor = original_tensor.copy()
        corrupted_tensor_outlier = corrupted_tensor[:, :, outlier_omega == 1]
        # 遍历异常通道
        for i in range(corrupted_tensor_outlier.shape[2]):
            image_outlier = corrupted_tensor_outlier[:, :, i]
            (image_x, image_y) = image_outlier.shape
            # 随机选取要破坏的区域方块位置，破坏其中10*10的区域
            x = np.random.randint(0, image_x - 10)
            y = np.random.randint(0, image_y - 10)
            image_outlier[x:x+10, y:y+10] = 0

        corrupted_tensor[:, :, outlier_omega == 1] = corrupted_tensor_outlier
    elif noise_mode == 'total_noise':
        corrupted_tensor = original_tensor.copy()
        corrupted_tensor_outlier = corrupted_tensor[:, :, outlier_omega == 1]
        # 遍历异常通道
        for i in range(corrupted_tensor_outlier.shape[2]):
            image_outlier = corrupted_tensor_outlier[:, :, i]
            # 对整张图加入随机噪声
            image_outlier += np.random.normal(0, 1, image_outlier.shape)
            image_outlier = np.clip(image_outlier, 0, 1)

        corrupted_tensor[:, :, outlier_omega == 1] = corrupted_tensor_outlier
    else:
        raise ValueError('Unsupported noise mode.')

    # outlier_tensor = corrupted_tensor - original_tensor

    return corrupted_tensor, original_tensor, outlier_omega


def highway_recovery(corrupted_tensor, original_tensor, outlier_omega):
    # 得到被破坏数据
    # corrupted_tensor, outlier_omega, original_tensor, outlier_tensor = tensor_highway_outlier()
    # 将数据置入outlier pursuit算法
    pursuit_methods = 'conv'
    input_shape = corrupted_tensor.shape
    kernel_size = (input_shape[0]//10, input_shape[1]//10, input_shape[2]//4)
    lambda1 = 1 * kernel_size[0] * kernel_size[1] * kernel_size[2] / (math.sqrt(input_shape[2]))
    (L, S) = tensor_convlr_outlier_pursuit(corrupted_tensor, kernel_size, lambda1=lambda1, max_iter=1000, tol=1e-6, display=True)
    np.save('../datasets/highway/L1.npy', L)
    np.save('../datasets/highway/S1.npy', S)
    np.save('../datasets/highway/original_tensor1.npy', original_tensor)
    np.save('../datasets/highway/corrupted_tensor1.npy', corrupted_tensor)
    # 按顺序展示原始数据、恢复的低秩数据、恢复的稀疏数据，将同一个通道上的数据绘制在一起
    # 由于数据集中的数据是按照时间顺序排列的，所以按照时间顺序展示前10帧
    for frame in range(10):
        plt.figure(figsize=(10, 10))
        # 计算当前帧Low Rank Image相比于原始图像的PSNR
        psnr_value = psnr(original_tensor[:, :, frame], L[:, :, frame])
        plt.suptitle('Frame ' + str(frame) + ' PSNR: ' + str(psnr_value))
        plt.subplot(2, 2, 1)
        plt.imshow(original_tensor[:, :, frame], cmap='gray')
        plt.title('Original Image')
        plt.subplot(2, 2, 2)
        plt.imshow(corrupted_tensor[:, :, frame], cmap='gray')
        plt.title('Corrupted Image')
        plt.subplot(2, 2, 3)
        plt.imshow(L[:, :, frame], cmap='gray')
        plt.title('Low Rank Image')
        plt.subplot(2, 2, 4)
        plt.imshow(S[:, :, frame], cmap='gray')
        plt.title('Sparse Image')
        plt.show()
    # 展示损坏帧的恢复结果
    frame_list = outlier_omega.nonzero()[0]
    for frame in frame_list:
        plt.figure(figsize=(10, 10))
        # 计算当前帧Low Rank Image相比于原始图像的PSNR
        psnr_value = psnr(original_tensor[:, :, frame], L[:, :, frame])
        plt.suptitle('Frame ' + str(frame) + ' PSNR: ' + str(psnr_value))
        plt.subplot(2, 2, 1)
        plt.imshow(original_tensor[:, :, frame], cmap='gray')
        plt.title('Original Image')
        plt.subplot(2, 2, 2)
        plt.imshow(corrupted_tensor[:, :, frame], cmap='gray')
        plt.title('Corrupted Image')
        plt.subplot(2, 2, 3)
        plt.imshow(L[:, :, frame], cmap='gray')
        plt.title('Low Rank Image')
        plt.subplot(2, 2, 4)
        plt.imshow(S[:, :, frame], cmap='gray')
        plt.title('Sparse Image')
        plt.show()

if __name__ == '__main__':
    import pylab
    def read_display():
        # 读取存储的文件
        L = np.load('../datasets/highway/L.npy')
        S = np.load('../datasets/highway/S.npy')
        original_tensor = np.load('../datasets/highway/original_tensor.npy')
        corrupted_tensor = np.load('../datasets/highway/corrupted_tensor.npy')
        # 按顺序展示原始数据、恢复的低秩数据、恢复的稀疏数据，将同一个通道上的数据绘制在一起
        # 由于数据集中的数据是按照时间顺序排列的，所以按照时间顺序展示
        for frame in range(30):
            plt.figure(figsize=(10, 10))
            # 计算当前帧Low Rank Image相比于原始图像的PSNR
            psnr_value = psnr(original_tensor[:, :, frame], L[:, :, frame])
            plt.suptitle('Frame ' + str(frame) + ' PSNR: ' + str(psnr_value))
            plt.subplot(2, 2, 1)
            plt.imshow(original_tensor[:, :, frame], cmap='gray')
            plt.title('Original Image')
            plt.subplot(2, 2, 2)
            plt.imshow(corrupted_tensor[:, :, frame], cmap='gray')
            plt.title('Corrupted Image')
            plt.subplot(2, 2, 3)
            plt.imshow(L[:, :, frame], cmap='gray')
            plt.title('Low Rank Image')
            plt.subplot(2, 2, 4)
            plt.imshow(S[:, :, frame], cmap='gray')
            plt.title('Sparse Image')
            plt.show()
        # 展示损坏帧的恢复结果
        frame_list = outlier_omega.nonzero()[0]
        for frame in frame_list:
            plt.figure(figsize=(10, 10))
            # 计算当前帧Low Rank Image相比于原始图像的PSNR
            psnr_value = psnr(original_tensor[:, :, frame], L[:, :, frame])
            plt.suptitle('Frame ' + str(frame) + ' PSNR: ' + str(psnr_value))
            plt.subplot(2, 2, 1)
            plt.imshow(original_tensor[:, :, frame], cmap='gray')
            plt.title('Original Image')
            plt.subplot(2, 2, 2)
            plt.imshow(corrupted_tensor[:, :, frame], cmap='gray')
            plt.title('Corrupted Image')
            plt.subplot(2, 2, 3)
            plt.imshow(L[:, :, frame], cmap='gray')
            plt.title('Low Rank Image')
            plt.subplot(2, 2, 4)
            plt.imshow(S[:, :, frame], cmap='gray')
            plt.title('Sparse Image')
            plt.show()

    corrupted_tensor, original_tensor, outlier_omega = tensor_highway_outlier(noise_mode='total_noise')
    highway_recovery(corrupted_tensor, original_tensor, outlier_omega)
    # read_display()


    exit()
