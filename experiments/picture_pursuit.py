import math
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, gridspec

from models.outlier_pursuit import convlr_outlier_pursuit, lr_outlier_pursuit


def picture_outlier_pursuit(dir='../datasets/lena_bin.png'):
    # 读取一张自然图像
    image = Image.open(os.path.join(os.path.dirname(__file__), dir)).convert('L')
    image = np.array(image).astype('float32') / 255.0
    # 将图像转换为矩阵

    # 图像以rou的概率损坏
    rou = 0.1
    image_outlier = image.copy()
    # 给随机的列附加整数型正态噪声，噪声均值为0，方差为1，适配图像范围

    # adversarial噪声方式为各列均相同的正态分布
    noise_type = 'natrual'
    # 随机选取损坏列，生成一个随机向量，向量中元素为0或1，1表示该列损坏
    Omega = np.random.rand(image.shape[1]) < rou
    if noise_type == 'natrual':
        # natrual噪声方式为各列均不相同的正态分布
        for i in range(image.shape[1]):
            if Omega[i]:
                # 产生一列整数噪声
                image_outlier[:, i] += np.random.normal(0, 1, image.shape[0]) * 0.02
                image_outlier[image_outlier < 0] = 0
                image_outlier[image_outlier > 1] = 1
    elif noise_type == 'adversarial':
        noise = np.random.normal(0, 25, image.shape[0])

        image_outlier[:, Omega] += noise.reshape((image.shape[0], 1)) * 0.02
        image_outlier[image_outlier < 0] = 0
        image_outlier[image_outlier > 1] = 1
    elif noise_type == 'zero':
        image_outlier[:, Omega] = 0


    # 将损坏图像置入outlier pursuit算法
    pursuit_methods = ['lr', 'conv']

    for pursuit_method in pursuit_methods:
        # 将损坏图像置入outlier pursuit算法
        if pursuit_method == 'conv':
            kernel_size = (5, 5)
            lambda1 = 1 * kernel_size[0] * kernel_size[1] / (math.sqrt(image_outlier.shape[0]))
            (L, S) = convlr_outlier_pursuit(image_outlier, kernel_size, lambda1=lambda1, max_iter=1000, tol=1e-6, display=True)
        elif pursuit_method == 'lr':
            lambda1 = 10 / (math.sqrt(image_outlier.shape[0]))
            (L, S) = lr_outlier_pursuit(image_outlier, lambda1=lambda1, max_iter=1000, tol=1e-6, display=True)
        else:
            print("No such pursuit method!")
            exit()

        # 可视化对比原图像、破损图像、低秩图像、列稀疏图像

        plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2)
        # 标题区分puruit方法
        plt.suptitle(pursuit_method)
        plt.subplot(gs[0, 0])
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.subplot(gs[0, 1])
        plt.imshow(image_outlier, cmap='gray')
        plt.title('Corrupted Image')
        plt.subplot(gs[1, 0])
        plt.imshow(L, cmap='gray')
        plt.title('Low Rank Image')
        plt.subplot(gs[1, 1])
        plt.imshow(S, cmap='gray')
        plt.title('Column Sparse Image')
        plt.show()
        # 对图像列进行l2范数统计，以列位置为横坐标，l2范数为纵坐标, 条形图, OMEGA为红色，非OMEGA为蓝色
        plt.figure(figsize=(10, 10))
        Sj_norm = np.linalg.norm(S, ord=2, axis=0) + 0.01
        # 控制图像显示高度
        plt.subplot()
        plt.xlim(0, S.shape[1])
        plt.ylim(0, Sj_norm.max())
        plt.bar(range(S.shape[1]), Sj_norm, color=['r' if Omega[i] else 'b' for i in range(S.shape[1])], width=1)
        plt.title('Column L2 Norm')
        plt.show()


if __name__ == '__main__':
    picture_outlier_pursuit()
    exit()