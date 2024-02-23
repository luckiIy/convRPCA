"""
使用本文算法进行华为数据集样本的异常检测
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from models.outlier_pursuit import lr_outlier_pursuit, convlr_outlier_pursuit

def read_huawei_data(data_path='datasets/huawei/纺织样本'):
    """
    读取华为数据集
    :param data_path: 数据集路径
    :return: fiber_data, lace_data
    """
    # 化纤数据集，建立字典，包含good, simple_anomaly, complex_anomaly
    fiber_path = os.path.join(data_path, '化纤样本数据')
    fiber_data = {
        'good': [],
        'simple_anomaly': [],
        'complex_anomaly': []
    }
    # 读取good数据，读取目录下所有图片
    good_path = os.path.join(fiber_path, 'good')
    for root, dirs, files in os.walk(good_path):
        for file in files:
            # numpy读取出BUG，这里使用plt.imread读取图片
            img = plt.imread(os.path.join(root, file))
            fiber_data['good'].append(img)
    # 读取simple_anomaly数据，读取目录下所有图片
    simple_anomaly_path = os.path.join(fiber_path, 'anomaly/简单')
    for root, dirs, files in os.walk(simple_anomaly_path):
        for file in files:
            img = plt.imread(os.path.join(root, file))
            fiber_data['simple_anomaly'].append(img)
    # 读取complex_anomaly数据，读取目录下所有图片
    complex_anomaly_path = os.path.join(fiber_path, 'anomaly/难例')
    for root, dirs, files in os.walk(complex_anomaly_path):
        for file in files:
            img = plt.imread(os.path.join(root, file))
            fiber_data['complex_anomaly'].append(img)

    # 蕾丝数据集，建立字典，包含good, simple_anomaly, complex_anomaly
    lace_path = os.path.join(data_path, '蕾丝样本数据')
    lace_data = {
        'good': [],
        'simple_anomaly': [],
        'complex_anomaly': []
    }
    # TODO: 读取lace数据集
    return fiber_data, lace_data

def gray_img_decomp(img, display=False):
    """
    灰度图像分解
    :param img: 灰度图像
    :return:
    """
    lambda_0 = 1 / (math.sqrt(img.shape[0])) * 5
    L, S = lr_outlier_pursuit(img, max_iter=1000, tol=1e-6, display=True, lambda1=lambda_0)
    kernel_size = (img.shape[0] // 32, img.shape[1] // 32)
    k = kernel_size[0] * kernel_size[1]
    lambda1 = 1 * k / (math.sqrt(k)) * 0.2
    conv_L, conv_S = convlr_outlier_pursuit(img, kernel_size, max_iter=1000, tol=1e-6, display=True, lambda1=lambda1)
    if display:
        # 限定显示0~1范围的数据对比度过差，更换S显示方式为保留原自适应归一化方案，但将归一化前数值范围显示出来
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title('Original Image')
        plt.subplot(2, 3, 2)
        plt.imshow(L, cmap='gray', vmin=0, vmax=1)
        plt.title('Low Rank Image')
        plt.subplot(2, 3, 3)
        plt.imshow(S, cmap='gray')
        # 显示S与conv_S的数值范围作为小标题，仅显示科学计数法1位
        S_min = np.min(S)
        S_max = np.max(S)
        plt.title(f'Sparse Image\n min: {np.min(S):.1f} max: {np.max(S):.1f}')
        plt.subplot(2, 3, 5)
        plt.imshow(conv_L, cmap='gray', vmin=0, vmax=1)
        plt.title('Conv Low Rank Image')
        plt.subplot(2, 3, 6)
        plt.imshow(conv_S, cmap='gray')
        plt.title(f'Conv Sparse Image\n min: {np.min(conv_S):.1e} max: {np.max(conv_S):.1e}')




        plt.show()
    img_decomp = {
        'L': L,
        'S': S,
        'conv_L': conv_L,
        'conv_S': conv_S
    }
    return img_decomp


if __name__ == '__main__':
    # 对华为数据集进行异常检测
    # 1. 读取数据集
    # 读取华为数据集
    fiber_data, lace_data = read_huawei_data()
    fiber_good = fiber_data['good']
    fiber_simple_anomaly = fiber_data['simple_anomaly']
    fiber_complex_anomaly = fiber_data['complex_anomaly']
    # 2. 使用本文算法进行异常检测
    # 对三个数据集中的每个样本进行异常检测，每张图像产生一个
    for img in fiber_complex_anomaly:
        # 将图像转换为灰度图像
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        # 图片太大了，这里分割图像为16块，先进行小实验
        gray_imgs = []
        for i in range(4):
            for j in range(4):
                gray_imgs.append(gray_img[i * gray_img.shape[0] // 4: (i + 1) * gray_img.shape[0] // 4,
                                          j * gray_img.shape[1] // 4: (j + 1) * gray_img.shape[1] // 4])
                img_decomp = gray_img_decomp(gray_imgs[-1], display=True)
        # 保存异常检测结果
        path = 'results/huawei/fiber/good'
        # 保存img_decomp



    pass



