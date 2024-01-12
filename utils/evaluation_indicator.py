"""
用来写一些评价指标函数，如PSNR、SSIM等
"""

import numpy as np
# 先自己写一个PSNR函数，不用skimage的
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr


def psnr(img1, img2):
    """
    计算两个图像的PSNR值
    :param img1: 图像1
    :param img2: 图像2
    :return: PSNR值
    """
    # 先将图像转换至0-255
    img1 = img1 / (img1.max() - img1.min()) * 255.0
    img2 = img2 / (img2.max() - img2.min()) * 255.0
    # 计算mse
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * np.log10(255.0 ** 2 / mse)



