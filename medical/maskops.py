# 导入必要的库

from functools import reduce

import numpy as np


def create_solid_sphere(radius, center, size, fill_value=1):
    """
    创建一个实心球体的掩膜。注意：只能用于三向等距的CT图像。

    参数:
    radius -- 球体的半径（体素单位）
    center -- 球体中心的图像坐标，形式为(x, y, z)
    size -- CT图像的尺寸，形式为(width, height, depth)，与SimpleITK GetSize()的结果相同
    fill_value -- 掩膜填充的数值，默认为1

    返回:
    一个表示实心球体的三维NumPy数组，其坐标轴顺序与SimpleITK GetArrayFromImage的相同。
    """
    # 创建空间，注意坐标轴顺序是(z, y, x)
    space = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    # 生成坐标网格，注意坐标轴顺序是(z, y, x)
    z, y, x = np.ogrid[: size[2], : size[1], : size[0]]
    # 计算每个点到中心的距离，注意中心坐标的顺序是(x, y, z)
    dist_from_center = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    )
    # 设置掩膜
    space[dist_from_center <= radius] = fill_value
    return space


def combine_masks(*masks, preserve_values=False):
    """
    合并多个形状相同的掩膜数组。

    参数:
    masks -- 一个或多个NumPy数组，代表要合并的掩膜。
    preserve_values -- 一个布尔值，表示是否保留原始掩膜值。

    返回:
    一个NumPy数组，表示合并后的掩膜。
    """
    # 确保至少有一个掩膜被传入
    if not masks:
        raise ValueError("至少需要一个掩膜来进行合并。")

    if preserve_values:
        # 使用np.maximum.reduce来保留原始掩膜值
        combined_mask = np.maximum.reduce(masks)
    else:
        # 使用reduce和np.logical_or合并所有掩膜
        combined_mask = reduce(np.logical_or, masks)

    combined_mask = combined_mask.astype(np.uint8)
    return combined_mask


def create_solid_ellipsoid(a, b, c, center, size, fill_value=1):
    """
    创建一个实心椭球体的掩膜。

    参数:
    a, b, c -- 椭球在x, y, z轴方向的半轴长度（体素单位）
    center -- 椭球中心的图像坐标，形式为(x, y, z)
    size -- CT图像的尺寸，形式为(width, height, depth)，与SimpleITK GetSize()的结果相同
    fill_value -- 掩膜填充的数值，默认为1

    返回:
    一个表示实心椭球体的三维NumPy数组。
    """
    # 创建空间，注意坐标轴顺序是(z, y, x)
    space = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    # 生成坐标网格，注意坐标轴顺序是(z, y, x)
    z, y, x = np.ogrid[: size[2], : size[1], : size[0]]
    # 计算每个点到中心的距离
    dist_from_center = (
        ((x - center[0]) ** 2 / a**2)
        + ((y - center[1]) ** 2 / b**2)
        + ((z - center[2]) ** 2 / c**2)
    )
    # 设置掩膜
    space[dist_from_center <= 1] = fill_value
    return space
