from typing import List, Union, Any

import numpy as np
import SimpleITK as sitk
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from scipy.ndimage import center_of_mass, label

from utils.iotools import get_array
from utils.mask_cleaning import filter_false_positive_nodules


def find_nodule_centers(binary_array):
    """
    计算二进制数组中每个连通域的重心坐标。

    参数：
    binary_array (ndarray): 输入的二进制数组，通常为3D图像的布尔值数组，
                            其中1表示目标区域，0表示背景。

    返回：
    ndarray: 一个数组，形状为(num_features, 3)，
             其中每一行包含一个连通域的重心坐标(z, y, x)。

    过程说明：
    1. 使用标记算法对二进制数组中的连通区域进行标记。
    2. 初始化一个数组用于存储每个连通域的重心坐标，数组的大小为(num_features, 3)，
       其中num_features是连通域的数量，3表示3个坐标(z, y, x)。
    3. 遍历每个连通域，通过计算其重心坐标来填充预分配的数组。
       重心的计算使用`center_of_mass`函数，该函数根据标记数组和输入数组
       计算指定连通域的重心。
    4. 返回包含所有连通域重心坐标的数组。

    注意：
    - 连通域的标记是从1开始的，因此循环范围为1到num_features + 1。
    - 重心坐标是以(z, y, x)的顺序存储的。
    """

    # 标记连通域
    labeled_array, num_features = label(binary_array)
    centers = np.zeros((num_features, 3))  # 预分配数组，3个坐标(z, y, x)

    for index in range(1, num_features + 1):
        # 计算每个连通域的重心
        # 必须循环计算。否则会得到一个拟合的中心。
        center = center_of_mass(input=binary_array, labels=labeled_array, index=index)
        centers[index - 1] = center  # 存储重心坐标

    return centers


def get_nodule_z_centers(
    pred_source: Any,
    pred_nodule_value: Union[List[int], int] =[12, 13],
) -> List[int]:
    """
    Analyze lung nodules in segmentation results, get z-axis coord of lung nodules center.

    Args:
        pred_source ( Union[str, sitk.Image, np.ndarray]): Data source of segmentation results, maybe path or Image or array
        pred_nodule_value (Union[List[int], int]): Lung nodule array value, should be a list or integer

    Return:
        list: A list of lung nodule center in z-axis
    """
    assert isinstance(pred_source, (str, sitk.Image, np.ndarray, Nifti1Image)), "Case must be provided correctly."

    pred_mask = get_array(pred_source)
    if pred_mask.ndim != 3:
        raise ValueError("Pred mask array should be 3D.")
    if pred_mask.size == 0:
        raise ValueError("Pred mask array should not be empty, please check!")

    processed_pred_mask = filter_false_positive_nodules(pred_mask) # TODO
    pred_binary = np.isin(processed_pred_mask, pred_nodule_value).astype(np.uint8)

    if np.sum(pred_binary) == 0:
        return []
    
    centers = find_nodule_centers(pred_binary)
    if not centers.any():
        return []
    
    centers_array = np.round(np.array(centers)[0, :]).astype(int)  # 仅提取 Z 轴坐标
    return np.unique(centers_array).tolist()

    # centers = find_nodule_centers(pred_binary)
    # center_zs = [int(round(z)) for z, y, x in centers]
    # unique_sorted_zs = sorted(set(center_zs))

    # return unique_sorted_zs
