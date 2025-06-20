import os

import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops

from naviutils.iotools import get_sitk_image_and_array


def remove_extrapulmonary_noise(
    organ_mask, processing_mask, output_dir, save_name="processed.nii.gz"
):
    """
    移除肺外噪声的三维分析函数

    参数:
    organ_mask (str): 器官掩膜文件路径
    processing_mask (str): 待处理掩膜文件路径
    output_dir (str): 输出目录
    save_name (str): 保存文件名，默认为 "processed.nii.gz"

    返回:
    None
    """
    output_path = os.path.join(output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)

    organ_image, organ_array = get_sitk_image_and_array(source=organ_mask)
    processing_image, processing_array = get_sitk_image_and_array(
        source=processing_mask
    )

    if organ_array.shape != processing_array.shape:
        raise ValueError("待处理的掩膜与器官掩膜尺寸不符。")

    # 三维连通区域分析
    lung_array = np.where(organ_array == 4, 1, 0)
    labeled_array, num_features = label(lung_array, connectivity=3, return_num=True)

    # 找到最大的两个连通区域成分
    regions = regionprops(labeled_array)
    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    largest_two_regions = sorted_regions[:2]

    # 找到包含这两个肺部数据的最小长方体
    min_bounds = []
    max_bounds = []
    for region in largest_two_regions:
        minr, minc, minz, maxr, maxc, maxz = region.bbox
        min_bounds.append((minr, minc, minz))
        max_bounds.append((maxr, maxc, maxz))

    min_bounds = np.array(min_bounds)
    max_bounds = np.array(max_bounds)

    # 计算最小长方体的边界
    min_z = np.min(min_bounds[:, 0])
    min_y = np.min(min_bounds[:, 1])
    min_x = np.min(min_bounds[:, 2])
    max_z = np.max(max_bounds[:, 0])
    max_y = np.max(max_bounds[:, 1])
    max_x = np.max(max_bounds[:, 2])

    output_array = np.zeros_like(processing_array)

    # 限制三个轴上的数据
    output_array[:, min_y : max_y + 1, min_x : max_x + 1] = processing_array[
        :, min_y : max_y + 1, min_x : max_x + 1
    ]

    # 肺外的血管去除，肺内的血管区域保留
    output_array[(lung_array != 1) & (output_array == 1)] = 0

    # 创建一个新的SimpleITK图像对象，并将处理后的结果数组复制到新图像中
    output_image = sitk.GetImageFromArray(output_array)
    output_image.CopyInformation(processing_image)

    # 保存新图像为nii格式的文件
    sitk.WriteImage(output_image, output_path)


def filter_false_positive_nodules(mask_array, lung_value=[2, 3, 4, 5], nodule_value=1):
    """
    过滤假阳性结节，保留位于肺部区域内的结节。
    适用于肺掩膜和结节掩膜在同一个文件中的情况。
    
    参数：
    - mask_array (numpy.ndarray): 一个三维numpy数组，表示掩膜。
    - lung_value (int或list): 表示肺部区域的值，可以是单个整数或整数列表。
    - nodule_value (int或list): 表示结节的值，可以是单个整数或整数列表。

    返回：
    - numpy.ndarray: 一个新的掩膜，已排除肺部区域外的假阳性结节。

    过程说明：
    1. 提取掩膜中肺部区域的二值化，以确定哪些结节位于肺部内部。
    2. 将肺部区域外的假阳性结节从掩膜中排除。
    3. 返回过滤后的掩膜。
    """

    # 1. 提取肺和结节的二值化
    modified_mask = mask_array.copy()
    # 处理逻辑尚未实现
    return modified_mask
