"""
Postprocess for organs in lungs.
Two segmentations: Dataset405 & Dataset501
Target organs: vein, artery, airway
Merged with multi-organ segmentations
"""

from typing import Union, List, Dict, Optional

import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label


def extract_lung_vessels(
    ct_path: str, 
    ct_image: sitk.Image, 
    label_mapping: Dict[int, int] = {3: 1, 4: 2, 5: 3},
    output_path: Optional[str] = None
) -> Optional[sitk.Image]:
    """
    提取Dataset501_LRAVANaug预测结果的动静脉、气道
    映射为3:1 4:2 5:3
    
    params:
        ct_path (str): Dataset501_LRAVANaug的预测结果路径
        ct_img (sitk.Image): Dataset501_LRAVANaug的预测结果
        label_mapping (dict): 默认为{3: 1, 4: 2, 5: 3}, 否则需要提供
        output_path (str): 如果提供则保存至output_path

    returns:
        sitk.Image 处理后的nii.gz文件
        如果提供output_path (str), 则保存
    """   
    if not isinstance(ct_path, str) or not os.path.isfile(ct_path):
        raise ValueError("Check ct_path...")
    if not isinstance(ct_image, sitk.Image):
        raise ValueError("Check ct_image...")
    if not isinstance(label_mapping, dict) or not all(isinstance(k, int) and isinstance(v, int) for k, v in label_mapping.items()):
        raise ValueError("Check label_mapping...")
    if output_path and not isinstance(output_path, str):
        raise ValueError("Check output_path...")

    if ct_path is not None:
        nii_img = sitk.ReadImage(ct_path)
        data = sitk.GetArrayFromImage(nii_img)
    else:
        data = sitk.GetArrayFromImage(nii_img)
    
    new_data = np.zeros_like(data, dtype=np.uint8)
    for old_label, new_label in label_mapping.items():
        new_data[data == old_label] = new_label
    
    new_nii = sitk.GetImageFromArray(new_data)
    new_nii.CopyInformation(nii_img)

    if output_path:
        sitk.WriteImage(new_nii, output_path)
    
    return new_nii


def modify_and_merge_labels(
    organ_path: str,
    organ_img: sitk.Image,
    lung_vessel_path: str,
    lung_vessel_img: sitk.Image,
    remove_labels: List[int] = [10, 11, 18],
    label_mapping: Dict[int, int] = {1: 10, 2: 18, 3: 11},
    output_path: Optional[str] = None
) -> Optional[sitk.Image]:
    """
    多标签文件(.nii.gz)标签修改与合并

    params:
        organ_path (str): Dataset405预测的多器官掩码路径
        organ_img (sitk.Image): 多器官掩码
        lung_vessel_path (str): 肺血管掩码路径
        lung_vessel_img: 肺血管掩码
        remove_labels (list): 需要置0的标签id, 默认为[10, 11, 18], 否则需要提供
        label_mapping (Dice): 新的id映射, 默认为{1: 10, 2: 18, 3: 11}, 否则需要提供
        output_path (str): 如果提供则保存至output_path

    returns:
        sitk.Image, 处理后的nii.gz文件
        如果提供output_path (str), 则保存
    """
    if not isinstance(organ_path, str) or not os.path.isfile(organ_path):
        raise ValueError("Check organ_path...")
    if not isinstance(organ_img, sitk.Image):
        raise ValueError("Check organ_img...")
    if not isinstance(lung_vessel_path, str) or not os.path.isfile(lung_vessel_path):
        raise ValueError("Check lung_vessel_path...")
    if not isinstance(lung_vessel_img, sitk.Image):
        raise ValueError("Check lung_vessel_img...")
    if not isinstance(remove_labels, list) or not all(isinstance(l, int) for l in remove_labels):
        raise ValueError("Check remove_labels...")
    if not isinstance(label_mapping, dict) or not all(isinstance(k, int) and isinstance(v, int) for k, v in label_mapping.items()):
        raise ValueError("Check label_mapping...")
    if output_path and not isinstance(output_path, str):
        raise ValueError("Check output_path...")
    
    if organ_path is not None:
        organ_img = sitk.ReadImage(organ_path)
        organ_data = sitk.GetArrayFromImage(organ_img)
    else:
        organ_data = sitk.GetArrayFromImage(organ_img)   
    for label in remove_labels:
        organ_data[organ_data == label] = 0

    if lung_vessel_path is not None:
        lung_vessel_img = sitk.ReadImage(lung_vessel_path)
        lung_vessel_data = sitk.GetArrayFromImage(lung_vessel_img)
    else:
        lung_vessel_data = sitk.GetArrayFromImage(lung_vessel_img)
    new_lung_vessel_data = np.zeros_like(lung_vessel_data, dtype=np.uint8)
    for old_label, new_label in label_mapping.items():
        new_lung_vessel_data[lung_vessel_data == old_label] = new_label
        
    mask = new_lung_vessel_data > 0
    organ_data[mask] = new_lung_vessel_data[mask]

    merged_img = sitk.GetImageFromArray(organ_data)
    merged_img.CopyInformation(organ_img)

    if output_path:
        sitk.WriteImage(merged_img, output_path)
    
    return merged_img


def remain_largest_regions(
    mask_path: str,
    mask_image: sitk.Image,
    organ_ids: List[int],
    output_path: Optional[str] = None
) -> Optional[sitk.Image]:
    """
    计算多标签掩码(.nii.gz)指定id的连通域, 保留最大连通域

    params:
        mask_path (str): 多标签文件路径
        mask_img (sitk.Image): 多标签文件
        organ_id (list): 指定的id列表
        output_path: 如果提供则保存至output_path
    
    returns:
        保留最大连通域后的nii.gz文件
        如果提供则保存至output_path
    """
    if not isinstance(mask_path, str) or not os.path.isfile(mask_path):
        raise ValueError("Check mask_path...")
    if not isinstance(mask_image, sitk.Image):
        raise ValueError("Check mask_img...")
    if not isinstance(organ_ids, list) or not all(isinstance(i, int) for i in organ_ids):
        raise ValueError("Check organ_ids...")
    if output_path and not isinstance(output_path, str):
        raise ValueError("Check output_path...")
    
    if mask_path is not None:
        mask_image = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_image)
    else:
        mask_array = sitk.GetArrayFromImage(mask_image)

    raw_array = mask_array.copy()
    for organ_id in organ_ids:
        raw_array[raw_array == organ_id] = 0

    new_mask_array = raw_array.copy()

    for organ_id in organ_ids:
        target_mask = (mask_array == organ_id).astype(np.uint8)
        
        labeled_array, num_features = label(target_mask)
        region_sizes = np.bincount(labeled_array.ravel())[1:] 
    
        if num_features > 0 and output_path:
            largest_label = np.argmax(region_sizes) + 1
            filtered_mask = (labeled_array == largest_label).astype(np.uint8)
            new_mask_array = np.where(filtered_mask == 1, organ_id, new_mask_array)
            
    if output_path:
        filtered_image = sitk.GetImageFromArray(new_mask_array)
        filtered_image.CopyInformation(mask_image)
        sitk.WriteImage(filtered_image, output_path)
        
    return filtered_image

