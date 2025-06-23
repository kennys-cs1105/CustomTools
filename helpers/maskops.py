from typing import Dict, Callable, List, Union, Optional, Any

import os
import SimpleITK as sitk
import logging
import concurrent.futures
import traceback
import time
import numpy as np
from nibabel.nifti1 import Nifti1Image
import nibabel as nib
from skimage import filters

from utils.iotools import load_mask, get_sitk_data, get_nib_data


def merge_masks(
    organ_mask_path: str,
    rib_mask_path: str,
    body_mask_path: str,
    fine_lobe_mask_path: Optional[str],
    lobe_mask_path: Optional[str],
    bedsplit_mask_path: str,
    input_nifti_path: str,
    organ_mapping: Dict[int, int],
    skin_mapping: Dict[int, int],
    rib_mapping: Dict[int, int],
    lobe_mapping: Dict[int, int],
    body_mapping: Dict[int, int],
    bedsplit_mapping: Dict[int, int],
    fine_seg: bool = False,
    bedsplit_seg: bool = False
) -> sitk.Image:
    """
    Merge multiple masks into a single mask using the provided mappings.

    Args:
        organ_mask_path: Path to the organ mask file.
        rib_mask_path: Path to the rib mask file.
        body_mask_path: Path to the body mask file.
        fine_lobe_mask_path: Path to the fine lobe mask file (if fine_seg is True).
        lobe_mask_path: Path to the lobe mask file (if fine_seg is False).
        bedsplit_mask_path: Path to the bedsplit mask file.
        input_nifti_path: Path to the input NIfTI file for metadata.
        organ_mapping: Mapping of organ mask values to new values.
        skin_mapping: Mapping of skin mask values to new values.
        rib_mapping: Mapping of rib mask values to new values.
        lobe_mapping: Mapping of lobe mask values to new values.
        body_mapping: Mapping of body mask values to new values.
        bedsplit_mapping: Mapping of bedsplit mask values to new values.
        fine_seg: Whether to use fine lobe segmentation (default: False).

    Returns:
        sitk.Image: The merged mask as a SimpleITK image.
    """
    # Load organ mask
    organ_mask_array = load_mask(organ_mask_path, "organ")

    # Load rib mask
    rib_mask_array = load_mask(rib_mask_path, "rib")

    # Load body mask
    body_mask_array = load_mask(os.path.join(body_mask_path, "body.nii.gz"), "body")

    # Generate skin mask using Sobel edge detection
    logging.info("Generating skin mask from body mask...")
    sobel_edge = filters.sobel(body_mask_array)
    skin_mask_array = (sobel_edge > 0).astype(np.uint8)
    if skin_mask_array.size == 0:
        logging.warning("Skin mask array is empty. Initializing as zero array.")
        skin_mask_array = np.zeros_like(body_mask_array, dtype=np.uint8)

    # Load lobe mask (fine or coarse)
    if fine_seg:
        if os.path.exists(fine_lobe_mask_path):
            fine_lobe_mask_array = load_mask(fine_lobe_mask_path, "lobe")
    else:
        if os.path.exists(lobe_mask_path):
            lobe_mask_array = load_mask(lobe_mask_path, "lobe")

    # Initialize combined mask array
    combined_mask_array = np.zeros_like(organ_mask_array, dtype=np.uint8)

    # Apply mappings to merge masks
    logging.info("Applying organ mapping...")
    for origin_value, new_value in organ_mapping.items():
        combined_mask_array[organ_mask_array == origin_value] = new_value

    logging.info("Applying skin mapping...")
    for origin_value, new_value in skin_mapping.items():
        combined_mask_array[skin_mask_array == origin_value] = new_value

    logging.info("Applying rib mapping...")
    rib_mask_array = np.isin(rib_mask_array, list(rib_mapping.keys()))
    combined_mask_array[rib_mask_array] = list(rib_mapping.values())[0]

    # Check whether use fine seg
    if fine_seg:
        logging.info("Applying fine lobe mapping...")
        for origin_value, new_value in lobe_mapping.items():
            combined_mask_array[fine_lobe_mask_array == origin_value] = new_value

    logging.info("Applying body mapping...")
    for origin_value, new_value in body_mapping.items():
        body_mask_array[body_mask_array == origin_value] = new_value

    if bedsplit_seg:
        bedsplit_mask_array = load_mask(bedsplit_mask_path, "bedsplit")
        logging.info("Applying bedsplit mapping...")
        for origin_value, new_value in bedsplit_mapping.items():
            combined_mask_array[bedsplit_mask_array == origin_value] = new_value

    # Merge combined mask with body mask
    logging.info("Merging combined mask with body mask...")
    merged_mask_array = np.where(combined_mask_array > 0, combined_mask_array, body_mask_array)

    # Convert merged mask to SimpleITK image
    logging.info("Converting merged mask to SimpleITK image...")
    merged_mask_image = sitk.GetImageFromArray(merged_mask_array)
    merged_mask_image.CopyInformation(sitk.ReadImage(input_nifti_path))

    return merged_mask_image


def get_target_region(ct_file: Union[str, Nifti1Image], 
                      mask_file: Union[str, Nifti1Image], 
                      mapping: Dict, 
                      output_dir:str):
    """
    功能: 基于掩码将人体ct扫描分为肺区域和非肺区域

    Args:
        ct_path: CT 扫描路径 (nii.gz 文件)
        mask_path: 目标掩码路径, 可以是单标签掩码文件(id=0,1) 也可以是多标签掩码文件
        mapping: 器官名称对应掩码id的Dict, 例如{"lung": [1, 2]}, id可以是int 也可以是list
        output_dir: 输出保存路径，将保存器官区域和非器官区域
    """

    ct_nii = get_nib_data(ct_file)
    mask_nii = get_nib_data(mask_file)
    
    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    if ct_data.shape != mask_data.shape:
        raise ValueError("CT 图像和掩码图像的尺寸不一致。")
    
    os.makedirs(output_dir, exist_ok=True)

    for organ_name, ids in mapping.items():
        if isinstance(ids, int):
            ids = [ids]

        organ_mask = np.isin(mask_data, ids)

        organ_region = ct_data * organ_mask
        organ_region_nii = nib.Nifti1Image(organ_region, affine=ct_nii.affine, header=ct_nii.header)
        organ_path = os.path.join(output_dir, f"{organ_name}.nii.gz")
        nib.save(organ_region_nii, organ_path)
        logging.info(f"{organ_name} region saved...")


def read_labels(file_path):
    """
    读取nii.gz, 获取标签id

    input: path to nii.gz

    return: label id list -> [0,1,2,3,4]
    """
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)
    unique_labels = set(image_array.flatten())

    return unique_labels


def extract_and_save_labels(input_file_path, label_id, output_path=None):
    """
    多标签nii.gz中的标签单独保存

    input_file_path: path to multi-label nii.gz file
    output_path: path to saved single-label nii.gz file
    label_id: label id
    """

    image = sitk.ReadImage(input_file_path)
    image_array = sitk.GetArrayFromImage(image)

    label_array = np.zeros_like(image_array)
    label_array[image_array == label_id] = 1
    label_image = sitk.GetImageFromArray(label_array)     
    label_image.CopyInformation(image)

    if output_path is not None:
        sitk.WriteImage(label_image, output_path)
    else:
        return label_image