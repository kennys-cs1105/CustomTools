from typing import Union, List, Any

from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import random

from utils.iotools import get_sitk_data


def generate_roi(
        ct_path: Union[str, sitk.Image],
        organ_mask_path: Union[str, sitk.Image],
        organ_ids: Union[List[int], int], 
        output_path: str, 
        save: bool = False
) -> sitk.Image:
    """
    Generate ROI region

    Args:
        ct_path ( Union[str, sitk.Image]):
    """
    assert isinstance(ct_path, (str, sitk.Image)), "CT path should be provided correctly."
    ct_image = get_sitk_data(ct_path)
    ct_array = sitk.GetArrayFromImage(ct_image)

    assert isinstance(organ_mask_path, (str, sitk.Image)), "Organ mask path should be provided correctly."
    mask_image = get_sitk_data(organ_mask_path)
    
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_image)
    
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = -float('inf'), -float('inf'), -float('inf')
    
    for organ_id in organ_ids:
        if organ_id in label_shape_filter.GetLabels():
            bbox = label_shape_filter.GetBoundingBox(organ_id)
            min_x = min(min_x, bbox[0])
            min_y = min(min_y, bbox[1])
            min_z = min(min_z, bbox[2])
            max_x = max(max_x, bbox[0] + bbox[3])
            max_y = max(max_y, bbox[1] + bbox[4])
            max_z = max(max_z, bbox[2] + bbox[5])

    ct_array[:min_z] = -2000
    ct_array[max_z:] = -2000
    ct_array[:, :min_y] = -2000
    ct_array[:, max_y:] = -2000
    ct_array[:, :, :min_x] = -2000
    ct_array[:, :, max_x:] = -2000
    
    roi_image = sitk.GetImageFromArray(ct_array)
    roi_image.CopyInformation(ct_image)

    if save:
        sitk.WriteImage(roi_image, output_path)
    return roi_image


def crop_region_ct(
        seriesuid: str,
        ct_path: Union[str, sitk.Image],
        ct_output_dir: str,
        min_slice_count: int = 80,
        max_slice_count: int = 125,
):
    """
    Crop ct file into region files in z-axis. Slice random in [80,125]

    Args:
        seriesuid (str): seriesuid of ct file
        ct_path (Union[str, sitk.Image]): ct source, could be provided as str or sitk.Image
        min_slice_count (int): minimum slices to crop, default=80
        max_slice_count (int): maximum slices to crop, default=125
        ct_output_dir (int): cropped regions files directory
    """
    assert isinstance(ct_path, (str, sitk.Image)), "CT path should be provided correctly."

    ct_img = get_sitk_data(ct_path)
    ct_size = ct_img.GetSize()
    z_size = ct_size[2]
    
    os.makedirs(ct_output_dir, exist_ok=True)

    i = 0
    while i < z_size:
        slice_count = random.randint(min_slice_count, max_slice_count)
        if i + slice_count > z_size:
            slice_count = z_size - i

        start_z = i
        end_z = i + slice_count

        region = [0, 0, start_z]  # (x, y, z)
        region_size = [ct_size[0], ct_size[1], end_z - start_z]
        
        ct_crop = sitk.RegionOfInterest(ct_img, region_size, region)

        i = end_z
        
        ct_output_path = os.path.join(ct_output_dir, f"{seriesuid}_{start_z}_{end_z}_0000.nii.gz")
        sitk.WriteImage(ct_crop, ct_output_path)


def reconstruct_mask_from_slices(
        seriesuid: str,
        original_ct_path: Union[str, sitk.Image],
        mask_slices_dir: str,
        output_mask_path: str
):
    """
    merge the region ct prediction to the completed one

    Args
        seriesuid (str): seriesuid of ct file
        original_ct_path (Union[str, sitk.Image]): original ct source to get ct size, could be provied as str or sitk.Image 
        mask_slices_dir (str): region segmentation files directory
        output_mask_path (str): merged mask file path
    """
    assert isinstance(original_ct_path, (str, sitk.Image)), "Original ct path should be provided correctly."

    ct_img = get_sitk_data(original_ct_path)
    ct_size = ct_img.GetSize()
    full_mask_array = np.zeros(ct_size[::-1], dtype=np.uint8)

    for filename in sorted(os.listdir(mask_slices_dir)):
        if filename.endswith(".nii.gz") and filename.startswith(seriesuid):
            parts = filename.replace(".nii.gz", "").split("_")
            start_z, end_z = int(parts[-2]), int(parts[-1])
            
            mask_path = os.path.join(mask_slices_dir, filename)
            mask_img = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_img)
            
            full_mask_array[start_z:end_z, :, :] = mask_array

    full_mask_img = sitk.GetImageFromArray(full_mask_array)
    full_mask_img.CopyInformation(ct_img) 
    sitk.WriteImage(full_mask_img, output_mask_path)