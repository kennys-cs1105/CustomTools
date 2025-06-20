import SimpleITK as sitk
import numpy as np
import os


def crop_ct_mask_in_z_axis(
        ct_img: sitk.Image = None,
        mask_img: sitk.Image = None,
        slice_count: int = 80,
        output_dir: str = None,
        ct_path: str = None,
        mask_path: str = None
):
    """
    对肺CT图像和其掩码在Z轴上进行裁切, 每次裁切80个切片。

    params:
        ct_img (sitk.Image): 输入的肺CT图像对象
        mask_img (sitk.Image): 输入的掩码图像对象
        ct_path (str): 输入的CT图像路径 (.nii.gz)
        mask_path (str): 输入的掩码图像路径 (.nii.gz)
        slice_count (int): 每次裁切的切片数

    return:
        None: 裁切结果会保存为新的NIfTI文件
    """
    # 加载CT图像和掩码图像
    if ct_img is None:
        if ct_path is None:
            raise ValueError("Please check the ct image.")
        ct_img = sitk.ReadImage(ct_path)

    if mask_img is None:
        if mask_path is None:
            raise ValueError("Please check the mask image.")
        mask_img = sitk.ReadImage(mask_path)

    seriesuid = os.path.basename(mask_path).split(".nii.gz")[0]

    # 获取CT图像的尺寸
    ct_size = ct_img.GetSize()
    z_size = ct_size[2]

    # 获取每次裁切的步长和大小
    slice_size = slice_count  # 每次裁切80个切片
    step_size = slice_size  # 每次裁切的步长为80个切片

    # 在z轴方向上进行裁切
    for i in range(0, z_size, step_size):
        start_z = i
        end_z = min(i + slice_size, z_size)
        
        # 创建裁切区域的起始点和大小
        region = [0, 0, start_z]  # (x, y, z)
        region_size = [ct_size[0], ct_size[1], end_z - start_z]  # 宽度，高度，z轴长度
        
        # 使用 RegionOfInterest 来裁切图像
        ct_crop = sitk.RegionOfInterest(ct_img, region_size, region)
        mask_crop = sitk.RegionOfInterest(mask_img, region_size, region)
        
        # 保存裁切后的图像
        ct_output_path = os.path.join(output_dir, f"{seriesuid}_{start_z}_{end_z}_0000.nii.gz")
        mask_output_path = os.path.join(output_dir, f"{seriesuid}_{start_z}_{end_z}.nii.gz")
        
        sitk.WriteImage(ct_crop, ct_output_path)
        sitk.WriteImage(mask_crop, mask_output_path)
        print(f"保存裁切的CT图像: {ct_output_path}")
        print(f"保存裁切的掩码图像: {mask_output_path}")
