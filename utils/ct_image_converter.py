import logging
import os

import SimpleITK as sitk

# 直接使用当前环境中已配置好的 logging 句柄
logger = logging.getLogger(__name__)


def dicom_to_nifti(dicom_dir, series_id=None, add_suffix=False, save_dir=None):
    """
    Convert a DICOM series to a NIfTI file.

    Parameters:
    dicom_dir (str): Path to the DICOM series folder.
    series_id (str, optional): Prefix for the saved file name. Defaults to None.
    add_suffix (bool, optional): If True, adds "_0000.nii.gz" to the file name. Defaults to False.
    save_dir (str, optional): Directory to save the NIfTI file. Defaults to None, which means dicom_dir will be used.

    Returns:
    tuple: (series_id, save_path) - Series ID and path to the saved NIfTI file.

    Raises:
    ValueError: If the provided path is not a directory.
    """
    if not os.path.isdir(dicom_dir):
        raise ValueError("Unsupported file or folder format")

    # 创建读取器对象
    reader = sitk.ImageSeriesReader()
    # 获取所有DICOM系列的ID
    series_IDs = reader.GetGDCMSeriesIDs(dicom_dir)
    # 获取DICOM序列文件名列表
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    # 确定保存文件的名称前缀
    if series_id is None:
        series_id = series_IDs[0]  # series_IDs是一个tuple

    # 确定保存目录
    if save_dir is None:
        save_dir = dicom_dir

    # 确定保存文件的后缀
    suffix = "_0000.nii.gz" if add_suffix else ".nii.gz"
    save_path = os.path.join(save_dir, f"{series_id}{suffix}")
    # 如果文件已经存在，则跳过保存步骤
    if os.path.exists(save_path):
        logger.info(f"{save_path} exists, skip convertion.")
        return series_id, save_path

    # 设置读取的文件名为DICOM序列文件名列表
    reader.SetFileNames(dicom_names)
    # 读取图像数据
    image = reader.Execute()
    os.makedirs(save_dir, exist_ok=True)
    # 保存图像为NIfTI格式
    sitk.WriteImage(image, save_path)

    return series_id, save_path


if __name__ == "__main__":

    save_path = dicom_to_nifti(
        "/path/to/dicom",
        series_id="example_series",
        add_suffix=True,
        save_dir="/path/to/save",
    )
    print(f"Saved NIfTI file at: {save_path}")
