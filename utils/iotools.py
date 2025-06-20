import SimpleITK as sitk
from nibabel.nifti1 import Nifti1Image
import nibabel as nib
import numpy as np
import os
import logging


def get_sitk_image_and_array(source):
    """
    Convert the input source to a SimpleITK image and a numpy array.

    Parameters:
    source (str or sitk.Image or np.ndarray): The input source which can be a file path,
                                              SimpleITK image, numpy array, or a directory.

    Returns:
    tuple: (image, array) - SimpleITK image and the converted numpy array.

    Raises:
    ValueError: If the source type or file format is unsupported.
    """
    if isinstance(source, str):
        if os.path.isdir(source):
            # 读取DICOM文件夹
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(source)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            array = sitk.GetArrayFromImage(image)
        elif source.endswith((".nii", ".nii.gz", ".mhd")):
            # 读取nifti或mhd文件
            image = sitk.ReadImage(source)
            array = sitk.GetArrayFromImage(image)
        elif source.endswith(".npy"):
            # 读取npy文件
            image = None
            array = np.load(source)
        else:
            raise ValueError("Unsupported file format")
    elif isinstance(source, sitk.Image):
        image = source
        array = sitk.GetArrayFromImage(image)
    elif isinstance(source, np.ndarray):
        image = None
        array = source
    else:
        raise TypeError("Unsupported source type")

    return image, array


# 示例使用
# image, array = get_array('/path/to/dicom/folder')
# image, array = get_array('/path/to/file.nii')
# image, array = get_array(sitk_image)
# image, array = get_array(numpy_array)
# image, array = get_array('/path/to/file.npy')


def read_mask(mask_path):
    """
    从给定的文件路径读取掩码。

    参数:
    mask_path (str): 掩码的文件路径。

    返回:
    np.ndarray: 读取的掩码数组。
    """
    if mask_path.endswith(".npy"):
        return np.load(mask_path)
    else:
        return sitk.GetArrayFromImage(sitk.ReadImage(mask_path))


def get_array(source):
    """
    根据提供的源数据类型，返回相应的NumPy数组。
    array统一为(x,y,z)

    参数:
    source: 可以是文件路径字符串、SimpleITK.Image对象或NumPy数组。

    返回:
    NumPy数组: 从给定的源数据中提取或直接返回的NumPy数组。

    如果源数据是字符串，假定它是掩码文件的路径，并调用read_mask函数来读取。
    如果源数据是SimpleITK.Image对象，使用SimpleITK将其转换为NumPy数组。
    如果源数据已经是NumPy数组，则直接返回。
    如果源数据类型不支持，将抛出ValueError。
    """
    if isinstance(source, str):
        # 假定source是文件路径，调用read_mask函数读取掩码文件
        return read_mask(source)
    elif isinstance(source, sitk.Image):
        # 如果source是SimpleITK.Image对象，转换为NumPy数组并转换维度
        transposed_data = sitk.GetArrayFromImage(source).transpose(2,1,0)
        return transposed_data
    elif isinstance(source, np.ndarray):
        # 如果source已经是NumPy数组，直接返回
        return source
    elif isinstance(source, Nifti1Image):
        return source.get_fdata()
    else:
        # 如果source不是上述支持的类型，抛出异常
        raise TypeError("Unsupported source type.")
    

def load_mask(mask_path: str, mask_name: str):
    """
    Helper function to load a mask and validate its shape
    
    Args:
        mask_path (str): Mask path
        mask_name (str): Specific mask name
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"{mask_name} mask file not found. Please check {mask_path}")
    logging.info(f"Loading {mask_name} mask from {mask_path}")
    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    if mask_array.ndim != 3:
        raise ValueError(f"{mask_name} mask must be 3D array, but got shape {mask_array.shape}")
    if mask_array.size == 0:
        raise ValueError(f"{mask_name} mask is empty, please check!")
    
    return mask_array


def get_sitk_data(source):
    """
    Get sitk.Image.

    Args:
        source: input file, could be str or sitk.Image
    """
    if isinstance(source, str):
        return sitk.ReadImage(source)
    elif isinstance(source, sitk.Image):
        return source
    else:
        raise TypeError("Unsupported source type.")
    


def get_nib_data(source):
    """
    Get nib data

    Args:
        source: input file, could be str or Nifti1Image
    """
    if isinstance(source, str):
        return nib.load(source)
    elif isinstance(source, Nifti1Image):
        return source
    else:
        raise TypeError("Unsupported source type.")
