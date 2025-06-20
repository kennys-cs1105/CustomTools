import os
import SimpleITK as sitk
import numpy as np


#------------------------------------------------------------------------------------------------


def convert_dcm_to_nifti(input_path):
    """
    source: 给定一个文件夹路径或文件
    如果文件夹中包含dcm文件或符合DICOM3.0协议的医学图像文件, 则将其转换为文件名对应的.nii.gz文件
    如果不符合以上条件, 则抛出Error
    """
    if os.path.isdir(input_path):
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(input_path)
        if not dicom_names:
            raise ValueError("No DICOM files found in the folder.")
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    else:
        raise ValueError("Unsupported file or folder format")
    
    # Save the image as NIfTI
    save_path = os.path.join(input_path, f"{dicom_names[0]}.nii.gz")
    return sitk.WriteImage(image, save_path)


#------------------------------------------------------------------------------------------------


def convert_mhd_to_nifti(input_path):
    """
    Convert MHD files in the given input_path to NII files and save them to the same directory.
    Args:
    - input_path (str): The input path with MHD file.
    """
    # Check if the input path is a valid file
    if not os.path.isfile(input_path) or not input_path.lower().endswith('.mhd'):
        raise ValueError("The input path is not a valid MHD file")

    # Read the image using SimpleITK
    itk_img = sitk.ReadImage(input_path)

    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    # Create the output file name in the same directory
    output_file = os.path.join(os.path.dirname(input_path), f"{file_name}.nii.gz")

    # Save the image to the output path
    return sitk.WriteImage(itk_img, output_file)


#------------------------------------------------------------------------------------------------


def get_image(source):
    """
    根据提供的源数据文件形式及后缀, 返回相应的医学图像读取方式, 统一转化为.nii.gz格式

    参数：
    source: 源数据文件路径, 可以是文件夹也可以是文件
    如果source为dcm文件的文件夹路径, 则将dcm转化为.nii.gz
    如果source为.nii或.nii.gz文件, 则读取
    如果source为mhd或mha格式, 则转化为.nii.gz
    """
    if os.path.isdir(source):
    # Check if the folder contains DICOM files
        files = os.listdir(source)
        if any(file.lower().endswith(('.dcm', '.ima')) for file in files):
            return convert_dcm_to_nifti(source)
        else:
            raise ValueError("The folder does not contain any DICOM files.")
        
    elif os.path.isfile(source):
        if source.lower().endswith(('.nii', '.nii.gz')):
            # Read and copy NIfTI file
            return sitk.ReadImage(source)

        elif os.path.isfile(source):
            if source.lower().endswith(('.mha', '.mhd')):
                return convert_mhd_to_nifti(source)
    else:
        raise ValueError("The source path is not a valid file or folder")


#------------------------------------------------------------------------------------------------

# test
# source = 'D:/kennys/temporary/ks_nodule/experiment/test_data/LKDS-00091.mhd' 

# get_image(source)

