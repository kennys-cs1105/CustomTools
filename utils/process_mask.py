import SimpleITK as sitk
import numpy as np


def remove_outside_lungmask(lungmask_path, lung_vessels_path, output_path):
    """
    根据lungmask过滤人体外噪声

    给定lungmask掩码:{lungmask_path}/{seriesuid}.nii.gz
    血管分割掩码:{lung_vessels_path}/{seriesuid}.nii
    
    比较lungmask掩码和血管分割掩码, 若血管分割掩码在lungmask掩码之外, 则将其置0
    
    return: 比较厚的掩码
    """

    lungmask_img = sitk.ReadImage(lungmask_path)
    lung_vessels_img = sitk.ReadImage(lung_vessels_path)

    lungmask_data = sitk.GetArrayFromImage(lungmask_img)
    lung_vessels_data = sitk.GetArrayFromImage(lung_vessels_img)

    filtered_vessels_data = lung_vessels_data.copy()
    filtered_vessels_data[lungmask_data == 0] = 0  # 将 lungmask 外的区域置为 0

    filtered_vessels_img = sitk.GetImageFromArray(filtered_vessels_data)
    filtered_vessels_img.CopyInformation(lung_vessels_img)  # 复制空间信息

    sitk.WriteImage(filtered_vessels_img, output_path)



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



def extract_and_save_labels(input_file_path, output_directory, label_ids):
    """
    多标签nii.gz中的标签单独保存

    input_file_path: path to multi-label nii.gz file
    output_directory: directory to saved single-label nii.gz file
    label_ids: label ids list
    """

    image = sitk.ReadImage(input_file_path)
    image_array = sitk.GetArrayFromImage(image)

    for label_id in label_ids:

        label_array = np.zeros_like(image_array)
        label_array[image_array == label_id] = 1
        label_image = sitk.GetImageFromArray(label_array)     
        label_image.CopyInformation(image)

        output_file_path = f"{output_directory}/label_{label_id}.nii.gz" 
        sitk.WriteImage(label_image, output_file_path)