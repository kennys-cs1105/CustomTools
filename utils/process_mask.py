import SimpleITK as sitk
import numpy as np




def merge_masks(mask1_path, mask2_path, output_path):
    """
    使用SimpleITK合并两个NIfTI格式的掩码文件，并保存结果。

    参数：
    mask1_path: str, mask1.nii.gz 文件的路径
    mask2_path: str, mask2.nii.gz 文件的路径
    output_path: str, 合并后mask3.nii.gz 文件的保存路径

    返回：
    output_path: str, 保存的合并后mask3.nii.gz 文件的路径
    """

    mask1 = sitk.ReadImage(mask1_path)
    mask2 = sitk.ReadImage(mask2_path)

    if mask1.GetSize() != mask2.GetSize() or mask1.GetOrigin() != mask2.GetOrigin() or mask1.GetSpacing() != mask2.GetSpacing() or mask1.GetDirection() != mask2.GetDirection():
        raise ValueError("The masks are not aligned. Please ensure the masks are in the same space.")

    mask1_data = mask1 > 0
    mask2_data = mask2 > 0
    merged_data = sitk.Or(mask1_data, mask2_data)

    sitk.WriteImage(merged_data, output_path)


def merge_multi_label_mask(mask_paths, output_path):
    """
    使用SimpleITK合并多个NIfTI格式的掩码文件, 并保存结果。

    参数：
    - mask_paths: list of str, 掩码文件路径的列表 -> ['path/to/mask1.nii.gz', 'path/to/mask2.nii.gz', 'path/to/mask3.nii.gz']
    - output_path: str, 合并后mask的保存路径

    返回：
    - output_path: str, 保存的合并后mask的路径
    """
    if not mask_paths:
        raise ValueError("The list of mask paths is empty.")

    # 初始化合并后的掩码数据
    merged_image = None

    for label, mask_path in enumerate(mask_paths, start=1):
        # 读取当前掩码文件
        mask = sitk.ReadImage(mask_path)
        mask_data = sitk.GetArrayFromImage(mask)
        
        # 检查合并后掩码的初始化
        if merged_image is None:
            # 初始化一个与第一个掩码具有相同尺寸和仿射的图像
            merged_image = sitk.Image(mask.GetSize(), sitk.sitkUInt8)
            merged_image.CopyInformation(mask)
            merged_image_array = sitk.GetArrayFromImage(merged_image)
        
        # 将当前掩码中大于0的部分设置为label
        label_data = (mask_data > 0).astype(merged_image_array.dtype) * label
        merged_image_array[label_data > 0] = label_data[label_data > 0]

    # 将合并后的数组设置回SimpleITK图像对象
    merged_image = sitk.GetImageFromArray(merged_image_array)
    merged_image.CopyInformation(mask)  # 保持空间信息

    # 将合并后的图像保存到指定路径
    sitk.WriteImage(merged_image, output_path)

"""
测试用例:
organ_list = ['lung_lower_lobe_left.nii.gz', 'lung_lower_lobe_right.nii.gz', 'lung_middle_lobe_right.nii.gz', 'lung_upper_lobe_left.nii.gz', 'lung_upper_lobe_right.nii.gz']

for seriesuid in tqdm(unique_seriesuids):

    path_list = []
    ct_path = os.path.join(totalseg_prediction_dir, f"{seriesuid}")

    for organ_name in organ_list:
        organ_path = os.path.join(ct_path, organ_name)
        path_list.append(organ_path)

    output_path = os.path.join(root_dir, "processing/combined_organ", f"{seriesuid}")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    save_path = os.path.join(output_path, f"{seriesuid}.nii.gz")
    
    merge_multi_label_mask(
        mask_paths=path_list,
        output_path=save_path
    )

标签与id须对应
"""


def merge_multiple_masks_sitk(mask_paths, output_path):
    """
    使用SimpleITK合并多个NIfTI格式的掩码文件, 并保存结果。

    参数：
    mask_paths: list of str, 掩码文件路径的列表 -> ['path/to/mask1.nii.gz', 'path/to/mask2.nii.gz', 'path/to/mask3.nii.gz']
    output_path: str, 合并后mask的保存路径

    返回：
    output_path: str, 保存的合并后mask的路径
    """

    if not mask_paths:
        raise ValueError("The list of mask paths is empty.")
    merged_data = sitk.ReadImage(mask_paths[0]) > 0

    for mask_path in mask_paths[1:]:
        mask = sitk.ReadImage(mask_path)
        
        if mask.GetSize() != merged_data.GetSize() or mask.GetOrigin() != merged_data.GetOrigin() or mask.GetSpacing() != merged_data.GetSpacing() or mask.GetDirection() != merged_data.GetDirection():
            raise ValueError(f"The mask at {mask_path} is not aligned with the previous masks.")
        
        mask_data = mask > 0
        merged_data = sitk.Or(merged_data, mask_data)

    sitk.WriteImage(merged_data, output_path)



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