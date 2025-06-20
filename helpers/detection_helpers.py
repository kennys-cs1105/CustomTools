"""
Created by KennyS on 2024/8/19

nodule_detection功能模块的函数和工具
"""

import logging
import os
import re
import shutil

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label
from tqdm.auto import tqdm


def crop_without_nodulemask(
    ct_image_path,
    organ_mask_path,
    ct_image=None,
    organ_mask_image=None,
    cube_size=64,
    step_size=40,
    series_uid=None,
    crop_cube_dir=None,
    air_hu_value=-1000,
):
    """
    读取CT和肺部掩码图像, 使用肺部掩码提取边界框，进行裁切。
    根据裁切尺寸和步长计算是否可以整数倍裁切。如果不能, 使用空气的HU值进行填充。
    执行裁切并将结果保存为nii.gz格式, 以及裁切的位置信息为txt文件, 并显示进度条。

    参数:
    ct_image_path (str): CT图像的文件路径。
    organ_mask_path (str): 用于提取边界框的器官掩膜文件路径。
    cube_size (int): 立方体的边长（以体素为单位）。
    step_size (int): 裁切的步长。
    crop_cube_dir (str): 输出cube文件夹路径。
    postion_txt_dir (str): 输出位置txt文件夹路径
    air_hu_value (int): 空气的HU值，默认为-1000。

    返回:
    None
    """
    if ct_image is None:
        ct_image = sitk.ReadImage(ct_image_path)
    if organ_mask_image is None:
        organ_mask_image = sitk.ReadImage(organ_mask_path)

    # 获取器官掩膜的边界框
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(organ_mask_image)
    # 正常情况下是1, 但是获取的是organs_masks, 其中lung==4
    bounding_box = label_shape_filter.GetBoundingBox(1)  # 手动修改id=5

    # 计算裁切空间的尺寸
    bb_size = [bounding_box[i + 3] for i in range(3)]
    padding = [(cube_size - (d % cube_size)) % cube_size for d in bb_size]
    padded_bb_size = [bb_size[i] + padding[i] for i in range(3)]

    # 创建填充图像
    padded_ct_image = sitk.ConstantPad(ct_image, [0] * 3, padding, air_hu_value)

    # 裁切并保存
    index = 0
    total_cubes = (
        (padded_bb_size[0] // step_size)
        * (padded_bb_size[1] // step_size)
        * (padded_bb_size[2] // step_size)
    )

    position_dict = {}

    pbar = tqdm(total=total_cubes, desc="裁切进度", unit="cube", leave=False)
    for z in range(0, padded_bb_size[2], step_size):
        for y in range(0, padded_bb_size[1], step_size):
            for x in range(0, padded_bb_size[0], step_size):
                # 定义裁切区域
                region_size = [
                    min(cube_size, padded_bb_size[i] - pos)
                    for i, pos in enumerate([x, y, z])
                ]
                region_index = [
                    bounding_box[i] + pos for i, pos in enumerate([x, y, z])
                ]

                # 裁切
                crop_region = sitk.RegionOfInterestImageFilter()
                crop_region.SetSize(region_size)
                crop_region.SetIndex(region_index)
                cropped_ct_cube = crop_region.Execute(padded_ct_image)

                ct_output_path = f"{crop_cube_dir}/{series_uid}.{index}_0000.nii.gz"

                sitk.WriteImage(cropped_ct_cube, ct_output_path, True)

                # 位置信息存入字典
                position_dict[f"{series_uid}.{index}"] = f"{x},{y},{z}"

                index += 1
                pbar.update(1)
    pbar.close()

    return position_dict


def check_prediction_ratio(prediction_path):
    """
    对prediction_cube_path路径下的所有.nii.gz掩码文件计算并返回掩码中为1的元素的比例
    若掩码中1的比例小于 (4/3 * np.pi * (1.5 ** 3) / 掩码体积), 则将掩码中为1的元素置为0
    若掩码中1的比例大于等于 (4/3 * np.pi * (1.5 ** 3) / 掩码体积), 则不做修改
    """

    for filename in tqdm(
        os.listdir(prediction_path), desc="Calculation Ratio...", unit="cube"
    ):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(prediction_path, filename)

            mask_img = sitk.ReadImage(file_path)
            mask_data = sitk.GetArrayFromImage(mask_img)

            mask_volume = np.prod(mask_data.shape)
            ones_ratio = np.sum(mask_data == 1) / mask_volume

            sphere_volume = (4 / 3) * np.pi * (1.5**3)
            threshold_ratio = sphere_volume / mask_volume

            if ones_ratio < threshold_ratio:
                mask_data[mask_data == 1] = 0

            modified_mask_img = sitk.GetImageFromArray(mask_data)
            modified_mask_img.CopyInformation(mask_img)

            sitk.WriteImage(modified_mask_img, file_path)


def get_bounding_box(mask_image):
    """
    获取掩膜图像的边界框。

    参数:
    mask_image (sitk.Image): 掩膜图像。

    返回:
    tuple: 包含边界框的起始位置和尺寸 (x_min, y_min, z_min, x_size, y_size, z_size)。
    """
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_image)
    return label_shape_filter.GetBoundingBox(1)


def load_and_reconstruct_nodule_mask_v2(
    cubes_dir,
    position_dict,
    original_size,
    lung_mask_path,
    output_path=None,
):
    """
    从立方体掩码图像中重建原始大小的掩码图像，并确保小立方体放置于原始大图像的肺掩膜区域。

    参数:
    cubes_dir (str): 小立方体掩码图像的文件夹路径。
    position_dir (str): 小立方体切割位置的txt文件夹路径
    original_size (tuple): 原始大图像的大小。
    lung_mask_path (str): 肺掩膜图像的文件路径。
    cube_size (int): 立方体的边长（以体素为单位）。
    step_size (int): 裁切的步长。
    output_dir (str): 输出大图像的路径。

    返回:
    None
    """
    # 读取肺掩膜图像并获取边界框
    lung_mask_image = sitk.ReadImage(lung_mask_path)
    bounding_box = get_bounding_box(lung_mask_image)

    # 初始化重建掩码图像
    concatenate_mask = np.zeros(original_size[::-1], dtype=np.uint8)

    # 获取所有小立方体的文件路径
    cube_files = sorted(
        [
            os.path.join(cubes_dir, f)
            for f in os.listdir(cubes_dir)
            if f.endswith(".nii.gz") and "_0000" not in f
        ]
    )

    # 读取并拼接小立方体掩码图像
    for cube_file in tqdm(cube_files, desc="Concatenate cube...", unit="cube"):
        # 获取立方体的索引
        base_name = os.path.basename(cube_file)

        parts = base_name.rsplit(".", 3)[:-2]
        series_uid = parts[0]
        index_str = parts[1]
        index = int(index_str)

        # 读取立方体的位置
        position_key = f"{series_uid}.{index}"
        if position_key in position_dict:
            x, y, z = map(int, position_dict[position_key].split(","))
        else:
            continue

        # 计算立方体在肺掩膜区域中的位置
        z += bounding_box[2]
        y += bounding_box[1]
        x += bounding_box[0]

        # 读取立方体掩码图像
        cube_mask = sitk.GetArrayFromImage(sitk.ReadImage(cube_file))
        cube_shape = cube_mask.shape

        # 确定实际可以放置的区域大小
        z_end = min(z + cube_shape[0], original_size[2])
        y_end = min(y + cube_shape[1], original_size[1])
        x_end = min(x + cube_shape[2], original_size[0])

        # 计算重叠区域的大小
        z_slice = max(0, z_end - z)
        y_slice = max(0, y_end - y)
        x_slice = max(0, x_end - x)

        # 放置立方体掩码图像到重建掩码图像中
        if z_slice > 0 and y_slice > 0 and x_slice > 0:
            # 放置立方体掩码图像到重建掩码图像中
            concatenate_mask[z : z + z_slice, y : y + y_slice, x : x + x_slice] = (
                np.maximum(
                    concatenate_mask[z : z + z_slice, y : y + y_slice, x : x + x_slice],
                    cube_mask[:z_slice, :y_slice, :x_slice],
                )
            )

    # 保存重建掩码图像
    if output_path:
        # 转换回SimpleITK图像，并确保轴顺序正确
        sitk_mask_image = sitk.GetImageFromArray(concatenate_mask)
        sitk.WriteImage(sitk_mask_image, output_path)


def cleanup_dir(dir):
    """
    检查dir路径是否为空，如果不为空，则删除该路径下的所有文件和子文件夹。

    参数:
    dir (str): 需要检查并清理的目录路径。

    返回:
    None
    """
    if not os.path.exists(dir):
        print(f"{dir} does not exist.")
        return

    # 检查目录是否为空
    if os.listdir(dir):
        print(f"{dir} is not empty. Deleting contents...")

        # 删除目录下的所有内容
        for item in os.listdir(dir):
            item_path = os.path.join(dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # 删除文件或符号链接
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除子文件夹及其内容
    else:
        print(f"{dir} is empty. Skipping deletion.")


def crop_with_nodulemask(
    organ_mask_path,
    position_dir,
    cutting_mask_dir,
    cube_size=64,
    step_size=40,
    series_uid=None,
    crop_mask_dir=None,
    # air_hu_value=-1000,
):
    """
    读取CT和两个掩膜图像，使用器官掩膜提取边界框，使用裁切掩膜进行裁切。
    根据裁切尺寸和步长计算是否可以整数倍裁切。如果不能，使用空气的HU值进行填充。
    执行裁切并将结果保存为nii.gz格式，并显示进度条。

    参数:
    ct_image_path (str): CT图像的文件路径。
    organ_mask_path (str): 用于提取边界框的器官掩膜文件路径。
    cutting_mask_path (str): 用于裁切的掩膜文件路径。
    cube_size (int): 立方体的边长（以体素为单位）。
    step_size (int): 裁切的步长。
    output_dir (str): 输出文件夹路径。
    air_hu_value (int): 空气的HU值，默认为-1000。

    返回:
    None
    """
    # 读取图像
    # ct_image = sitk.ReadImage(ct_image_path)
    organ_mask_image = sitk.ReadImage(organ_mask_path)
    cutting_mask_path = os.path.join(
        cutting_mask_dir, f"{series_uid}_concatenate.nii.gz"
    )
    cutting_mask_image = sitk.ReadImage(cutting_mask_path)

    # 获取器官掩膜的边界框
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(organ_mask_image)
    bounding_box = label_shape_filter.GetBoundingBox(1)  # lung id == 4

    # 计算裁切空间的尺寸
    bb_size = [bounding_box[i + 3] for i in range(3)]
    padding = [(cube_size - (d % cube_size)) % cube_size for d in bb_size]
    padded_bb_size = [bb_size[i] + padding[i] for i in range(3)]

    # 创建填充图像
    # padded_ct_image = sitk.ConstantPad(ct_image, [0] * 3, padding, air_hu_value)
    padded_cutting_mask_image = sitk.ConstantPad(
        cutting_mask_image, [0] * 3, padding, 0
    )

    # 裁切并保存
    index = 0
    total_cubes = (
        (padded_bb_size[0] // step_size)
        * (padded_bb_size[1] // step_size)
        * (padded_bb_size[2] // step_size)
    )

    pbar = tqdm(total=total_cubes, desc="裁切进度", unit="cube", leave=False)
    for z in range(0, padded_bb_size[2], step_size):
        for y in range(0, padded_bb_size[1], step_size):
            for x in range(0, padded_bb_size[0], step_size):
                # 定义裁切区域
                region_size = [
                    min(cube_size, padded_bb_size[i] - pos)
                    for i, pos in enumerate([x, y, z])
                ]
                region_index = [
                    bounding_box[i] + pos for i, pos in enumerate([x, y, z])
                ]

                # 裁切
                crop_region = sitk.RegionOfInterestImageFilter()
                crop_region.SetSize(region_size)
                crop_region.SetIndex(region_index)
                # cropped_ct_cube = crop_region.Execute(padded_ct_image)
                cropped_cutting_mask_cube = crop_region.Execute(
                    padded_cutting_mask_image
                )

                # 保存
                crop_mask_path = os.path.join(crop_mask_dir, series_uid)
                try:
                    os.makedirs(crop_mask_path)
                except FileExistsError:
                    print(f"Crop mask file of {series_uid} already exists!")
                mask_output_path = f"{crop_mask_path}/{series_uid}.{index}.nii.gz"

                position_path = os.path.join(position_dir, series_uid)
                if not os.path.exists(position_path):
                    os.makedirs(position_path)
                position_output_path = f"{position_path}/{series_uid}.{index}.txt"

                sitk.WriteImage(cropped_cutting_mask_cube, mask_output_path, True)

                # 保存立方体的位置
                with open(position_output_path, "w") as f:
                    f.write(f"{x},{y},{z}")

                index += 1
                pbar.update(1)
    pbar.close()


def split_nodules(concatenate_path, split_path, uid, id):
    """
    将单个肺结节从预测掩码中分离出来并单独保存。

    参数:
    pred_mask_path (str): 预测掩码图像的文件路径。
    output_dir (str): 输出文件夹路径。

    返回:
    None
    """
    concatenate_mask = sitk.ReadImage(concatenate_path)
    concatenate_mask_array = sitk.GetArrayFromImage(concatenate_mask)
    nodule_array = (concatenate_mask_array == 1).astype(np.uint8)  # nodule id == 3

    # 链接组件找寻独立的肺结节
    labeled_array, num_features = label(nodule_array)
    print(f"{os.path.basename(concatenate_path)} has {num_features} nodules!")

    for i in range(1, num_features + 1):
        individual_nodulemask = (labeled_array == i).astype(np.uint8)

        individual_nodulemask_image = sitk.GetImageFromArray(individual_nodulemask)
        individual_nodulemask_image.CopyInformation(concatenate_mask)

        output_path = f"{split_path}/{uid}.{id}.{i}.nii.gz"

        sitk.WriteImage(individual_nodulemask_image, output_path)
        # print(f"Saved: {output_file}")


def using_split_nodule(nodule_dir, split_dir, seriesuid):

    split_path = os.path.join(split_dir, seriesuid)
    os.makedirs(split_path, exist_ok=True)

    # 需要注意nifti_name的命名方式
    nodule_path = os.path.join(nodule_dir, seriesuid)
    sorted_files = sorted(
        os.listdir(nodule_path), key=lambda x: int(x.rsplit(".", 3)[-3])
    )

    for idx, filename in enumerate(sorted_files):
        nodule_cube_path = os.path.join(nodule_path, filename)
        split_nodules(
            concatenate_path=nodule_cube_path,
            split_path=split_path,
            uid=seriesuid,
            id=idx,
        )


def check_split_nodule_file(nodule_dir, non_nodule_dir, seriesuid):
    """
    检查split_nodule中有无肺结节, 若没有则删除该文件
    """
    nodule_path = os.path.join(nodule_dir, seriesuid)
    non_nodule_path = os.path.join(non_nodule_dir, seriesuid)
    if not os.path.exists(non_nodule_path):
        os.makedirs(non_nodule_path)

    for filename in os.listdir(nodule_path):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(nodule_path, filename)

            img = sitk.ReadImage(file_path)
            data = sitk.GetArrayFromImage(img)

            if np.count_nonzero(data == 1) == 0:  # nodule id == 3
                shutil.move(file_path, os.path.join(non_nodule_path, filename))
            else:
                print(f"{file_path} contains nodules!")


def check_prediction_ratio(prediction_dir, seriesuid):
    """
    对prediction_cube_path路径下的所有.nii.gz掩码文件计算并返回掩码中为1的元素的比例
    若掩码中1的比例小于 (4/3 * np.pi * (1.5 ** 3) / 掩码体积), 则将掩码中为1的元素置为0
    若掩码中1的比例大于等于 (4/3 * np.pi * (1.5 ** 3) / 掩码体积), 则不做修改
    """
    prediction_path = os.path.join(prediction_dir, seriesuid)
    for filename in tqdm(
        os.listdir(prediction_path), desc="Calculation Ratio...", unit="cube"
    ):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(prediction_path, filename)

            mask_img = sitk.ReadImage(file_path)
            mask_data = sitk.GetArrayFromImage(mask_img)

            mask_volume = np.prod(mask_data.shape)
            nodule_ratio = np.sum(mask_data == 1) / mask_volume  # nodule id == 3

            sphere_volume = (4 / 3) * np.pi * (1.5**3)
            threshold_ratio = sphere_volume / mask_volume

            if nodule_ratio < threshold_ratio:
                mask_data[mask_data == 1] = 0  # noudle id == 3

            modified_mask_img = sitk.GetImageFromArray(mask_data)
            modified_mask_img.CopyInformation(mask_img)

            sitk.WriteImage(modified_mask_img, file_path)


def check_and_pad_nifti(
    file_path, save_dir, target_size=(64, 64, 64), padding_value=None
):
    """
    检查 .nii.gz 文件的大小, 如果不是 target_size, 则对其进行填充.

    :param file_path: .nii.gz 文件的路径
    :param target_size: 目标大小，默认为 (64, 64, 64)
    :param padding_value: 填充值，默认为 0
    :return: None
    """
    # 加载nii.gz文件
    img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(img)
    img_size = img_array.shape

    # 检查大小
    if not img_size == target_size:
        # print(f"{file_path} already has the target size {target_size}.")
        # return

        print(f"{file_path} size is {img_size}, padding to {target_size}.")

    # 计算需要填充的大小
    pad_z = max(0, target_size[0] - img_size[0])
    pad_y = max(0, target_size[1] - img_size[1])
    pad_x = max(0, target_size[2] - img_size[2])

    # 计算前后需要填充的数量
    pad_z_front = pad_z // 2
    pad_z_back = pad_z - pad_z_front
    pad_y_front = pad_y // 2
    pad_y_back = pad_y - pad_y_front
    pad_x_front = pad_x // 2
    pad_x_back = pad_x - pad_x_front

    # 使用指定的填充值进行填充
    padded_img_array = np.pad(
        img_array,
        (
            (pad_z_front, pad_z_back),
            (pad_y_front, pad_y_back),
            (pad_x_front, pad_x_back),
        ),
        mode="constant",
        constant_values=padding_value,
    )

    # 确保填充后的大小正确
    assert (
        padded_img_array.shape == target_size
    ), "Padding did not result in the correct size."

    # 创建新的SimpleITK图像
    padded_img = sitk.GetImageFromArray(padded_img_array)

    # 设置新图像的元数据信息
    padded_img.SetDirection(img.GetDirection())
    padded_img.SetOrigin(img.GetOrigin())
    padded_img.SetSpacing(img.GetSpacing())

    output_path = os.path.join(save_dir, os.path.basename(file_path))
    sitk.WriteImage(padded_img, output_path)
    # print(f"Processed and saved {output_path} with new size {target_size}.")


def using_check_pad(cube_dir, pad_dir, seriesuid, value):

    cube_path = os.path.join(cube_dir, seriesuid)
    output_dir = os.path.join(pad_dir, seriesuid)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cube_file in os.listdir(cube_path):
        cube_file_path = os.path.join(cube_path, cube_file)

        check_and_pad_nifti(
            file_path=cube_file_path, save_dir=output_dir, padding_value=value
        )


def write_paths_to_csv(nodule_cube_path, ct_cube_path, output_csv_path):
    """
    将split_fpr_cube文件以及对应的ct_cube_v2文件的绝对路径写入csv,
    ct_cube_v2对应列名'CT_Path', split_fpr_cube对应列名'Nodule_Path'
    split_fpr_cube文件路径命名方式为 {split_fpr_cube_path}/{seriesuid}/{seriesuid}.{index}.{i}.nii.gz
    对应的ct_cube_v2文件路径命名方式为{ct_cube_v2_path}/{seriesuid}/{seriesuid}.{index}_0000.nii.gz

    :param split_fpr_cube_path: split_fpr_cube 文件的根路径
    :param ct_cube_v2_path: ct_cube_v2 文件的根路径
    :param output_csv_path: 输出的 CSV 文件路径
    :return: None
    """
    # 初始化列表以存储路径对
    paths = []

    # 遍历 split_fpr_cube_path 下的所有文件
    for root, dirs, files in os.walk(nodule_cube_path):
        for file in files:
            if file.endswith(".nii.gz"):
                # 获取 seriesuid 和索引
                parts = file.rsplit(".", 4)
                if len(parts) >= 4:
                    seriesuid = parts[0]
                    index = parts[1]

                    # 构造 split_fpr_cube 文件路径
                    nodule_path = os.path.join(root, file)

                    # 构造对应的 ct_cube_v2 文件路径
                    ct_file_name = f"{seriesuid}.{index}_0000.nii.gz"
                    ct_path = os.path.join(ct_cube_path, seriesuid, ct_file_name)

                    # 检查 ct_path 是否存在
                    if os.path.exists(ct_path):
                        paths.append({"CT_Path": ct_path, "Nodule_Path": nodule_path})
                    else:
                        print(
                            f"Warning: Corresponding CT file not found for {nodule_path}"
                        )

    # 创建 DataFrame 并写入 CSV
    df = pd.DataFrame(paths)
    df.to_csv(output_csv_path, index=False)
    print(f"Paths have been written to {output_csv_path}")


def filter_predictions(df, threshold):
    """
    筛选出概率值大于阈值的记录
    :param df: pandas DataFrame
    :param threshold: 概率阈值
    :return: 筛选后的 DataFrame
    """
    return df[df["Probability"] > threshold]


def copy_files(filtered_df, split_nodule_dir, combine_dir):
    """
    根据筛选的文件路径，将文件从 split_nodule_path 复制到 combine_path
    :param filtered_df: 筛选后的 DataFrame
    :param split_nodule_dir: 源文件路径的根目录
    :param combine_dir: 目标文件路径的根目录
    """

    for _, row in filtered_df.iterrows():
        seriesuid = row["CT_Path"].split("/")[-2]
        file_name = row["Nodule_Path"].split("/")[-1]

        source_file = os.path.join(split_nodule_dir, file_name)

        target_file = os.path.join(combine_dir, file_name)

        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")
        else:
            print(f"Source file {source_file} does not exist")


def list_files(combine_path):
    """
    列出combine_path下的所有文件
    :param combine_path: 路径
    :return: 文件列表
    """
    files = []
    for dirpath, _, filenames in os.walk(combine_path):
        for filename in filenames:
            if filename.endswith(".nii.gz"):
                files.append(os.path.join(dirpath, filename))
    return files


def group_files_by_seriesuid_and_index(files):
    """
    根据{seriesuid}.{index}对文件进行分组
    :param files: 文件列表
    :return: 分组后的字典
    """
    pattern = re.compile(r"^(.*)\.(\d+)\.(\d+)\.nii\.gz$")
    grouped_files = {}
    for file in files:
        match = pattern.search(os.path.basename(file))
        if match:
            seriesuid, index, i = match.groups()
            key = f"{seriesuid}.{index}"
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file)
    return grouped_files


def merge_and_rename_files(grouped_files, combine_path):
    """
    合并和重命名文件
    :param grouped_files: 分组后的文件字典
    :param combine_path: 路径
    """
    for key, file_list in grouped_files.items():
        if len(file_list) > 1:
            # 读取和合并掩码文件
            merged_image = None
            for file in file_list:
                img = sitk.ReadImage(file)
                img_array = sitk.GetArrayFromImage(img)
                if merged_image is None:
                    merged_image = img_array
                else:
                    merged_image = np.maximum(merged_image, img_array)

            # 保存合并后的文件
            merged_img = sitk.GetImageFromArray(merged_image)
            merged_img.CopyInformation(sitk.ReadImage(file_list[0]))
            output_path = os.path.join(combine_path, f"{key}.nii.gz")
            sitk.WriteImage(merged_img, output_path)
            print(f"Merged files into {output_path}")

            # 删除原始文件
            for file in file_list:
                os.remove(file)
                print(f"Deleted {file}")
        else:
            # 只有一个文件，重命名
            file = file_list[0]
            new_name = os.path.join(combine_path, f"{key}.nii.gz")
            os.rename(file, new_name)
            print(f"Renamed {file} to {new_name}")


def list_files_in_directory(directory):
    """
    列出目录中的所有文件
    :param directory: 目录路径
    :return: 文件列表（仅文件名，不包含路径）
    """
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        return []

    return [entry.name for entry in os.scandir(directory) if entry.is_file()]


def copy_missing_files(source_path, destination_path):
    """
    将 source_path 中存在, destination_path 中不存在的文件复制到 destination_path
    :param source_path: 源目录路径
    :param destination_path: 目标目录路径
    """
    if not os.path.exists(source_path):
        logging.error(f"Source directory does not exist: {source_path}")
        return

    if not os.path.exists(destination_path):
        logging.error(f"Destination directory does not exist: {destination_path}")
        return

    source_files = set(list_files_in_directory(source_path))
    destination_files = set(list_files_in_directory(destination_path))

    missing_files = source_files - destination_files

    for file_name in missing_files:
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        shutil.copy2(source_file, destination_file)
        logging.info(f"Copied {source_file} to {destination_file}")


def remove_nodules_outside_lungmask(
    lungmask_path, nodulemask_path, compared_output_path
):
    """
    比较肺的掩码图以及肺结节的掩码图, 如果肺结节出现在肺区域以外, 则去除, 将肺结节掩码图另存为新的掩码图。

    参数:
    lung_mask_path (str): 肺掩码图的文件路径。
    nodule_mask_path (str): 肺结节掩码图的文件路径。
    output_nodule_mask_path (str): 输出的新的肺结节掩码图的文件路径。

    返回:
    None
    """

    lung_mask = sitk.ReadImage(lungmask_path)
    nodule_mask = sitk.ReadImage(nodulemask_path)

    lung_mask_array = sitk.GetArrayFromImage(lung_mask)
    nodule_mask_array = sitk.GetArrayFromImage(nodule_mask)

    new_nodule_mask_array = np.where(lung_mask_array == 1, nodule_mask_array, 0)

    new_nodule_mask = sitk.GetImageFromArray(new_nodule_mask_array)
    new_nodule_mask.CopyInformation(nodule_mask)

    sitk.WriteImage(new_nodule_mask, compared_output_path)
