import os

import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def apply_windowing(
    image, window_center=None, window_width=None, apply_limits=False, preset="lung"
):
    """
    对图像应用窗宽窗位调整，以增强图像的对比度。
    参考资料：CT检查技术专家共识，https://gpedu.yiigle.com/uploads/guide_html/CT%E6%A3%80%E6%9F%A5%E6%8A%80%E6%9C%AF%E4%B8%93%E5%AE%B6%E5%85%B1%E8%AF%86.html


    参数:
    image (numpy.ndarray): 输入的图像数组。
    window_center (int, optional): 窗位的中心值。
    window_width (int, optional): 窗宽的大小。
    apply_limits (bool, optional): 是否应用图像的最小和最大值限制（根据图像密度值，计算出纯黑和纯白）。
    preset (str, optional): 预设的窗宽窗位参数组合名称。默认肺窗。

    返回:
    numpy.ndarray: 窗宽窗位调整后的图像数组。
    """

    # 预设的窗宽和窗位参数组合
    window_presets = {
        "lung": {"center": -600, "width": 1500},
        "liver": {"center": 40, "width": 250},  # 肝脏
        "bile_duct": {"center": 40, "width": 250},  # 胆管
        "pancreas": {"center": 40, "width": 250},  # 胰腺
        "spleen": {"center": 40, "width": 250},  # 脾脏
        "kidney": {"center": 40, "width": 250},  # 肾脏
        "retroperitoneal_space": {"center": 40, "width": 250},  # 腹膜后腔
        "stomach": {"center": 40, "width": 250},  # 胃部
        "adrenal_gland": {"center": 40, "width": 300},  # 肾上腺
        "mediastinum": {"center": 40, "width": 500},  # 纵隔窗
        "bone": {"center": 600, "width": 4000},  # 骨窗
    }

    # 如果指定了预设值，则使用预设的窗宽和窗位
    if preset and preset in window_presets:
        window_center = window_presets[preset]["center"]
        window_width = window_presets[preset]["width"]
    elif window_center is None or window_width is None:
        raise ValueError("必须提供窗宽和窗位参数，或者指定一个预设值")

    # 计算窗宽窗位的最小和最大值
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2

    # 如果启用了限制，则将窗宽窗位的最小和最大值与图像的最小和最大值进行比较
    if apply_limits:
        window_min = max(window_min, image.min())
        window_max = min(window_max, image.max())

    # 应用窗宽窗位调整
    windowed_image = np.clip(image, window_min, window_max)
    windowed_image = ((windowed_image - window_min) / (window_max - window_min)) * 255
    windowed_image = windowed_image.astype(np.uint8)

    return windowed_image


def read_dicom_series(dicom_dir):
    """
    读取DICOM序列并返回图像和像素数组。

    参数:
    dicom_dir (str): DICOM序列的文件夹路径。

    返回:
    ct_image (SimpleITK.Image): DICOM序列的图像。
    ct_array (numpy.ndarray): DICOM序列的像素数组。
    """
    # 使用SimpleITK的ImageSeriesReader获取DICOM序列的文件名
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    # 设置文件名并执行读取操作，获取DICOM图像
    reader.SetFileNames(dicom_names)
    ct_image = reader.Execute()

    # 使用SimpleITK的GetArrayFromImage方法获取像素数组
    ct_array = sitk.GetArrayFromImage(ct_image)

    return ct_image, ct_array


def split_and_save(
    input_file,
    num_slices=5,
    overlap=15,
    is_dicom=False,
    target_length=None,
    series_id=None,
    save_dir=".",
):
    """
    拆分CT图像并保存。

    参数:
    input_file (str): 输入的DICOM或NIfTI文件。
    num_slices (int): 要拆分成的子图像的数量，若指定则忽略target_length。
    overlap (int): 子图像之间的重叠长度。
    is_dicom (bool): 输入文件是否为DICOM格式。
    target_length (int): 目标子图像序列长度，若指定则忽略num_slices。
    series_id (str): 序列ID。
    save_dir (str): 保存目录。

    返回:
    无
    """
    # 读取图像
    if is_dicom:
        if series_id is None:
            raise ValueError("If input is DICOM, series_id must be specified")
        else:
            ct_image, _ = read_dicom_series(dicom_dir=input_file)
    else:
        ct_image = sitk.ReadImage(input_file)

    # 获取图像的尺寸和间距
    ct_size = ct_image.GetSize()
    ct_spacing = ct_image.GetSpacing()

    # 计算子图像的长度
    total_length = ct_size[2]
    if target_length is None:
        # 最短50mm，不能再短。
        target_length = int(round(50 / ct_spacing[2]))
        # 已验证这个公式。
        slice_length = max((total_length - (num_slices - 1) * overlap) // num_slices, target_length)
    else:
        slice_length = target_length
        num_slices = int(np.ceil((total_length - overlap) / (target_length - overlap)))

    # 如果series_id为None，根据输入文件的类型和名称来确定series_id
    if series_id is None:
        series_id = os.path.basename(input_file).replace('_0000.nii.gz', '').replace('.nii.gz', '')

    # 使用tqdm.auto进度条
    pbar = tqdm(total=num_slices)

    # 沿着横断面拆分图像
    for i in range(num_slices):
        # 构造保存文件的名称
        save_file = f"{save_dir}/{series_id}.L{slice_length}.S{overlap}.{i}"
        save_file += '_0000.nii.gz' if input_file.endswith("_0000.nii.gz") else '.nii.gz'
        # 如果保存文件已存在，跳过执行
        if os.path.exists(save_file):
            continue
        # 计算子图像的范围
        start = i * (slice_length - overlap)
        end = start + slice_length
        if end > total_length:
            end = total_length
            start = end - slice_length

        # 提取子图像
        sub_image = ct_image[:, :, start:end]
        # 保存子图像
        sitk.WriteImage(sub_image, save_file)
        # 更新进度条
        pbar.update(1)
    # 关闭进度条
    pbar.close()


def evaluate_nodule_inclusion(
    ct_slices,
    lung_mask,
    nodule_mask=None,
    nodule_center_voxel=None,
    nodule_center_world=None,
    nodule_radius_world=None,
    ct_image=None,
    ct_spacing=None,
    display=False,
    inclusion_threshold=0.95,
    save_images=False,
    save_dir="",
    ct_id="",
):
    """
    评估结节是否位于肺部掩码内，并可选地在CT图像上绘制结节。

    参数:
    ct_slices (numpy.ndarray): CT图像的横断面数组，每个横断面为二维数组。
    lung_mask (numpy.ndarray): 与CT图像相对应的肺部掩码数组，用于标识肺部区域。
    nodule_center_voxel (tuple): 结节中心点的体素坐标 (x, y, z)，表示结节在CT图像中的位置。
    nodule_radius_world (float): 结节的世界坐标系半径，表示结节的实际大小。
    ct_image (SimpleITK.Image): CT图像，用于从世界坐标转换到体素坐标（可选）。
    ct_spacing (tuple): 图像的体素间距，即每个体素在世界坐标系中的实际大小（可选）。
    display (bool): 是否在CT图像上显示结节，如果为True，则绘制结节并显示（默认为False）。
    inclusion_threshold (float): 判断结节是否被肺部掩码包含的阈值，如果结节与掩码的交集比例高于此值，则认为结节被包含（默认为0.95）。
    save_images (bool): 是否保存绘制的结节图像。
    save_dir (str): 保存图像的目录路径。
    ct_id (str): CT的编号，用于命名保存的图像。

    返回:
    float: 结节被肺部掩码包含的比例，值域为[0, 1]。
    bool: 结节是否被肺部掩码包含，根据inclusion_threshold判断。

    该函数首先计算结节在每个横断面上的投影半径，然后创建一个圆形掩码来模拟结节的投影。
    接着，计算圆形掩码与肺部掩码的交集面积，并累加所有横断面的交集面积。
    最后，根据累加的交集面积与结节投影面积的比例，评估结节是否被肺部掩码包含。
    如果需要，还可以在CT图像上绘制结节的投影并显示出来。
    """
    import cv2
    if save_images:
        os.makedirs(save_dir, exist_ok=True)

    # 如果提供了nodule_mask，则不再考虑坐标计算问题：
    if nodule_mask is None:
        # 如果结节中心的体素坐标未提供，并且有CT图像，则使用CT图像将世界坐标转换为体素坐标
        if nodule_center_voxel is None and ct_image is not None:
            nodule_center_voxel = ct_image.TransformPhysicalPointToIndex(
                nodule_center_world
            )

        # 从体素坐标中提取结节中心的x, y, z坐标
        center_x, center_y, center_z = nodule_center_voxel

        # 如果图像的体素间距未提供，并且有CT图像，则获取CT图像的体素间距
        if ct_spacing is None and ct_image is not None:
            ct_spacing = ct_image.GetSpacing()
    else:
        # 有结节的mask。对它进行二值化处理：
        nodule_mask = (nodule_mask > 0).astype(np.int8)

    # 初始化交集面积和圆形面积的总和
    ## 这里的设计考虑：球形的结节标注，其横断面是个圆形。
    total_intersection = 0
    total_circle_area = 0

    # 遍历CT图像的每个横断面
    for slice_index in range(ct_slices.shape[0]):
        if nodule_mask is not None:
            # mask模式：
            nodule_slice = nodule_mask[slice_index, :, :]
            if not nodule_slice.any():
                continue
            circle_area = np.sum(nodule_slice)
            # 获取对应横断面的肺部掩码
            mask_slice = lung_mask[slice_index, :, :]
            # 计算肺部掩码与结节投影的交集面积
            intersection_area = np.sum(np.logical_and(mask_slice, nodule_slice))
        else:
            # 坐标模式：
            # 计算当前横断面与结节中心的距离
            distance_from_center = abs(slice_index - center_z) * ct_spacing[2]
            # 如果距离小于等于结节的世界半径，则在该横断面上计算结节的投影
            if distance_from_center > nodule_radius_world:
                continue

            # 根据结节的世界半径和中心距离计算横断面上的半径
            slice_radius = np.sqrt(nodule_radius_world**2 - distance_from_center**2)
            # 将世界半径转换为体素半径
            # 横断面上。假定X和Y方向的spacing数值相等。
            slice_radius_voxel = round(slice_radius / ct_spacing[0])
            # 创建一个与CT横断面同样大小的空白掩码
            circle_mask = np.zeros(ct_slices[slice_index].shape, dtype=np.uint8)
            # 在掩码上绘制结节的投影（圆形）
            cv2.circle(circle_mask, (center_x, center_y), slice_radius_voxel, 1, -1)
            # 将掩码转换为整数类型
            circle_mask = circle_mask.astype(np.int8)
            # 获取对应横断面的肺部掩码
            mask_slice = lung_mask[slice_index, :, :]
            # 计算肺部掩码与结节投影的交集面积
            intersection_area = np.sum(np.logical_and(mask_slice, circle_mask))

            # 计算圆形投影的面积
            circle_area = np.sum(
                circle_mask
            )  # 使用pi * r ** 2的方法会引入误差，不如直接用numpy计算像素点数。
        # 累加交集面积
        total_intersection += intersection_area
        # 累加圆形面积
        total_circle_area += circle_area

        # debug用：
        nodule_inclusion_ratio = intersection_area / circle_area
        # 如果设置为显示或者交集面积与圆形面积的比例小于包含阈值，则在CT横断面上绘制结节投影和肺部掩码轮廓
        if display or (nodule_inclusion_ratio < inclusion_threshold):
            # 复制CT横断面图像
            slice_img = ct_slices[slice_index, :, :].copy()
            # 将图像从灰度转换为BGR颜色空间
            slice_img = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
            if nodule_mask is not None:
                nodule_slice = nodule_slice.astype(np.uint8)
                contours, _ = cv2.findContours(
                    nodule_slice,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                # 在背景图像上绘制红色轮廓线
                cv2.drawContours(slice_img, contours, -1, (0, 0, 255), 2)
            else:
                # 在图像上绘制结节投影（红色圆形）
                cv2.circle(
                    slice_img, (center_x, center_y), slice_radius_voxel, (0, 0, 255), 2
                )
            # 如果肺部掩码不为空，则使用OpenCV找到外轮廓
            if mask_slice.any():
                contours, _ = cv2.findContours(
                    mask_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                # 在背景图像上绘制黄色轮廓线
                cv2.drawContours(
                    slice_img, contours, -1, (0, 255, 255), 2
                )  # 绿色轮廓线，线宽为2个像素

            # 将交集面积与圆形面积的比例写在图片的左上角
            cv2.putText(
                slice_img,
                f"{nodule_inclusion_ratio:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            # 根据是否保存图片决定是显示还是保存
            if save_images:
                # 保存图像
                cv2.imwrite(
                    os.path.join(save_dir, f"{ct_id}.{slice_index}.png"),
                    cv2.cvtColor(slice_img, cv2.COLOR_BGR2RGB),
                )
            else:
                # 显示图像，将BGR转换为RGB
                plt.imshow(slice_img[:, :, ::-1])
                plt.show()

    # 计算结节被肺部掩码包含的比例
    inclusion_ratio = (
        total_intersection / total_circle_area if total_circle_area > 0 else 0
    )
    # 根据包含比例和阈值判断结节是否被肺部掩码包含
    is_included = inclusion_ratio >= inclusion_threshold

    return inclusion_ratio, is_included


"""
# 使用示例
ct_img = "/data/PublicDatasets/AliTianchiPulmonaryNodules/correct_images/LKDS-00052.mhd"
lung_mask_nii = "/data/niubing/experiments/CTImaging/lungmask/masks_by_lungmask/R231/ali_tianchi/LKDS-00052.lung_mask.nii"

ct_image = sitk.ReadImage(ct_img)
ct_array = sitk.GetArrayFromImage(ct_image)
lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_mask_nii))
ct_array_windowed = utils.apply_windowing(ct_array, -600, 1500)
# 获得结节的世界坐标和半径：
# 筛选 'seriesuid' 为 'LKDS-00670' 的行，并获取特定的列
row = total_df.loc[
    total_df["seriesuid"] == "LKDS-00052", ["coordX", "coordY", "coordZ", "diameter_mm"]
].iloc[0]

# 将提取的数值赋值给变量
x = row["coordX"]
y = row["coordY"]
z = row["coordZ"]
r = row["diameter_mm"]

# 输出变量值
print(f"x: {x}, y: {y}, z: {z}, r: {r}")
# 将世界坐标转换为体素坐标
voxel_coordinates = ct_image.TransformPhysicalPointToIndex((x, y, z))
ct_spacing = ct_image.GetSpacing()
inclusion_ratio, is_included = evaluate_nodule_inclusion(
    ct_slices=ct_array_windowed,
    lung_mask=lung_mask,
    nodule_center_voxel=voxel_coordinates,
    nodule_radius_world=r,
    ct_img=ct_image,
    img_spacing=ct_spacing,
    display=True,
)

print(inclusion_ratio, is_included)
"""


# def world_to_voxel_coord_and_radii(
#     x_world, y_world, z_world, radius_mm, image_path=None, image=None
# ):
#     """
#     将世界坐标和半径转换为图像坐标和体素半径。

#     参数:
#     x_world (float): 结节中心点的世界坐标x。
#     y_world (float): 结节中心点的世界坐标y。
#     z_world (float): 结节中心点的世界坐标z。
#     radius_mm (float): 结节的半径（毫米）。
#     image_path (str, optional): 图像文件的路径。如果提供了image参数，则忽略此参数。
#     image (SimpleITK.Image, optional): 已加载的SimpleITK图像对象。如果未提供，则从image_path加载图像。

#     返回:
#     tuple: 包含图像坐标 (i, j, k) 和体素半径的元组。
#     """
#     # 如果未提供图像对象，则从路径加载图像
#     if image is None and image_path is not None:
#         image = sitk.ReadImage(image_path)

#     # 从图像中提取元数据
#     origin = image.GetOrigin()
#     spacing = image.GetSpacing()
#     direction = np.array(image.GetDirection()).reshape(3, 3)

#     # 将世界坐标转换为体素坐标
#     stretched_voxel_coord = np.array([x_world, y_world, z_world]) - np.array(origin)
#     voxel_coord = stretched_voxel_coord / np.array(spacing)

#     # 如果方向矩阵不是单位矩阵，则需要应用方向矩阵的逆变换
#     if not np.allclose(direction, np.eye(3)):
#         inv_direction = np.linalg.inv(direction)
#         voxel_coord = np.dot(inv_direction, voxel_coord)

#     # 将世界坐标的半径转换为体素半径
#     voxel_radius_x = radius_mm / spacing[0]
#     voxel_radius_y = radius_mm / spacing[1]
#     voxel_radius_z = radius_mm / spacing[2]

#     # 四舍五入坐标和半径到最近的整数
#     voxel_coord = np.round(voxel_coord).astype(int)
#     voxel_radius_x = round(voxel_radius_x)
#     voxel_radius_y = round(voxel_radius_y)
#     voxel_radius_z = round(voxel_radius_z)

#     return tuple(voxel_coord), (voxel_radius_x, voxel_radius_y, voxel_radius_z)


# # 示例使用
# image_path = 'path/to/your/image.mhd'  # 替换为您的图像文件路径
# x_world, y_world, z_world, radius_mm = -73.3527560764, 120.0576413, 761.5, 12.5381245616

# # 调用函数转换坐标和半径
# voxel_coord, (voxel_radius_x, voxel_radius_y, voxel_radius_z) = world_to_voxel_coord_and_radii(x_world, y_world, z_world, radius_mm, image_path=image_path)
# print('图像坐标:', voxel_coord)
# print('体素半径 X:', voxel_radius_x)
# print('体素半径 Y:', voxel_radius_y)
# print('体素半径 Z:', voxel_radius_z)
