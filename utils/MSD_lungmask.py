import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label



"""
1. MSD数据集的lungmask

	- liver
	- lung
	- pancreas
	- spleen

2. 分割模型

	- nnunet

3. 分割结果存在问题

	- 肺外噪声，解剖上毫无关联，可能是因为CT值接近被误分割
	- 肺边缘被过分割，形态上接近

4. 后处理方式

	- 正常情况下，lungmask的连通域为2个，且肺部相比外部噪声体积更大，以连通域体积大小进行排序并设定num_features=2进行过滤
	- 有较多样本的lungmask粘连在一起，导致体积较大的连通域只有1个，所以根据连通域的体积设定阈值进行过滤
	- 查看了部分样本，设定阈值
		- liver: volume_threshold=50000
		- lung: volume_threshold=50000
		- spleen: volume_threshold=50000
		- pancreas: volume_threshold=900

5. 仍旧存在极少数个例无法过滤，有问题了再查看
"""



def test_and_filter_lung_volume(lungmask_path, output_path, min_size=50000):
    """
    计算肺部掩码图像中的连通域数量及其体积，并将体积小于min_size的连通域置为0。

    参数：
    - lungmask_path: str, 肺部掩码文件的路径 (NIfTI格式)
    - output_path: str, 处理后结果的保存路径
    - min_size: int, 连通域的最小保留体积阈值

    返回：
    - num_features: int, 连通域的数量
    - component_sizes: list of int, 每个连通域的体积列表
    """
    # 读取肺部掩码图像
    lungmask = sitk.ReadImage(lungmask_path)
    lungmask_array = sitk.GetArrayFromImage(lungmask)

    # 标记连通域
    labeled_array, num_features = label(lungmask_array)

    # 计算每个连通域的体积
    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]

    # 过滤体积小于min_size的连通域
    filtered_mask_array = np.zeros_like(lungmask_array)
    for i, size in enumerate(component_sizes):
        if size >= min_size:
            filtered_mask_array[labeled_array == (i + 1)] = 1

    # 将过滤后的结果转换为SimpleITK图像并保存
    filtered_mask = sitk.GetImageFromArray(filtered_mask_array.astype(np.uint8))
    filtered_mask.CopyInformation(lungmask)
    sitk.WriteImage(filtered_mask, output_path)



def test_lung_volume(lungmask_path):

    lungmask = sitk.ReadImage(lungmask_path)
    lungmask_array = sitk.GetArrayFromImage(lungmask)

    # 标记连通域
    labeled_array, num_features = label(lungmask_array)

    # 计算每个连通域的体积
    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]

    return num_features, component_sizes



def remain_two_lung(lungmask_path, output_path, num_largest_components=2):
    """
    过滤肺部分割结果中的过分割区域，保留体积最大的两个连通域。

    参数：
    - lungmask_path: str, 肺部分割结果的路径 (NIfTI格式)
    - output_path: str, 处理后结果的保存路径
    - num_largest_components: int, 要保留的最大连通域的数量 (默认: 2)

    返回：
    - output_path: str, 保存的结果文件路径
    """
    # 读取肺部分割结果
    lungmask = sitk.ReadImage(lungmask_path)
    lungmask_array = sitk.GetArrayFromImage(lungmask)

    # 标记连通域
    labeled_array, num_features = label(lungmask_array)
    print(f"This mask has {num_features} features!")

    # 计算每个连通域的体积
    component_sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]

    # 获取体积最大的连通域的索引
    largest_components_indices = np.argsort(component_sizes)[-num_largest_components:]

    # 创建一个空的掩码用于存储保留的连通域
    filtered_mask_array = np.zeros_like(lungmask_array)

    # 保留体积最大的连通域
    for index in largest_components_indices:
        filtered_mask_array[labeled_array == (index + 1)] = 1  # +1因为label从1开始

    # 将结果转换为SimpleITK图像并保存
    filtered_mask = sitk.GetImageFromArray(filtered_mask_array.astype(np.uint8))
    filtered_mask.CopyInformation(lungmask)
    sitk.WriteImage(filtered_mask, output_path)