import logging
from pathlib import Path
from typing import List, Optional, Tuple

import fill_voids
import numpy as np
import SimpleITK as sitk
import skimage.measure
import skimage.morphology
from scipy import ndimage
from tqdm.auto import tqdm

# 配置日志记录器 #################################################################################
# --- 全局日志配置 ---
# 在模块加载时进行一次性基本配置。
# 应用程序的其他部分可以添加更复杂的处理器或修改级别。
logging.basicConfig(
    level=logging.INFO,  # 默认级别
    format="%(asctime)s|%(levelname)s|%(message)s",
)
LOGGER = logging.getLogger(__name__)
################################################################################################


def lungmask_postprocess(
    label_image: np.ndarray,
    spare: list = [],
    disable_tqdm: bool = False,
    skip_below: int = 3,
) -> np.ndarray:
    """some post-processing mapping small label patches to the neighbout whith which they share the
        largest border. Only largest connected components (CC) for each label will be kept. If a label is member of the spare list it will be mapped to neighboring labels and not present in the final labelling.

    Args:
        label_image (np.ndarray): Label image (int) to be processed
        spare (list, optional): Labels that are used for mapping to neighbors but not considered for final labelling. This is used for label fusion with a filling model. Defaults to [].
        disable_tqdm (bool, optional): If true, tqdm will be diabled. Defaults to False.
        skip_below (int, optional): If a CC is smaller than this value. It will not be merged but removed. This is for performance optimization.

    Returns:
        np.ndarray: Postprocessed volume
    """

    # CC analysis
    regionmask = skimage.measure.label(label_image)  # 连通区域标记
    origlabels = np.unique(label_image)  # 唯一标签
    origlabels_maxsub = np.zeros(
        (max(origlabels) + 1,), dtype=np.uint32
    )  # will hold the largest component for a label
    regions = skimage.measure.regionprops(regionmask, label_image)
    regions.sort(key=lambda x: x.area)
    regionlabels = [x.label for x in regions]

    # will hold mapping from regionlabels to original labels
    region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
    for r in regions:
        r_max_intensity = int(r.max_intensity)
        if r.area > origlabels_maxsub[r_max_intensity]:
            origlabels_maxsub[r_max_intensity] = r.area
            region_to_lobemap[r.label] = r_max_intensity

    for r in tqdm(regions, disable=disable_tqdm):
        r_max_intensity = int(r.max_intensity)
        if (
            r.area < origlabels_maxsub[r_max_intensity] or r_max_intensity in spare
        ) and r.area >= skip_below:  # area>2 improves runtime because small areas 1 and 2 voxel will be ignored
            bb = bbox_3D(regionmask == r.label)
            sub = regionmask[bb[0] : bb[1], bb[2] : bb[3], bb[4] : bb[5]]
            dil = ndimage.binary_dilation(sub == r.label)
            neighbours, counts = np.unique(sub[dil], return_counts=True)
            mapto = r.label
            maxmap = 0
            myarea = 0
            for ix, n in enumerate(neighbours):
                if n != 0 and n != r.label and counts[ix] > maxmap and n not in spare:
                    maxmap = counts[ix]
                    mapto = n
                    myarea = r.area
            regionmask[regionmask == r.label] = mapto

            # print(str(region_to_lobemap[r.label]) + ' -> ' + str(region_to_lobemap[mapto])) # for debugging
            if (
                regions[regionlabels.index(mapto)].area
                == origlabels_maxsub[
                    int(regions[regionlabels.index(mapto)].max_intensity)
                ]
            ):
                origlabels_maxsub[
                    int(regions[regionlabels.index(mapto)].max_intensity)
                ] += myarea
            regions[regionlabels.index(mapto)].__dict__["_cache"]["area"] += myarea

    outmask_mapped = region_to_lobemap[regionmask]
    outmask_mapped[np.isin(outmask_mapped, spare)] = 0

    if outmask_mapped.shape[0] == 1:
        holefiller = (
            lambda x: skimage.morphology.area_closing(
                x[0].astype(int), area_threshold=64
            )[None, :, :]
            == 1
        )
    else:
        holefiller = fill_voids.fill

    outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
    for i in np.unique(outmask_mapped)[1:]:
        outmask[holefiller(keep_largest_connected_component(outmask_mapped == i))] = i

    return outmask


def bbox_3D(labelmap, margin=2):
    """Compute bounding box of a 3D labelmap.

    Args:
        labelmap (np.ndarray): Input labelmap
        margin (int, optional): Margin to add to the bounding box. Defaults to 2.

    Returns:
        np.ndarray: Bounding box as [zmin, zmax, ymin, ymax, xmin, xmax]
    """
    shape = labelmap.shape
    dimensions = np.arange(len(shape))
    bmins = []
    bmaxs = []
    margin = [margin] * len(dimensions)
    for dim, dim_margin, dim_shape in zip(dimensions, margin, shape):
        margin_label = np.any(labelmap, axis=tuple(dimensions[dimensions != dim]))
        bmin, bmax = np.where(margin_label)[0][[0, -1]]
        bmin -= dim_margin
        bmax += dim_margin + 1
        bmin = max(bmin, 0)
        bmax = min(bmax, dim_shape)
        bmins.append(bmin)
        bmaxs.append(bmax)

    bbox = np.array(list(zip(bmins, bmaxs))).flatten()
    return bbox


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keeps largest connected component (CC)

    Args:
        mask (np.ndarray): Input label map

    Returns:
        np.ndarray: Binary label map with largest CC
    """
    mask = skimage.measure.label(mask)
    regions = skimage.measure.regionprops(mask)
    resizes = np.asarray([x.area for x in regions])
    max_region = np.argsort(resizes)[-1] + 1
    mask = mask == max_region
    return mask


def _create_neighborhood_mask(
    binary_mask_slice: np.ndarray, radius_voxels: int
) -> np.ndarray:
    """
    通过膨胀和减法操作，为二值掩码中的前景体素（True值）生成一个邻域掩码。

    Args:
        binary_mask_slice: 一个N维NumPy数组，代表二值掩码（前景为True，背景为False）。
        radius_voxels: 膨胀操作的半径（以体素为单位），用于定义邻域范围。

    Returns:
        一个N维NumPy数组，代表邻域掩码。此掩码中的True值表示那些
        在原始二值掩码前景体素的指定半径范围内，但本身不属于原始前景的体素。
    """
    if not np.any(binary_mask_slice):  # 如果输入掩码为空，则其邻域也为空
        return np.zeros_like(binary_mask_slice, dtype=bool)

    # 创建结构元素用于膨胀。connectivity=1表示面连接（例如3D中的6邻域）。
    # iterations参数控制膨胀次数，近似于半径效果。
    structuring_element = ndimage.generate_binary_structure(
        rank=binary_mask_slice.ndim, connectivity=1
    )
    dilated_mask = ndimage.binary_dilation(
        binary_mask_slice, structure=structuring_element, iterations=radius_voxels
    )
    # 从膨胀后的掩码中减去原始掩码，得到纯邻域
    neighborhood_mask = dilated_mask & ~binary_mask_slice
    return neighborhood_mask


def _check_lobe_integrity(
    segmentation_array: np.ndarray,
    target_lobe_labels_to_check: Tuple[int, ...],
    connectivity_structure: np.ndarray,
) -> bool:
    """
    检查指定的每个目标肺叶标签在分割数组中是否都表示为单个连通域，
    或者该标签在数组中已不存在（例如，所有小碎片都被清除了）。

    Args:
        segmentation_array: 包含分割标签的NumPy数组。
        target_lobe_labels_to_check: 需要检查的肺叶标签元组。
        connectivity_structure: 用于连通域分析的结构元素。

    Returns:
        bool: 如果所有指定的肺叶都满足条件（单个连通域或不存在），则返回True，否则返回False。
    """
    for lobe_label in target_lobe_labels_to_check:
        current_lobe_binary_mask = segmentation_array == lobe_label
        if not np.any(
            current_lobe_binary_mask
        ):  # 如果该肺叶标签在数组中不存在，也视为满足条件
            continue
        _, num_components = ndimage.label(
            current_lobe_binary_mask, structure=connectivity_structure
        )
        if num_components > 1:
            LOGGER.debug(
                f"肺叶标签 {lobe_label} 仍有 {num_components} 个连通域，完整性检查未通过。"
            )
            return False
    LOGGER.debug("所有指定肺叶均满足完整性检查（单个连通域或不存在）。")
    return True


def _perform_single_lobe_refinement_pass(
    current_segmentation_array: np.ndarray,
    all_target_lobe_labels: Tuple[int, ...],
    connectivity_structure: np.ndarray,
    min_fragment_volume_voxels: int = 5,
    neighborhood_radius_voxels: int = 1,
) -> Tuple[np.ndarray, bool]:
    """
    对当前分割掩码执行单次完整的肺叶清理和优化流程。

    核心逻辑：
    1. 对于每个目标肺叶标签：
       a. 识别所有连通域（“碎片”）。
       b. 体积小于 `min_fragment_volume_voxels` 的碎片直接标记为背景 (0)。
       c. 保留体积最大的连通域（“主要区域”）。
       d. 其他非主要且体积不小于阈值的碎片，将根据其邻域中最主要的、
          且非自身的其他肺叶标签，或背景标签，来进行重新标记。
          邻域的投票基于本次迭代开始时的、只包含目标肺叶和背景的参考掩码。

    Args:
        current_segmentation_array: 当前迭代周期的分割掩码NumPy数组。
        all_target_lobe_labels: 需要处理的所有目标肺叶标签的元组。
        connectivity_structure: 用于连通域分析的结构元素。
        min_fragment_volume_voxels: 小于此体积（体素数量）的连通域将被直接移除（标记为0）。
        neighborhood_radius_voxels: 定义邻域范围的半径（用于 `_create_neighborhood_mask`）。

    Returns:
        Tuple[np.ndarray, bool]:
            - 第一个元素 (np.ndarray): 执行本次优化后得到的分割掩码数组。
            - 第二个元素 (bool): 如果本次优化对掩码进行了任何修改，则为True，否则为False。
    """
    array_after_pass = (
        current_segmentation_array.copy()
    )  # 创建副本进行修改，避免影响原始输入
    modifications_made_in_pass = False

    # 创建一个参考数组，用于邻域投票决策。
    # 该数组基于本次迭代开始时的掩码状态，并且只包含指定的目标肺叶标签或背景(0)。
    # 这样做可以确保在一个pass内部，对所有肺叶碎片的修复决策都基于一个统一的、稳定的初始参考。
    decision_reference_array = current_segmentation_array.copy()
    # 将所有不在目标肺叶标签列表中的体素值在参考数组中置为0
    labels_to_zero_out_mask = ~np.isin(decision_reference_array, all_target_lobe_labels)
    decision_reference_array[labels_to_zero_out_mask] = 0

    # 遍历每个指定的目标肺叶标签
    for current_processing_lobe_label in all_target_lobe_labels:
        # 获取当前处理的肺叶的二值掩码
        # 注意：这里是在 array_after_pass 上操作，因为它可能已被上一个肺叶标签的处理修改过
        current_lobe_binary_mask = array_after_pass == current_processing_lobe_label

        if not np.any(current_lobe_binary_mask):  # 如果该肺叶已不存在，则跳过
            LOGGER.debug(
                f"肺叶标签 {current_processing_lobe_label} 在当前掩码中不存在，跳过处理。"
            )
            continue

        # 对当前目标肺叶进行一次连通域分析
        labeled_fragments_array, num_fragments = ndimage.label(
            current_lobe_binary_mask, structure=connectivity_structure
        )

        if (
            num_fragments <= 1
        ):  # 如果只有一个连通域或没有（理论上 np.any 已处理），则无需处理
            LOGGER.debug(
                f"肺叶标签 {current_processing_lobe_label} 已是单连通域或无碎片，跳过。"
            )
            continue

        LOGGER.debug(
            f"处理肺叶标签: {current_processing_lobe_label}，发现 {num_fragments} 个碎片。"
        )

        fragment_ids, fragment_volumes = np.unique(
            labeled_fragments_array[labeled_fragments_array > 0], return_counts=True
        )

        # fragment_ids 此时是当前 labeled_fragments_array 中的标签值 (从1到num_fragments)
        # component_bounding_boxes 的索引是 0 到 num_fragments-1
        component_bounding_boxes = ndimage.find_objects(input=labeled_fragments_array)
        dominant_component_id = fragment_ids[np.argmax(fragment_volumes)]

        LOGGER.debug(
            f"肺叶 {current_processing_lobe_label}: 主要区域ID {dominant_component_id}，体积 {fragment_volumes.max()}。"
        )

        for (
            current_fragment_id,
            current_fragment_volume,
            current_fragment_bbox_slice,
        ) in zip(fragment_ids, fragment_volumes, component_bounding_boxes):
            fragment_global_voxels_mask = labeled_fragments_array == current_fragment_id

            if current_fragment_volume < min_fragment_volume_voxels:
                if np.any(array_after_pass[fragment_global_voxels_mask] != 0):
                    LOGGER.debug(
                        f"  碎片ID {current_fragment_id} (属肺叶 {current_processing_lobe_label}), "
                        f"体积 {current_fragment_volume} < {min_fragment_volume_voxels}，标记为背景。"
                    )
                    array_after_pass[fragment_global_voxels_mask] = 0
                    modifications_made_in_pass = True
                continue

            if current_fragment_id == dominant_component_id:
                continue

            LOGGER.debug(
                f"  处理碎片ID {current_fragment_id} (属肺叶 {current_processing_lobe_label}), 体积 {current_fragment_volume}。"
                f" 边界框: {current_fragment_bbox_slice}"
            )

            fragment_mask_in_bbox = (
                labeled_fragments_array[current_fragment_bbox_slice]
                == current_fragment_id
            )
            neighborhood_mask_in_bbox = _create_neighborhood_mask(
                fragment_mask_in_bbox, radius_voxels=neighborhood_radius_voxels
            )
            neighbor_voxels_for_voting = decision_reference_array[
                current_fragment_bbox_slice
            ][neighborhood_mask_in_bbox]

            new_label_for_fragment = 0
            if neighbor_voxels_for_voting.size > 0:
                unique_neighbor_labels, counts_neighbor_labels = np.unique(
                    neighbor_voxels_for_voting, return_counts=True
                )
                # 将标签和其计数打包，并按计数降序排列
                sorted_neighbor_label_counts = sorted(
                    zip(unique_neighbor_labels, counts_neighbor_labels),
                    key=lambda item: item[1],
                    reverse=True,
                )

                for potential_new_label, count in sorted_neighbor_label_counts:
                    if potential_new_label == current_processing_lobe_label:
                        continue
                    if potential_new_label == 0:  # 背景标签是最后的选择
                        continue
                    # 如果是一个其他有效的目标肺叶标签
                    new_label_for_fragment = potential_new_label
                    LOGGER.debug(
                        f"    碎片ID {current_fragment_id} 邻域投票决定新标签为: {new_label_for_fragment} "
                        f"(基于邻域标签 {potential_new_label}，数量 {count})"
                    )
                    break  # 已找到最主要的其他肺叶标签
                    # 如果一直找不到，则会赋值0

                array_after_pass[fragment_global_voxels_mask] = new_label_for_fragment
                modifications_made_in_pass = True

    return array_after_pass, modifications_made_in_pass


def refine_lung_lobe_segmentation_iteratively(
    segmentation_mask_filepath: str,
    target_lobe_labels: Tuple[int, ...] = (1, 2, 3, 4, 5),
    output_filepath: Optional[str] = None,
    max_iterations: int = 10,
    min_fragment_volume_voxels: int = 5,
    neighborhood_radius_voxels: int = 1,
    verbose_logging: bool = False,
) -> sitk.Image:
    """
    迭代式地清理和优化3D肺叶分割掩码。

    在每次迭代中，针对每个指定的目标肺叶标签：
    1. 体积小于 `min_fragment_volume_voxels` 的连通域（碎片）将被设置为背景标签 (0)。
    2. 对于其他较小的、非主要的连通域，其新标签将根据其邻域中出现频率最高的有效标签
       （即 `target_lobe_labels` 中排除该碎片当前标签的其他标签，或背景标签0）来确定。
       邻域的确定使用 `_create_neighborhood_mask` 函数。
       邻域投票基于该次迭代开始时，掩码中仅包含目标肺叶和背景的快照状态。

    迭代停止条件：
    1. 在完整的一次处理流程（pass）中，没有对任何目标肺叶相关的体素进行修改。
    2. 所有指定的目标肺叶标签都已成为单个连通域（或在掩码中已不存在，例如被完全清理）。
    3. 达到 `max_iterations` 定义的最大迭代次数。

    Args:
        segmentation_mask_filepath: NIFTI格式的肺叶分割掩码文件路径。
        target_lobe_labels: 一个包含代表不同肺叶的整数标签的元组。
                            默认为 (1, 2, 3, 4, 5)。
        output_filepath: 清理后的分割掩码的可选保存路径。如果为None，则不保存。
        max_iterations: 执行清理和优化的最大迭代次数。
        min_fragment_volume_voxels: 体积小于此值的连通域将被移除（标记为0）。
        neighborhood_radius_voxels: 定义邻域范围的半径（以体素为单位）。
        verbose_logging: 如果为True，日志级别将设置为DEBUG，输出更详细的信息。

    Returns:
        一个SimpleITK.Image对象，包含清理和优化后的肺叶分割掩码。

    Raises:
        FileNotFoundError: 如果 `segmentation_mask_filepath` 无效或无法读取。
        ValueError: 如果初始掩码中未找到任何指定的 `target_lobe_labels`。
    """
    if verbose_logging:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)  # 确保至少INFO级别被记录

    LOGGER.info(f"开始肺叶分割迭代优化，目标文件: {segmentation_mask_filepath}")
    LOGGER.info(f"目标肺叶标签: {target_lobe_labels}, 最大迭代次数: {max_iterations}")
    LOGGER.info(
        f"最小碎片体积: {min_fragment_volume_voxels} 体素, 邻域半径: {neighborhood_radius_voxels} 体素"
    )

    try:
        segmentation_image = sitk.ReadImage(segmentation_mask_filepath)
        LOGGER.debug(f"成功读取图像: {segmentation_mask_filepath}")
    except RuntimeError as e:
        LOGGER.error(f"读取图像文件失败: {segmentation_mask_filepath} - {e}")
        raise FileNotFoundError(f"无法读取图像文件: {segmentation_mask_filepath}: {e}")

    segmentation_array = sitk.GetArrayFromImage(segmentation_image)
    LOGGER.debug(
        f"图像转换为NumPy数组，形状: {segmentation_array.shape}, 数据类型: {segmentation_array.dtype}"
    )

    # 初始检查：确认至少存在一个指定的目标肺叶标签
    initial_labels_present = [
        label for label in target_lobe_labels if np.any(segmentation_array == label)
    ]
    if not initial_labels_present:
        message = f"在初始掩码 {segmentation_mask_filepath} 中未找到任何指定的目标肺叶标签 {target_lobe_labels}。"
        LOGGER.error(message)
        raise ValueError(message)

    if len(initial_labels_present) < len(target_lobe_labels):
        missing_labels = set(target_lobe_labels) - set(initial_labels_present)
        LOGGER.warning(
            f"警告: 初始掩码缺少以下肺叶标签: {missing_labels}。"
            f"将仅处理找到的标签: {initial_labels_present}"
        )
    else:
        LOGGER.debug("初始掩码中包含所有指定的目标肺叶标签。")

    # 定义3D图像的6邻域（面连接）结构元素
    connectivity_structure_3d = ndimage.generate_binary_structure(
        rank=3, connectivity=1
    )

    for iteration_count in range(max_iterations):
        LOGGER.info(
            f"--- 开始迭代优化: 第 {iteration_count + 1}/{max_iterations} 轮 ---"
        )

        # 执行单次完整的清理和优化流程
        segmentation_array_after_pass, modifications_made_this_iteration = (
            _perform_single_lobe_refinement_pass(
                current_segmentation_array=segmentation_array,
                all_target_lobe_labels=target_lobe_labels,  # 传递所有原始指定标签
                connectivity_structure=connectivity_structure_3d,
                min_fragment_volume_voxels=min_fragment_volume_voxels,
                neighborhood_radius_voxels=neighborhood_radius_voxels,
            )
        )

        # 更新当前分割掩码数组
        segmentation_array = segmentation_array_after_pass

        # 检查迭代终止条件
        if not modifications_made_this_iteration:
            LOGGER.info("本轮迭代未对目标肺叶标签进行任何修改。优化过程提前终止。")
            break

        if _check_lobe_integrity(
            segmentation_array, target_lobe_labels, connectivity_structure_3d
        ):
            LOGGER.info(
                "所有指定的目标肺叶均已满足完整性条件（单个连通域或不存在）。优化过程提前终止。"
            )
            break

        if iteration_count == max_iterations - 1:
            LOGGER.info("已达到最大迭代次数。优化过程结束。")

    LOGGER.info("肺叶分割迭代优化完成。")

    # 将最终的NumPy数组转换回SimpleITK图像对象
    refined_segmentation_image = sitk.GetImageFromArray(segmentation_array)
    # 复制原始图像的元数据（如spacing, origin, direction）到新图像
    refined_segmentation_image.CopyInformation(segmentation_image)
    LOGGER.debug("已将优化后的NumPy数组转换回SimpleITK图像并复制元数据。")

    if output_filepath:
        try:
            sitk.WriteImage(refined_segmentation_image, output_filepath)
            LOGGER.info(f"已将优化后的分割掩码保存至: {output_filepath}")
        except RuntimeError as e:
            LOGGER.error(f"保存优化后的掩码失败: {output_filepath} - {e}")

    return refined_segmentation_image
