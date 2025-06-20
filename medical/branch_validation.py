import os
import sys
import time
from functools import wraps

import networkx as nx
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PWD)
from valtools import ct_seg_perf, get_array

"""
设计说明：

1. 主要参考ATM22的github仓库：https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work
2. 其次参考：https://github.com/Nandayang/FANN-for-airway-segmentation/tree/main
"""

EPSILON = 1e-32
FACE_CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(3, 1)
CONNECTIVITY_26_3D = ndimage.generate_binary_structure(3, 3)
# 全局开关
ENABLE_TIMING = False


def timeit(func):
    """Decorator to time the execution of a function."""
    if not ENABLE_TIMING:
        return func  # 直接返回原函数，不包装

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


@timeit
def get_largest_connected_component_array(mask_array: np.ndarray) -> np.ndarray:
    """
    Identifies and extracts the largest connected component from a 3D binary mask.

    Args:
        mask_array: A 3D NumPy array where the foreground is represented by non-zero values.

    Returns:
        A 3D NumPy array containing only the largest connected component (labeled as 1),
        with all other voxels set to 0. Returns the original mask if no connected
        components are found.
    """
    # Label each connected region in the binary mask
    labeled_mask, num_components = ndimage.label(
        mask_array, structure=FACE_CONNECTIVITY_STRUCTURE
    )

    # If no connected components are found, return the original mask
    if num_components == 0:
        return mask_array

    # Calculate the volume (number of voxels) of each connected component
    component_labels, component_sizes = np.unique(
        labeled_mask[labeled_mask > 0], return_counts=True
    )

    # Find the label of the largest connected component
    largest_component_label = component_labels[np.argmax(component_sizes)]

    # Create a binary mask containing only the largest component
    largest_component_mask = (labeled_mask == largest_component_label).astype(np.uint8)

    # Fill any holes within the largest connected component
    filled_largest_component = ndimage.binary_fill_holes(largest_component_mask).astype(
        np.uint8
    )

    return filled_largest_component


@timeit
def parse_skeleton_into_branches(
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Identifies and segments individual branches within a binary skeleton.

    Args:
        skeleton: A 3D NumPy array representing the binary skeleton.

    Returns:
        A tuple containing:
            - branch_skeleton: A 3D NumPy array with points removed that are likely branching or complex.
            - labeled_branches: A 3D NumPy array where each distinct connected branch is assigned a unique integer label.
            - num_branches: The total number of identified branches.
    """
    # Count the number of neighboring skeleton voxels for each point using convolution
    neighbor_counts = ndimage.convolve(skeleton, CONNECTIVITY_26_3D) * skeleton
    # Create a copy of the skeleton to mark branch points for removal
    branch_skeleton = skeleton.copy()
    # Identify and remove voxels with more than 3 neighbors (likely branching or complex points)
    branch_skeleton[neighbor_counts > 3] = 0
    # Label the connected components (individual branches) in the simplified skeleton
    labeled_branches, num_branches = ndimage.label(
        branch_skeleton, structure=CONNECTIVITY_26_3D
    )
    # Identify and remove small, potentially spurious branches
    branch_labels, branch_sizes = np.unique(
        labeled_branches[labeled_branches > 0], return_counts=True
    )
    small_branch_threshold = 5
    small_branch_ids = branch_labels[branch_sizes < small_branch_threshold]
    # Create a refined skeleton by removing the small branches
    refined_branch_skeleton = branch_skeleton.copy()
    refined_branch_skeleton[np.isin(branch_skeleton, small_branch_ids)] = 0
    # Re-label the refined skeleton to get the final set of branches
    labeled_branches, num_branches = ndimage.label(
        refined_branch_skeleton, structure=CONNECTIVITY_26_3D
    )

    return refined_branch_skeleton, labeled_branches, num_branches


@timeit
def fill_airway_with_branch_labels(
    skeleton_parse, segmentation_label, labeled_branches
):
    """
    Assigns branch IDs from the skeleton parsing to the corresponding voxels in the original label.

    Args:
        skeleton_parse (np.ndarray): The parsed skeleton.
        segmentation_label (np.ndarray): The original binary segmentation label.
        labeled_branches (np.ndarray): Labeled array of connected skeleton branches.

    Returns:
        np.ndarray: A NumPy array where each voxel belonging to the airway is labeled with its branch ID.
    """
    # Calculate the distance transform from the background to the parsed skeleton
    # 输入： 1 - skeleton_parse。这里对解析后的骨架进行了反转，使得骨架像素的值为 0，背景像素的值为 1。
    # ndimage.distance_transform_edt 函数计算了图像中每个像素到最近的非零像素（在我们的例子中，是原始的骨架像素）
    # 的欧几里得距离。return_indices=True 使得该函数不仅返回距离值，还返回最近非零像素的索引。
    # 原理： 这一步的目的是为了找到原始气道分割中的每个体素，离哪个骨架分支上的点最近。通过计算距离变换，我们可以得到每个体素的“最近的骨架邻居”的信息。

    distance_map, nearest_skeleton_indices = ndimage.distance_transform_edt(
        1 - skeleton_parse, return_indices=True
    )
    # Initialize an array for the tree parsing with the same shape as the segmentation label
    tree_parsing = np.zeros(segmentation_label.shape, dtype=np.uint16)
    # Use the indices of the nearest skeleton points to assign the branch ID from 'labeled_branches' to the 'segmentation_label'
    # 1. labeled_branches: 这个数组包含了骨架各个连通分支的唯一 ID 标签。
    # 2. nearest_skeleton_indices: 这是上一步距离变换返回的最近骨架点的索引。它是一个包含三个数组的元组，分别对应 Z、Y、X 三个维度的索引。
    # 3. labeled_branches[nearest_skeleton_indices[0, ...], nearest_skeleton_indices[1, ...], nearest_skeleton_indices[2, ...]]:
    ### 这部分代码利用 nearest_skeleton_indices 来查找 labeled_branches 中对应位置的标签值。本质上，对于原始分割中的每一个体素，我们都找到了它最近的骨架点，
    ### 然后从 labeled_branches 中提取了那个骨架点所属的分支的 ID。
    # 4. * segmentation_label: 最后，将上述结果与原始的 segmentation_label 相乘。由于 segmentation_label 是一个二值化数组（1 代表气道，0 代表背景），
    ### 这个乘法操作确保了只有在原始分割中属于气道的体素才会被赋予分支的 ID。背景区域的体素与 0 相乘后仍然是 0。
    # 原理： 通过这种方式，原始分割中的每个气道体素都被分配了距离它最近的骨架分支的 ID。这样，分割出的整个气道结构就被按照其骨架分支进行了“划分”和标记。
    tree_parsing = (
        labeled_branches[
            nearest_skeleton_indices[0, ...],
            nearest_skeleton_indices[1, ...],
            nearest_skeleton_indices[2, ...],
        ]
        * segmentation_label
    )
    return tree_parsing


@timeit
def locate_trachea(tree_parsing, num_branches):
    """
    Identifies the label ID of the trachea based on the volume of the branches.
    loc_trachea 函数的目的是在已经标记了各个分支的 tree_parsing 数组中找到代表气管的那一个分支的标签 ID。
    它的原理基于一个假设：气管是整个气道树中体积最大的分支。

    Args:
        tree_parsing (np.ndarray): Array where each branch is labeled with a unique ID.
        num_branches (int): The total number of branches.

    Returns:
        int: The label ID of the trachea.
    """
    # Calculate the volume of each branch
    labels, branch_volumes = np.unique(
        tree_parsing[tree_parsing > 0], return_counts=True
    )
    # The trachea is assumed to be the branch with the largest volume
    trachea_label_id = labels[np.argmax(branch_volumes)]
    return trachea_label_id


@timeit
def parse_airway_tree(airway_mask_array):
    """
    从二值化气道掩模执行气道树解析的高级函数。

    Args:
        airway_mask_array (np.ndarray): 一个3D NumPy数组，表示二值化气道掩模。前景1，背景0；np.uint8类型。

    Returns:
        tuple[np.ndarray, np.ndarray, int]: 一个包含解析后的气道树、骨架和分支数量的元组。
        - np.ndarray: The parsed airway tree where each branch has a unique ID.
        - np.ndarray: The skeleton of the airway mask.
        - int: The number of detected branches.
    """
    airway_mask_array = get_largest_connected_component_array(airway_mask_array)
    skeleton = skeletonize(airway_mask_array).astype(np.uint8)
    skeleton_parse, labeled_branches, num_branches = parse_skeleton_into_branches(
        skeleton
    )
    tree_parsing = fill_airway_with_branch_labels(
        skeleton_parse, airway_mask_array, labeled_branches
    )
    # tree_parsing: airway数组，有branch标签。
    return tree_parsing, skeleton, num_branches


@timeit
def build_branch_adjacency_map(
    branch_labels: np.ndarray, num_branches: int
) -> np.ndarray:
    """
    使用 ndimage.find_objects 更高效地构建3D树状结构中分支的邻接矩阵 (z, y, x 轴顺序).

    Args:
        branch_labels: 一个3D NumPy数组，不同的正整数代表不同的分支（ndimage.label 的结果）。
        num_branches: 检测到的分支总数。

    Returns:
        一个形状为 (num_branches, num_branches) 的NumPy邻接矩阵，表示分支间的邻接关系。
    """
    adjacency_matrix = np.zeros((num_branches, num_branches), dtype=np.uint8)
    # 定义用于膨胀的结构元素（6-连通性）
    # dilation_structure = ndimage.generate_binary_structure(3, 1)

    # 使用 find_objects 获取每个分支的切片对象
    # 这个函数比for循环计算bbox速度快很多。
    branch_object_slices = ndimage.find_objects(branch_labels)

    for label_index, object_slice in enumerate(branch_object_slices):
        if object_slice is None:
            continue  # 跳过空对象

        branch_label = label_index + 1
        if branch_label > num_branches:
            continue

        # 使用切片提取当前分支的局部标签区域
        cropped_labels = branch_labels[object_slice]
        # 创建当前分支的局部二值掩码
        current_branch_mask = (cropped_labels == branch_label).astype(np.uint8)

        # 对局部掩码进行膨胀
        dilated_mask = ndimage.binary_dilation(
            current_branch_mask, structure=FACE_CONNECTIVITY_STRUCTURE
        ).astype(np.uint8)

        # 计算膨胀后的边界
        boundary = dilated_mask - current_branch_mask

        # 在原始标签数组中找到边界区域对应的标签
        adjacent_region_labels = branch_labels[object_slice]
        boundary_labels = np.unique(adjacent_region_labels[boundary > 0])
        # 排除背景标签 (0)
        boundary_labels = boundary_labels[boundary_labels > 0]

        # 更新邻接矩阵
        for neighbor_label in boundary_labels:
            # 确保不是当前分支并且在有效标签范围内
            if neighbor_label != branch_label and 1 <= neighbor_label <= num_branches:
                adjacency_matrix[branch_label - 1, neighbor_label - 1] = 1
                adjacency_matrix[neighbor_label - 1, branch_label - 1] = (
                    1  # 邻接关系是对称的
                )

    return adjacency_matrix


def build_parent_children_maps_nx(
    adjacency_matrix: np.ndarray, root_label: int, num_branches: int
):
    """
    使用 NetworkX 库构建父节点和子节点映射，以及计算每个节点的代数 (Generation).

    Args:
        adjacency_matrix: 一个 NumPy 数组，表示分支之间的邻接关系（邻接矩阵）。
        root_label: 整数，表示树的根节点（例如气管）的标签。
        num_branches: 整数，表示树中分支的总数。

    Returns:
        包含父节点映射、子节点映射和代数信息的元组：
        - parent_map: 一个 NumPy 数组，parent_map[i, j] = 1 表示节点 j 是节点 i 的父节点。
        - children_map: 一个 NumPy 数组，children_map[i, j] = 1 表示节点 j 是节点 i 的子节点。
        - generation: 一个 NumPy 数组，generation[i] 表示节点 i 的代数。
    """
    # 创建一个空的 NetworkX 图
    graph = nx.Graph()

    # 将每个分支添加为图的节点
    for i in range(1, num_branches + 1):
        graph.add_node(i)

    # 根据邻接矩阵添加边到图中
    for i in range(num_branches):
        for j in range(i + 1, num_branches):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(i + 1, j + 1)

    # 指定根节点
    root_node = root_label

    # 初始化存储父节点、子节点和代数的 NumPy 数组
    parent_map = np.zeros((num_branches, num_branches), dtype=np.uint8)
    children_map = np.zeros((num_branches, num_branches), dtype=np.uint8)
    generation = np.zeros(num_branches, dtype=np.uint8)

    # 检查根节点是否存在于图中
    if root_node not in graph:
        raise ValueError(f"根节点标签 {root_node} 未在图中找到。")

    # 使用广度优先搜索 (BFS) 查找前驱节点 (父节点)
    predecessors = nx.bfs_predecessors(graph, root_node)

    # 构建父节点映射和子节点映射
    parent_dict = {}
    for child, parent in predecessors:
        if parent is not None:
            parent_dict[child] = parent
            parent_index = parent - 1
            child_index = child - 1
            parent_map[child_index, parent_index] = 1
            children_map[parent_index, child_index] = 1

    # 计算每个节点的代数（从根节点开始的距离）
    shortest_path_lengths = nx.shortest_path_length(graph, source=root_node)
    for node, length in shortest_path_lengths.items():
        generation[node - 1] = length

    return parent_map, children_map, generation


@timeit
def refine_tree_structure(
    tree_parsing: np.ndarray,
    num_branches: int,
    trachea_label: int,
    max_iterations: int = 10,
):
    """
    迭代地细化树状结构，直到没有进一步的细化发生或达到最大迭代次数。

    该函数通过构建邻接矩阵和父子关系映射，识别并处理树结构中常见的错误，
    例如具有多个父节点的子分支和只有一个子节点的中间分支。

    Args:
        tree_parsing: 初始的树状结构标签数组，不同的正整数代表不同的分支。
        num_branches: 初始的分支数量。
        trachea_label: 气管的标签，作为树的根节点。
        max_iterations: 最大细化迭代次数，防止无限循环。默认为 10。

    Returns:
        细化后的树状结构标签数组和最终的分支数量。
    """
    refined = True  # 标记在当前迭代中是否发生了细化
    iteration = 0
    current_tree_parsing = tree_parsing.copy()
    current_num_branches = num_branches
    previous_num_branches = -1  # 用于检测分支数量是否发生变化，判断是否收敛

    while (
        refined
        and iteration < max_iterations
        and current_num_branches != previous_num_branches
    ):
        previous_num_branches = current_num_branches
        print(f"Refinement iteration: {iteration}")

        # 步骤 1: 构建当前树结构的邻接矩阵，表示分支之间的连接关系
        adjacency_matrix = build_branch_adjacency_map(
            branch_labels=current_tree_parsing, num_branches=current_num_branches
        )

        # 步骤 2: 定位当前树结构中的气管标签 (如果需要在每次迭代中重新定位)
        trachea_label = locate_trachea(
            tree_parsing=current_tree_parsing, num_branches=current_num_branches
        )

        # 步骤 3: 基于邻接矩阵和气管标签，构建父子关系映射和代数信息
        parent_map, children_map, generation = build_parent_children_maps_nx(
            adjacency_matrix=adjacency_matrix,
            root_label=trachea_label,
            num_branches=current_num_branches,
        )

        # 步骤 4: 进行树结构的细化，解决多父节点和单子节点问题
        current_tree_parsing, current_num_branches = refine_branches(
            parent_map,
            children_map,
            current_tree_parsing,
            current_num_branches,
            trachea_label,
        )

        # 步骤 5: 检查在当前迭代中是否发生了任何细化
        refined = has_refinement_occurred(parent_map, children_map)
        iteration += 1

    return current_tree_parsing, current_num_branches


@timeit
def refine_branches(
    parent_map: np.ndarray,
    children_map: np.ndarray,
    branch_labels: np.ndarray,
    num_branches: int,
    trachea_label: int,
) -> tuple[np.ndarray, int]:
    """
    根据父节点和子节点映射细化树状结构标签，解决多父节点和单子节点问题。

    Args:
        parent_map: 父节点映射矩阵，parent_map[i, j] = 1 表示节点 j 是节点 i 的父节点。
        children_map: 子节点映射矩阵，children_map[i, j] = 1 表示节点 j 是节点 i 的子节点。
        branch_labels: 当前的树状结构标签数组。
        num_branches: 当前的分支数量。
        trachea_label: 气管的标签。

    Returns:
        细化后的树状结构标签数组和更新后的分支数量。
    """
    refined_labels = branch_labels.copy()
    deleted_branch_ids = set()  # 用于存储被删除（合并）的分支标签 ID

    # 步骤 1: 处理多父节点的子分支
    for i in range(num_branches):
        # 查找分支 i 的所有父节点标签
        parent_indices = np.where(parent_map[i, :] > 0)[0] + 1
        # 如果分支 i 有多个父节点
        if len(parent_indices) > 1:
            # 选择第一个父节点作为主要父节点
            primary_parent = parent_indices[0]
            # 将后续的父节点标签在 refined_labels 中替换为主要父节点的标签，实现融合
            for parent_to_fuse in parent_indices[1:]:
                refined_labels[refined_labels == parent_to_fuse] = primary_parent
                # 将被融合的父节点的标签 ID 添加到 deleted_branch_ids 集合
                deleted_branch_ids.add(parent_to_fuse)

    # 步骤 2: 处理只有一个子节点的中间分支
    for i in range(num_branches):
        # 查找分支 i 的所有子节点标签
        child_indices = np.where(children_map[i, :] > 0)[0] + 1
        # 如果分支 i 只有一个子节点
        if len(child_indices) == 1:
            child_label = child_indices[0]
            # 如果子节点和当前分支都没有被标记为删除
            if (
                child_label not in deleted_branch_ids
                and i + 1 not in deleted_branch_ids
            ):
                # 将子节点的标签在 refined_labels 中替换为当前分支的标签，实现合并
                refined_labels[refined_labels == child_label] = i + 1
                # 将被合并的子节点的标签 ID 添加到 deleted_branch_ids 集合
                deleted_branch_ids.add(child_label)

    # 步骤 3: 重新编号剩余的分支标签，确保标签的连续性
    # 获取所有原始标签 ID 中未被删除的标签 ID
    remaining_labels = sorted(
        list(set(range(1, num_branches + 1)) - deleted_branch_ids)
    )
    # 创建一个旧标签到新标签的映射字典
    label_map = {
        old_label: new_label
        for new_label, old_label in enumerate(remaining_labels, start=1)
    }
    # 创建一个新的标签数组用于存储重新编号后的结果
    final_labels = np.zeros_like(refined_labels)
    # 根据映射字典替换旧标签为新标签
    for old_label, new_label in label_map.items():
        final_labels[refined_labels == old_label] = new_label
    # 更新细化后的分支数量
    num_refined_branches = len(remaining_labels)

    return final_labels, num_refined_branches


@timeit
def has_refinement_occurred(parent_map: np.ndarray, children_map: np.ndarray) -> bool:
    """
    检查是否需要对树状结构进行细化（是否存在多父节点或单子节点的情况）。

    Args:
        parent_map: 父节点映射矩阵。
        children_map: 子节点映射矩阵。

    Returns:
        如果需要细化返回 True，否则返回 False。
    """
    # 如果存在任何节点的父节点数量大于 1，则返回 True
    if np.any(np.sum(parent_map, axis=1) > 1):
        return True
    # 如果存在任何节点的子节点数量等于 1，则返回 True
    if np.any(np.sum(children_map, axis=1) == 1):
        return True
    # 如果以上两种情况都不存在，则返回 False
    return False


@timeit
def evaluation_branch_metrics(
    ground_truth=None,
    prediction=None,
    ground_truth_label=1,
    prediction_label=1,
    series_uid="ATM_011",
    save_dir="output",
    reference_image=None,
    calculate_dice=False,
):
    """
    评估分支层面的分割性能，使用 Dice、IoU、Detected Length Ratio (DLR) 和 Detected Branch Ratio (BDR) 等指标。

    Args:
        ground_truth: 可以是真实值（GT）掩码的文件路径、SimpleITK.Image 对象或 NumPy 数组。
        prediction: 可以是预测值（Prediction）掩码的文件路径、SimpleITK.Image 对象或 NumPy 数组 (当 calculate_dice 为 False 时可以为 None)。
        ground_truth_label: 真实值掩码中目标（气道）的标签值。默认为 1。
        prediction_label: 预测值掩码中目标（气道）的标签值。默认为 1。
        series_uid: 病例序列的唯一标识符，用于保存结果和中间文件。默认为 "ATM_011"。
        save_dir: 保存中间结果（如骨架和分支标签）的目录路径。默认为 "output"。
        reference_image: 用于复制元数据的参考 SimpleITK.Image 对象。仅当需要保存新的骨架和分支标签文件且 ground_truth 不是文件路径时才需要。
        calculate_dice: 一个布尔标志，指示是否计算 Dice 和 IoU 等基于体素的指标。默认为 False。

    Returns:
        包含评估指标的元组：
        - series_uid (str): 病例序列的唯一标识符。
        - result (dict): 一个字典，包含评估指标。
    """
    result = {}

    # ---------------------- 准备数据 ----------------------
    ground_truth_image = None
    ground_truth_array = None
    prediction_image = None
    prediction_array = None

    os.makedirs(save_dir, exist_ok=True)
    ground_truth_skeleton_path = os.path.join(
        save_dir, f"{series_uid}.gt_skeleton.nii.gz"
    )
    ground_truth_branches_path = os.path.join(
        save_dir, f"{series_uid}.gt_branches.nii.gz"
    )
    gt_files_exist = os.path.exists(ground_truth_skeleton_path) and os.path.exists(
        ground_truth_branches_path
    )

    if calculate_dice and ground_truth is None:
        raise ValueError("`calculate_dice`为True时，必须提供`ground_truth`")
    
    # 读取真实值掩码
    if calculate_dice or (ground_truth is not None) or (not gt_files_exist):
        if isinstance(ground_truth, str):
            ground_truth_image = sitk.ReadImage(ground_truth)
            ground_truth_array = sitk.GetArrayFromImage(ground_truth_image).astype(np.uint8)
        elif isinstance(ground_truth, sitk.Image):
            ground_truth_image = ground_truth
            ground_truth_array = sitk.GetArrayFromImage(ground_truth_image).astype(np.uint8)
        elif isinstance(ground_truth, np.ndarray):
            ground_truth_array = ground_truth.astype(np.uint8)
            ground_truth_image = sitk.GetImageFromArray(ground_truth_array)
        elif ground_truth is None:
            pass
        else:
            raise ValueError(
                "ground_truth must be a file path, SimpleITK.Image object, or NumPy array."
            )
        ground_truth_array = (ground_truth_array == ground_truth_label).astype(np.uint8)

    # 读取预测值掩码 (如果需要)
    if calculate_dice or prediction is not None:
        if isinstance(prediction, str):
            prediction_image = sitk.ReadImage(prediction)
        elif isinstance(prediction, sitk.Image):
            prediction_image = prediction
        elif isinstance(prediction, np.ndarray):
            prediction_image = sitk.GetImageFromArray(prediction)
        elif prediction is not None:
            raise ValueError(
                "prediction must be a file path, SimpleITK.Image object, or NumPy array."
            )
        if prediction_image is not None:
            prediction_array = sitk.GetArrayFromImage(prediction_image).astype(np.uint8)
            prediction_array = (prediction_array == prediction_label).astype(np.uint8)

    # ---------------------- 计算基于体素的指标（如 DICE 和 IoU） ----------------------
    if calculate_dice and prediction_array is not None:
        metrics = ct_seg_perf(
            pred_mask=prediction_array,
            true_mask=ground_truth_array,
            binarize=False,
            pred_mask_path=None,
            true_mask_path=None,
            pred_class_ids=[1],
            true_class_ids=[1],
        )
        # 将 ct_seg_perf 返回的指标解析到 result 字典中
        if "1" in metrics:
            result.update(metrics["1"])

    # ---------------------- 准备计算分支层面指标所需的数据 ----------------------
    ground_truth_skeleton = None
    ground_truth_branches = None
    num_branches = 0

    if gt_files_exist:
        ground_truth_skeleton_image = sitk.ReadImage(ground_truth_skeleton_path)
        ground_truth_skeleton = sitk.GetArrayFromImage(ground_truth_skeleton_image)
        ground_truth_branches_image = sitk.ReadImage(ground_truth_branches_path)
        ground_truth_branches = sitk.GetArrayFromImage(ground_truth_branches_image)
        num_branches = int(ground_truth_branches.max())
    else:
        ground_truth_branches_0, ground_truth_skeleton, num_branches = (
            parse_airway_tree(ground_truth_array)
        )
        trachea_label = locate_trachea(
            tree_parsing=ground_truth_branches_0, num_branches=num_branches
        )
        ground_truth_branches, final_num_branches = refine_tree_structure(
            tree_parsing=ground_truth_branches_0,
            num_branches=num_branches,
            trachea_label=trachea_label,
            max_iterations=10,
        )
        if reference_image is None:
            if ground_truth_image is not None:
                reference_image = ground_truth_image
            elif prediction_image is not None:
                reference_image = prediction_image
            else:
                raise ValueError("reference_image cannot be None if both ground_truth_image and prediction_image are None.")

        ground_truth_skeleton_image = sitk.GetImageFromArray(ground_truth_skeleton)
        ground_truth_skeleton_image.CopyInformation(reference_image)
        sitk.WriteImage(ground_truth_skeleton_image, ground_truth_skeleton_path)
        ground_truth_branches_image = sitk.GetImageFromArray(ground_truth_branches)
        ground_truth_branches_image.CopyInformation(reference_image)
        sitk.WriteImage(ground_truth_branches_image, ground_truth_branches_path)
        ground_truth_branches_image_0 = sitk.GetImageFromArray(
            ground_truth_branches_0
        )
        ground_truth_branches_image_0.CopyInformation(reference_image)
        sitk.WriteImage(
            ground_truth_branches_image_0,
            os.path.join(save_dir, f"{series_uid}.gt_branches_0.nii.gz"),
        )

    # ---------------------- 计算分支层面指标 ----------------------
    if ground_truth_skeleton is not None and prediction_array is not None:
        # 计算检测长度比 (DLR)
        detected_length = np.logical_and(prediction_array, ground_truth_skeleton).sum()
        total_length_gt = ground_truth_skeleton.sum()
        tree_length_detection_rate = detected_length / (
            total_length_gt + EPSILON
        )
        result["DLR"] = tree_length_detection_rate

        # 计算检测分支比 (BDR)
        ground_truth_branch_skeleton = ground_truth_skeleton * ground_truth_branches
        predicted_branch_skeleton = ground_truth_branch_skeleton * prediction_array

        predicted_labels, predicted_counts = np.unique(
            predicted_branch_skeleton[predicted_branch_skeleton > 0], return_counts=True
        )
        predicted_length_df = pd.DataFrame(
            {"branch_id": predicted_labels, "predicted_length": predicted_counts}
        )

        ground_truth_labels, ground_truth_counts = np.unique(
            ground_truth_branch_skeleton[ground_truth_branch_skeleton > 0],
            return_counts=True,
        )
        ground_truth_length_df = pd.DataFrame(
            {
                "branch_id": ground_truth_labels,
                "ground_truth_length": ground_truth_counts,
            }
        )

        merged_df = pd.merge(
            predicted_length_df, ground_truth_length_df, on="branch_id", how="outer"
        )
        merged_df["predicted_length"] = merged_df["predicted_length"].fillna(0)
        merged_df.loc[:, "overlap_ratio"] = (
            merged_df["predicted_length"] / merged_df["ground_truth_length"]
        )
        branch_detection_rate = (
            merged_df["overlap_ratio"] > 0.8
        ).sum() / merged_df.shape[0]
        result["BDR"] = branch_detection_rate

    return series_uid, result
