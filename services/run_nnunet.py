import logging
import os
import subprocess


def set_environment_variables(
    nnunet_raw, nnunet_preprocessed, nnunet_results, cuda_device
):
    """
    Set environment variables for nnUNet.

    Parameters:
    nnunet_raw (str): Path to the raw data directory.
    nnunet_preprocessed (str): Path to the preprocessed data directory.
    nnunet_results (str): Path to the results directory.
    cuda_device (int): CUDA device number.
    """
    if nnunet_raw is not None:
        os.environ["nnUNet_raw"] = os.path.abspath(nnunet_raw)
    if nnunet_preprocessed is not None:
        os.environ["nnUNet_preprocessed"] = os.path.abspath(nnunet_preprocessed)
    if nnunet_results is not None:
        os.environ["nnUNet_results"] = os.path.abspath(nnunet_results)
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)


def run_command(command):
    """
    Run a shell command and log the output.

    Parameters:
    command (str): The command to run.

    Raises:
    subprocess.CalledProcessError: If the command returns a non-zero exit status.
    """
    try:
        logging.info(f"Running command: {command}")
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        logging.info(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr}")
        raise RuntimeError()


def run_nnunet(
    input_dir,
    output_dir,
    dataset_id="Dataset006_KeyPoints",
    config="2d",
    nnunet_raw=None,
    nnunet_preprocessed=None,
    nnunet_results="../models/",
    post_processing=False,
    postprocessing_file=None,
    plans_file=None,
    dataset_json_file=None,
    fold="0",
    cuda_device=0,
):
    """
    Run nnUNet prediction and postprocessing.

    Parameters:
    input_dir (str): Path to the input directory.
    output_dir (str): Path to the output directory.
    dataset_id (str): Dataset ID or name.
    config (str): Configuration name.
    fold (str): Fold number.
    nnunet_raw (str): Path to the raw data directory.
    nnunet_preprocessed (str): Path to the preprocessed data directory.
    nnunet_results (str): Path to the results directory.
    post_processing (bool): Whether to run postprocessing.
    postprocessing_file (str): Path to the postprocessing file.
    plans_file (str): Path to the plans file.
    dataset_json_file (str): Path to the dataset JSON file.
    cuda_device (int, optional): CUDA device number. Default is 0.

    Raises:
    ValueError: If any required parameter is missing or empty.
    """
    # 验证输入参数
    if not all(
        [
            input_dir,
            output_dir,
            dataset_id,
            config,
            fold,
            nnunet_results,
        ]
    ):
        raise ValueError("All parameters must be provided and non-empty.")

    # 设置环境变量
    set_environment_variables(
        nnunet_raw, nnunet_preprocessed, nnunet_results, cuda_device
    )

    # 构建并执行 nnUNetv2_predict 命令
    predict_command = (
        f"nnUNetv2_predict -i {os.path.abspath(input_dir)} "
        f"-o {os.path.abspath(output_dir)} "
        f"-d {dataset_id} "
        f"-c {config} "
        f"-f {fold}"
    )
    run_command(predict_command)

    if post_processing:
        # 构建并执行 nnUNetv2_apply_postprocessing 命令
        postprocess_command = (
            f"nnUNetv2_apply_postprocessing -i {os.path.abspath(output_dir)} "
            f"-o {os.path.abspath(output_dir)} "
            f"--pp_pkl_file {os.path.abspath(postprocessing_file)} "
            f"-plans_json {os.path.abspath(plans_file)} "
            f"-dataset_json {os.path.abspath(dataset_json_file)}"
        )
        run_command(postprocess_command)


def run_nnunet_python(
    input_directory="../testdata/nnunet",
    output_directory="../testdata/nnunet/keypoints",
    folds=(0,),
    cuda_device_id=0,
    num_threads_preprocessing=3,
    model_weights_directory="../models/Dataset006_KeyPoints/nnUNetTrainer__nnUNetPlans__2d",
    enable_post_processing=False,
    postprocessing_file=None,
    plans_file=None,
    dataset_json_file=None,
):
    """
    运行 nnUNet 预测管道。

    参数:
    input_directory (str): 包含原始数据的输入目录路径。
    output_directory (str): 保存预测结果的输出目录路径。
    folds (tuple): 用于预测的折叠索引元组。
    cuda_device_id (int): 要使用的 CUDA 设备 ID。
    num_threads_preprocessing (int): 用于预处理和分割导出的线程数。
    model_weights_directory (str): 包含模型权重的目录路径。
    enable_post_processing (bool): 是否启用后处理。
    postprocessing_file (str, optional): 后处理文件的路径。
    plans_file (str, optional): 计划文件的路径。
    dataset_json_file (str, optional): 数据集 JSON 文件的路径。

    返回:
    None
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # 设置环境变量，主要是为了消除nnU-net的警告，没有实际作用。
    set_environment_variables(
        nnunet_raw=input_directory,
        nnunet_preprocessed=input_directory,
        nnunet_results=os.path.dirname(os.path.dirname(model_weights_directory)),
        cuda_device=None,
    )

    # 实例化 nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", cuda_device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # 初始化网络架构，加载检查点
    predictor.initialize_from_trained_model_folder(
        model_weights_directory,
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
    )
    # 预测
    predictor.predict_from_files(
        input_directory,
        output_directory,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=num_threads_preprocessing,
        num_processes_segmentation_export=num_threads_preprocessing,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    if enable_post_processing:
        # 构建并执行 nnUNetv2_apply_postprocessing 命令
        postprocess_command = (
            f"nnUNetv2_apply_postprocessing -i {os.path.abspath(output_directory)} "
            f"-o {os.path.abspath(output_directory)} "
            f"--pp_pkl_file {os.path.abspath(postprocessing_file)} "
            f"-plans_json {os.path.abspath(plans_file)} "
            f"-dataset_json {os.path.abspath(dataset_json_file)}"
        )
        run_command(postprocess_command)


def get_props_dict(input_nifti):
    """
    获取NIFTI文件的图像与参数字典

    参数：
    input_nifti (str): 输入 NIfTI 文件路径

    返回：
    tuple: 包含输入图像数组、图像属性
    """
    
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    img, props = SimpleITKIO().read_images([input_nifti])

    return img, props


def run_nnunet_on_single_array(
    img=None,
    props=None,
    input_nifti="../testdata/nnunet/1.2.840.113619.2.55.3.2831178506.503.1719278360.298_0000.nii.gz",
    output_path="../testdata/nnunet/keypoints/1.2.840.113619.2.55.3.2831178506.503.1719278360.298.nii.gz",
    folds=(0,),
    cuda_device_id=0,
    model_weights_directory="../models/Dataset006_KeyPoints/nnUNetTrainer__nnUNetPlans__2d",
    background_save=False,
):
    """
    运行 nnUNet 预测管道，处理单个数组。

    参数:
    img (numpy.ndarray, optional): 输入图像数组，需要对三维数组扩展一个维度。如果为 None，则从 input_nifti 读取图像。 -> (1,z,y,x)
    props (dict, optional): 图像属性。如果为 None，则从 input_nifti 读取图像属性。
    input_nifti (str): 输入 NIfTI 文件路径。
    output_path (str): 保存预测结果的输出文件路径。
    folds (tuple): 用于预测的折叠索引元组。
    cuda_device_id (int): 要使用的 CUDA 设备 ID。
    model_weights_directory (str): 包含模型权重的目录路径。

    返回:
    tuple: 包含输入图像数组、图像属性和预测结果的元组。
    """
    import os

    import torch
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    if img is None or props is None:

        img, props = SimpleITKIO().read_images([input_nifti])

    # set_environment_variables(
    #     nnunet_raw="/tmp/",
    #     nnunet_preprocessed="/tmp/",
    #     nnunet_results=os.path.dirname(os.path.dirname(model_weights_directory)),
    #     cuda_device=None,
    # )

    # 实例化 nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", cuda_device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    # 初始化网络架构，加载检查点
    predictor.initialize_from_trained_model_folder(
        model_weights_directory,
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
    )

    ret = predictor.predict_single_npy_array(
        input_image=img,
        image_properties=props,
        segmentation_previous_stage=None,
        output_file_truncated=None,
        save_or_return_probabilities=False,
    )
    # 如果save_or_return_probabilities为True，则ret是一个tuple，[0]是掩膜，[1]是概率。
    # 如果save_or_return_probabilities为False，则ret只有一个numpy.array
    if output_path:
        if background_save:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                executor.submit(
                    SimpleITKIO().write_seg,
                    seg=ret,
                    output_fname=output_path,
                    properties=props,
                )
        else:
            SimpleITKIO().write_seg(seg=ret, output_fname=output_path, properties=props)
    return img, props, ret


def run_nnunet_on_list_array(
    img_list=None,
    props_list=None,
    input_nifti_list=None,
    output_path_list=None,
    folds=(0,),
    cuda_device_id=0,
    model_weights_directory="../models/Dataset006_KeyPoints/nnUNetTrainer__nnUNetPlans__2d",
):
    """
    运行 nnUNet 预测管道，处理多个数组。

    参数:
    img_list (list of numpy.ndarray, optional): 输入图像数组列表。如果为 None，则从 input_nifti_list 读取图像。
    props_list (list of dict, optional): 图像属性列表。如果为 None，则从 input_nifti_list 读取图像属性。
    input_nifti_list (list of str, optional): 输入 NIfTI 文件路径列表。
    output_path_list (list of str, optional): 保存预测结果的输出文件路径列表。
    folds (tuple): 用于预测的折叠索引元组。
    cuda_device_id (int): 要使用的 CUDA 设备 ID。
    model_weights_directory (str): 包含模型权重的目录路径。

    返回:
    list of tuple: 包含输入图像数组、图像属性和预测结果的元组列表。
    """
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    if img_list is None or props_list is None:
        img_list = []
        props_list = []
        for nifti_path in input_nifti_list:
            img, props = SimpleITKIO().read_images([nifti_path])
            img_list.append(img)
            props_list.append(props)

    if output_path_list is None:
        output_path_list = [None] * len(img_list)

    # 设置环境变量
    set_environment_variables(
        nnunet_raw="/tmp/",
        nnunet_preprocessed="/tmp/",
        nnunet_results=os.path.dirname(os.path.dirname(model_weights_directory)),
        cuda_device=None,
    )

    # 实例化 nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", cuda_device_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    # 初始化网络架构，加载检查点
    predictor.initialize_from_trained_model_folder(
        model_weights_directory,
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
    )

    # 调用 predict_from_list_of_npy_arrays 进行预测
    ret_list = predictor.predict_from_list_of_npy_arrays(
        image_or_list_of_images=img_list,
        segs_from_prev_stage_or_list_of_segs_from_prev_stage=None,
        properties_or_list_of_properties=props_list,
        truncated_ofname=output_path_list,
        num_processes=3,
        save_probabilities=False,
        num_processes_segmentation_export=None,
    )

    # 如果指定了输出路径，保存预测结果
    for output_path, ret, props in zip(output_path_list, ret_list, props_list):
        if output_path:
            SimpleITKIO().write_seg(seg=ret, output_fname=output_path, properties=props)

    return [(img, props, ret) for img, props, ret in zip(img_list, props_list, ret_list)]


# 示例调用
if __name__ == "__main__":

    # 示例调用
    run_nnunet(
        input_dir="INPUT_FOLDER",
        output_dir="OUTPUT_FOLDER",
        dataset_id="DATASET_NAME_OR_ID",
        config="CONFIGURATION",
        fold="FOLD",
        nnunet_raw="/FlashCache/niubing/nnU-net/nnUNet_raw",
        nnunet_preprocessed="/FlashCache/niubing/nnU-net/nnUNet_preprocessed",
        nnunet_results="/FlashCache/niubing/nnU-net/nnUNet_results",
        postprocessing_file="POSTPROCESSING_FILE",
        plans_file="PLANS_FILE",
        dataset_json_file="DATASET_JSON_FILE",
        cuda_device=0,
    )
