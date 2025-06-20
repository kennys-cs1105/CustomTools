from typing import Dict, Callable, List, Union, Optional, Any

import os
import SimpleITK as sitk
import logging
import concurrent.futures
import traceback
import time
import numpy as np
from nibabel.nifti1 import Nifti1Image
import nibabel as nib
from skimage import filters

from utils.iotools import load_mask, get_sitk_data, get_nib_data
from libs.totalsegmentator.python_api import totalsegmentator


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def segment(
        input_nifti_path: Union[str, Nifti1Image], 
        task: str, 
        output_path: str, 
        **kwargs
):
    """
    Using totalsegmentator method api

    Args:

    """
    assert isinstance(input_nifti_path, (str, Nifti1Image)), "Input nifti path should be provided correctly."

    if not os.path.exists(output_path):
        logging.info(f"Starting {task} prediction for {input_nifti_path}")
        totalsegmentator(input=input_nifti_path, output=output_path, task=task, **kwargs)
        logging.info(f"{task} prediction finished and saved to {output_path}")


def run_segmentation_task(task_name: str, task_callable: Callable[[], None], attempt: int = 1, retry_attempts: int = 2):
    """
    Run a single task with retries

    Args:
        task_name (str): Task names.
        task_callable (Callabel): Callable functions corresponding to task names.
        attempt (int): Number of times to retry.
        retry_attempts (int): Number of times to retry a failed task before giving up.
    """
    try:
        logging.info(f"[{task_name}] Attempt {attempt} started.")
        start_time = time.time()
        task_callable()
        elapsed_time = time.time() - start_time
        logging.info(f"[{task_name}] Completed successfully in {elapsed_time:.2f}s.")
    except Exception as e:
        logging.error(f"[{task_name}] Attempt {attempt} failed with error: {e}\n{traceback.format_exc()}")
        if attempt < retry_attempts:
            logging.info(f"[{task_name}] Retrying attempt {attempt + 1}/{retry_attempts}...")
            run_segmentation_task(task_name, task_callable, attempt + 1)
        else:
            logging.critical(f"[{task_name}] Failed after {retry_attempts} attempts. Skipping.")
    

def execute_segmentation_tasks(task_group: Dict[str, Callable[[], None]], max_worker=None, timeout=None, retry_attempts=3):
        """
        Execute a group of tasks in parallel and log their status
        
        Args:
            task_groups (Dict[str, Callable[[], None]]): A dict of task name and corresponding callable tasks
            max_worker (int): Maximum number of parallel workers. Defaults to CPU count
            timeout (int): Maximum time (in seconds) to wait for each task. Defaults to None.
            retry_attempts (int): Number of retry attempts for failed tasks. Defaults to 3
        """
        if max_worker is None:
            max_worker = min(4, os.cpu_count() or 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker) as executor:
            future_to_task: Dict[concurrent.futures.Future, str] = {
                executor.submit(run_segmentation_task, name, task): name for name, task in task_group.items()
            }
            for future in concurrent.futures.as_completed(future_to_task, timeout=timeout):
                task_name = future_to_task[future]
                try:
                    future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logging.error(f"[{task_name}] Timed out after {timeout} seconds.")
                except Exception as exc:
                    logging.error(f"[{task_name}] Encountered an unexpected error: {exc}")


def process_region_ct_file(
        region_ct_file: str,
        fine_chest_ct_dir: str,
        fine_lobe_prediction_dir: str,
        lobe_segmentation_function: Callable
):
    """
    Processing region ct files of chest roi file for fine lobe segmentation
    """
    region_ct_path = os.path.join(fine_chest_ct_dir, region_ct_file)
    basename = region_ct_file.split("_0000.nii.gz")[0]

    output_path = os.path.join(fine_lobe_prediction_dir, f"{basename}.nii.gz")
    lobe_segmentation_function(task="lobe", input_path=region_ct_path, lobe_output_path=output_path, ml=True, fast=True)

    logging.info(f"Processed: {region_ct_file}")


def infer_fine_lobe_parallel(
        fine_chest_ct_dir: str,
        fine_lobe_prediction_dir: str,
        lobe_segmentation_function: Callable,
        max_worker: int,
):
    """
    Perform fine lobe segmentation in parallel
    """
    ct_files = [f for f in os.listdir(fine_chest_ct_dir) if f.endswith(".nii.gz")]

    os.makedirs(fine_lobe_prediction_dir, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = []
        for region_ct_file in ct_files:
            futures.append(
                executor.submit(process_region_ct_file, region_ct_file, fine_chest_ct_dir, fine_lobe_prediction_dir, lobe_segmentation_function)
            )
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Exception occurred: {exc}")


def merge_masks(
    organ_mask_path: str,
    rib_mask_path: str,
    body_mask_path: str,
    fine_lobe_mask_path: Optional[str],
    lobe_mask_path: Optional[str],
    bedsplit_mask_path: str,
    input_nifti_path: str,
    organ_mapping: Dict[int, int],
    skin_mapping: Dict[int, int],
    rib_mapping: Dict[int, int],
    lobe_mapping: Dict[int, int],
    body_mapping: Dict[int, int],
    bedsplit_mapping: Dict[int, int],
    fine_seg: bool = False,
    bedsplit_seg: bool = False
) -> sitk.Image:
    """
    Merge multiple masks into a single mask using the provided mappings.

    Args:
        organ_mask_path: Path to the organ mask file.
        rib_mask_path: Path to the rib mask file.
        body_mask_path: Path to the body mask file.
        fine_lobe_mask_path: Path to the fine lobe mask file (if fine_seg is True).
        lobe_mask_path: Path to the lobe mask file (if fine_seg is False).
        bedsplit_mask_path: Path to the bedsplit mask file.
        input_nifti_path: Path to the input NIfTI file for metadata.
        organ_mapping: Mapping of organ mask values to new values.
        skin_mapping: Mapping of skin mask values to new values.
        rib_mapping: Mapping of rib mask values to new values.
        lobe_mapping: Mapping of lobe mask values to new values.
        body_mapping: Mapping of body mask values to new values.
        bedsplit_mapping: Mapping of bedsplit mask values to new values.
        fine_seg: Whether to use fine lobe segmentation (default: False).

    Returns:
        sitk.Image: The merged mask as a SimpleITK image.
    """
    # Load organ mask
    organ_mask_array = load_mask(organ_mask_path, "organ")

    # Load rib mask
    rib_mask_array = load_mask(rib_mask_path, "rib")

    # Load body mask
    body_mask_array = load_mask(os.path.join(body_mask_path, "body.nii.gz"), "body")

    # Generate skin mask using Sobel edge detection
    logging.info("Generating skin mask from body mask...")
    sobel_edge = filters.sobel(body_mask_array)
    skin_mask_array = (sobel_edge > 0).astype(np.uint8)
    if skin_mask_array.size == 0:
        logging.warning("Skin mask array is empty. Initializing as zero array.")
        skin_mask_array = np.zeros_like(body_mask_array, dtype=np.uint8)

    # Load lobe mask (fine or coarse)
    if fine_seg:
        if os.path.exists(fine_lobe_mask_path):
            fine_lobe_mask_array = load_mask(fine_lobe_mask_path, "lobe")
    else:
        if os.path.exists(lobe_mask_path):
            lobe_mask_array = load_mask(lobe_mask_path, "lobe")

    # Initialize combined mask array
    combined_mask_array = np.zeros_like(organ_mask_array, dtype=np.uint8)

    # Apply mappings to merge masks
    logging.info("Applying organ mapping...")
    for origin_value, new_value in organ_mapping.items():
        combined_mask_array[organ_mask_array == origin_value] = new_value

    logging.info("Applying skin mapping...")
    for origin_value, new_value in skin_mapping.items():
        combined_mask_array[skin_mask_array == origin_value] = new_value

    logging.info("Applying rib mapping...")
    rib_mask_array = np.isin(rib_mask_array, list(rib_mapping.keys()))
    combined_mask_array[rib_mask_array] = list(rib_mapping.values())[0]

    # Check whether use fine seg
    if fine_seg:
        logging.info("Applying fine lobe mapping...")
        for origin_value, new_value in lobe_mapping.items():
            combined_mask_array[fine_lobe_mask_array == origin_value] = new_value

    logging.info("Applying body mapping...")
    for origin_value, new_value in body_mapping.items():
        body_mask_array[body_mask_array == origin_value] = new_value

    if bedsplit_seg:
        bedsplit_mask_array = load_mask(bedsplit_mask_path, "bedsplit")
        logging.info("Applying bedsplit mapping...")
        for origin_value, new_value in bedsplit_mapping.items():
            combined_mask_array[bedsplit_mask_array == origin_value] = new_value

    # Merge combined mask with body mask
    logging.info("Merging combined mask with body mask...")
    merged_mask_array = np.where(combined_mask_array > 0, combined_mask_array, body_mask_array)

    # Convert merged mask to SimpleITK image
    logging.info("Converting merged mask to SimpleITK image...")
    merged_mask_image = sitk.GetImageFromArray(merged_mask_array)
    merged_mask_image.CopyInformation(sitk.ReadImage(input_nifti_path))

    return merged_mask_image


def generate_task_lambda(task_name, input_nifti_path, output_path, ml=False, fast=False):
    return lambda: (
        segment(task=task_name, input_nifti_path=input_nifti_path, output_path=output_path, ml=ml, fast=fast)
        if not os.path.exists(output_path)
        else logging.info(f"{task_name.capitalize()} mask already exists. Skipping.")
    )


def assign_tasks_to_loops(task_list, paths_map, input_nifti_path):
    loops = {}
    loop_index = 1
    current_group = {}

    for i, task in enumerate(task_list):
        task_name = task["name"]
        ml = task.get("ml", False)
        fast = task.get("fast", False)

        if task_name not in paths_map:
            logging.warning(f"Unknown task '{task_name}', skipping.")
            continue

        output_path = paths_map[task_name]
        current_group[f"{task_name}_segmentation"] = generate_task_lambda(
            task_name, input_nifti_path, output_path, ml, fast
        )

        # Flush group if full or last element
        if len(current_group) == 2 or i == len(task_list) - 1:
            loops[f"tasks_loop_{loop_index}"] = current_group
            loop_index += 1
            current_group = {}

    return loops


def get_target_region(ct_file: Union[str, Nifti1Image], 
                      mask_file: Union[str, Nifti1Image], 
                      mapping: Dict, 
                      output_dir:str):
    """
    功能: 基于掩码将人体ct扫描分为肺区域和非肺区域

    Args:
        ct_path: CT 扫描路径 (nii.gz 文件)
        mask_path: 目标掩码路径, 可以是单标签掩码文件(id=0,1) 也可以是多标签掩码文件
        mapping: 器官名称对应掩码id的Dict, 例如{"lung": [1, 2]}, id可以是int 也可以是list
        output_dir: 输出保存路径，将保存器官区域和非器官区域
    """

    ct_nii = get_nib_data(ct_file)
    mask_nii = get_nib_data(mask_file)
    
    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()

    if ct_data.shape != mask_data.shape:
        raise ValueError("CT 图像和掩码图像的尺寸不一致。")
    
    os.makedirs(output_dir, exist_ok=True)

    for organ_name, ids in mapping.items():
        if isinstance(ids, int):
            ids = [ids]

        organ_mask = np.isin(mask_data, ids)

        organ_region = ct_data * organ_mask
        organ_region_nii = nib.Nifti1Image(organ_region, affine=ct_nii.affine, header=ct_nii.header)
        organ_path = os.path.join(output_dir, f"{organ_name}.nii.gz")
        nib.save(organ_region_nii, organ_path)
        logging.info(f"{organ_name} region saved...")