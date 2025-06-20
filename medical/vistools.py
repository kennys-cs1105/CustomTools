# 可视化工具

import colorsys
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

import CTtools
from valtools import get_array


def show_img(img):
    if img.ndim == 3:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img)
    plt.show()


def gen_contrast_colors(n, order="BGR"):
    """# Example: Generate 5 BGR colors with high contrast
    print(gen_contrast_colors(5))

    # Example: Generate 5 RGB colors with high contrast
    print(gen_contrast_colors(5, 'RGB'))

    Args:
        n (_type_): _description_
        order (str, optional): _description_. Defaults to 'BGR'.
    """

    def format_clr(r, g, b, ord):
        # Format the color in the specified order
        clr = (r, g, b)
        if ord == "BGR":
            return (clr[2], clr[1], clr[0])
        elif ord == "RGB":
            return clr
        else:
            raise ValueError("Invalid order. Use 'BGR' or 'RGB'.")

    colors = []
    for i in range(n):
        # Generate a color in HSV space, then convert it to RGB
        h = i / n
        s = 0.7 + random.random() / 3  # Random saturation between 0.7 and 1
        v = 0.7 + random.random() / 3  # Random brightness between 0.7 and 1
        r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
        r = min(abs(r), 255)
        g = min(abs(g), 255)
        b = min(abs(b), 255)
        colors.append(format_clr(r, g, b, order))
    return colors


def display_ct_with_lesion_contours(
    ct_source=None,
    lesion_source=None,
    apply_limits=False,
    class_num=None,
    preset="lung",
    window_center=None,
    window_width=None,
    save=False,
    pdf=False,
    serial_number=None,
    save_folder="output",
):
    """

    参数:
    ct_source: 可以是CT图像的文件路径、SimpleITK.Image对象或NumPy数组。
    lesion_source: 可以是结节掩码的文件路径、SimpleITK.Image对象或NumPy数组。


    返回:

    """
    if save and serial_number is None:
        raise ValueError("'serial_number' cannot be None when 'save' is True.")

    ct_array = get_array(ct_source) if ct_source is not None else None
    ct_array = CTtools.apply_windowing(
        ct_array,
        window_center=window_center,
        window_width=window_width,
        apply_limits=apply_limits,
        preset=preset,
    )
    lesion_array = get_array(lesion_source) if lesion_source is not None else None
    # Check if the dimensions match
    if ct_array.shape != lesion_array.shape:
        raise ValueError("CT and lesion images must have the same dimensions.")

    if class_num is None:
        class_num = np.unique(lesion_array[lesion_array != 0]).shape[
            0
        ]  # 除去背景0。直接移除0，剩下稀疏的标注，运行效率更高。
    # Generate colors for each class
    colors = gen_contrast_colors(class_num)

    # Create a legend for the colors
    plt.figure(figsize=(2, class_num))
    plt.title("Legend")
    for i, color in enumerate(colors):
        plt.barh(i, 1, color=np.array(color) / 255.0)
        plt.text(0.5, i, f"Class {i+1}", ha="center", va="center", color="white")
    plt.yticks([])
    if not save:
        plt.show()

    # Create a PDF object if needed
    if save and pdf:
        import matplotlib.backends.backend_pdf

        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"{save_folder}/{serial_number}.pdf"
        )

    if save:
        os.makedirs(save_folder, exist_ok=True)
    # Process each slice
    for slice_index in range(lesion_array.shape[0]):
        ct_slice = ct_array[slice_index].copy()
        lesion_slice = lesion_array[slice_index].copy()

        # Skip slices without lesions
        if not np.any(lesion_slice):
            continue

        # Draw contours for each class
        ct_slice_rgb = cv2.cvtColor(ct_slice, cv2.COLOR_GRAY2RGB)
        for class_id in range(1, class_num + 1):
            mask = lesion_slice == class_id
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(ct_slice_rgb, contours, -1, colors[class_id - 1], 2)

        # Display the original and contoured slices
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(ct_slice, cmap="gray")
        ax[0].set_title(f"Original CT Slice - Slice {slice_index}")
        ax[1].imshow(ct_slice_rgb)
        ax[1].set_title(f"CT Slice with Lesion Contours - Slice {slice_index}")
        if not save:
            plt.show()

        # Save the figure if needed
        if save:
            if pdf:
                pdf.savefig(fig)
            else:
                plt.savefig(f"{save_folder}/{serial_number}.{slice_index}.png")
        plt.close(fig)  # Close the figure

    # Close the PDF object if needed
    if save and pdf:
        pdf.close()
