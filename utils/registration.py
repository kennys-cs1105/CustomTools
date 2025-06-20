import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def resample_image(image, target_size, target_spacing):
    """
    重采样图像以匹配目标尺寸和spacing。

    参数:
    image (sitk.Image): 需要重采样的图像
    target_size (list): 目标尺寸
    target_spacing (list): 目标spacing

    返回:
    sitk.Image: 重采样后的图像
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def calculate_target_size_and_spacing(fixed_image, moving_image):
    """
    计算目标尺寸和spacing。

    参数:
    fixed_image (sitk.Image): 固定图像
    moving_image (sitk.Image): 移动图像

    返回:
    tuple: 目标尺寸和spacing
    """
    fixed_size, moving_size = fixed_image.GetSize(), moving_image.GetSize()
    fixed_spacing, moving_spacing = fixed_image.GetSpacing(), moving_image.GetSpacing()

    target_physical_size = [
        max(fixed_size[i] * fixed_spacing[i], moving_size[i] * moving_spacing[i])
        for i in range(3)
    ]
    target_spacing = [min(fixed_spacing[i], moving_spacing[i]) for i in range(3)]
    target_size = [int(target_physical_size[i] / target_spacing[i]) for i in range(3)]

    return target_size, target_spacing


def affine_registration(
    fixed_image_path,
    moving_image_path,
    transform_type="affine",
    shrink_factors=[16, 8],
    smoothing_sigmas=[2, 1],
    num_threads=1,
):
    """
    读取两个NIfTI文件，并执行仿射或刚性配准，使用多分辨率调度。

    参数:
    fixed_image_path (str): 固定图像的路径
    moving_image_path (str): 移动图像的路径
    transform_type (str): 选择变换类型，'affine' 或 'rigid'
    shrink_factors (list): 缩小因子列表，默认值为 [16, 8]
    smoothing_sigmas (list): 平滑参数列表，默认值为 [2, 1]
    num_threads (int): 使用的线程数，默认值为 1

    返回:
    sitk.Image: 变换后的移动图像
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    target_size, target_spacing = calculate_target_size_and_spacing(
        fixed_image, moving_image
    )

    if (
        fixed_image.GetSize() != moving_image.GetSize()
        or fixed_image.GetSpacing() != moving_image.GetSpacing()
    ):
        fixed_image = resample_image(fixed_image, target_size, target_spacing)
        moving_image = resample_image(moving_image, target_size, target_spacing)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        (
            sitk.AffineTransform(fixed_image.GetDimension())
            if transform_type == "affine"
            else sitk.Euler3DTransform()
        ),
    )

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetShrinkFactorsPerLevel(shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    final_transform = registration_method.Execute(fixed_image, moving_image)

    transformed_moving_image = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID(),
    )

    print("Final metric value: {0}".format(registration_method.GetMetricValue()))
    print(
        "Optimizer's stopping condition, {0}".format(
            registration_method.GetOptimizerStopConditionDescription()
        )
    )

    return fixed_image, transformed_moving_image


def visualize_images(
    fixed_image,
    transformed_image,
    window_width=1500,
    window_level=-600,
    slice_range=None,
):
    """
    可视化固定图像和变换后的图像。

    参数:
    fixed_image (sitk.Image): 固定图像
    transformed_image (sitk.Image): 变换后的图像
    window_width (int): 窗宽，默认值为1500
    window_level (int): 窗位，默认值为-600
    slice_range (tuple): 查看范围（起始-结束），如果不提供则查看全部
    """
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    transformed_array = sitk.GetArrayFromImage(transformed_image)

    def apply_window(image_array, window_width, window_level):
        min_val = window_level - (window_width / 2)
        max_val = window_level + (window_width / 2)
        windowed_image = np.clip(image_array, min_val, max_val)
        return (windowed_image - min_val) / (max_val - min_val)

    fixed_windowed = apply_window(fixed_array, window_width, window_level)
    transformed_windowed = apply_window(transformed_array, window_width, window_level)

    num_slices = fixed_windowed.shape[0]
    start_slice, end_slice = (0, num_slices) if slice_range is None else slice_range

    for i in range(start_slice, end_slice):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(fixed_windowed[i, :, :], cmap="gray")
        axes[0].set_title("Fixed Image - Slice {}".format(i))

        axes[1].imshow(transformed_windowed[i, :, :], cmap="gray")
        axes[1].set_title("Resampled Image - Slice {}".format(i))

        plt.show()


# 示例调用
if __name__ == "__main__":
    fixed_image_path = r"D:\DataSets\DICOM\R009\1.2.392.200036.9116.2.5.1.37.2418725123.1722420178.318187_0000.nii.gz"
    moving_image_path = r"D:\DataSets\DICOM\R009\1.2.392.200036.9116.2.5.1.37.2418725123.1722420770.171286_0000.nii.gz"

    # 示例调用
    fixed_image, transformed_image = affine_registration(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        transform_type="affine",  # 或 'rigid'
        shrink_factors=[8, 4, 2, 1],
        smoothing_sigmas=[4, 2, 1, 0],
        num_threads=8
    )

    visualize_images(fixed_image, transformed_image, slice_range=(135, 142))
