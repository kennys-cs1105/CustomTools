import pathlib
import SimpleITK as sitk


def dicom_to_nifti(dicom_dir, nifti_path=None):
    # 使用pathlib获取路径
    dicom_dir = pathlib.Path(dicom_dir)

    # 如果nifti_path为None，使用dicom路径的basename作为名字，加上.nii.gz的后缀
    if nifti_path is None:
        nifti_path = dicom_dir.parent / (dicom_dir.name + ".nii.gz")
    else:
        nifti_path = pathlib.Path(nifti_path)

    # 创建nii.gz的存储目录
    nifti_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果nii.gz已经存在则提示并跳过执行
    if nifti_path.exists():
        print(f"File {nifti_path} already exists. Skipping conversion.")
        return None

    # 读取DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    reader.SetFileNames(dicom_names)

    # 读取DICOM数据，包括元数据
    image = reader.Execute()

    # 将DICOM转换为NIfTI格式（.nii.gz）
    sitk.WriteImage(image, str(nifti_path), True)  # True表示使用压缩

    print(f"Converted {dicom_dir} to {nifti_path}")
