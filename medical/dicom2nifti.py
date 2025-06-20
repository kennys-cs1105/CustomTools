import argparse
import imgconverters


def main():
    parser = argparse.ArgumentParser(description="将DICOM文件转换为NIfTI文件。")
    parser.add_argument("dicom_dir", help="DICOM文件所在的目录。")
    parser.add_argument(
        "--nifti_path", help="输出NIfTI文件的路径 (如果指定)。", default=None
    )

    args = parser.parse_args()

    dicom_directory = args.dicom_dir
    nifti_output_path = args.nifti_path

    imgconverters.dicom_to_nifti(dicom_directory, nifti_path=nifti_output_path)
    print(f"成功将DICOM目录 '{dicom_directory}' 转换为 NIfTI 文件。")
    if nifti_output_path:
        print(f"NIfTI 文件保存在: {nifti_output_path}")
    else:
        print("NIfTI 文件保存在 imgconverters 模块决定的默认位置。")


if __name__ == "__main__":
    main()
