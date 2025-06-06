import SimpleITK as sitk
from pathlib import Path

def convert_dicom_to_nifti(dicom_dir, output_file):
    """
    Convert a DICOM series to NIfTI format using SimpleITK.
    
    Parameters:
    -----------
    dicom_dir : str or Path
        Directory containing the DICOM series
    output_file : str or Path
        Output NIfTI file path (.nii.gz)
    """
    print(f"Reading DICOM series from {dicom_dir}")
    
    # Create image series reader
    reader = sitk.ImageSeriesReader()
    
    # Get the DICOM filenames
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    
    print(f"Found {len(dicom_names)} DICOM files")
    
    # Set the filenames and read
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Write as NIfTI
    print(f"Writing NIfTI file to {output_file}")
    sitk.WriteImage(image, str(output_file))
    print("Conversion complete!")

if __name__ == "__main__":
    # Set up paths
    notebook_dir = Path(__file__).parent.parent
    dicom_dir = notebook_dir / "data/example-dicom-structural/dicoms"
    output_file = notebook_dir / "data/example-dicom-structural/structural.nii.gz"
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert
    convert_dicom_to_nifti(dicom_dir, output_file) 