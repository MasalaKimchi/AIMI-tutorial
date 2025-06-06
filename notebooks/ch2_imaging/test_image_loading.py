import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import nibabel as nib
import SimpleITK as sitk

# Set up plotting style
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.color'] = '#cccccc'

def visualize_array_orientation(img_array, title, axis_labels=None):
    """
    Visualize a 2D image array with its axis orientations.
    """
    if img_array is None:
        print("Error: No image data provided")
        return
        
    if len(img_array.shape) > 2:
        print(f"Warning: Input array has {len(img_array.shape)} dimensions. Using first 2D slice.")
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        else:
            img_array = img_array[:, :, 0, 0]
            
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.set_facecolor('white')
    
    # Normalize image data for better visualization
    img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    # Plot the image with improved contrast
    im = ax.imshow(img_normalized, cmap='gray')
    
    # Add a more informative colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Normalized Intensity', rotation=270, labelpad=15)
    
    # Add grid with better visibility
    ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
    
    # Add arrows with improved visibility
    arrow_props = dict(arrowstyle='->', color='red', lw=2.5,
                     mutation_scale=15, shrinkA=0, shrinkB=0)
    
    # Add directional arrows
    ax.annotate('', xy=(img_array.shape[1]-1, img_array.shape[0]/2),
                xytext=(0, img_array.shape[0]/2), arrowprops=arrow_props)
    ax.annotate('', xy=(img_array.shape[1]/2, img_array.shape[0]-1),
                xytext=(img_array.shape[1]/2, 0), arrowprops=arrow_props)
    
    # Add axis labels with better positioning and visibility
    if axis_labels:
        ax.text(img_array.shape[1]/2, -img_array.shape[0]*0.05, 
               axis_labels[0], ha='center', va='center', 
               color='red', fontsize=12, fontweight='bold')
        ax.text(-img_array.shape[1]*0.05, img_array.shape[0]/2, 
               axis_labels[1], va='center', ha='center',
               rotation=90, color='red', fontsize=12, fontweight='bold')
    
    # Add title with better styling
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    # Improve layout
    plt.tight_layout()
    plt.show()

def load_and_compare_image(dicom_path, nifti_path=None, slice_idx=None):
    """
    Load and compare medical images using different libraries.
    
    Parameters:
    -----------
    dicom_path : str or Path
        Path to the DICOM file
    nifti_path : str or Path, optional
        Path to the NIfTI file (if None, only DICOM will be loaded)
    slice_idx : int, optional
        Specific slice index to visualize for 3D data (if None, middle slice is used)
    """
    results = {}
    
    # Load with PyDICOM
    try:
        dcm = pydicom.dcmread(str(dicom_path))
        dcm_array = dcm.pixel_array
        results['pydicom'] = {
            'array': dcm_array,
            'metadata': {
                'shape': dcm_array.shape,
                'dtype': str(dcm_array.dtype),
                'PatientPosition': getattr(dcm, 'PatientPosition', 'N/A'),
                'ImageOrientation': getattr(dcm, 'ImageOrientationPatient', 'N/A'),
                'PixelSpacing': getattr(dcm, 'PixelSpacing', 'N/A'),
                'SliceThickness': getattr(dcm, 'SliceThickness', 'N/A')
            }
        }
        print("\nPyDICOM loading:")
        for key, value in results['pydicom']['metadata'].items():
            print(f"{key}: {value}")
        print("-" * 50)
        
        # Visualize PyDICOM array
        visualize_array_orientation(dcm_array, "PyDICOM Array Orientation", 
                                 ("Column (Width)", "Row (Height)"))
    except Exception as e:
        print(f"Error loading DICOM: {e}")
    
    if nifti_path is not None:
        # Load with NiBabel
        try:
            nii = nib.load(str(nifti_path))
            nii_array = nii.get_fdata()
            results['nibabel'] = {
                'array': nii_array,
                'metadata': {
                    'shape': nii_array.shape,
                    'dtype': str(nii_array.dtype),
                    'affine': nii.affine,
                    'header': dict(nii.header)
                }
            }
            print("\nNiBabel loading:")
            print(f"Shape: {nii_array.shape}")
            print(f"Data type: {nii_array.dtype}")
            print(f"Affine matrix:\n{nii.affine}")
            print("-" * 50)
            
            # Get appropriate slice
            if len(nii_array.shape) == 3:
                if slice_idx is None:
                    slice_idx = nii_array.shape[2]//2
                middle_slice = nii_array[:, :, slice_idx]
            else:
                middle_slice = nii_array
            visualize_array_orientation(middle_slice, f"NiBabel Array Orientation (Slice {slice_idx})",
                                     ("R -> L", "P -> A"))
        except Exception as e:
            print(f"Error loading NIfTI: {e}")
        
        # Load with SimpleITK
        try:
            sitk_img = sitk.ReadImage(str(nifti_path))
            sitk_array = sitk.GetArrayFromImage(sitk_img)
            results['sitk'] = {
                'array': sitk_array,
                'metadata': {
                    'shape': sitk_array.shape,
                    'dtype': str(sitk_array.dtype),
                    'origin': sitk_img.GetOrigin(),
                    'spacing': sitk_img.GetSpacing(),
                    'direction': sitk_img.GetDirection()
                }
            }
            print("\nSimpleITK loading:")
            for key, value in results['sitk']['metadata'].items():
                print(f"{key}: {value}")
            print("-" * 50)
            
            # Get appropriate slice
            if len(sitk_array.shape) == 3:
                if slice_idx is None:
                    slice_idx = sitk_array.shape[0]//2
                middle_slice = sitk_array[slice_idx, :, :]
            else:
                middle_slice = sitk_array
            visualize_array_orientation(middle_slice, f"SimpleITK Array Orientation (Slice {slice_idx})",
                                     ("Column", "Row"))
        except Exception as e:
            print(f"Error loading with SimpleITK: {e}")
    
    return results

if __name__ == "__main__":
    # Set up paths
    notebook_dir = Path(__file__).parent
    data_dir = notebook_dir.parent / 'data'
    dicom_dir = data_dir / 'example-dicom-structural/dicoms'
    nifti_file = data_dir / 'example-dicom-structural/structural.nii.gz'

    print(f"DICOM directory exists: {dicom_dir.exists()}")
    print(f"NIfTI file exists: {nifti_file.exists()}")

    if dicom_dir.exists() and nifti_file.exists():
        # Get first DICOM file
        dicom_files = sorted(list(dicom_dir.glob('*.dcm')))
        if dicom_files:
            dicom_file = dicom_files[0]
            print(f"\nUsing DICOM file: {dicom_file.name}")
            
            # Load and compare
            results = load_and_compare_image(dicom_file, nifti_file)
            
            # Print additional comparison information
            if 'pydicom' in results and 'sitk' in results:
                print("\nShape comparison:")
                print(f"PyDICOM shape: {results['pydicom']['metadata']['shape']}")
                print(f"SimpleITK shape: {results['sitk']['metadata']['shape']}")
                if 'nibabel' in results:
                    print(f"NiBabel shape: {results['nibabel']['metadata']['shape']}")
        else:
            print("No DICOM files found in the directory.") 