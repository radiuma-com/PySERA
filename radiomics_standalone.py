#!/usr/bin/env python3
"""
Radiomics Standalone Processing Pipeline (Legacy Interface)

This script maintains backward compatibility with the original radiomics_standalone.py
while now using the pysera library under the hood.

For new projects, consider using the pysera library directly:
    import pysera
    processor = pysera.RadiomicsProcessor()
    result = processor.process_batch(image_input, mask_input)

Supported Input Formats:
-----------------------
Image Files:
  - NIfTI (.nii, .nii.gz)
  - NRRD (.nrrd, .nhdr)
  - DICOM (.dcm, .dicom)
  - NumPy arrays (.npy)
  - Multi-dcm: Directory with subfolders (patients), each containing DICOM files
  - Any other format supported by SimpleITK
  - Type: Medical images (CT, MRI, PET, etc.)
  - Bit depth: Any supported by SimpleITK

Mask Files:
  - Same formats as image files (NIfTI, NRRD, DICOM, NumPy)
  - Type: Binary or labeled segmentation masks
  - Requirement: Must have same dimensions as corresponding image
  - For multi-dcm, mask may be a single DICOM file

Usage Examples:
---------------
# Single NIfTI image and mask
python radiomics_standalone.py --image_input path/to/image.nii.gz --mask_input path/to/mask.nii.gz --output results/

# DICOM series (folder) and mask
python radiomics_standalone.py --image_input path/to/dicom_folder --mask_input path/to/mask.dcm --output results/

# Multi-dcm (batch of patients)
python radiomics_standalone.py --image_input path/to/patients_dir --mask_input path/to/masks_dir --output results/

# NRRD image and mask
python radiomics_standalone.py --image_input path/to/image.nrrd --mask_input path/to/mask.nrrd --output results/

# NumPy array image and mask (file-based)
python radiomics_standalone.py --image_input path/to/image.npy --mask_input path/to/mask.npy --output results/


# Any format supported by SimpleITK
python radiomics_standalone.py --image_input path/to/image.ext --mask_input path/to/mask.ext --output results/
"""

import sys
import os

# Add the pysera package to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # Use the new pysera library CLI
    from pysera._cli import main
except ImportError as e:
    print(f"Error importing pysera library: {e}")
    print("Make sure you have installed the pysera library:")
    print("pip install -e .")
    sys.exit(1)


if __name__ == "__main__":
    # Delegate to the new library's CLI
    exit_code = main()
    sys.exit(exit_code) 