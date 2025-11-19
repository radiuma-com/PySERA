# PySERA: Python-Based Standardized Extraction for Radiomics Analysis – Python Radiomics Script and Library

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-stable-green.svg)](https://pypi.org/project/pysera/)

**PySERA** (Python-based Standardized Extraction for Radiomics Analysis), published in "[PySERA](https://pypi.org/project/pysera/)"  is a comprehensive Python library for radiomics feature extraction from medical imaging data. It provides a **simple, single-function API** with built-in multiprocessing support, comprehensive report capabilities, and optimized performance through OOP architecture, RAM optimization, and CPU-efficient parallel processing. PySERA supports both **traditional handcrafted radiomics** (557  features including 487 IBSI-compliant, 60 diagnostic, and 10 moment-invariant features) and **deep learning-based feature extraction** using pre-trained models like ResNet50, VGG16, and DenseNet121.

## 🔍 Table of Contents
- [🧩IBSI (Image Biomarker Standardisation Initiative) Standardization-1.0](#IBSI-Standardization)
- [✨Key Features](#key-features)
- [🤖Deep Learning Feature Extraction](#deep-learning-feature-extraction)
- [📚Library Usage](#library-usage)
  - [📂Single File Processing](#single-file-processing)
  - [🧠In-Memory Array Processing](#in-memory-array-processing)
  - [⚡Parallel Batch Processing](#parallel-batch-processing)
  - [🤖Deep Features Extraction](#deep-features-extraction)
  - [🔧Advanced Configuration](#advanced-configuration)
- [📥Installation](#installation)
  - [🌐GitHub Installation](#github-installation)
  - [🖥️Python Script - Command Line Interface (CLI)](#python-script---command-line-interface-cli)
  - [📦Library Installation via pip](#library-installation-via-pip)

- [📂Data Structure Requirements](#data-structure-requirements)
- [📋PySERA Parameters Reference](#pysera-parameters-reference)
- [📊Parameter Compatibility](#parameter-compatibility)
- [📚API Reference](#api-reference)
- [📊Output Structure](#output-structure)
- [🔢Feature Extraction Modes](#feature-extraction-modes)
- [📁Supported File Formats](#supported-file-formats)
- [🎯Library Examples](#library-examples)
- [⚡Performance Tips](#performance-tips)
- [💻Integration of PySERA to GUI](#integration-of-pysera-to-gui)
- [❓Troubleshooting](#troubleshooting)
- [🕒Version History](#Version-History)
- [📬Contact](#contact)
- [👥Authors](#authors)
- [🙏Acknowledgment](#acknowledgment)
- [📜License](#license)

## 🧩IBSI Standardization
Both the script and library have been rigorously standardized based on **IBSI** (Image Biomarker Standardisation Initiative) Standardization 1.0. PySERA returns IBSI-compliant feature values that match the reference standard, ensuring reproducibility and comparability across studies.
See the detailed evaluation and test cases here: [IBSI_Evaluation Folder](https://github.com/radiuma-com/PySERA/tree/main/IBSI_Evaluation)

## ✨Key Features

PySERA provides a **single-function API** that handles all radiomics processing:

```python
import pysera

result = pysera.process_batch(
    image_input="image.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results"
)
```

That's it! 🎉 All the complexity of multiprocessing, error & warning reports, file format handling, and feature extraction is handled automatically.

- **Single Function API**: One function does everything - `pysera.process_batch()`
- **Multi-format Support**: NIfTI, DICOM, NRRD, RTSTRUCT, NumPy arrays, and more
- **Automatic Multiprocessing**: Built-in parallel processing for maximum performance
- **Comprehensive Report**: Excel export functionality for detailed analysis
- **Extensive Features**: 557 total radiomics features across multiple categories (morphological, statistical, texture, etc.) and dimensions (1st, 2D, 2.5D, 3D) including:
  - **487 IBSI-compliant features** (standardized radiomics)
  - **60 diagnostic features** (image quality and metadata)
  - **10 moment-invariant features** (shape descriptors) 
- **Medical Image Optimized**: Designed for CT, MRI, PET, SPECT, X-Ray, Ultrasound, and other medical imaging modalities.
- **Dual Extraction Modes**: Both traditional IBSI-compliant radiomics (557 features) and deep learning features (ResNet50, VGG16, DenseNet121)

## 🤖Deep Learning Feature Extraction

PySERA supports advanced **deep learning-based** feature extraction alongside traditional radiomics, providing multiple pre-trained models for comprehensive feature representation. When using **extraction_mode="deep_feature"**, the categories parameter is automatically handled by the **deep learning model**. Deep features are extracted in 3D dimension by default for comprehensive volumetric analysis. All deep learning features are extracted specifically from the ROI regions defined by the mask and model outputs provide complementary feature representations to traditional radiomics.

**Available Deep Learning Models**:

- **`resnet50`** - 2048 features: Residual Network with 50 layers, balanced performance and accuracy
- **`vgg16`** - 512 features: Visual Geometry Group with 16 layers, strong hierarchical feature representation  
- **`densenet121`** - 1024 features: Dense Convolutional Network with 121 layers, efficient feature reuse
  
### 📦Library Installation via pip

Install the PySERA library directly from PyPI:

```bash
pip install pysera
```

## 📚Library Usage

Once installed, you can use PySERA directly in your Python code.

### 📂Single File Processing

```python
import pysera

# Process single image-mask pair
result = pysera.process_batch(
    image_input="scan.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results"
)

print(f"Success: {result['success']}")
print(f"Features extracted: {result['features_extracted']}")
print(f"Processing time: {result['processing_time']:.2f} seconds")
```

### 🧠In-Memory Array Processing

```python
import numpy as np
import nibabel as nib
import pysera

# Load image and mask as NumPy arrays (for example, using nibabel)
image_array = nib.load("patient002_image.nii.gz").get_fdata()
mask_array = nib.load("patient002_mask.nii.gz").get_fdata()

# Process the image and mask directly from memory
result = pysera.process_batch(
    image_input=image_array,
    mask_input=mask_array,
    output_path="./results"
)

# Display results
print(f"Success: {result['success']}")
print(f"Features extracted: {result['features_extracted']}")
print(f"Processing time: {result['processing_time']:.2f} seconds")
```

### ⚡Parallel Batch Processing

```python
import pysera

# Process multiple files with 4 CPU cores
result = pysera.process_batch(
    image_input="./patient_scans",
    mask_input="./patient_masks", 
    output_path="./results",
    num_workers="4",              # Use 4 CPU cores
    categories="glcm, glrlm",     # Extract specific feature categories
    dimensions="1st, 2_5d, 3d",   # Extract features in specified dimensions
    apply_preprocessing=True,   # Apply ROI preprocessing
)

print(f"Processed {result['processed_files']} files")
print(f"Total processing time: {result['processing_time']:.2f} seconds")
```

### 🤖Deep Features Extraction


```python
import pysera

# Process multiple files with 4 CPU cores
result = pysera.process_batch(
    image_input="./patient_scans",
    mask_input="./patient_masks", 
    output_path="./results",

    # Deep learning configuration
    extraction_mode="deep_feature",      # Enable deep learning features
    deep_learning_model="resnet50",      # Use ResNet50 model (2047 features)

    roi_num=5,                           # Number of ROIs to process
    roi_selection_mode="per_img",        # ROI selection strategy


    # Logging options
    report="warning"            # Report detail level: "all" (full processing details), 
                              # "info" (essential information), "warning" (warnings only), 
                              # "error" (errors only), "none" (no reporting). Default: "all"

)

print(f"Processed {result['processed_files']} files")
print(f"Total processing time: {result['processing_time']:.2f} seconds")
```


## 🔧Advanced Configuration

```python
import pysera

# Comprehensive processing with custom parameters
result = pysera.process_batch(
    image_input="image.nii.gz",
    mask_input="mask.nii.gz",
    output_path="./results",
    
    # Performance settings
    num_workers="2",           # Use 2 CPU cores
    enable_parallelism=False ,     # Disable multiprocessing
    
    # Image feature extraction settings
    categories="glcm, glrlm, glszm",  # Extract specific texture feature categories
    dimensions="1st, 2_5d, 3d",       # Extract features in 1st order, 2.5D and 3D dimensions
    # Alternative examples for categories and dimensions:
    # categories="all",                 # Extract all 557 features
    # categories="stat, morph, glcm",   # Statistical, morphological and GLCM features
    # dimensions="2D",                  # Extract only 2D features
    # dimensions="all",                 # Extract features in all dimensions
    
    bin_size=25,               # Texture analysis bin size
    roi_num=2,                # Number of ROIs to process
    roi_selection_mode="per_region",  # ROI selection strategy
    min_roi_volume=5,          # Minimum ROI volume threshold
    
    # Processing options
    apply_preprocessing=True,   # Apply ROI preprocessing
    feature_value_mode="APPROXIMATE_VALUE",  #	Strategy for handling NaN values.

    # IBSI parameters (advanced, overrides defaults)
    IBSI_based_parameters={
        "radiomics_DataType": "CT",
        "radiomics_DiscType": "FBN",
        "radiomics_isScale": 1
    },
    
    # Logging options
    report="info"             # Report detail level: "all" (full processing details), 
                              # "info" (essential information), "warning" (warnings only), 
                              # "error" (errors only), "none" (no reporting). Default: "all"
)
```

## 📥Installation

PySERA can be installed as a Python library for integration into your projects or as a standalone script for command-line usage. It supports Windows, macOS, and Linux. Below are the installation options.

### 🌐GitHub Installation 

For users who want to develop with the source code or run PySERA as a standalone command-line tool (CLI) without installing it as a Python package, you can clone the repository from GitHub.
This gives you access to the standalone script radiomics_standalone.py and all example files. After installing the dependencies, you can run the script directly (see the [💻Python Script - Command Line Interface (CLI)](#python-script---command-line-interface-cli) section).

```bash
# Clone the repository
git clone https://github.com/radiuma-com/PySERA.git
cd pysera
```
### macOS/Linux Installation
#### Quick Setup (Recommended):

```bash
# Quick setup (creates a virtual environment and installs everything)
./dev_setup.sh
```
#### Manual Setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .

```
### Windows Setup

#### Manual Setup

```bash

python -m venv venv
.\venv\Scripts\activate
cd PySERA
pip install -r requirements.txt

```

### 💻Python Script - Command Line Interface (CLI)

If you just want to run the CLI without installing the library into Python,the standalone script 'radiomics_standalone.py' provides a command-line interface for radiomics processing :

```bash
# Process single files
python radiomics_standalone.py \
    --image-input image.nii.gz \
    --mask-input mask.nii.gz \
    --output ./results

# Batch processing (folders)
python radiomics_standalone.py \
    --image-input ./images \
    --mask-input ./masks \
    --output ./results \
    --num-workers 4
```



## 📂Data Structure Requirements

For batch processing or multi-DICOM inputs, the folder structure for images and masks must follow these rules:
   - The **final folders** containing images and masks (e.g., ``images/`` and ``masks/``) must **not contain additional subfolders**. Only the image and mask files should be present in these folders.
   - There must be **only one folder level** between the parent folder and the image/mask files (e.g., ``parent/images/image001.nii.gz`` or ``parent/masks/mask001.nii.gz``).
   - **Warning**: Any additional internal subfolders within the final images or masks folders will cause PySERA to **malfunction** and fail to process the data correctly.

## Patient-Subfolder Organization (NIfTI/DICOM)

**Works with both:**

1. **DICOM Series** (multiple `.dcm` files per patient)  
2. **NIfTI Files** (single `.nii.gz` per patient)


### 🏷️Example Structures

**Note:**  PySERA supports all major formats, including DICOM, multi-slice DICOM, NIfTI, NRRD, RT Struct, and NumPy arrays.

#### 1️⃣**Flat NIfTI/NRRD Structure** 

**✅Correct:**
    
      parent/
      ├── images/ # All scan files directly here
      │   ├── patient001.nii.gz
      │   ├── patient002.nii.gz
      │   └── patient003.nii.gz
      └── masks/  # All mask files directly here
          ├── patient001.nii.gz
          ├── patient002.nii.gz
          └── patient003.nii.gz

#### 2️⃣**Patient-Subfolder NIfTI Structure**

**✅Correct:**

    parent/
    ├── CT_Images/ # Each patient has own folder
    │ ├── patient_01/
    │ │ └── scan.nii.gz # Single NIfTI file
    │ └── patient_02/
    │ └── scan.nii.gz
    └── CT_Masks/ # Mirrored structure
    ├── patient_01/
    │ └── segmentation.nii.gz
    └── patient_02/
    └── segmentation.nii.gz
    
**Notes:**  

- PySERA automatically processes DICOM series organized in patient subfolders.  
- **Patient subfolders are required** (one folder per patient).  
- **All DICOM slices for one series must be in the same patient folder.**  
- **Mask files must mirror the image folder structure.**  
  If there is a folder for `patient_01` under `CT_Images/`, there must be a corresponding `patient_01` folder under `CT_Masks/` containing the RTSTRUCT or mask.
    
    
### 3️⃣DICOM Series Structure

**✅Correct:**
    

    parent/
    ├── CT_Images/  # --image-input
    │ ├── patient_01/ # DICOM series folder
    │ │ ├── slice1.dcm  # Any number of slices
    │ │ ├── slice2.dcm
    │ │ └── slice3.dcm
    │ └── patient_02/
    │ ├── slice1.dcm
    │ └── slice2.dcm
    └── CT_Masks/   # --mask-input
    ├── patient_01/
    │ └── mask.dcm 
    └── patient_02/
    └── mask.dcm

   
**❌Incorrect Structure (Will Fail):**

      parent/
      ├── images/
      │   ├── subfolder1/
      │   │   ├── patient001.nii.gz
      │   └── subfolder2/
      │       ├── patient002.nii.gz
      └── masks/
          ├── subfolderA/
          │   ├── patient001.nii.gz
          └── patient002.nii.gz

### 🧪Sample Multi-Modality Dataset (CT, MRI, PET, RTSTRUCT)

To help users quickly test **PySERA** and understand the correct folder structure, we provide sample multimodal datasets (CT, MRI, PET), including:

- **NIfTI files**  
- **Multi-DICOM series**  
- **RTSTRUCT (Radiotherapy Structure Set)**  

You can download the sample dataset here:

🔗 **Sample Dataset (CT/MRI/PET/RTSTRUCT – NIfTI + Multi-DICOM formats)**  
https://github.com/MohammadRSalmanpour/PySERATest/tree/main/library_examples/Data


### 📋PySERA Parameters Reference


| Parameter            | Type        | Default                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|----------------------|-------------|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **image_input**       | str / .npy  | Required                                    | Path to the image file, directory, or NumPy file containing the image data.                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **mask_input**        | str / .npy  | Required                                    | Path to the mask file, directory, or NumPy file defining the regions of interest.                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **output_path**      | str         | `"./output_result"`                         | Directory where the processing results will be saved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **num_workers**      | str         | `"auto"`                                    | Number of CPU cores used for processing. If set to "auto", it will use 70% of the total available CPU cores, rounded down (e.g., 5.7 becomes 5).                                                                                                                                                                                                                                                                                                                                                                                                                        
|  **apply_preprocessing** | bool        | False                                       | If True, rounds mask array values to nearest integers. If False, uses raw mask values without rounding. |                                                                                                                                                                                                                                                                                                 
| **enable_parallelism**  | bool        | True                                        | If True, enables parallel processing for the analysis.                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| **min_roi_volume**      | int         | 10                                          | Minimum volume threshold for regions of interest (ROI).                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **bin_size**            | int         | 25                                          | **Number of bins** used for texture analysis in gray-level discretization. This determines how many intensity levels the image is divided into for feature computation. Note: This parameter specifies the *number* of bins                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **roi_selection_mode**  | str         | `"per_img"`                                 | **ROI selection strategy:**<br>- **"per_Img"** (default): Selects the top `roi_num` ROIs per image based on size, regardless of label category.<br>  • Suitable for single or dominant lesions per scan.<br>  • Preserves original spatial relationships.<br>- **"per_region"**: Selects up to `roi_num` ROIs separately for each label category, ensuring balanced representation across regions.<br>  • Useful in multi-lesion, multi-label, or longitudinal studies.<br>  • Requires consistent ROI labeling across datasets.<br> |
| **roi_num**             | int         | 10                                          | Number of ROIs to process.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **feature_value_mode**  | str         | `"REAL_VALUE"`                              | Strategy for handling NaN values. Options:`"APPROXIMATE_VALUE"` or `"REAL_VALUE"`. **"APPROXIMATE_VALUE"**: Replaces NaN features with substitutes (e.g., very small constants like `1e-30` or synthetic masks) to maintain pipeline continuity.<br>- **"REAL_VALUE"** (default): Keeps NaN values whenever feature extraction fails (e.g., small ROI, numerical instability), preserving the raw outcome without substitution.<br>                                                                                                     |
| **categories**          | str         | `"diag,morph,glcm,glrlm,glszm,ngtdm,ngldm"` | Feature categories to extract. Choices: "diag" (diagnostics), "morph" (morphological/shape), "ip" (intensity peak), "stat" (first-order statistical), "ih" (intensity histogram), "ivh" (intensity-volume histogram), "glcm" (Gray-Level Co-occurrence Matrix), "glrlm" (Gray-Level Run Length Matrix), "glszm" (Gray-Level Size Zone Matrix), "gldzm" (Gray-Level Distance Zone Matrix), "ngtdm" (Neighboring Gray-Tone Difference Matrix), "ngldm" (Neighboring Gray-Level Dependence Matrix), "mi" (moment-invariant). Example: "glcm, glrlm". |
| **dimensions**          | str         | `"1st,2d"`                                  | Spatial dimensions for feature extraction. Choices: "1st" (first-order intensity-based features), "2D" (features extracted per 2D slice), "2_5D" (features aggregated across slices with limited inter-slice context), "3D" (fully volumetric features across entire ROI). Example: "1st, 2_5d, 3d". Combine with categories for specific feature sets. |
| **aggregation_lesion**  | bool        | False                                       | When enabled, this parameter performs lesion-level feature aggregation across ROIs belonging to the same image or anatomical region, depending on the `roi_selection_mode` setting. Specifically, if `roi_selection_mode` is set to `"per_img"`, aggregation is performed by PatientID; if set to `"per_region"`, grouping is based on both PatientID and label ID. Feature aggregation is conducted on a per-feature basis. For the `"deep_feature"` extraction mode, all features are averaged. For morphological descriptors, including `morph_volume_mesh`, `morph_volume_count`, `morph_surface_area`, `morph_max_3d_diameter`, `morph_major_axis_length`, `morph_minor_axis_length`, and `morph_least_axis_length`, a weighted average based on `morph_volume_mesh` is applied. `Diagnostic` features are selected from the largest lesion, while all remaining features are summed across ROIs. Missing values are excluded from the aggregation process. |
| **callback_fn**          | function    | None                                        | Callback function for external notifications. Receives parameters: flag (`"START"`\|`"END"`), image_id (str), roi_name (str). Useful for integration with notification platforms. |
| **extraction_mode**      | str         | `"handcrafted_feature"`                     | Feature extraction mode. Options: `"handcrafted_feature"` (traditional radiomics), `"deep_feature"` (deep learning features).  |
| **deep_learning_model**  | str         | `"resnet50"`                                | Deep learning model for feature extraction when extraction_mode="deep_feature". Options:`"resnet50"`, `"vgg16"`, `"densenet121". |
| **temporary_files_path** | str         | `"./temporary_files_path"`                  | Directory for caching intermediate NumPy masks during DICOM-RT (RTSTRUCT) processing. Prevents memory spikes by writing per-ROI masks to disk and streaming them on demand. Automatically created if missing; contents are automatically cleared after processing. Not used for other image formats. |
| **report**              | str         | `"all"`                                     | Report detail level: "all" (full processing details), "info" (essential information), "warning" (warnings only), "error" (errors only), "none" (no reporting). Default: "all". |
| **IBSI_based_parameters** | dict / JSON | See defaults                                | Advanced configuration parameters. See the table below for detailed descriptions. |


#### 🔧Advanced configuration parameters (IBSI_based_parameters)


| Parameter                   | Type   | Default                | Description                                                                 |
|-----------------------------|--------|------------------------|-----------------------------------------------------------------------------|
| **radiomics_DataType**      | str    |  `"OTHER"`                 | Image modality type (CT / PET / MRI / OTHER).                               |
| **radiomics_DiscType**      | str    | `"FBS"`                 | Specifies the discretization type used for gray-level calculation — either "FBN" (fixed bin numbers) or "FBS" (fixed bin size or fixed bin width). |
| **radiomics_isScale**       | int    | 0                      | Determines whether image resampling is performed. Set to 1 to enable resampling or 0 to retain the original voxel dimensions.              |
| **radiomics_VoxInterp**     | str    | `"Nearest"`              | Defines the interpolation type used for image resampling. Accepted values include `"Nearest"`, `"linear"`, `"bilinear"`, `"trilinear"`, `"tricubic-spline"`, `"cubic"`, `"bspline"`, `"None"`.                |
| **radiomics_ROIInterp**     | str    | `"Nearest"`              | Specifies the interpolation type for ROI resampling (`"Nearest"`, `"linear"`, `"bilinear"`, `"trilinear"`, `"tricubic-spline"`, `"cubic"`, `"bspline"`, `"None"`.)                                       |
| **radiomics_isotVoxSize**   | int    | 2                      | Sets the new isotropic voxel size for 3D resampling, applied equally to the X, Y, and Z dimensions.                               |
| **radiomics_isotVoxSize2D** | int    | 2                      | Defines the voxel size for resampling in 2D mode, keeping the Z dimension unchanged while rescaling X and Y.                                |
| **radiomics_isIsot2D**      | int    | 0                      | Indicates whether to resample to isotropic 2D voxels (1) or isotropic 3D voxels (0). Applicable mainly for first-order features, as higher-order 2D features always use the original slice thickness.                           |
| **radiomics_isGLround**     | int    | 0                      | Determines whether to round voxel intensities to the nearest integer (commonly 1 for CT, 0 for PET and SPECT).                       |
| **radiomics_isReSegRng**    | int    | 0                      | Enables range-based re-segmentation. The valid intensity range is specified in radiomics_ReSegIntrvl01 and radiomics_ReSegIntrvl02. Note: not recommended for arbitrary-unit modalities such as MRI or SPECT.                            |
| **radiomics_isOutliers**    | int    | 0                      | Controls outlier removal, where 1 removes intensities beyond ±3σ.                         |
| **radiomics_isQuntzStat**   | int    | 1                      | Determines whether quantized images are used to compute first-order statistics. Set to 0 to use raw intensities (preferred for PET).                 |
| **radiomics_ReSegIntrvl01** | int    | -1000                  | Specifies the lower bound for range re-segmentation; intensities below this value are replaced with NaN.                          |
| **radiomics_ReSegIntrvl02** | int    | 400                    | Specifies the upper bound for range re-segmentation; intensities above this value are replaced with NaN.                          |
| **radiomics_ROI_PV**        | float  | 0.5                    | Defines the partial volume threshold for ROI binarization after resampling. Voxels with values below this threshold are excluded.                       |
| **radiomics_qntz**          | str    |`"Uniform"`              | Sets the quantization strategy for fixed bin number discretization. Options are "Uniform" or "Lloyd" (for Max-Lloyd quantization).                              |
| **radiomics_IVH_Type**      | int    | 3                      | {0: Definite (PET, CT), 1: Arbitrary (MRI, SPECT), 2: 1000 bins, 3: same discretization as histogram (CT)}.                         |
| **radiomics_IVH_DiscCont**  | int    | 1                      | Defines IVH continuity: {0: Discrete (CT), 1: Continuous (CT, PET; for FBS)}.                                  |
| **radiomics_IVH_binSize**   | float    | 2.0                    | for Intensity-Volume Histogram (different from `bin_size` parameter which specifies number of bins). Sets the width of each bin in intensity units. Sets the bin size for the IVH in applicable configurations (FBN with setting 1, or when IVH_DiscCont is enabled).                                                   |

## 📊Parameter Compatibility

Parameter compatibility across different extraction modes.

| Parameter | Handcrafted Feature Mode | Deep Learning Feature Mode |
|-----------|-------------------------|---------------------------|
| **image_input** | ✓ | ✓ |
| **mask_input** | ✓ | ✓ |
| **output_path** | ✓ | ✓ |
| **num_workers** | ✓ | ✓ |
| **apply_preprocessing** | ✓ | ✓ |
| **enable_parallelism** | ✓ | ✓ |
| **min_roi_volume** | ✓ | ✓ |
| **bin_size** | ✓ | ✕ |
| **roi_selection_mode** | ✓ | ✓ |
| **roi_num** | ✓ | ✓ |
| **feature_value_mode** | ✓ | ✕ |
| **categories** | ✓ | ✕ |
| **dimensions** | ✓ | ✕ |
| **aggregation_lesion** | ✓ | ✓ |
| **callback_fn** | ✓ | ✓ |
| **extraction_mode** | ✓ | ✓ |
| **deep_learning_model** | ✕ | ✓ |
| **temporary_files_path** | ✓ | ✓ |
| **report** | ✓ | ✓ |
| **IBSI_based_parameters** | ✓ | ✕ |

## 📚API Reference

### `pysera.process_batch()`

The main and only function you need for radiomics processing.


## 📊Output Structure

The ``pysera.process_batch()`` function produces two types of output: a **Python dictionary** with processing results and an **Excel file** containing detailed analysis data. Ensure your data follows `Data Structure Requirements` to avoid errors affecting output.

**Python Dictionary Output**

The function returns a dictionary with the following keys:

```python
{
    'success': bool,              # True if processing completed
    'output_path': str,           # Path to results directory
    'processed_files': int,       # Number of files processed
    'features_extracted': Dataframe,    # extracted features
    'processing_time': float,     # Processing time in seconds
    'logs': list,                # Log messages (if logging enabled)
    'error': str                 # Error message (if failed)
}
```
**Excel File Output**

**PySERA** generates an Excel file with three sheets:

📑1. **Radiomics_Features**: Lists all extracted radiomics features with IBSI-compliant naming conventions, exactly matching the standardized feature names from the Image Biomarker Standardisation Initiative. Contains computed feature values for each processed image-mask pair across all selected categories and dimensions.

⚙️2. **Parameters**: Details the parameters used for the run (e.g., ``bin_size``, ``min_roi_volume``, ``roi_selection_mode``).

⚠️3. **Report**: Logs issues for each patient sample, including ROI labels, warnings (e.g., small ROI volume), and errors (e.g., “No matching mask found for patient001.nii.gz”).


## 📁Supported File Formats

### Image Files
- **NIfTI**: `.nii`, `.nii.gz`
- **DICOM**: `.dcm`, `.dicom`, directories with DICOM files
- **NRRD**: `.nrrd`, `.nhdr`
- **NumPy**: `.npy` arrays
- **Multi-DICOM**: Directory structure with patient subdirectories
- **RTSTRUCT**: DICOM-RT Structure Set files for contour-based images.
- **Other**: Any format readable by SimpleITK (e.g., CT, MRI, PET medical images).

### Mask Files
- Same formats as image files: NIfTI, DICOM, NRRD, NumPy, RTSTRUCT.
   - **Type**: Binary or labeled segmentation masks.

   - **Requirements**:
     - Must have the **same dimensions and geometry** as the corresponding image.
     - When loading folders containing images and masks, mask file names must **exactly match** the corresponding image file names.

## 🎯Library Examples

See the [`library_examples`](https://github.com/radiuma-com/PySERA/tree/main/library_examples) directory for comprehensive usage examples:

```bash
# Run library_examples
cd library_examples
python basic_usage.py
```

Example use cases:
- Basic single-file processing
- Batch processing with multiprocessing
- High-performance processing
- Custom parameter configuration
- Single-core processing
- Comprehensive analysis with full reporting
- Selective radiomics by category and dimension
- IBSI-compliant research reproducible radiomics for scientific studies
- Deep learning feature extraction using pre-trained models (ResNet50, VGG16, DenseNet121)
- Real-time monitoring progress tracking with callback function integration
- Multi-modal analysis across CT, MRI, PET, SPECT, X-Ray, and Ultrasound

## ⚡Performance Tips

1. **Optimize CPU Utilization**: Set `num_workers="auto"` to leverage all available CPU cores for maximum parallel processing throughput
2. **Targeted Feature Extraction**: Use `categories` and `dimensions` parameters to extract only relevant features, significantly reducing computational overhead
3. **ROI Volume Filtering**: Configure appropriate `min_roi_volume` thresholds to exclude small regions and enhance processing stability
4. **Robust Feature Computation**: Use `feature_value_mode="APPROXIMATE_VALUE"` to enable synthetic voxel generation for ROIs with insufficient data (<10 voxels) OR for some features requiring specific mathematical operations (even in larger ROIs), preventing computational errors. Use `feature_value_mode="REAL_VALUE"` to preserve raw outcomes with NaN values for unreliable features in both small ROIs and mathematically constrained scenarios.
5. **Advanced Feature Representation**: Leverage `extraction_mode="deep_feature"` with pre-trained models ("resnet50", "vgg16", "densenet121") for complementary deep learning features
6. **Data Quality Enhancement**: Enable `apply_preprocessing=True` for improved mask integrity through integer value normalization
7. **Real-time Monitoring**: Implement `callback_fn` for external progress tracking and notification system integration
8. **Batch Processing Efficiency**: Process multiple files in single operations to minimize I/O overhead and maximize computational throughput
9. **Memory Optimization**: PySERA's OOP architecture automatically manages RAM utilization during large-scale batch operations
10. **Logging Optimization**: Use `report="info"` or `report="warning"` to reduce logging overhead in production environments while maintaining essential monitoring

## 💻Integration of PySERA to GUI

**PySERA** will be available as a 3D Slicer extension in the near future, providing seamless radiomics feature extraction within the popular medical imaging platform. Additionally, PySERA currently serves as the core feature extraction engine for **Radiuma**, another product in our suite that offers a comprehensive graphical interface for radiomics analysis.


## ❓Troubleshooting

### Common Issues & Fixes

- **Installation Problems**  
  **Issue:** Dependency conflicts or missing packages  
  **Fix:** Create a fresh virtual environment and run:  
  `pip install --upgrade pip wheel setuptools` before installing PySERA.

- **Memory Overload**  
  **Issue:** Crashes or slowdowns with large datasets  
  **Fix:** Set `num_workers="2"` or `enable_parallelism=False`.

- **File Processing Errors**  
  **Issue:** “No matching files found” or dimension mismatch errors  
  **Fix:** Check folder structure and ensure image–mask filenames correspond correctly.

- **ROI Volume Warnings**  
  **Issue:** Very small ROIs or NaN features  
  **Fix:** Increase `min_roi_volume=50` or use `feature_value_mode="APPROXIMATE_VALUE"`.

- **Performance Bottlenecks**  
  **Issue:** Slow processing  
  **Fix:** Use `num_workers="auto"` and target fewer feature categories with the `categories` parameter.

  ### Get Help
  
  - **Examples**: Run `python examples/basic_usage.py`

## 🕒Version History

For detailed release notes, explanations of updates, and technical changes, please see the  
👉 [Development Report](https://github.com/radiuma-com/PySERA/blob/main/development_report.md)

    v2
    ├── v2.1
    │   ├── v2.1.4 - 2025-11-19
    │   │   - Bug fix
    │   │   - use instance series UID for rt-struct files
    │   ├── v2.1.3 - 2025-11-07
    │   │   - Bug fix
    │   │   - modify default cpu core allocation
    │   │   - remove additional packages
    │   ├── v2.1.2 - 2025-10-24
    │   │   - Bug fix
    │   ├── v2.1.1 - 2025-10-22
    │   │   - Bug fix
    │   ├── v2.1.0 - 2025-10-22
    │   │   - Bug fix (configuration)
    │   │   - add aggregation_lesion parameter for aggregating radiomics features
    │   │   - update parameter default
    ├── v2.0
    │   ├── v2.0.2 - 2025-10-20
    │   │   - bug fix (configuration)
    │   ├── v2.0.1 - 2025-10-20
    │   │   - remove additional packages
    │   ├── v2.0.0 - 2025-10-19
    │   │   - ✨Major Feature Expansion, 557 features including 487 IBSI-compliant, 60 diagnostic, and 10 moment-invariant features
    │   │   - 🎯New `categories` parameter for feature category selection
    │   │   - 📐New `dimensions` parameter for 1st, 2D, 2.5D, 3D feature extraction
    │   │   - 🤖`extraction_mode="deep_feature"` with ResNet50, VGG16, DenseNet121
    │   │   - 🔔`callback_fn` for external notification platform integration
    │   │   - ⚡Enhanced OOP architecture with improved RAM and CPU efficiency
    │   │   - 📊Multi-level report system ("all", "info", "warning", "error", "none")
    │   │   - 🐛Bug Fixes, Enhanced stability and error handling
    │   │
    v1
    ├── v1.0
    │   ├── v1.0.2 - 2025-08-20
    │   │   - 🛠️change PySera name to pysera
    │   │
    │   ├── v1.0.1 - 2025-08-20
    │   │   - 🐛fixing bug in numpy array file processing in in-memory mode
    │   │
    │   └── v1.0.0 - 2025-08-19
    │       - 🛠️Structural modifications
    │       - ⚡Improved image loader 
    │       - ✨Added two strategies for feature value mode (real vs. approximate)
    │       - 🔢New parameter for number of ROIs to select
    │       - ✨Synthetic generation for ROI lesions smaller than 10 voxels
    │       - ✨New strategy for ROI selection (image-based vs. region-based)
    │       - 💾Disk-based processing to prevent RAM overflow
    │       - 🐛Fixed NaN value bug in some features
    │       - ✨Added support for processing NumPy array files in addition to file paths
    │       - ✅IBSI compliance validation
    │       - 📊New output structure including parameter set, error log, and warning report
    │       - 📦Updated package dependencies
    v0
    ├── v0.0
    │   └── v0.0.0 - 2025-08-13
    │       - 🔧IBSI Standardization 
    │       - 🐛Some Bug fix
    │
    └── initial version - 2022-02-12
       - 🎉Initial implementation  
       - 🛠️Core radiomics pipeline  
       - 📄Support for some types of files

## 📬Contact
SERA is available **free of charge**.
For access, questions, or feedback:

**Dr. Mohammad R. Salmanpour (Team Lead)**  
📧[msalman@bccrc.ca](mailto:msalman@bccrc.ca) | [m.salmanpoor66@gmail.com](mailto:m.salmanpoor66@gmail.com), [m.salmanpour@ubc.ca](mailto:m.salmanpour@ubc.ca)

---

## 🛠️Maintenance
For technical support and maintenance inquiries, please contact:

**Dr. Mohammad R. Salmanpour (Team Lead)**  
 msalman@bccrc.ca – m.salmanpoor66@gmail.com – m.salmanpour@ubc.ca

**Amir Hossein Pouria**  
amirporia99.1378@gmail.com  

## 👥Authors
- **Dr. Mohammad R. Salmanpour (Team Lead, Fund Provider, Evaluator, Medical Imaging Expert, Backend Development, Code Refactoring, Debugging, Library Management, IBSI Standardization, and Activation of the PySERA Library, and GUI Development in 3D Slicer)** – [msalman@bccrc.ca](mailto:msalman@bccrc.ca), [m.salmanpoor66@gmail.com](mailto:m.salmanpoor66@gmail.com), [m.salmanpour@ubc.ca](mailto:m.salmanpour@ubc.ca)
- **Amir Hossein Pouria (Assistant Team Lead; Backend Development, Code Refactoring, Debugging, and Library Management)** – [amirporia99.1378@gmail.com](mailto:amirporia99.1378@gmail.com)
- **Sirwan Barichin (IBSI Standardization, Debugging, and Activation of the PySERA Library, and GUI Development in 3D Slicer)** – [sirwanbarichin@gmail.com](mailto:sirwanbarichin@gmail.com)
- **Yasaman Salehi (Backend Development, Code Refactoring, and Debugging)** – [y.salehi7698@gmail.com](mailto:y.salehi7698@gmail.com)
- **Sonya Falahati (Tesing and Data prepration)** – [falahati.sonya@gmail.com](mailto:falahati.sonya@gmail.com)
- **Dr. Mehrdad Oveisi (Evaluator, Software Engineer, and Advisor)** – [moveisi@cs.ubc.ca](mailto:moveisi@cs.ubc.ca)
- **Dr. Arman Rahmim (Fund Provider, Medical Imaging Expert, Evaluator, and Advisor)** – [arman.rahmim@ubc.ca](mailto:arman.rahmim@ubc.ca), [arahmim@bccrc.ca ](mailto:arahmim@bccrc.ca)

## 📚Citation

```bibtex
@software{pysera2025,
  title={pysera: A Simple Python Library for Radiomics Feature Extraction},
  author={pysera Team},
  year={2025},
  url={https://github.com/radiuma-com/PySERA}
}
```
## 📜License

This open-source software is released under the **MIT License**, which grants permission to use, modify, and distribute it for any purpose, including research or commercial use, without requiring modified versions to be shared as open source. See the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/radiuma-com/PySERA/issues)
- **Documentation**: This README and the included guides
- **Examples**: See `examples/basic_usage.py`

## Acknowledgment

This study was supported by:  

- [🔬 **Qu**antitative **R**adiomolecular **I**maging and **T**herapy (Qurit) Lab, University of British Columbia, Vancouver, BC, Canada](https://www.qurit.ca)  
- [🏥 BC Cancer Research Institute, Department of Basic and Translational Research, Vancouver, BC, Canada](https://www.bccrc.ca/)  
- [💻 **Vir**tual **Collaboration (VirCollab) Group, Vancouver, BC, Canada](https://www.vircollab.com)  
- [🏭 **Tec**hnological **Vi**rtual **Co**llaboration **Corp**oration (TECVICO Corp.), Vancouver, BC, Canada](https://www.tecvico.com)  
We gratefully acknowledge funding from the💰 Natural Sciences and Engineering Research Council of Canada (**NSERC**) Idea to Innovation [**I2I**] Grant **GR034192**.
---

*PySERA - Simple, powerful radiomics in one function call. 🚀*
