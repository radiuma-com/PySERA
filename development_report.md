# Development Report

This document provides a detailed history of the development progress, major improvements, and bug fixes made throughout the lifecycle of the project.  
It serves as both a changelog and a reference for design decisions, allowing users and developers to track the evolution of the radiomics processing pipeline.  
Each version entry documents new features, optimizations, supported formats, compliance updates, and structural refinements that have contributed to making the system more robust, efficient, and standardized.  

---

## Version History
        
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
        ├── ├── v2.1.0 - 2025-10-22
        │   │   - Bug fix (configuration)
        │   │   - add aggregation_lesion parameter for aggregating radiomics features
        │   │   - update parameter default
        ├── v2.0
        ├── ├── v2.0.2 - 2025-10-20
        │   │   - Bug fix (configuration)
        ├── ├── v2.0.1 - 2025-10-20
        │   │   - remove additional packages
        │   ├── v2.0.0 - 2025-10-19
        │   │   - Expanded feature library to 497 IBSI-compliant radiomics features, 10 moment invariant features, 60 diagnostics features across multiple categories and dimensions, providing comprehensive coverage for advanced radiomics analysis.
        │   │   - Introduced selective feature extraction with `categories` parameter, enabling users to target specific feature types: diagnostics (diag), morphological (morph), intensity peak (ip), first-order statistical (stat), intensity histogram (ih), intensity-volume histogram (ivh), Gray-Level Co-occurrence Matrix (glcm), Gray-Level Run Length Matrix (glrlm), Gray-Level Size Zone Matrix (glszm), Gray-Level Distance Zone Matrix (gldzm), Neighboring Gray-Tone Difference Matrix (ngtdm), Neighboring Gray-Level Dependence Matrix (ngldm), and moment-invariant (mi) features.
        │   │   - Added dimensional control with `dimensions` parameter, supporting first-order (1st), 2D slice-based (2D), 2.5D aggregated (2_5D), and fully volumetric 3D (3D) feature extraction strategies.
        │   │   - Integrated deep learning feature extraction with `extraction_mode="deep_feature"` supporting pre-trained models including ResNet50, VGG16, and DenseNet121 for advanced feature representation.
        │   │   - Implemented callback function system (`callback_fn`) for real-time progress tracking and external notification platform integration, receiving parameters: flag ("START"|"END"), image_id (str), roi_name (str).
        │   │   - Enhanced performance through optimized OOP architecture with improved RAM management and CPU-efficient parallel processing for large-scale batch operations.
        │   │   - Developed multi-level reporting system with `report` parameter offering "all" (full details), "info" (essential information), "warning" (warnings only), "error" (errors only), and "none" (no reporting) options.
        │   │   - Resolved stability issues and enhanced error handling across the feature extraction pipeline.
        │   │   - Improved documentation with comprehensive examples demonstrating selective feature extraction combinations and deep learning integration workflows.
        │   │
        v1
        ├── v1.0
        │   ├── v1.0.2 - 2025-08-20
        │   │   - change PySera name to pysera
        │   │
        │   ├── v1.0.1 - 2025-08-20
        │   │   - fixing bug in numpy array file processing in in-memory mode
        │   │
        │   └── v1.0.0 - 2025-08-19
        │       - Optimized memory management by storing large RoIs as float32 NumPy arrays on disk with memory mapping, ensuring efficient management and preventing RAM overflows.  
        │       - Enabled robust feature extraction for very small RoIs (fewer than 10 voxels) using epsilon corrections (𝜀) and synthetic augmentation, supporting cases as small as single-voxel inputs.  
        │       - Corrected mathematical operations prone to errors (e.g., log(0), division by zero, sqrt of negatives, covariance matrix for single voxel, ConvexHull requirements, percentile estimation).  
        │       - Improved data loading performance with smoother and more efficient handling of medical imaging inputs.  
        │       - Extended supported input formats to include RT-Struct data paths and direct NumPy array inputs for both images and masks.  
        │       - Enhanced Excel output with two new tabs: Parameters (record of all parameters used) and Warnings and Errors (detailed runtime issues).  
        │       - Added flexible RoI selection controls:  
        │            • roi_num: defines how many RoIs to select.  
        │            • roi_selection_mode: supports per_img and per_region strategies.  
        │       - Exposed radiomics parameters as configurable arguments, replacing hardcoded values.  
        │       - Implemented automatic padding to resolve image and mask dimension mismatches.  
        │       - Improved RoI naming in Excel outputs by including both label and lesion information when available.  
        │       - Performed codebase restructuring: renamed variables for clarity, modularized the implementation into a standalone library, updated requirements, and verified IBSI standard compliance.  
        │
        v0
        ├── v0.0
        │   └── v0.0.0 - 2025-08-13
        │       - Implemented IBSI standardization to ensure feature extraction follows international radiomics guidelines.  
        │       - Fixed several bugs and stability issues identified in earlier development versions.  
        │
        └── initial version - 2022-02-12
            - Released the first working implementation of the project.  
            - Developed the core radiomics processing pipeline, providing the foundation for feature extraction.  
            - Added support for initial file formats, enabling users to process a limited set of medical imaging data types.  
        
