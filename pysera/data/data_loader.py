import gc
import logging
import os
from collections import defaultdict
from typing import Tuple, Optional, List, Dict, Any, Callable

import SimpleITK as sitk
import cc3d
import cv2
import numpy as np
import pydicom
from rt_utils import RTStructBuilder

from pysera.preprocessing.data_preprocessing import apply_mask_roundup
from ..utils.helpers import detect_file_format
from ..utils.helpers import save_numpy_on_disk

logger = logging.getLogger("Dev_logger")


# ============================================================
# ------------------- OOP Facade ------------------------------
# ============================================================

class DataLoader:
    """Single class OOP interface for image/mask conversion.

    This class provides a clean, cohesive API while delegating to the
    underlying helper functions in this module. Configuration is held on the
    instance; methods perform a single responsibility and compose helpers.
    """

    def __init__(
            self,
            roi_num: int,
            roi_selection_mode: str,
            min_roi_volume: int,
            temporary_files_path: str,
            apply_preprocessing: bool,
    ) -> None:
        self.roi_num = roi_num
        self.roi_selection_mode = roi_selection_mode
        self.min_roi_volume = min_roi_volume
        self.temporary_files_path = temporary_files_path
        self.apply_preprocessing = apply_preprocessing

    def convert(
            self,
            image_input: str | np.ndarray,
            mask_input: str | np.ndarray,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[Dict],
        Optional[np.ndarray | Dict[str, str]],
        Optional[dict],
    ]:
        """Convert image and mask paths into aligned arrays and metadata.

        - Detects formats
        - Loads image and mask
        - Applies optional preprocessing
        - Returns processed arrays and metadata
        """
        if isinstance(image_input, np.ndarray) and isinstance(mask_input, np.ndarray):
            return image_input, None, self.filter_rois(mask_array=mask_input), None

        img_fmt = detect_file_format(image_input)
        mask_fmt = detect_file_format(mask_input)

        logger.info(f"Image format: {img_fmt}, Mask format: {mask_fmt}")

        img_arr, img_meta, img_sitk = load_image(image_input, img_fmt)
        if img_arr is None:
            return None, None, None, None

        mask_data, mask_meta, combined_mask = self.load_mask(
            mask_input,
            mask_fmt,
            img_arr.shape,
            img_sitk,
            image_input
        )

        if mask_data is None:
            return None, None, None, None

        # if self.apply_preprocessing:
        #     processed_image = apply_image_intensity_preprocessing(
        #         img_arr,
        #         (combined_mask if isinstance(mask_data, dict) else mask_data > 0),
        #     )
        #     return processed_image, img_meta, mask_data, mask_meta

        return img_arr, img_meta, mask_data, mask_meta

    def load_mask(self, path: str, fmt: str, ref_shape: Tuple, ref_img: Optional[sitk.Image], ref_path: str) -> tuple[
        Optional[np.ndarray | dict], Optional[dict], Optional[np.ndarray]]:
        """Main entry point: load mask depending on format."""
        try:
            if fmt == "dicom":
                return self._load_dicom_mask(path, ref_img, ref_path)
            elif fmt == "npy":
                return self._load_numpy_mask(path, fmt, ref_shape)
            elif fmt == "nrrd":
                return self._load_nrrd_mask(path, fmt, ref_shape)
            elif fmt == "OpenCV_supported":
                return self._load_opencv_mask(path)
            else:
                return self._load_nifti_mask(path, fmt, ref_shape)
        except Exception as e:
            logger.error(f"Mask load failed ({fmt}): {e}")
            return None, None, None

    def _load_dicom_mask(self, path: str, ref_img: Optional[sitk.Image], ref_dir: str) -> tuple[
        Optional[np.ndarray | dict], Optional[dict], Optional[np.ndarray]]:
        """
        Load a DICOM RT-STRUCT, SEG, or DICOM mask file and align it with the reference image if provided.
        Returns a tuple of (mask_path | roi_masks, metadata, None).
        """

        dicom_file = find_first_dicom_file(path)
        if dicom_file is None:
            return None, None, None

        dicom_type = detect_dicom_type(dicom_file)
        loader = self._get_dicom_loader(dicom_type, dicom_file, ref_dir)

        if loader is None:
            logger.error(f"Unsupported DICOM type for file: {dicom_file}")
            return None, None, None

        roi_masks, roi_names, meta, combined_mask = loader()

        # Handle RT-STRUCT (multiple ROI masks)
        if isinstance(roi_masks, dict):
            return roi_masks, meta, combined_mask

        # Handle raw DICOM slices
        if roi_masks is None:
            return self.handle_dicom_file_alignment(path, ref_img)

        # Handle single mask array
        return self._process_mask_array(roi_masks, roi_names, ref_img, path)

    def _load_numpy_mask(self, path: str, fmt: str, ref_shape: Tuple):
        """Load numpy (.npy) mask."""
        arr, meta, _ = load_numpy_image(path)
        if arr.shape[::-1] == ref_shape:
            arr, meta = fix_axis_order(arr, meta, fmt, (2, 1, 0))

        if self.apply_preprocessing:
            arr = apply_mask_roundup(arr)

        return self.filter_rois(arr), meta, None

    def _load_nrrd_mask(self, path: str, fmt: str, ref_shape: Tuple):
        """Load numpy (.npy) mask."""
        arr, meta, _ = load_nrrd_image(path)
        if arr.shape[::-1] == ref_shape:
            arr, meta = fix_axis_order(arr, meta, fmt)

        if self.apply_preprocessing:
            arr = apply_mask_roundup(arr)

        return self.filter_rois(arr), meta, None

    def _load_nifti_mask(self, path: str, fmt: str, ref_shape: Tuple):
        """Load numpy (.npy) mask."""
        arr, meta, _ = load_nifti_image(path)
        if arr.shape[::-1] == ref_shape:
            arr, meta = fix_axis_order(arr, meta, fmt)

        if self.apply_preprocessing:
            arr = apply_mask_roundup(arr)

        return self.filter_rois(arr), meta, None

    def _load_opencv_mask(self, path: str):
        """Load numpy (.npy) mask."""
        arr, meta, _ = load_opencv_image(path)
        if self.apply_preprocessing:
            arr = apply_mask_roundup(arr)

        return self.filter_rois(arr), meta, None

    def _get_dicom_loader(self, dicom_type: str, dicom_file: str, ref_dir: str) -> Callable[
        [], tuple[Optional[np.ndarray | Dict[str, str]], Optional[list[str]], Optional[Dict], Optional[np.ndarray]]]:
        """Return the appropriate DICOM loader function based on type."""
        loaders = {
            "rtstruct": lambda: self.load_rtstruct_file(dicom_file, ref_dir),
            "seg": lambda: load_seg_file(dicom_file, "seg"),
            "dicom_mask": lambda: load_seg_file(dicom_file, "dicom_mask"),
        }
        return loaders.get(dicom_type)

    def load_rtstruct_file(self, rtstruct_path: str, ref_dir: str) -> \
            Tuple[Optional[Dict[str, str]], Optional[List[str]], Optional[Dict], Optional[np.ndarray]]:
        """Load RT-STRUCT and extract ROI masks."""

        ref_img = load_reference_image(ref_dir)
        if not ref_img:
            logger.error(f"No valid reference image in {ref_dir}")
            return None, None, None, None

        try:
            roi_masks, metadata, combined_mask = self.extract_rois_from_rtstruct(rtstruct_path, ref_img, ref_dir)

            roi_names = list(roi_masks.keys()) if roi_masks else []
            return roi_masks, roi_names, metadata, combined_mask
        except Exception as e:
            logger.error(f"RT-STRUCT load failed: {e}")
            return None, None, None, None

    def extract_rois_from_rtstruct(self, rtstruct_path: str, ref_img: sitk.Image, ref_dir: str) -> Tuple[
        Dict[str, str], Dict, Optional[np.ndarray]]:
        """Main entry to extract selected ROIs efficiently."""
        try:
            rtstruct = RTStructBuilder.create_from(ref_dir, rtstruct_path)
        except Exception as e:
            logger.warning("RT-STRUCT and image multi-dicom instance series UID are not equal.")
            try:
                rtstruct = load_rtstruct_without_uid_check(rtstruct_path, ref_img)
            except Exception as e:
                logger.error(e)

        # Step 1: Collect metadata (no saving yet)
        all_lesions_meta = self.collect_lesion_metadata(rtstruct, ref_img)

        # Step 2: Select top lesions based on metadata
        selected_meta = self.select_lesions_metadata(all_lesions_meta)

        # Step 3: Load, save, and record only selected lesions
        roi_masks, metadata, combined_mask = self.save_selected_lesions(selected_meta, rtstruct, ref_img, rtstruct_path)

        return roi_masks, metadata, combined_mask

    def collect_lesion_metadata(self, rtstruct, ref_img: sitk.Image) -> \
            list[dict]:
        """Collect lesion metadata (ROI name, label, volume) without saving arrays."""
        all_meta = []

        for roi_name in rtstruct.get_roi_names():
            try:
                mask = rtstruct.get_roi_mask_by_name(roi_name)
                if mask is None or not np.any(mask):
                    logger.warning(
                        f"ROI '{roi_name}' produced empty mask — likely modality mismatch or out-of-bounds contour.")
                    continue

                if self.apply_preprocessing:
                    mask = apply_mask_roundup(mask)

                aligned_mask = align_mask_shape(mask.astype(np.float32), ref_img)
                if aligned_mask is None or not np.any(aligned_mask):
                    continue

                labeled, volumes = label_and_measure_lesions_from_rtstruct(aligned_mask)
                valid_labels = np.where(volumes >= self.min_roi_volume)[0] + 1

                for label_id in valid_labels:
                    all_meta.append({
                        "roi_name": roi_name,
                        "label_id": label_id,
                        "volume": volumes[label_id - 1],
                    })

                del mask, aligned_mask, labeled, volumes

            except Exception as e:
                logger.warning(f"Failed to process ROI '{roi_name}': {e}")

        return all_meta

    def select_lesions_metadata(self, all_meta: list[dict]) -> list[dict]:
        """Select lesion metadata only (no file I/O)."""
        if not all_meta:
            logger.warning("No lesions found for selection.")
            return []

        if self.roi_selection_mode == "per_img":
            # Global top N lesions
            return sorted(all_meta, key=lambda x: x["volume"], reverse=True)[:self.roi_num]

        if self.roi_selection_mode == "per_region":
            # Top N per ROI
            selected = []
            grouped = defaultdict(list)
            for lesion in all_meta:
                grouped[lesion["roi_name"]].append(lesion)

            for roi_name, lesions in grouped.items():
                lesions.sort(key=lambda x: x["volume"], reverse=True)
                selected.extend(lesions[:self.roi_num])
            return selected

        raise ValueError(f"Invalid roi_selection_mode: {self.roi_selection_mode}")

    def save_selected_lesions(self, selected_meta: list[dict], rtstruct, ref_img: sitk.Image, rtstruct_path: str) -> \
            Tuple[Dict[str, str], Dict, Optional[np.ndarray]]:
        """Load and save only selected lesions from RT-STRUCT."""
        roi_masks, type_map, shape_map, combined_mask = {}, {}, {}, None
        if self.apply_preprocessing:
            combined_mask = np.zeros(tuple(reversed(ref_img.GetSize())), dtype=bool)

        for idx, lesion_meta in enumerate(selected_meta, start=1):
            roi_name = lesion_meta["roi_name"]
            label_id = lesion_meta["label_id"]

            mask = rtstruct.get_roi_mask_by_name(roi_name)
            aligned_mask = align_mask_shape(mask.astype(np.float32), ref_img)
            labeled, _ = cc3d.connected_components(aligned_mask, connectivity=26, return_N=True)

            lesion = (labeled == label_id).astype(np.float32)
            lesion = expand_if_2d(lesion, axis=-1)

            if self.apply_preprocessing:
                combined_mask |= (lesion > 0)

            lesion_path = save_array(lesion, prefix="roi_mask", custom_path=self.temporary_files_path)

            lesion_id = f"{roi_name}_lesion_{idx}"
            roi_masks[lesion_id] = lesion_path
            type_map[lesion_id] = lesion.dtype
            shape_map[lesion_id] = lesion.shape

            del lesion, labeled, aligned_mask, mask

        metadata = build_rtstruct_metadata(rtstruct_path, ref_img, roi_masks, type_map, shape_map)
        return roi_masks, metadata, combined_mask

    def handle_dicom_file_alignment(self, path: str, ref_img: sitk.Image) -> Tuple[
        Optional[np.ndarray], Optional[Dict], None]:
        """Handle case where mask is provided as a directory of DICOM slices."""
        mask_files = collect_dicom_files(path)
        if not mask_files:
            return None, None, None

        logger.info(f"Attempting spatial alignment of DICOM mask files: {mask_files}")
        mask_array, _dtype, _shape = self.align_dicom_mask_to_image(mask_files, ref_img)
        if mask_array is None:
            return None, None, None

        meta = build_metadata(
            _dtype=_dtype,
            _shape=_shape,
            fmt="dicom",
            files=mask_files,
            roi_names=["DICOM_MASK_ALIGNED"],
            ref_img=ref_img,
        )
        return mask_array, meta, None

    def align_dicom_mask_to_image(self, mask_files: List[str], ref_img: sitk.Image):
        """
        Align DICOM mask slices to a reference image volume.
        """
        try:
            ref_array = sitk.GetArrayFromImage(ref_img)
            ref_shape = ref_array.shape
            ref_positions = get_image_positions(ref_img)

            mask_slices, mask_positions = read_mask_slices(mask_files)
            mask_volume = map_mask_to_volume(mask_slices, mask_positions, ref_positions, ref_shape)

            if mask_volume is None:
                logger.error("No mask slices aligned.")
                return None, None

            if self.apply_preprocessing:
                mask_volume = apply_mask_roundup(mask_volume)

            mask_volume = self.filter_rois(mask_volume)
            return mask_volume, mask_volume.dtype, mask_volume.shape
        except Exception as e:
            logger.error(f"Mask alignment failed: {e}")
            return None, None, None

    def filter_rois(self, mask_array: np.ndarray) -> np.ndarray:
        """
        Filter ROIs from mask based on selection mode and volume (FAST version).
        """
        output_mask = np.zeros_like(mask_array, dtype=np.int32)
        unique_labels = [unique_label for unique_label in np.unique(mask_array) if unique_label != 0]

        if self.roi_selection_mode == "per_region":
            for label_value in unique_labels:
                output_mask += self.process_label(mask_array, label_value)

        elif self.roi_selection_mode == "per_img":
            candidates = []
            for label_value in unique_labels:
                labeled_mask, roi_ids = find_connected_rois(mask_array, label_value)
                if roi_ids.size == 0:
                    continue
                volumes = calculate_roi_volumes(labeled_mask, roi_ids)
                for idx, vol in enumerate(volumes):
                    if vol >= self.min_roi_volume:
                        candidates.append((vol, label_value, idx, labeled_mask))

            # Use partition instead of full sort
            if len(candidates) > self.roi_num:
                vols = np.array([c[0] for c in candidates])
                top_idx = np.argpartition(vols, -self.roi_num)[-self.roi_num:]
                candidates = [candidates[i] for i in top_idx]
                candidates.sort(key=lambda x: x[0], reverse=True)
            else:
                candidates.sort(key=lambda x: x[0], reverse=True)

            for _, label_value, idx, labeled_mask in candidates:
                output_mask += recreate_mask(labeled_mask, [idx], label_value)

        else:
            raise ValueError("roi_selection_mode must be 'per_img' or 'per_region'")

        return output_mask.astype(mask_array.dtype)

    def _process_mask_array(
            self,
            mask_array: np.ndarray,
            roi_names: list[str],
            ref_img: Optional[sitk.Image],
            path: str
    ) -> tuple[Optional[np.ndarray], Optional[dict], None]:
        """Align, validate, and save a single DICOM mask array."""
        if ref_img is None:
            return None, None, None

        ref_array = sitk.GetArrayFromImage(ref_img)

        # Align if necessary
        mask_array = align_if_single_slice(mask_array, ref_array, ref_img, path)

        # preprocessing
        if self.apply_preprocessing:
            mask_array = apply_mask_roundup(mask_array)

        # roi selection
        mask_array = self.filter_rois(mask_array)

        # Build metadata
        meta = build_metadata(
            _dtype=mask_array.dtype,
            _shape=mask_array.shape,
            fmt="dicom",
            files=[path],
            roi_names=roi_names,
            ref_img=ref_img,
        )

        # Validate shape and fix mismatch if needed
        mask_array, meta = _ensure_alignment(mask_array, ref_array, meta)
        if mask_array is None:
            return None, None, None

        return mask_array, meta, None

    def select_top_rois(self, volumes: np.ndarray) -> np.ndarray:
        """Select top ROIs using partition (faster than full sort)."""
        valid = np.where(volumes >= self.min_roi_volume)[0]
        if valid.size == 0:
            return np.array([], dtype=int)

        if valid.size <= self.roi_num:
            return valid

        # Partial top-k selection (O(n) instead of O(n log n))
        top_k_idx = np.argpartition(volumes[valid], -self.roi_num)[-self.roi_num:]
        sorted_top = top_k_idx[np.argsort(volumes[valid][top_k_idx])[::-1]]
        return valid[sorted_top]

    def process_label(self, mask: np.ndarray, label_value: int) -> np.ndarray:
        """Process a single label to extract selected ROIs."""
        labeled_mask, roi_ids = find_connected_rois(mask, label_value)
        if roi_ids.size == 0:
            return np.zeros_like(mask, dtype=np.int32)

        volumes = calculate_roi_volumes(labeled_mask, roi_ids)
        selected = self.select_top_rois(volumes)
        return recreate_mask(labeled_mask, selected, label_value)


# ============================================================
# ------------------- Utility Helpers ------------------------
# ============================================================

def expand_if_2d(array: np.ndarray, axis: int = -1) -> np.ndarray:
    """Ensure a 2D array becomes 3D by expanding one axis."""
    return np.expand_dims(array, axis=axis) if array.ndim == 2 else array


def save_array(array: np.ndarray, prefix: str, suffix: str = ".npy", custom_path: Optional[str] = None) -> str:
    """Save numpy array on disk and free RAM."""
    path = save_numpy_on_disk(array, prefix=prefix, suffix=suffix, custom_path=custom_path)
    del array
    gc.collect()
    return path


def fix_axis_order(mask_array: np.ndarray, meta: dict, fmt: str, trans_order: tuple = (2, 0, 1)):
    mask_array = np.transpose(mask_array, trans_order)
    meta["shape_file"] = mask_array.shape
    logger.info(f"Transposed {fmt} mask axes from {mask_array.shape[::-1]} to {mask_array.shape} to match image.")
    return mask_array, meta


def collect_dicom_files(path: str) -> List[str]:
    """Recursively collect all .dcm/.dicom files under a path."""
    dicom_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".dcm", ".dicom")):
                dicom_files.append(os.path.join(root, f))
    return dicom_files


def is_single_slice_mask(mask_array: np.ndarray, image_array: np.ndarray) -> bool:
    """Check if the mask is 2D while the reference image is 3D."""
    return mask_array.ndim == 2 and image_array.ndim == 3


def get_slice_positions(image_sitk: sitk.Image) -> list[list[float]]:
    """Get physical positions for each slice in the reference image."""
    depth = image_sitk.GetSize()[2]
    return [list(image_sitk.TransformIndexToPhysicalPoint((0, 0, i))) for i in range(depth)]


def align_single_slice_mask(mask_array: np.ndarray, image_array: np.ndarray, image_sitk: sitk.Image,
                            mask_input: str) -> np.ndarray:
    """
    Align a 2D mask to the closest slice of the reference 3D image.
    Falls back to placing in the first slice if no spatial info is found.
    """
    import pydicom

    ds = pydicom.dcmread(mask_input, stop_before_pixels=True)
    mask_pos = ds.get("ImagePositionPatient", None)

    mask_volume = np.zeros_like(image_array)

    if mask_pos is not None:
        mask_pos = [float(x) for x in mask_pos]
        img_positions = get_slice_positions(image_sitk)

        # Find the closest slice
        distances = [np.linalg.norm(np.array(mask_pos) - np.array(pos)) for pos in img_positions]
        best_idx = int(np.argmin(distances))
        best_dist = distances[best_idx]

        tolerance = 1e-2
        if best_dist < tolerance:
            mask_volume[best_idx, :, :] = mask_array
            logger.info(f"Single-slice mask aligned to slice {best_idx} (dist {best_dist:.4f} mm)")
        else:
            mask_volume[0, :, :] = mask_array
            logger.warning(f"Single-slice mask not aligned, placed in first slice (closest dist {best_dist:.4f} mm)")
    else:
        mask_volume[0, :, :] = mask_array
        logger.warning("Single-slice mask has no spatial info; placed in first slice.")

    return mask_volume


def fallback_pad_or_crop(
        mask_array: np.ndarray,
        image_array: np.ndarray,
        mask_metadata: dict
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Pad or crop mask along slice axis to match image shape.
    Updates metadata and returns (mask_array, new_path, metadata).
    """
    if mask_array.ndim != 3 or image_array.ndim != 3:
        logger.error(f"Fallback not implemented for non-3D shapes: "
                     f"image {image_array.shape}, mask {mask_array.shape}")
        return None, None

    img_slices, mask_slices = image_array.shape[0], mask_array.shape[0]

    if mask_slices < img_slices:
        pad_width = ((0, img_slices - mask_slices), (0, 0), (0, 0))
        mask_array = np.pad(mask_array, pad_width, mode="constant", constant_values=0)
        mask_array = mask_array[:img_slices, :, :]
        logger.warning(f"Mask padded from {mask_slices} to {img_slices} slices.")
    elif mask_slices > img_slices:
        mask_array = mask_array[:img_slices, :, :]
        logger.warning(f"Mask cropped from {mask_slices} to {img_slices} slices.")

    if mask_array.shape != image_array.shape:
        logger.error(f"Fallback failed: image {image_array.shape} vs mask {mask_array.shape}")
        return None, None

    # Update metadata
    mask_metadata["shape_file"] = mask_array.shape
    logger.info(f"Mask shape after fallback: {mask_array.shape}")

    return mask_array, mask_metadata


def align_if_single_slice(arr: np.ndarray, image_array: np.ndarray, ref_img: sitk.Image,
                          mask_input: str) -> np.ndarray:
    """Align a 2D mask to the correct slice in a 3D reference image if needed."""
    if is_single_slice_mask(arr, image_array):
        return align_single_slice_mask(arr, image_array, ref_img, mask_input)
    return arr


def apply_fallback_if_needed(arr: np.ndarray, image_array: np.ndarray, meta: dict):
    """Apply pad/crop fallback if mask shape does not match reference image."""
    if arr.shape == image_array.shape:
        return arr, meta
    logger.warning(
        f"Shape mismatch: image {image_array.shape} vs mask {arr.shape}. "
        f"Attempting robust fallback alignment by slice index."
    )
    return fallback_pad_or_crop(arr, image_array, meta)


def get_series_ids(dicom_dir: str) -> List[str]:
    return get_series_instance_uids(dicom_dir)


def build_metadata(
        _dtype,
        _shape,
        fmt: str,
        files: List[str],
        roi_names: List[str],
        ref_img: sitk.Image,
) -> Dict[str, Any]:
    """Construct metadata dictionary for mask arrays."""
    return {
        "type_file": _dtype,
        "shape_file": _shape,
        "format": fmt,
        "file": files,
        "roi_names": roi_names,
        "origin": ref_img.GetOrigin(),
        "spacing": ref_img.GetSpacing(),
        "direction": ref_img.GetDirection(),
    }


# ============================================================
# ------------------- ROI selection ---------------------------
# ============================================================


def find_connected_rois(mask: np.ndarray, label_value: int) -> tuple[np.ndarray, np.ndarray]:
    """Find connected ROIs for a given label using CC3D (very fast)."""
    binary_mask = (mask == label_value).astype(np.float32)
    labeled_mask = cc3d.connected_components(binary_mask, connectivity=26)  # 26-connected 3D
    roi_ids = np.unique(labeled_mask)
    roi_ids = roi_ids[roi_ids != 0]  # remove background
    return labeled_mask, roi_ids


def calculate_roi_volumes(labeled_mask: np.ndarray, roi_ids: np.ndarray) -> np.ndarray:
    """Calculate ROI volumes using bincount (fast)."""
    counts = np.bincount(labeled_mask.ravel())
    return counts[roi_ids]


def recreate_mask(labeled_mask: np.ndarray, selected_ids: np.ndarray | list, label_value: int) -> np.ndarray:
    """Recreate mask for selected ROIs while preserving original label value."""
    new_mask = np.zeros_like(labeled_mask, dtype=np.int32)
    for roi_idx in selected_ids:
        roi_id = roi_idx + 1
        new_mask[labeled_mask == roi_id] = label_value
    return new_mask


# ============================================================
# ------------------- RT-STRUCT ------------------------
# ============================================================

def find_first_dicom_file(path: str) -> Optional[str]:
    """Return the first DICOM file found in a directory or the file itself."""
    if os.path.isfile(path):
        return path
    for root, _, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            if is_dicom_file(full_path):
                return full_path
    return None


def is_dicom_file(filepath: str) -> bool:
    """Quickly check if a file is a valid DICOM file."""
    try:
        pydicom.dcmread(filepath, stop_before_pixels=True)
        return True
    except Exception:
        return False


def detect_dicom_type(filepath: str) -> str:
    """Detect whether DICOM file is RT-STRUCT, SEG, or a plain DICOM mask."""
    ds = pydicom.dcmread(filepath, stop_before_pixels=True)
    sop_uid = getattr(ds, "SOPClassUID", "")
    modality = getattr(ds, "Modality", "")
    if modality == "RTSTRUCT" or sop_uid == "1.2.840.10008.5.1.4.1.1.481.3":
        return "rtstruct"
    if modality == "SEG":
        return "seg"
    if hasattr(ds, "PixelData"):
        return "dicom_mask"
    return "unknown"


# ==============================
# RT-STRUCT: RT-STRUCT Handling
# ==============================
class MinimalRTStruct:
    """Lightweight RTStruct alternative with only two required methods."""
    def __init__(self, roi_masks):
        self.roi_masks = roi_masks  # dict: name → numpy array mask

    def get_roi_names(self):
        return list(self.roi_masks.keys())

    def get_roi_mask_by_name(self, name):
        return self.roi_masks.get(name, None)


def load_rtstruct_without_uid_check(rtstruct_path: str, ref_img: sitk.Image):

    ds = pydicom.dcmread(rtstruct_path)

    if "ROIContourSequence" not in ds:
        raise ValueError("RTSTRUCT has no ROIContourSequence.")

    ref_array = sitk.GetArrayFromImage(ref_img)
    roi_masks = {}

    for roi_contour in ds.ROIContourSequence:

        roi_number = roi_contour.ReferencedROINumber

        roi_name = None
        for item in ds.StructureSetROISequence:
            if item.ROINumber == roi_number:
                roi_name = item.ROIName
                break

        if roi_name is None:
            continue

        mask = np.zeros(ref_array.shape, dtype=np.uint8)

        if not hasattr(roi_contour, "ContourSequence"):
            continue

        for contour in roi_contour.ContourSequence:

            pts = np.array(contour.ContourData).reshape(-1, 3)   # EXACT world coords

            # Convert each point using SITK native conversion
            voxel_pts = []
            for p in pts:
                try:
                    idx = ref_img.TransformPhysicalPointToIndex(tuple(p))
                    voxel_pts.append(idx)
                except:
                    continue

            voxel_pts = np.array(voxel_pts)

            # Process slice by slice
            slice_idx = voxel_pts[:, 2]

            for sl in np.unique(slice_idx):
                if sl < 0 or sl >= mask.shape[0]:
                    continue

                poly = voxel_pts[slice_idx == sl][:, :2]

                if len(poly) < 3:
                    continue

                from skimage.draw import polygon
                rr, cc = polygon(poly[:, 1], poly[:, 0], shape=mask[sl].shape)
                mask[sl, rr, cc] = 1

        roi_masks[roi_name] = mask.astype(np.uint8)

    return MinimalRTStruct(roi_masks)


def label_and_measure_lesions_from_rtstruct(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Label connected lesions using CC3D and compute their volumes."""
    labeled, num_labels = cc3d.connected_components(mask, connectivity=26, return_N=True)
    if num_labels == 0:
        return labeled, np.array([])
    lesion_volumes = np.bincount(labeled.ravel())[1:]  # skip background
    return labeled, lesion_volumes


# ==============================
# RT-STRUCT: SEG / DICOM Mask Loader
# ==============================

def load_seg_file(filepath: str, tag: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], None, None]:
    """Load and save SEG/DICOM mask file."""
    try:
        sitk_mask = sitk.ReadImage(filepath)
        arr = sitk.GetArrayFromImage(sitk_mask).astype(np.float32)
        return arr, [tag.upper()], None, None
    except Exception as e:
        logger.error(f"Failed to load {tag} mask: {e}")
        return None, None, None, None


# ==============================
# RT-STRUCT: Reference Image Loader
# ==============================
def get_series_instance_uids(dicom_dir):
    """
    Extract Series Instance UIDs from DICOM directory
    """
    series_uids = set()

    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                if hasattr(ds, 'SeriesInstanceUID'):
                    suid = ds.SeriesInstanceUID
                    series_uids.add(suid)

            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

    return list(series_uids)


def load_reference_image(dicom_dir: str) -> Optional[sitk.Image]:
    """Load first series image from DICOM directory."""
    reader = sitk.ImageSeriesReader()
    series_ids = get_series_instance_uids(dicom_dir)
    if not series_ids:
        return None
    try:
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
        reader.SetFileNames(dicom_files)
        return reader.Execute()
    except Exception as e:
        logger.warning(f"Failed to load DICOM series: {e}")
        return None


# ==============================
# RT-STRUCT: Mask Alignment
# ==============================

def align_mask_shape(mask: np.ndarray, ref_image: sitk.Image) -> Optional[np.ndarray]:
    """Align mask shape to reference image array shape."""
    ref_array = sitk.GetArrayFromImage(ref_image)

    if mask.shape == ref_array.shape:
        return mask

    possible_shapes = [
        lambda: mask.transpose(2, 0, 1) if mask.ndim == 3 else None,
        lambda: mask.squeeze() if mask.ndim > 3 else None,
        lambda: np.expand_dims(mask, axis=-1) if mask.ndim == 2 else None,
    ]

    for fix in possible_shapes:
        try:
            candidate = fix()
            if candidate is not None and candidate.shape == ref_array.shape:
                return candidate
        except Exception:
            continue

    logger.error(f"Mask shape {mask.shape} does not match reference {ref_array.shape}")
    return None


# ==============================
# RT-STRUCT: Mask Array Handler
# ==============================


def build_rtstruct_metadata(
        path: str, ref_img: sitk.Image, rois_dict: Dict[str, str],
        type_map: Dict, shape_map: Dict
) -> Dict[str, Any]:
    """Build metadata dictionary for RT-STRUCT arrays."""
    return {
        "type_file": type_map,
        "shape_file": shape_map,
        "format": "dicom-rtstruct",
        "file": [path],
        "roi_names": list(rois_dict.keys()),
        "origin": ref_img.GetOrigin(),
        "spacing": ref_img.GetSpacing(),
        "direction": ref_img.GetDirection(),
    }


# ============================================================
# ------------------- Mask Loading ---------------------------
# ============================================================

def _ensure_alignment(
        roi_masks: np.ndarray,
        ref_array: np.ndarray,
        meta: dict
) -> tuple[np.ndarray | None, dict]:
    """Ensure mask and reference image are aligned; fix mismatch if needed."""
    aligned_mask, updated_meta = apply_fallback_if_needed(roi_masks, ref_array, meta)
    return aligned_mask, updated_meta


# ============================================================
# ------------------- Image Loading ---------------------------
# ============================================================

def load_single_dicom_image(path: str) -> Optional[sitk.Image]:
    try:
        return sitk.ReadImage(path)
    except Exception as e:
        logger.warning(f"Failed to load single dicom image: {e}")
        return None


def load_series_by_id(dicom_dir: str, series_id: str) -> Optional[sitk.Image]:
    try:
        reader = sitk.ImageSeriesReader()
        dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        reader.SetFileNames(dicom_files)
        return reader.Execute()
    except Exception as e:
        logger.warning(f"Failed to load series {series_id}: {e}")
        return None


def load_dicom_series(dicom_dir: str) -> Tuple[Optional[sitk.Image], Optional[str]]:
    """
    Load the largest available DICOM series (by slice count).
    Falls back to a single image if no series found.
    """
    series_ids = get_series_ids(dicom_dir)

    if not series_ids:
        image = load_single_dicom_image(dicom_dir)
        if image:
            logger.info("Loaded single DICOM image.")
            return image, None
        logger.error(f"No DICOM found in {dicom_dir}")
        return None, None

    series_images = []
    for sid in series_ids:
        image = load_series_by_id(dicom_dir, sid)
        if image:
            n_slices = image.GetSize()[-1]
            series_images.append((sid, image, n_slices))

    if not series_images:
        logger.error("No valid DICOM series found.")
        return None, None

    # Pick largest
    sid, image, n_slices = max(series_images, key=lambda x: x[2])
    logger.info(f"Using series {sid} with {n_slices} slices.")
    return image, sid


def load_dicom_image(path: str) -> Tuple[np.ndarray | None, Dict | None, sitk.Image | None]:
    img_sitk, series_id = load_dicom_series(path)
    if img_sitk is None:
        return None, None, None
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    arr = expand_if_2d(arr, axis=0)
    meta = {
        "type_file": arr.dtype,
        "shape_file": arr.shape,
        "format": "dicom",
        "file": [path],
        "series_id": series_id,
        "origin": img_sitk.GetOrigin(),
        "spacing": img_sitk.GetSpacing(),
        "direction": img_sitk.GetDirection(),
    }
    return arr, meta, img_sitk


def load_nifti_image(path: str) -> Tuple[np.ndarray, Dict, sitk.Image]:
    img_sitk = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    arr = expand_if_2d(arr, axis=-1)
    arr = np.transpose(arr, (2, 1, 0))
    meta = {
        "type_file": arr.dtype,
        "shape_file": arr.shape,
        "format": "nifti",
        "file": [path],
        "origin": img_sitk.GetOrigin(),
        "spacing": img_sitk.GetSpacing(),
        "direction": img_sitk.GetDirection(),
    }
    return arr, meta, img_sitk


def load_nrrd_image(path: str) -> Tuple[np.ndarray, Dict, None]:
    import nrrd
    arr, header = nrrd.read(path)
    arr = arr.astype(np.float32)
    arr = expand_if_2d(arr, axis=-1)
    meta = {"type_file": arr.dtype, "shape_file": arr.shape, "format": "nrrd", "file": [path], "header": header}
    return arr, meta, None


def load_numpy_image(path: str) -> Tuple[np.ndarray, Dict, None]:
    arr = np.load(path).astype(np.float32)
    arr = expand_if_2d(arr, axis=-1)
    meta = {
        "type_file": arr.dtype,
        "shape_file": arr.shape,
        "format": "npy",
        "file": [path],
        "origin": (0, 0, 0),
        "spacing": (1, 1, 1),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return arr, meta, None


def load_opencv_image(path: str) -> Tuple[np.ndarray, Dict, None]:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    arr = expand_if_2d(arr, axis=-1)
    arr = np.transpose(arr, (1, 0, 2))
    meta = {
        "type_file": arr.dtype,
        "shape_file": arr.shape,
        "format": "opencv",
        "file": [path],
        "origin": (0, 0, 0),
        "spacing": (1, 1, 1),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
    }
    return arr, meta, None


# ============================================================
# ------------------- RT-STRUCT -----------------------
# ============================================================


def get_image_positions(image: sitk.Image) -> List[List[float]]:
    """Return physical positions of all slices in an image."""
    depth = image.GetSize()[2]
    return [list(image.TransformIndexToPhysicalPoint((0, 0, i))) for i in range(depth)]


def read_mask_slices(mask_files: List[str]) -> Tuple[np.ndarray, List[Optional[List[float]]]]:
    """Read mask slices and extract their positions."""
    slices, positions = [], []
    for f in mask_files:
        ds = pydicom.dcmread(f)
        arr = expand_if_2d(ds.pixel_array, axis=0)
        slices.append(arr)
        pos = ds.get("ImagePositionPatient")
        positions.append([float(x) for x in pos] if pos is not None else None)
    return np.concatenate(slices, axis=0), positions


def map_mask_to_volume(mask_slices: np.ndarray, mask_positions: List, ref_positions: List, ref_shape: Tuple[int]) -> \
        Optional[np.ndarray]:
    """Map 2D mask slices into 3D volume based on slice positions."""
    volume = np.zeros(ref_shape, dtype=mask_slices.dtype)
    tolerance = 1e-2
    used = 0

    for m_idx, m_pos in enumerate(mask_positions):
        if m_pos is None:
            continue
        distances = [np.linalg.norm(np.array(m_pos) - np.array(rp)) for rp in ref_positions]
        best_idx = int(np.argmin(distances))
        if distances[best_idx] < tolerance:
            volume[best_idx] = mask_slices[m_idx]
            used += 1
            logger.debug(f"Slice {m_idx} → {best_idx}, dist={distances[best_idx]:.4f} mm")
    return volume if used > 0 else None


# ============================================================
# ------------------- Main -----------------------
# ============================================================

def load_image(path: str, fmt: str):
    try:
        loaders = {
            "dicom": load_dicom_image,
            "nifti": load_nifti_image,
            "nrrd": load_nrrd_image,
            "npy": load_numpy_image,
            "OpenCV_supported": load_opencv_image,
        }
        if fmt not in loaders:
            return load_nifti_image(path)  # fallback
        return loaders[fmt](path)

    except Exception as e:
        logger.error(f"Failed to load image file: {path}, error: {e}")
        return None, None, None
