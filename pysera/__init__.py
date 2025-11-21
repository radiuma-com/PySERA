__version__ = "2.1.5"
__author__ = "Dr. Mohammad R. Salmanpour, A.H. Pouria"
__email__ = "m.salmanpoor66@gmail.com"

from typing import Any, Dict, Optional, Union, Callable

import numpy as np

from .config.settings import DEFAULT_RADIOICS_PARAMS, DEFAULT_EXTRACTION_MODES, DEFAULT_DEEP_LEARNING_MODELS
from .processing.radiomics_processor import RadiomicsProcessor
from .utils.log_record import MemoryLogHandler, initialize_logging


def process_batch(
        image_input: Union[str, np.ndarray],
        mask_input: Union[str, np.ndarray],
        output_path: Optional[str] = None,
        temporary_files_path: Optional[str] = None,
        apply_preprocessing: Optional[bool] = None,
        min_roi_volume: Optional[int] = None,
        num_workers: Optional[Union[str, int]] = None,
        enable_parallelism: Optional[bool] = None,
        categories: Optional[str] = None,
        dimensions: Optional[str] = None,
        bin_size: Optional[int] = None,
        roi_num: Optional[int] = None,
        roi_selection_mode: Optional[str] = None,
        feature_value_mode: Optional[str] = None,
        report: Optional[str] = None,
        callback_fn: Optional[Callable[..., None]] = None,
        extraction_mode: str = DEFAULT_EXTRACTION_MODES,
        deep_learning_model: str = DEFAULT_DEEP_LEARNING_MODELS,
        aggregation_lesion: bool = False,
        IBSI_based_parameters: Optional[Dict[str, Any]] = None,
):
    import time
    import logging
    logger = logging.getLogger("Dev_logger")

    start_time = time.time()

    # Set up logging if requested
    logger, memory_handler = initialize_logging(report)
    logger.info("Starting pysera radiomics processing")

    try:
        # Create processor with logging if enabled
        processor = RadiomicsProcessor(
            output_path=output_path,
            memory_handler=memory_handler,
            temporary_files_path=temporary_files_path,
            apply_preprocessing=apply_preprocessing,
            min_roi_volume=min_roi_volume,
            num_workers=num_workers,
            enable_parallelism=enable_parallelism,
            categories=categories,
            dimensions=dimensions,
            bin_size=bin_size,
            roi_num=roi_num,
            roi_selection_mode=roi_selection_mode,
            feature_value_mode=feature_value_mode,
            report=report,
            extraction_mode=extraction_mode,
            deep_learning_model=deep_learning_model,
            aggregation_lesion=aggregation_lesion,
            IBSI_based_parameters=IBSI_based_parameters
        )

        # Process the batch
        result = processor.process_batch(
            image_input=image_input,
            mask_input=mask_input
        )

        processing_time = time.time() - start_time

        # Prepare return results
        return_result = {
            'success': bool(result),
            'output_path': processor.output_path,
            'processed_files': result.get('processed_files', 0) if result else 0,
            'features_extracted': result["out"][1] if result else 0,
            'processing_time': processing_time,
            'logs': memory_handler.get_logs() if memory_handler else []
        }

        if report != "none":
            if bool(result):
                logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            else:
                logger.error("Processing failed")

        return return_result

    except Exception as e:
        processing_time = time.time() - start_time

        if report != "none" and 'logger' in locals():
            logger.error(f"Processing failed with error: {e}")

        return {
            'success': False,
            'output_path': output_path,
            'processed_files': 0,
            'features_extracted': 0,
            'processing_time': processing_time,
            'logs': memory_handler.get_logs() if memory_handler else [],
            'error': str(e)
        }


# Main exports - only process_batch is public
__all__ = [
    'process_batch',  # Main public function
    '__version__',
    '__author__',
    '__email__'
]
