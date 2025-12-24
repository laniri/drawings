"""
Data Pipeline Service for Children's Drawing Anomaly Detection System

This module provides image preprocessing, validation, and metadata management
functionality for the drawing analysis pipeline.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field, field_validator

from app.core.exceptions import ImageProcessingError, ValidationError

logger = logging.getLogger(__name__)

# Optional OpenCV import for advanced image processing
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
except Exception as e:
    # Handle other OpenCV-related errors (e.g., missing system libraries)
    logger.warning(f"OpenCV import failed: {e}")
    HAS_OPENCV = False
    cv2 = None


class ValidationResult(BaseModel):
    """Result of image validation process"""

    is_valid: bool
    error_message: Optional[str] = None
    image_format: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    file_size: Optional[int] = None


class DrawingMetadata(BaseModel):
    """Metadata extracted from drawing upload"""

    age_years: float = Field(..., ge=2.0, le=18.0, description="Child's age in years")
    subject: Optional[str] = Field(None, max_length=100, description="Drawing subject")
    expert_label: Optional[str] = Field(
        None, max_length=50, description="Expert annotation"
    )
    drawing_tool: Optional[str] = Field(
        None, max_length=50, description="Drawing tool used"
    )
    prompt: Optional[str] = Field(
        None, max_length=500, description="Drawing prompt given"
    )

    @field_validator("age_years")
    @classmethod
    def validate_age(cls, v):
        if not isinstance(v, (int, float)):
            raise ValidationError("Age must be a number")
        if v < 2.0 or v > 18.0:
            raise ValidationError("Age must be between 2 and 18 years")
        return float(v)


# Keep the old exception for backward compatibility in tests
class ImagePreprocessingError(ImageProcessingError):
    """Custom exception for image preprocessing errors (deprecated, use ImageProcessingError)"""

    pass


class DataPipelineService:
    """Service for handling image preprocessing and validation"""

    SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "BMP"}
    DEFAULT_TARGET_SIZE = (224, 224)  # Standard size for Vision Transformers
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

    def __init__(self, target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE):
        """
        Initialize the data pipeline service

        Args:
            target_size: Target dimensions for image resizing (width, height)
        """
        self.target_size = target_size
        logger.info(f"DataPipelineService initialized with target size: {target_size}")

    def validate_image(
        self, image_data: bytes, filename: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate image data for format, size, and integrity

        Args:
            image_data: Raw image bytes
            filename: Optional filename for additional validation

        Returns:
            ValidationResult with validation status and details
        """
        try:
            # Check file size
            if len(image_data) > self.MAX_FILE_SIZE:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"File size {len(image_data)} bytes exceeds maximum {self.MAX_FILE_SIZE} bytes",
                    file_size=len(image_data),
                )

            # Check if data is empty
            if len(image_data) == 0:
                return ValidationResult(
                    is_valid=False, error_message="Empty file provided", file_size=0
                )

            # Try to open and validate the image
            try:
                image = Image.open(io.BytesIO(image_data))
                image.verify()  # Verify image integrity

                # Re-open for format checking (verify() closes the image)
                image = Image.open(io.BytesIO(image_data))

                # Check format
                if image.format not in self.SUPPORTED_FORMATS:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unsupported image format: {image.format}. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}",
                        image_format=image.format,
                        dimensions=image.size,
                        file_size=len(image_data),
                    )

                # Check dimensions
                width, height = image.size
                if width < 32 or height < 32:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Image too small: {width}x{height}. Minimum size is 32x32 pixels",
                        image_format=image.format,
                        dimensions=image.size,
                        file_size=len(image_data),
                    )

                # Check if image has valid mode
                if image.mode not in ["RGB", "RGBA", "L", "P"]:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Unsupported image mode: {image.mode}",
                        image_format=image.format,
                        dimensions=image.size,
                        file_size=len(image_data),
                    )

                return ValidationResult(
                    is_valid=True,
                    image_format=image.format,
                    dimensions=image.size,
                    file_size=len(image_data),
                )

            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Corrupted or invalid image file: {str(e)}",
                    file_size=len(image_data),
                )

        except Exception as e:
            logger.error(f"Unexpected error during image validation: {str(e)}")
            return ValidationResult(
                is_valid=False, error_message=f"Validation error: {str(e)}"
            )

    def preprocess_image(
        self, image_data: bytes, target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess image data into standardized tensor format

        Args:
            image_data: Raw image bytes
            target_size: Optional target size override

        Returns:
            Preprocessed image as numpy array with shape (H, W, C) and values in [0, 1]

        Raises:
            ImagePreprocessingError: If preprocessing fails
        """
        try:
            target_size = target_size or self.target_size

            # Validate image first
            validation_result = self.validate_image(image_data)
            if not validation_result.is_valid:
                raise ImageProcessingError(
                    f"Image validation failed: {validation_result.error_message}"
                )

            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                if image.mode == "RGBA":
                    # Handle transparency by compositing on white background
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(
                        image, mask=image.split()[-1]
                    )  # Use alpha channel as mask
                    image = background
                elif image.mode == "P":
                    # Convert palette mode to RGB
                    image = image.convert("RGB")
                elif image.mode == "L":
                    # Convert grayscale to RGB
                    image = image.convert("RGB")
                else:
                    # For any other mode, convert to RGB
                    image = image.convert("RGB")

            # Resize image while maintaining aspect ratio
            image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)

            # Normalize pixel values to [0, 1]
            image_array = image_array / 255.0

            # Ensure correct shape (H, W, C)
            if len(image_array.shape) == 2:
                # Grayscale image, add channel dimension
                image_array = np.expand_dims(image_array, axis=-1)
                image_array = np.repeat(image_array, 3, axis=-1)

            if image_array.shape[-1] != 3:
                raise ImageProcessingError(
                    f"Unexpected number of channels: {image_array.shape[-1]}"
                )

            logger.debug(
                f"Image preprocessed successfully: shape={image_array.shape}, dtype={image_array.dtype}"
            )
            return image_array

        except ImagePreprocessingError:
            raise
        except ImageProcessingError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise ImageProcessingError(f"Preprocessing failed: {str(e)}")

    def extract_metadata(self, upload_data: Dict[str, Any]) -> DrawingMetadata:
        """
        Extract and validate metadata from upload data

        Args:
            upload_data: Dictionary containing metadata fields

        Returns:
            Validated DrawingMetadata object

        Raises:
            ValueError: If metadata validation fails
        """
        try:
            # Extract required and optional fields
            metadata_dict = {
                "age_years": upload_data.get("age_years"),
                "subject": upload_data.get("subject"),
                "expert_label": upload_data.get("expert_label"),
                "drawing_tool": upload_data.get("drawing_tool"),
                "prompt": upload_data.get("prompt"),
            }

            # Remove None values and empty strings for optional fields
            metadata_dict = {
                k: v
                for k, v in metadata_dict.items()
                if v is not None and (not isinstance(v, str) or v.strip())
            }

            # Validate using Pydantic model
            metadata = DrawingMetadata(**metadata_dict)

            logger.debug(f"Metadata extracted successfully: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction failed: {str(e)}")
            raise ValueError(f"Invalid metadata: {str(e)}")

    def validate_and_preprocess(
        self, image_data: bytes, metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, DrawingMetadata]:
        """
        Combined validation and preprocessing pipeline

        Args:
            image_data: Raw image bytes
            metadata: Metadata dictionary

        Returns:
            Tuple of (preprocessed_image, validated_metadata)

        Raises:
            ImagePreprocessingError: If image processing fails
            ValueError: If metadata validation fails
        """
        # Extract and validate metadata first
        validated_metadata = self.extract_metadata(metadata)

        # Validate and preprocess image
        preprocessed_image = self.preprocess_image(image_data)

        return preprocessed_image, validated_metadata


# Global data pipeline service instance
_data_pipeline_service = None


def get_data_pipeline_service() -> DataPipelineService:
    """Get the global data pipeline service instance."""
    global _data_pipeline_service
    if _data_pipeline_service is None:
        _data_pipeline_service = DataPipelineService()
    return _data_pipeline_service
