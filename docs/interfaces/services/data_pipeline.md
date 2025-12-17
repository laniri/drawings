# Data Pipeline Service

Data Pipeline Service for Children's Drawing Anomaly Detection System

This module provides image preprocessing, validation, and metadata management
functionality for the drawing analysis pipeline.

## Class: ValidationResult

Result of image validation process

## Class: DrawingMetadata

Metadata extracted from drawing upload

### validate_age

**Signature**: `validate_age(cls, v)`

## Class: ImagePreprocessingError

Custom exception for image preprocessing errors (deprecated, use ImageProcessingError)

## Class: DataPipelineService

Service for handling image preprocessing and validation

### validate_image

Validate image data for format, size, and integrity

Args:
    image_data: Raw image bytes
    filename: Optional filename for additional validation
    
Returns:
    ValidationResult with validation status and details

**Signature**: `validate_image(image_data, filename)`

### preprocess_image

Preprocess image data into standardized tensor format

Args:
    image_data: Raw image bytes
    target_size: Optional target size override
    
Returns:
    Preprocessed image as numpy array with shape (H, W, C) and values in [0, 1]
    
Raises:
    ImagePreprocessingError: If preprocessing fails

**Signature**: `preprocess_image(image_data, target_size)`

### extract_metadata

Extract and validate metadata from upload data

Args:
    upload_data: Dictionary containing metadata fields
    
Returns:
    Validated DrawingMetadata object
    
Raises:
    ValueError: If metadata validation fails

**Signature**: `extract_metadata(upload_data)`

### validate_and_preprocess

Combined validation and preprocessing pipeline

Args:
    image_data: Raw image bytes
    metadata: Metadata dictionary
    
Returns:
    Tuple of (preprocessed_image, validated_metadata)
    
Raises:
    ImagePreprocessingError: If image processing fails
    ValueError: If metadata validation fails

**Signature**: `validate_and_preprocess(image_data, metadata)`

