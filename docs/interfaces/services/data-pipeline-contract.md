# Data Pipeline Contract

## Overview
Service contract for Data Pipeline (service)

**Source File**: `app/services/data_pipeline.py`

## Interface Specification

### Classes

#### ValidationResult

Result of image validation process

**Inherits from**: BaseModel

**Attributes**:

- `is_valid: bool`
- `error_message: Optional[str]`
- `image_format: Optional[str]`
- `dimensions: <ast.Subscript object at 0x11042f150>`
- `file_size: Optional[int]`

#### DrawingMetadata

Metadata extracted from drawing upload

**Inherits from**: BaseModel

**Attributes**:

- `age_years: float`
- `subject: Optional[str]`
- `expert_label: Optional[str]`
- `drawing_tool: Optional[str]`
- `prompt: Optional[str]`

#### ImagePreprocessingError

Custom exception for image preprocessing errors (deprecated, use ImageProcessingError)

**Inherits from**: ImageProcessingError

#### DataPipelineService

Service for handling image preprocessing and validation

## Methods

### validate_age

**Signature**: `validate_age(cls, v)`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `cls` | `Any` | Parameter description |
| `v` | `Any` | Parameter description |

### validate_image

Validate image data for format, size, and integrity

Args:
    image_data: Raw image bytes
    filename: Optional filename for additional validation
    
Returns:
    ValidationResult with validation status and details

**Signature**: `validate_image(image_data: bytes, filename: Optional[str]) -> ValidationResult`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image_data` | `bytes` | Parameter description |
| `filename` | `Optional[str]` | Parameter description |

**Returns**: `ValidationResult`

### preprocess_image

Preprocess image data into standardized tensor format

Args:
    image_data: Raw image bytes
    target_size: Optional target size override
    
Returns:
    Preprocessed image as numpy array with shape (H, W, C) and values in [0, 1]
    
Raises:
    ImagePreprocessingError: If preprocessing fails

**Signature**: `preprocess_image(image_data: bytes, target_size: <ast.Subscript object at 0x110387450>) -> np.ndarray`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image_data` | `bytes` | Parameter description |
| `target_size` | `<ast.Subscript object at 0x110387450>` | Parameter description |

**Returns**: `np.ndarray`

### extract_metadata

Extract and validate metadata from upload data

Args:
    upload_data: Dictionary containing metadata fields
    
Returns:
    Validated DrawingMetadata object
    
Raises:
    ValueError: If metadata validation fails

**Signature**: `extract_metadata(upload_data: Dict[str, Any]) -> DrawingMetadata`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `upload_data` | `Dict[str, Any]` | Parameter description |

**Returns**: `DrawingMetadata`

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

**Signature**: `validate_and_preprocess(image_data: bytes, metadata: Dict[str, Any]) -> Tuple[<ast.Attribute object at 0x11058eb50>, DrawingMetadata]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image_data` | `bytes` | Parameter description |
| `metadata` | `Dict[str, Any]` | Parameter description |

**Returns**: `Tuple[<ast.Attribute object at 0x11058eb50>, DrawingMetadata]`

## Defined Interfaces

### DrawingMetadataInterface

**Type**: Protocol
**Implemented by**: DrawingMetadata

**Methods**:

- `validate_age(cls: Any, v: Any)`

### DataPipelineServiceInterface

**Type**: Protocol
**Implemented by**: DataPipelineService

**Methods**:

- `validate_image(image_data: bytes, filename: Optional[str]) -> ValidationResult`
- `preprocess_image(image_data: bytes, target_size: <ast.Subscript object at 0x110387450>) -> np.ndarray`
- `extract_metadata(upload_data: Dict[str, Any]) -> DrawingMetadata`
- `validate_and_preprocess(image_data: bytes, metadata: Dict[str, Any]) -> Tuple[<ast.Attribute object at 0x11058eb50>, DrawingMetadata]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/data_pipeline.py`
- Last validated: 2025-12-16 15:47:04

