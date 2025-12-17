# Dataset Preparation Contract

## Overview
Service contract for Dataset Preparation (service)

**Source File**: `app/services/dataset_preparation.py`

## Interface Specification

### Classes

#### MetadataFormat

Supported metadata file formats.

**Inherits from**: str, Enum

#### DatasetSplit

Container for dataset split information.

**Attributes**:

- `train_files: List[Path]`
- `train_metadata: List[DrawingMetadata]`
- `validation_files: List[Path]`
- `validation_metadata: List[DrawingMetadata]`
- `test_files: List[Path]`
- `test_metadata: List[DrawingMetadata]`

#### SplitConfig

Configuration for dataset splitting.

**Attributes**:

- `train_ratio: float`
- `validation_ratio: float`
- `test_ratio: float`
- `random_seed: int`
- `stratify_by_age: bool`
- `age_group_size: float`

#### DatasetPreparationService

Service for preparing datasets for training.

## Methods

### train_count

**Signature**: `train_count() -> int`

**Type**: Property

**Returns**: `int`

### validation_count

**Signature**: `validation_count() -> int`

**Type**: Property

**Returns**: `int`

### test_count

**Signature**: `test_count() -> int`

**Type**: Property

**Returns**: `int`

### total_count

**Signature**: `total_count() -> int`

**Type**: Property

**Returns**: `int`

### load_dataset_from_folder

Load dataset from folder structure with metadata file.

Expected folder structure:
dataset_folder/
├── image1.png
├── image2.jpg
└── metadata.csv (or metadata.json)

Args:
    dataset_folder: Path to folder containing drawing images
    metadata_file: Path to metadata file (CSV or JSON)
    
Returns:
    Tuple of (image_files, metadata_list)
    
Raises:
    ValidationError: If dataset structure is invalid
    FileNotFoundError: If folder or metadata file doesn't exist

**Signature**: `load_dataset_from_folder(dataset_folder: Union[str, Path], metadata_file: Union[str, Path]) -> Tuple[<ast.Subscript object at 0x110476350>, <ast.Subscript object at 0x1104764d0>]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `dataset_folder` | `Union[str, Path]` | Parameter description |
| `metadata_file` | `Union[str, Path]` | Parameter description |

**Returns**: `Tuple[<ast.Subscript object at 0x110476350>, <ast.Subscript object at 0x1104764d0>]`

### create_dataset_splits

Split dataset into train/validation/test sets.

Args:
    files: List of image files
    metadata: List of metadata objects
    split_config: Configuration for splitting
    
Returns:
    DatasetSplit object containing the splits
    
Raises:
    ValidationError: If splitting fails

**Signature**: `create_dataset_splits(files: List[Path], metadata: List[DrawingMetadata], split_config: SplitConfig) -> DatasetSplit`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `files` | `List[Path]` | Parameter description |
| `metadata` | `List[DrawingMetadata]` | Parameter description |
| `split_config` | `SplitConfig` | Parameter description |

**Returns**: `DatasetSplit`

### prepare_dataset

Complete dataset preparation pipeline.

Args:
    dataset_folder: Path to folder containing drawing images
    metadata_file: Path to metadata file
    split_config: Optional split configuration
    
Returns:
    DatasetSplit object with train/validation/test splits

**Signature**: `prepare_dataset(dataset_folder: Union[str, Path], metadata_file: Union[str, Path], split_config: Optional[SplitConfig]) -> DatasetSplit`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `dataset_folder` | `Union[str, Path]` | Parameter description |
| `metadata_file` | `Union[str, Path]` | Parameter description |
| `split_config` | `Optional[SplitConfig]` | Parameter description |

**Returns**: `DatasetSplit`

### validate_dataset_for_training

Validate dataset split for training readiness.

Args:
    dataset_split: Dataset split to validate
    
Returns:
    Dictionary with validation results and warnings

**Signature**: `validate_dataset_for_training(dataset_split: DatasetSplit) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `dataset_split` | `DatasetSplit` | Parameter description |

**Returns**: `Dict[str, Any]`

## Dependencies

- `app.services.data_pipeline.DataPipelineService`
- `app.services.data_pipeline.DrawingMetadata`

## Defined Interfaces

### DatasetSplitInterface

**Type**: Protocol
**Implemented by**: DatasetSplit

**Methods**:

- `train_count() -> int`
- `validation_count() -> int`
- `test_count() -> int`
- `total_count() -> int`

### DatasetPreparationServiceInterface

**Type**: Protocol
**Implemented by**: DatasetPreparationService

**Methods**:

- `load_dataset_from_folder(dataset_folder: Union[str, Path], metadata_file: Union[str, Path]) -> Tuple[<ast.Subscript object at 0x110476350>, <ast.Subscript object at 0x1104764d0>]`
- `create_dataset_splits(files: List[Path], metadata: List[DrawingMetadata], split_config: SplitConfig) -> DatasetSplit`
- `prepare_dataset(dataset_folder: Union[str, Path], metadata_file: Union[str, Path], split_config: Optional[SplitConfig]) -> DatasetSplit`
- `validate_dataset_for_training(dataset_split: DatasetSplit) -> Dict[str, Any]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/dataset_preparation.py`
- Last validated: 2025-12-16 15:47:04

