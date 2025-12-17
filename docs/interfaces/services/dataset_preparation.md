# Dataset Preparation Service

Dataset Preparation Service for Children's Drawing Anomaly Detection System

This module provides utilities for preparing datasets for training, including
folder-based dataset loading, metadata parsing, and train/validation/test splitting.

## Class: MetadataFormat

Supported metadata file formats.

## Class: DatasetSplit

Container for dataset split information.

### train_count

**Signature**: `train_count()`

### validation_count

**Signature**: `validation_count()`

### test_count

**Signature**: `test_count()`

### total_count

**Signature**: `total_count()`

## Class: SplitConfig

Configuration for dataset splitting.

## Class: DatasetPreparationService

Service for preparing datasets for training.

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

**Signature**: `load_dataset_from_folder(dataset_folder, metadata_file)`

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

**Signature**: `create_dataset_splits(files, metadata, split_config)`

### prepare_dataset

Complete dataset preparation pipeline.

Args:
    dataset_folder: Path to folder containing drawing images
    metadata_file: Path to metadata file
    split_config: Optional split configuration
    
Returns:
    DatasetSplit object with train/validation/test splits

**Signature**: `prepare_dataset(dataset_folder, metadata_file, split_config)`

### validate_dataset_for_training

Validate dataset split for training readiness.

Args:
    dataset_split: Dataset split to validate
    
Returns:
    Dictionary with validation results and warnings

**Signature**: `validate_dataset_for_training(dataset_split)`

