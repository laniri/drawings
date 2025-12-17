"""
Dataset Preparation Service for Children's Drawing Anomaly Detection System

This module provides utilities for preparing datasets for training, including
folder-based dataset loading, metadata parsing, and train/validation/test splitting.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split

from app.core.exceptions import ValidationError
from app.services.data_pipeline import DataPipelineService, DrawingMetadata

logger = logging.getLogger(__name__)


class MetadataFormat(str, Enum):
    """Supported metadata file formats."""
    CSV = "csv"
    JSON = "json"


@dataclass
class DatasetSplit:
    """Container for dataset split information."""
    train_files: List[Path]
    train_metadata: List[DrawingMetadata]
    validation_files: List[Path]
    validation_metadata: List[DrawingMetadata]
    test_files: List[Path]
    test_metadata: List[DrawingMetadata]
    
    @property
    def train_count(self) -> int:
        return len(self.train_files)
    
    @property
    def validation_count(self) -> int:
        return len(self.validation_files)
    
    @property
    def test_count(self) -> int:
        return len(self.test_files)
    
    @property
    def total_count(self) -> int:
        return self.train_count + self.validation_count + self.test_count


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    train_ratio: float = 0.7
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    random_seed: int = 42
    stratify_by_age: bool = True
    age_group_size: float = 1.0  # Age group size in years for stratification
    
    def __post_init__(self):
        """Validate split ratios sum to 1.0."""
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


class DatasetPreparationService:
    """Service for preparing datasets for training."""
    
    SUPPORTED_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    def __init__(self, data_pipeline: Optional[DataPipelineService] = None):
        """
        Initialize the dataset preparation service.
        
        Args:
            data_pipeline: Optional data pipeline service for validation
        """
        self.data_pipeline = data_pipeline or DataPipelineService()
        logger.info("DatasetPreparationService initialized")
    
    def load_dataset_from_folder(
        self, 
        dataset_folder: Union[str, Path], 
        metadata_file: Union[str, Path]
    ) -> Tuple[List[Path], List[DrawingMetadata]]:
        """
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
        """
        dataset_path = Path(dataset_folder)
        metadata_path = Path(metadata_file)
        
        # Validate paths exist
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        logger.info(f"Loading dataset from {dataset_path} with metadata {metadata_path}")
        
        # Load metadata
        metadata_dict = self._load_metadata_file(metadata_path)
        
        # Find image files in dataset folder
        image_files = []
        for ext in self.SUPPORTED_IMAGE_EXTENSIONS:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValidationError(f"No image files found in {dataset_path}")
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Match images with metadata
        matched_files, matched_metadata = self._match_files_with_metadata(
            image_files, metadata_dict
        )
        
        # Validate matched data
        self._validate_dataset(matched_files, matched_metadata)
        
        logger.info(f"Successfully loaded dataset: {len(matched_files)} files with metadata")
        return matched_files, matched_metadata
    
    def _load_metadata_file(self, metadata_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load metadata from CSV or JSON file.
        
        Args:
            metadata_path: Path to metadata file
            
        Returns:
            Dictionary mapping filename to metadata
            
        Raises:
            ValidationError: If metadata format is invalid
        """
        try:
            if metadata_path.suffix.lower() == '.csv':
                return self._load_csv_metadata(metadata_path)
            elif metadata_path.suffix.lower() == '.json':
                return self._load_json_metadata(metadata_path)
            else:
                raise ValidationError(f"Unsupported metadata format: {metadata_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {str(e)}")
            raise ValidationError(f"Invalid metadata file: {str(e)}")
    
    def _load_csv_metadata(self, csv_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load metadata from CSV file."""
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_columns = {'filename', 'age_years'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                raise ValidationError(f"Missing required columns in CSV: {missing}")
            
            # Convert to dictionary
            metadata_dict = {}
            for _, row in df.iterrows():
                filename = row['filename']
                metadata = {
                    'age_years': float(row['age_years']),
                    'subject': row.get('subject'),
                    'expert_label': row.get('expert_label'),
                    'drawing_tool': row.get('drawing_tool'),
                    'prompt': row.get('prompt')
                }
                # Remove None values
                metadata = {k: v for k, v in metadata.items() if pd.notna(v)}
                metadata_dict[filename] = metadata
            
            return metadata_dict
            
        except Exception as e:
            raise ValidationError(f"Failed to parse CSV metadata: {str(e)}")
    
    def _load_json_metadata(self, json_path: Path) -> Dict[str, Dict[str, Any]]:
        """Load metadata from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of objects with filename field
                metadata_dict = {}
                for item in data:
                    if 'filename' not in item:
                        raise ValidationError("JSON list items must have 'filename' field")
                    filename = item.pop('filename')
                    metadata_dict[filename] = item
                return metadata_dict
            elif isinstance(data, dict):
                # Dictionary mapping filename to metadata
                return data
            else:
                raise ValidationError("JSON must be a list or dictionary")
                
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValidationError(f"Failed to parse JSON metadata: {str(e)}")
    
    def _match_files_with_metadata(
        self, 
        image_files: List[Path], 
        metadata_dict: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[Path], List[DrawingMetadata]]:
        """
        Match image files with their metadata.
        
        Args:
            image_files: List of image file paths
            metadata_dict: Dictionary mapping filename to metadata
            
        Returns:
            Tuple of (matched_files, matched_metadata)
        """
        matched_files = []
        matched_metadata = []
        
        for image_file in image_files:
            filename = image_file.name
            
            # Try exact match first
            if filename in metadata_dict:
                metadata = metadata_dict[filename]
            else:
                # Try without extension
                stem = image_file.stem
                matching_keys = [k for k in metadata_dict.keys() 
                               if Path(k).stem == stem]
                if len(matching_keys) == 1:
                    metadata = metadata_dict[matching_keys[0]]
                elif len(matching_keys) > 1:
                    logger.warning(f"Multiple metadata entries for {filename}, using first match")
                    metadata = metadata_dict[matching_keys[0]]
                else:
                    logger.warning(f"No metadata found for {filename}, skipping")
                    continue
            
            try:
                # Validate and create metadata object
                drawing_metadata = DrawingMetadata(**metadata)
                matched_files.append(image_file)
                matched_metadata.append(drawing_metadata)
            except Exception as e:
                logger.warning(f"Invalid metadata for {filename}: {str(e)}, skipping")
                continue
        
        return matched_files, matched_metadata
    
    def _validate_dataset(self, files: List[Path], metadata: List[DrawingMetadata]) -> None:
        """
        Validate dataset consistency and quality.
        
        Args:
            files: List of image files
            metadata: List of metadata objects
            
        Raises:
            ValidationError: If dataset validation fails
        """
        if len(files) != len(metadata):
            raise ValidationError("Mismatch between number of files and metadata entries")
        
        if len(files) == 0:
            raise ValidationError("No valid files found in dataset")
        
        # Check age distribution
        ages = [m.age_years for m in metadata]
        age_range = max(ages) - min(ages)
        
        logger.info(f"Dataset validation: {len(files)} files, age range {min(ages):.1f}-{max(ages):.1f} years")
        
        # Warn about potential issues
        if len(files) < 50:
            logger.warning(f"Small dataset size: {len(files)} files (recommended: 50+)")
        
        if age_range < 1.0:
            logger.warning(f"Narrow age range: {age_range:.1f} years")
    
    def create_dataset_splits(
        self, 
        files: List[Path], 
        metadata: List[DrawingMetadata], 
        split_config: SplitConfig
    ) -> DatasetSplit:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            files: List of image files
            metadata: List of metadata objects
            split_config: Configuration for splitting
            
        Returns:
            DatasetSplit object containing the splits
            
        Raises:
            ValidationError: If splitting fails
        """
        if len(files) != len(metadata):
            raise ValidationError("Files and metadata lists must have same length")
        
        if len(files) == 0:
            raise ValidationError("Cannot split empty dataset")
        
        logger.info(f"Creating dataset splits with config: train={split_config.train_ratio}, "
                   f"val={split_config.validation_ratio}, test={split_config.test_ratio}")
        
        # Prepare data for splitting
        indices = np.arange(len(files))
        
        # Create stratification labels if requested
        stratify_labels = None
        if split_config.stratify_by_age and len(files) >= 10:  # Need minimum samples for stratification
            ages = [m.age_years for m in metadata]
            # Group ages into bins for stratification
            age_bins = np.arange(
                min(ages), 
                max(ages) + split_config.age_group_size, 
                split_config.age_group_size
            )
            stratify_labels = np.digitize(ages, age_bins)
            
            # Check if we have enough samples per stratum for stratified splitting
            unique_labels, counts = np.unique(stratify_labels, return_counts=True)
            min_count = min(counts)
            n_classes = len(unique_labels)
            
            # Calculate minimum samples needed for stratified split
            min_test_size = max(1, int(len(files) * split_config.test_ratio))
            min_val_size = max(1, int(len(files) * split_config.validation_ratio))
            
            # Need at least 2 samples per class and enough samples for each split
            if (min_count < 2 or 
                n_classes > min_test_size or 
                n_classes > min_val_size or
                min_count < 3):  # Need at least 3 samples per stratum for reliable splitting
                logger.warning(f"Insufficient samples for age stratification: {n_classes} classes, "
                             f"min_count={min_count}, test_size={min_test_size}, val_size={min_val_size}. "
                             f"Using random split")
                stratify_labels = None
        
        try:
            # First split: separate test set
            if split_config.test_ratio > 0:
                try:
                    train_val_indices, test_indices = train_test_split(
                        indices,
                        test_size=split_config.test_ratio,
                        random_state=split_config.random_seed,
                        stratify=stratify_labels
                    )
                except ValueError as e:
                    if "test_size" in str(e) and "classes" in str(e):
                        # Stratification failed, fall back to random split
                        logger.warning(f"Stratified split failed ({str(e)}), using random split")
                        train_val_indices, test_indices = train_test_split(
                            indices,
                            test_size=split_config.test_ratio,
                            random_state=split_config.random_seed,
                            stratify=None
                        )
                        stratify_labels = None  # Disable stratification for subsequent splits
                    else:
                        raise
            else:
                train_val_indices = indices
                test_indices = np.array([])
            
            # Second split: separate train and validation
            if split_config.validation_ratio > 0 and len(train_val_indices) > 1:
                # Adjust validation ratio for remaining data
                val_ratio_adjusted = split_config.validation_ratio / (
                    split_config.train_ratio + split_config.validation_ratio
                )
                
                # Prepare stratification for remaining data
                remaining_stratify = None
                if stratify_labels is not None:
                    remaining_stratify = stratify_labels[train_val_indices]
                
                try:
                    train_indices, val_indices = train_test_split(
                        train_val_indices,
                        test_size=val_ratio_adjusted,
                        random_state=split_config.random_seed,
                        stratify=remaining_stratify
                    )
                except ValueError as e:
                    if "test_size" in str(e) and "classes" in str(e):
                        # Stratification failed, fall back to random split
                        logger.warning(f"Validation stratified split failed ({str(e)}), using random split")
                        train_indices, val_indices = train_test_split(
                            train_val_indices,
                            test_size=val_ratio_adjusted,
                            random_state=split_config.random_seed,
                            stratify=None
                        )
                    else:
                        raise
            else:
                train_indices = train_val_indices
                val_indices = np.array([])
            
            # Create split data
            dataset_split = DatasetSplit(
                train_files=[files[i] for i in train_indices],
                train_metadata=[metadata[i] for i in train_indices],
                validation_files=[files[i] for i in val_indices],
                validation_metadata=[metadata[i] for i in val_indices],
                test_files=[files[i] for i in test_indices],
                test_metadata=[metadata[i] for i in test_indices]
            )
            
            logger.info(f"Dataset split created: train={dataset_split.train_count}, "
                       f"val={dataset_split.validation_count}, test={dataset_split.test_count}")
            
            return dataset_split
            
        except Exception as e:
            logger.error(f"Failed to create dataset splits: {str(e)}")
            raise ValidationError(f"Dataset splitting failed: {str(e)}")
    
    def prepare_dataset(
        self, 
        dataset_folder: Union[str, Path], 
        metadata_file: Union[str, Path],
        split_config: Optional[SplitConfig] = None
    ) -> DatasetSplit:
        """
        Complete dataset preparation pipeline.
        
        Args:
            dataset_folder: Path to folder containing drawing images
            metadata_file: Path to metadata file
            split_config: Optional split configuration
            
        Returns:
            DatasetSplit object with train/validation/test splits
        """
        if split_config is None:
            split_config = SplitConfig()
        
        # Load dataset
        files, metadata = self.load_dataset_from_folder(dataset_folder, metadata_file)
        
        # Create splits
        dataset_split = self.create_dataset_splits(files, metadata, split_config)
        
        return dataset_split
    
    def validate_dataset_for_training(self, dataset_split: DatasetSplit) -> Dict[str, Any]:
        """
        Validate dataset split for training readiness.
        
        Args:
            dataset_split: Dataset split to validate
            
        Returns:
            Dictionary with validation results and warnings
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'statistics': {},
            'recommendations': []
        }
        
        # Check minimum sample counts
        min_train_samples = 20
        min_val_samples = 5
        
        if dataset_split.train_count < min_train_samples:
            validation_result['warnings'].append(
                f"Training set has only {dataset_split.train_count} samples "
                f"(recommended: {min_train_samples}+)"
            )
        
        if dataset_split.validation_count < min_val_samples:
            validation_result['warnings'].append(
                f"Validation set has only {dataset_split.validation_count} samples "
                f"(recommended: {min_val_samples}+)"
            )
        
        # Analyze age distributions
        train_ages = [m.age_years for m in dataset_split.train_metadata]
        val_ages = [m.age_years for m in dataset_split.validation_metadata]
        
        validation_result['statistics'] = {
            'total_samples': dataset_split.total_count,
            'train_samples': dataset_split.train_count,
            'validation_samples': dataset_split.validation_count,
            'test_samples': dataset_split.test_count,
            'age_range': {
                'min': min(train_ages + val_ages),
                'max': max(train_ages + val_ages),
                'mean': np.mean(train_ages + val_ages)
            }
        }
        
        # Check age distribution overlap
        train_age_range = (min(train_ages), max(train_ages))
        val_age_range = (min(val_ages), max(val_ages))
        
        if not (train_age_range[0] <= val_age_range[1] and val_age_range[0] <= train_age_range[1]):
            validation_result['warnings'].append(
                "Training and validation sets have non-overlapping age ranges"
            )
        
        # Generate recommendations
        if dataset_split.total_count < 100:
            validation_result['recommendations'].append(
                "Consider collecting more data for better model performance"
            )
        
        if len(validation_result['warnings']) > 0:
            validation_result['is_valid'] = False
        
        return validation_result


# Global service instance
_dataset_preparation_service = None


def get_dataset_preparation_service() -> DatasetPreparationService:
    """Get the global dataset preparation service instance."""
    global _dataset_preparation_service
    if _dataset_preparation_service is None:
        _dataset_preparation_service = DatasetPreparationService()
    return _dataset_preparation_service