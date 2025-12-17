"""
Property-based tests for dataset preparation service.

**Feature: children-drawing-anomaly-detection, Property 7: Training Data Split Consistency**
**Validates: Requirements 3.1**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import json
import csv
import os
from typing import List, Dict, Any

from app.services.dataset_preparation import (
    DatasetPreparationService, 
    SplitConfig, 
    DatasetSplit,
    MetadataFormat
)
from app.services.data_pipeline import DrawingMetadata
from app.core.exceptions import ValidationError


def create_test_image_file(temp_dir: Path, filename: str) -> Path:
    """Create a minimal test image file."""
    from PIL import Image
    
    # Create a simple 100x100 RGB image
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    
    # Add some pattern
    pixels = image.load()
    for i in range(0, 100, 10):
        for j in range(0, 100, 10):
            pixels[i, j] = (255, 0, 0)
    
    file_path = temp_dir / filename
    image.save(file_path, 'PNG')
    return file_path


def create_test_dataset(
    temp_dir: Path, 
    num_files: int, 
    age_range: tuple = (3.0, 12.0),
    metadata_format: str = 'csv'
) -> tuple:
    """Create a test dataset with specified number of files."""
    
    # Generate file names and metadata
    files_data = []
    for i in range(num_files):
        filename = f"drawing_{i:03d}.png"
        age = np.random.uniform(age_range[0], age_range[1])
        
        # Create image file
        create_test_image_file(temp_dir, filename)
        
        # Create metadata entry
        metadata = {
            'filename': filename,
            'age_years': age,
            'subject': np.random.choice(['person', 'house', 'tree', None]),
            'expert_label': np.random.choice(['normal', 'concern', None]),
        }
        files_data.append(metadata)
    
    # Create metadata file
    if metadata_format == 'csv':
        metadata_file = temp_dir / 'metadata.csv'
        with open(metadata_file, 'w', newline='') as f:
            if files_data:
                writer = csv.DictWriter(f, fieldnames=files_data[0].keys())
                writer.writeheader()
                writer.writerows(files_data)
    else:  # json
        metadata_file = temp_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(files_data, f)
    
    return temp_dir, metadata_file


# Hypothesis strategies
dataset_size_strategy = st.integers(min_value=20, max_value=100)

# Generate valid split ratios that sum to 1.0
@st.composite
def split_ratios_strategy(draw):
    # Generate two ratios and calculate the third
    train_ratio = draw(st.floats(min_value=0.5, max_value=0.8))
    val_ratio = draw(st.floats(min_value=0.1, max_value=0.3))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Ensure test ratio is reasonable
    if test_ratio < 0.05 or test_ratio > 0.4:
        # Adjust ratios to be valid
        test_ratio = 0.1
        val_ratio = 0.2
        train_ratio = 0.7
    
    return (train_ratio, val_ratio, test_ratio)

age_range_strategy = st.tuples(
    st.floats(min_value=3.0, max_value=8.0),  # min_age - narrower range
    st.floats(min_value=9.0, max_value=15.0)  # max_age - narrower range
)

metadata_format_strategy = st.sampled_from(['csv', 'json'])


@given(
    dataset_size=dataset_size_strategy,
    split_ratios=split_ratios_strategy(),
    age_range=age_range_strategy,
    metadata_format=metadata_format_strategy,
    random_seed=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=50, deadline=None)
def test_training_data_split_consistency(dataset_size, split_ratios, age_range, metadata_format, random_seed):
    """
    **Feature: children-drawing-anomaly-detection, Property 7: Training Data Split Consistency**
    **Validates: Requirements 3.1**
    
    Property: For any dataset with specified split ratios, the training environment should 
    create splits with correct proportions and no data overlap between splits.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # Normalize ratios to ensure they sum to exactly 1.0
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test dataset
        dataset_folder, metadata_file = create_test_dataset(
            temp_path, dataset_size, age_range, metadata_format
        )
        
        # Create split configuration
        split_config = SplitConfig(
            train_ratio=train_ratio,
            validation_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed,
            stratify_by_age=True
        )
        
        # Initialize service and prepare dataset
        service = DatasetPreparationService()
        dataset_split = service.prepare_dataset(dataset_folder, metadata_file, split_config)
        
        # Property 1: Split proportions should match configuration (within tolerance)
        total_files = dataset_split.total_count
        expected_train = int(total_files * train_ratio)
        expected_val = int(total_files * val_ratio)
        expected_test = total_files - expected_train - expected_val  # Remainder goes to test
        
        # Allow for rounding differences (±1 file per split)
        assert abs(dataset_split.train_count - expected_train) <= 1, \
            f"Train split size {dataset_split.train_count} not close to expected {expected_train}"
        
        assert abs(dataset_split.validation_count - expected_val) <= 1, \
            f"Validation split size {dataset_split.validation_count} not close to expected {expected_val}"
        
        assert abs(dataset_split.test_count - expected_test) <= 1, \
            f"Test split size {dataset_split.test_count} not close to expected {expected_test}"
        
        # Property 2: No data overlap between splits
        train_files = set(f.name for f in dataset_split.train_files)
        val_files = set(f.name for f in dataset_split.validation_files)
        test_files = set(f.name for f in dataset_split.test_files)
        
        # Check no overlap between any splits
        assert len(train_files & val_files) == 0, "Training and validation sets should not overlap"
        assert len(train_files & test_files) == 0, "Training and test sets should not overlap"
        assert len(val_files & test_files) == 0, "Validation and test sets should not overlap"
        
        # Property 3: All original files should be accounted for
        all_split_files = train_files | val_files | test_files
        assert len(all_split_files) == total_files, \
            f"Split files count {len(all_split_files)} doesn't match total {total_files}"
        
        # Property 4: Metadata should match files
        assert len(dataset_split.train_files) == len(dataset_split.train_metadata), \
            "Train files and metadata counts should match"
        assert len(dataset_split.validation_files) == len(dataset_split.validation_metadata), \
            "Validation files and metadata counts should match"
        assert len(dataset_split.test_files) == len(dataset_split.test_metadata), \
            "Test files and metadata counts should match"
        
        # Property 5: Age ranges should be reasonable across splits
        if dataset_split.train_count > 0:
            train_ages = [m.age_years for m in dataset_split.train_metadata]
            assert all(age_range[0] <= age <= age_range[1] for age in train_ages), \
                "All training ages should be within expected range"
        
        if dataset_split.validation_count > 0:
            val_ages = [m.age_years for m in dataset_split.validation_metadata]
            assert all(age_range[0] <= age <= age_range[1] for age in val_ages), \
                "All validation ages should be within expected range"


@given(
    dataset_size=st.integers(min_value=5, max_value=50),
    random_seed1=st.integers(min_value=1, max_value=1000),
    random_seed2=st.integers(min_value=1, max_value=1000)
)
@settings(max_examples=50, deadline=None)
def test_split_reproducibility_with_same_seed(dataset_size, random_seed1, random_seed2):
    """
    Property: Splits with the same random seed should be identical, 
    splits with different seeds should be different (with high probability).
    """
    assume(random_seed1 != random_seed2)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test dataset
        dataset_folder, metadata_file = create_test_dataset(
            temp_path, dataset_size, (3.0, 12.0), 'csv'
        )
        
        service = DatasetPreparationService()
        
        # Create two splits with same seed
        split_config1 = SplitConfig(random_seed=random_seed1)
        split1a = service.prepare_dataset(dataset_folder, metadata_file, split_config1)
        split1b = service.prepare_dataset(dataset_folder, metadata_file, split_config1)
        
        # Create split with different seed
        split_config2 = SplitConfig(random_seed=random_seed2)
        split2 = service.prepare_dataset(dataset_folder, metadata_file, split_config2)
        
        # Same seed should produce identical splits
        train_files1a = set(f.name for f in split1a.train_files)
        train_files1b = set(f.name for f in split1b.train_files)
        assert train_files1a == train_files1b, "Same seed should produce identical train splits"
        
        # Different seeds should produce different splits (with high probability for reasonable dataset sizes)
        if dataset_size >= 10:
            train_files2 = set(f.name for f in split2.train_files)
            # Allow for some overlap, but expect significant difference
            overlap_ratio = len(train_files1a & train_files2) / len(train_files1a)
            assert overlap_ratio < 0.9, "Different seeds should produce different splits"


def test_split_config_validation():
    """Test that split configuration validates ratios sum to 1.0."""
    
    # Valid configuration
    config = SplitConfig(train_ratio=0.7, validation_ratio=0.2, test_ratio=0.1)
    assert abs(config.train_ratio + config.validation_ratio + config.test_ratio - 1.0) < 0.01
    
    # Invalid configuration - doesn't sum to 1.0
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitConfig(train_ratio=0.5, validation_ratio=0.3, test_ratio=0.3)  # sums to 1.1
    
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        SplitConfig(train_ratio=0.4, validation_ratio=0.2, test_ratio=0.2)  # sums to 0.8


def test_empty_dataset_handling():
    """Test that empty datasets are handled gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create empty metadata file
        metadata_file = temp_path / 'metadata.csv'
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'age_years'])  # Header only
        
        service = DatasetPreparationService()
        
        with pytest.raises(ValidationError, match="No image files found"):
            service.prepare_dataset(temp_path, metadata_file)


def test_missing_metadata_handling():
    """Test handling of files without metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create image file
        create_test_image_file(temp_path, 'test.png')
        
        # Create metadata for different file
        metadata_file = temp_path / 'metadata.csv'
        with open(metadata_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'age_years'])
            writer.writerow(['other.png', 5.0])  # Different filename
        
        service = DatasetPreparationService()
        
        with pytest.raises(ValidationError, match="No valid files found"):
            service.prepare_dataset(temp_path, metadata_file)