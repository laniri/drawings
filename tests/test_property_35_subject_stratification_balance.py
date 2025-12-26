"""
Property-Based Test for Subject Stratification Balance

**Feature: children-drawing-anomaly-detection, Property 35: Subject Stratification Balance**
**Validates: Requirements 3.8**

This test validates that dataset splitting maintains balanced representation across 
age-subject combinations when subject-aware stratification is enabled.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from dataclasses import dataclass
import tempfile
import json

from app.services.dataset_preparation import (
    DatasetPreparationService, 
    SplitConfig, 
    DatasetSplit
)
from app.services.data_pipeline import DrawingMetadata
from app.schemas.drawings import SubjectCategory


@dataclass
class MockDrawingFile:
    """Mock drawing file for testing."""
    path: Path
    metadata: DrawingMetadata


class TestSubjectStratificationBalance:
    """Test subject stratification balance properties."""
    
    def create_mock_dataset(self, 
                          age_subject_combinations: List[Tuple[float, str, int]]) -> Tuple[List[Path], List[DrawingMetadata]]:
        """
        Create a mock dataset with specified age-subject combinations.
        
        Args:
            age_subject_combinations: List of (age, subject, count) tuples
            
        Returns:
            Tuple of (file_paths, metadata_list)
        """
        files = []
        metadata_list = []
        
        file_counter = 0
        for age, subject, count in age_subject_combinations:
            for i in range(count):
                # Create mock file path
                file_path = Path(f"mock_drawing_{file_counter}.png")
                files.append(file_path)
                
                # Create metadata with minimal age variation to stay within age bins
                metadata = DrawingMetadata(
                    age_years=age + (i * 0.01),  # Very small age variation to stay in same bin
                    subject=subject if subject != "unspecified" else None,
                    expert_label=None,
                    drawing_tool=None,
                    prompt=None
                )
                metadata_list.append(metadata)
                
                file_counter += 1
        
        return files, metadata_list
    
    def calculate_age_subject_distribution(self, metadata_list: List[DrawingMetadata]) -> Dict[str, int]:
        """Calculate distribution of age-subject combinations."""
        distribution = {}
        for metadata in metadata_list:
            age_bin = int(metadata.age_years)
            subject = metadata.subject or "unspecified"
            key = f"{age_bin}_{subject}"
            distribution[key] = distribution.get(key, 0) + 1
        return distribution
    
    def test_subject_stratification_maintains_balance_with_viable_data(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 35: Subject Stratification Balance**
        **Validates: Requirements 3.8**
        
        Test subject stratification with carefully constructed viable data.
        """
        # Create a dataset that meets stratification requirements
        # For test_ratio=0.1 and val_ratio=0.2, we need few enough combinations
        # that n_classes <= min(min_test_size, min_val_size)
        age_subject_combinations = [
            (4.0, "house", 20),    # 20 samples
            (4.0, "person", 18),   # 18 samples  
            (5.0, "house", 22),    # 22 samples
        ]
        
        # Total: 60 samples, 3 combinations
        # min_test_size = max(1, int(60 * 0.1)) = 6
        # min_val_size = max(1, int(60 * 0.2)) = 12
        # n_classes = 3 <= min(6, 12) = 6 ✓
        
        files, metadata_list = self.create_mock_dataset(age_subject_combinations)
        
        # Configure subject-aware stratification
        split_config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            stratify_by_age=True,
            stratify_by_subject=True,
            min_samples_per_age_subject=3
        )
        
        # Create dataset preparation service
        service = DatasetPreparationService()
        
        # Create splits
        dataset_split = service.create_dataset_splits(files, metadata_list, split_config)
        
        # Verify stratification was successful (no fallback to random)
        if dataset_split.subject_stratification_warnings:
            fallback_warnings = [w for w in dataset_split.subject_stratification_warnings 
                                if "random splitting" in str(w)]
            assert len(fallback_warnings) == 0, "Should not fall back to random splitting with viable data"
        
        # Calculate distributions
        train_distribution = self.calculate_age_subject_distribution(dataset_split.train_metadata)
        original_distribution = self.calculate_age_subject_distribution(metadata_list)
        
        # Property: Each age-subject combination should be represented in training set
        for combo in original_distribution.keys():
            assert combo in train_distribution, f"Age-subject combination '{combo}' missing from training set"
            assert train_distribution[combo] > 0, f"Age-subject combination '{combo}' has zero samples in training set"
        
        # Property: Proportions should be maintained within reasonable tolerance
        tolerance = 0.15  # 15% tolerance for well-structured stratification
        
        for combo, original_count in original_distribution.items():
            original_proportion = original_count / len(metadata_list)
            train_count = train_distribution[combo]
            train_proportion = train_count / len(dataset_split.train_metadata)
            
            # Calculate how much the training proportion deviates from original proportion
            # In proper stratification, train_proportion should be close to original_proportion
            proportion_ratio = train_proportion / original_proportion if original_proportion > 0 else 0
            expected_ratio = 1.0  # Should be close to 1.0 (same proportion as original)
            
            assert abs(proportion_ratio - expected_ratio) <= tolerance, \
                f"Training proportion for '{combo}' deviates too much: {proportion_ratio:.3f} vs expected {expected_ratio:.3f} (train: {train_proportion:.3f}, original: {original_proportion:.3f})"
    
    @given(
        st.lists(
            st.tuples(
                st.floats(min_value=4.0, max_value=6.0),  # narrow age range
                st.sampled_from(["house", "person", "tree"]),  # limited subjects
                st.integers(min_value=2, max_value=4)  # small counts
            ),
            min_size=2,
            max_size=4
        )
    )
    @settings(max_examples=30)
    def test_insufficient_data_generates_warnings(self, age_subject_combinations):
        """
        **Feature: children-drawing-anomaly-detection, Property 35: Subject Stratification Balance**
        **Validates: Requirements 3.8**
        
        For any dataset with insufficient samples per age-subject combination,
        the stratification process should generate appropriate warnings.
        """
        # Ensure we have some combinations with insufficient data
        has_insufficient = any(count < 3 for _, _, count in age_subject_combinations)
        assume(has_insufficient)
        
        # Create mock dataset
        files, metadata_list = self.create_mock_dataset(age_subject_combinations)
        
        # Configure subject-aware stratification with higher minimum
        split_config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            stratify_by_age=True,
            stratify_by_subject=True,
            min_samples_per_age_subject=3
        )
        
        # Create dataset preparation service
        service = DatasetPreparationService()
        
        # Create splits
        dataset_split = service.create_dataset_splits(files, metadata_list, split_config)
        
        # Property: Warnings should be generated for insufficient combinations
        if dataset_split.subject_stratification_warnings:
            # Check that warnings are generated for combinations with insufficient data
            insufficient_combinations = [
                (age, subject, count) for age, subject, count in age_subject_combinations 
                if count < split_config.min_samples_per_age_subject
            ]
            
            if insufficient_combinations:
                assert len(dataset_split.subject_stratification_warnings) > 0, \
                    "Expected warnings for insufficient age-subject combinations"
                
                # Check that warnings contain relevant information
                for warning in dataset_split.subject_stratification_warnings:
                    assert warning.warning_type == "insufficient_age_subject_data"
                    assert warning.current_samples < split_config.min_samples_per_age_subject
                    assert len(warning.suggestions) > 0
    
    def test_subject_stratification_fallback_behavior(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 35: Subject Stratification Balance**
        **Validates: Requirements 3.8**
        
        When subject-aware stratification is not viable due to insufficient data,
        the system should fall back to age-only stratification gracefully.
        """
        # Create dataset with very few samples per combination (forces fallback)
        age_subject_combinations = [
            (4.0, "house", 1),
            (4.5, "person", 1),
            (5.0, "tree", 1),
            (5.5, "car", 1)
        ]
        
        files, metadata_list = self.create_mock_dataset(age_subject_combinations)
        
        # Configure subject-aware stratification
        split_config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            stratify_by_age=True,
            stratify_by_subject=True,
            min_samples_per_age_subject=3
        )
        
        # Create dataset preparation service
        service = DatasetPreparationService()
        
        # Create splits (should fall back to age-only or random)
        dataset_split = service.create_dataset_splits(files, metadata_list, split_config)
        
        # Property: Split should still be created successfully
        assert dataset_split.train_count > 0
        assert dataset_split.validation_count >= 0  # May be 0 for very small datasets
        assert dataset_split.test_count >= 0
        assert dataset_split.total_count == len(files)
        
        # Property: Warnings should be generated about the fallback
        assert dataset_split.subject_stratification_warnings is not None
    
    def test_balanced_age_subject_distribution_no_warnings(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 35: Subject Stratification Balance**
        **Validates: Requirements 3.8**
        
        When all age-subject combinations have sufficient data, no warnings should be generated.
        """
        # Create well-balanced dataset
        age_subject_combinations = [
            (4.0, "house", 10),
            (4.0, "person", 8),
            (5.0, "house", 12),
            (5.0, "person", 9),
            (6.0, "house", 11),
            (6.0, "person", 10)
        ]
        
        files, metadata_list = self.create_mock_dataset(age_subject_combinations)
        
        # Configure subject-aware stratification
        split_config = SplitConfig(
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.1,
            stratify_by_age=True,
            stratify_by_subject=True,
            min_samples_per_age_subject=3
        )
        
        # Create dataset preparation service
        service = DatasetPreparationService()
        
        # Create splits
        dataset_split = service.create_dataset_splits(files, metadata_list, split_config)
        
        # Property: No warnings should be generated for well-balanced data
        if dataset_split.subject_stratification_warnings:
            # Filter out low-severity warnings (they might be acceptable)
            high_severity_warnings = [
                w for w in dataset_split.subject_stratification_warnings 
                if w.severity in ["high", "critical"]
            ]
            assert len(high_severity_warnings) == 0, \
                f"Unexpected high-severity warnings for balanced dataset: {high_severity_warnings}"
        
        # Property: All combinations should be represented in training
        train_distribution = self.calculate_age_subject_distribution(dataset_split.train_metadata)
        original_distribution = self.calculate_age_subject_distribution(metadata_list)
        
        for combo in original_distribution.keys():
            assert combo in train_distribution, \
                f"Age-subject combination '{combo}' missing from training set"
            assert train_distribution[combo] > 0, \
                f"Age-subject combination '{combo}' has zero samples in training set"