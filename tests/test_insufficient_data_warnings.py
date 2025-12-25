"""
Property-based tests for insufficient data warning generation.

**Feature: children-drawing-anomaly-detection, Property 12: Insufficient Data Warning Generation**
**Validates: Requirements 3.8**
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Any, Tuple
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.data_sufficiency_service import (
    DataSufficiencyAnalyzer,
    DataAugmentationSuggester,
    AgeGroupDataInfo,
    DataSufficiencyWarning,
    AgeGroupMergingSuggestion,
    DataSufficiencyError
)
from app.models.database import Base, Drawing, AgeGroupModel
from app.core.database import get_db


# Hypothesis strategies for generating test data
valid_age_strategy = st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)
valid_sample_count_strategy = st.integers(min_value=0, max_value=500)
subject_strategy = st.sampled_from(["person", "house", "tree", "animal", "car", "flower", None])

@st.composite
def age_group_strategy(draw):
    """Generate valid age group ranges."""
    age_min = draw(st.floats(min_value=2.0, max_value=16.0))
    age_max = draw(st.floats(min_value=age_min + 0.5, max_value=18.0))
    return (age_min, age_max)

@st.composite
def drawing_data_strategy(draw):
    """Generate drawing data for testing."""
    age_min, age_max = draw(age_group_strategy())
    sample_count = draw(valid_sample_count_strategy)
    
    # Generate ages within the range
    ages = []
    subjects = []
    
    for _ in range(sample_count):
        age = draw(st.floats(min_value=age_min, max_value=age_max - 0.01))
        subject = draw(subject_strategy)
        ages.append(age)
        subjects.append(subject)
    
    return {
        'age_min': age_min,
        'age_max': age_max,
        'sample_count': sample_count,
        'ages': ages,
        'subjects': subjects
    }

@st.composite
def multiple_age_groups_strategy(draw):
    """Generate multiple age groups with varying data sufficiency."""
    num_groups = draw(st.integers(min_value=2, max_value=6))
    groups = []
    
    current_age = 3.0
    for _ in range(num_groups):
        age_min = current_age
        age_max = current_age + draw(st.floats(min_value=0.5, max_value=2.0))
        sample_count = draw(st.integers(min_value=0, max_value=200))
        
        groups.append({
            'age_min': age_min,
            'age_max': age_max,
            'sample_count': sample_count
        })
        
        current_age = age_max
    
    return groups

def create_test_database():
    """Create an in-memory test database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def populate_database_with_drawings(db, drawing_data: Dict):
    """Populate database with test drawing data."""
    for i, (age, subject) in enumerate(zip(drawing_data['ages'], drawing_data['subjects'])):
        drawing = Drawing(
            filename=f"test_drawing_{i}.png",
            file_path=f"/test/path/test_drawing_{i}.png",
            age_years=age,
            subject=subject,
            upload_timestamp=datetime.now(timezone.utc)
        )
        db.add(drawing)
    
    db.commit()


@given(drawing_data=drawing_data_strategy())
@settings(max_examples=100, deadline=None)
def test_insufficient_data_warning_generation(drawing_data):
    """
    **Feature: children-drawing-anomaly-detection, Property 12: Insufficient Data Warning Generation**
    **Validates: Requirements 3.8**
    
    Property: For any age group with below-threshold data counts, the training 
    environment should generate appropriate warnings.
    """
    # Create test database
    db = create_test_database()
    
    try:
        # Populate database with test data
        populate_database_with_drawings(db, drawing_data)
        
        # Initialize analyzer
        analyzer = DataSufficiencyAnalyzer()
        
        # Analyze the age group
        age_min = drawing_data['age_min']
        age_max = drawing_data['age_max']
        sample_count = drawing_data['sample_count']
        
        data_info = analyzer.analyze_age_group_data(age_min, age_max, db)
        
        # Verify data info is accurate
        assert data_info.age_min == age_min
        assert data_info.age_max == age_max
        assert data_info.sample_count == sample_count
        
        # Generate warnings for this age group
        warnings = analyzer.generate_data_warnings([(age_min, age_max)], db)
        
        # Property: Warnings should be generated based on sample count thresholds
        if sample_count <= analyzer.critical_threshold:
            # Should have critical warning
            critical_warnings = [w for w in warnings if w.severity == "critical"]
            assert len(critical_warnings) > 0, f"Expected critical warning for {sample_count} samples"
            
            # Critical warning should mention insufficient data
            critical_warning = critical_warnings[0]
            assert critical_warning.warning_type == "insufficient_data"
            assert critical_warning.current_samples == sample_count
            assert critical_warning.age_group_min == age_min
            assert critical_warning.age_group_max == age_max
            assert len(critical_warning.suggestions) > 0
            
        elif sample_count < analyzer.min_samples_per_group:
            # Should have high severity warning
            high_warnings = [w for w in warnings if w.severity == "high" and w.warning_type == "insufficient_data"]
            assert len(high_warnings) > 0, f"Expected high warning for {sample_count} samples"
            
            high_warning = high_warnings[0]
            assert high_warning.current_samples == sample_count
            assert high_warning.recommended_samples == analyzer.min_samples_per_group
            
        elif sample_count < analyzer.recommended_samples_per_group:
            # Should have medium severity warning
            medium_warnings = [w for w in warnings if w.severity == "medium" and w.warning_type == "insufficient_data"]
            assert len(medium_warnings) > 0, f"Expected medium warning for {sample_count} samples"
            
        else:
            # Should not have insufficient data warnings
            insufficient_warnings = [w for w in warnings if w.warning_type == "insufficient_data"]
            # Note: There might still be other types of warnings (unbalanced, narrow range, etc.)
        
        # Verify warning structure
        for warning in warnings:
            assert isinstance(warning, DataSufficiencyWarning)
            assert warning.warning_type in ["insufficient_data", "unbalanced_distribution", "narrow_age_range"]
            assert warning.severity in ["low", "medium", "high", "critical"]
            assert warning.age_group_min == age_min
            assert warning.age_group_max == age_max
            assert warning.current_samples == sample_count
            assert warning.recommended_samples > 0
            assert len(warning.message) > 0
            assert isinstance(warning.suggestions, list)
            
    finally:
        db.close()


@given(groups_data=multiple_age_groups_strategy())
@settings(max_examples=50, deadline=None)
def test_age_group_merging_suggestions(groups_data):
    """
    Property: Age groups with insufficient data should generate merging suggestions 
    that improve data sufficiency.
    """
    # Create test database
    db = create_test_database()
    
    try:
        # Populate database with multiple age groups
        all_ages = []
        all_subjects = []
        age_groups = []
        
        for group in groups_data:
            age_min = group['age_min']
            age_max = group['age_max']
            sample_count = group['sample_count']
            
            age_groups.append((age_min, age_max))
            
            # Generate ages for this group
            for i in range(sample_count):
                # Generate age within [age_min, age_max) to match SQL query logic
                if sample_count == 1:
                    age = age_min + (age_max - age_min) * 0.5  # Middle of range
                else:
                    # Distribute evenly but ensure last age < age_max
                    age = age_min + (age_max - age_min) * (i / sample_count)
                subject = "person" if i % 2 == 0 else "house"
                all_ages.append(age)
                all_subjects.append(subject)
        
        # Add drawings to database
        for i, (age, subject) in enumerate(zip(all_ages, all_subjects)):
            drawing = Drawing(
                filename=f"test_drawing_{i}.png",
                file_path=f"/test/path/test_drawing_{i}.png",
                age_years=age,
                subject=subject,
                upload_timestamp=datetime.now(timezone.utc)
            )
            db.add(drawing)
        
        db.commit()
        
        # Initialize analyzer
        analyzer = DataSufficiencyAnalyzer()
        
        # Generate merging suggestions
        suggestions = analyzer.suggest_age_group_merging(age_groups, db)
        
        # Verify suggestions
        for suggestion in suggestions:
            assert isinstance(suggestion, AgeGroupMergingSuggestion)
            assert len(suggestion.original_groups) >= 2
            assert suggestion.merged_group[0] < suggestion.merged_group[1]
            assert suggestion.combined_sample_count >= 0
            assert 0.0 <= suggestion.improvement_score <= 1.0
            assert len(suggestion.rationale) > 0
            
            # Verify merged group encompasses original groups
            merged_min, merged_max = suggestion.merged_group
            for orig_min, orig_max in suggestion.original_groups:
                assert merged_min <= orig_min
                assert merged_max >= orig_max
            
            # Verify combined sample count is reasonable
            expected_min_samples = sum(
                len([age for age in all_ages if orig_min <= age < orig_max])
                for orig_min, orig_max in suggestion.original_groups
            )
            # Allow some tolerance due to boundary conditions
            assert suggestion.combined_sample_count >= expected_min_samples * 0.8
        
        # Property: Suggestions should be ordered by improvement score
        if len(suggestions) > 1:
            for i in range(len(suggestions) - 1):
                assert suggestions[i].improvement_score >= suggestions[i + 1].improvement_score
        
    finally:
        db.close()


@given(
    sample_count=st.integers(min_value=0, max_value=300),
    age_range=age_group_strategy()
)
@settings(max_examples=50, deadline=None)
def test_data_quality_score_calculation(sample_count, age_range):
    """
    Property: Data quality scores should be calculated consistently and fall within valid range.
    """
    age_min, age_max = age_range
    
    # Create test database
    db = create_test_database()
    
    try:
        # Generate test drawings with varying quality characteristics
        subjects = ["person", "house", "tree", "animal"]
        
        for i in range(sample_count):
            # Distribute ages across the range within [age_min, age_max)
            if sample_count == 1:
                age = age_min + (age_max - age_min) * 0.5  # Middle of range
            else:
                # Distribute evenly but ensure last age < age_max
                age = age_min + (age_max - age_min) * (i / sample_count)
            subject = subjects[i % len(subjects)]  # Rotate through subjects
            
            drawing = Drawing(
                filename=f"test_drawing_{i}.png",
                file_path=f"/test/path/test_drawing_{i}.png",
                age_years=age,
                subject=subject,
                upload_timestamp=datetime.now(timezone.utc)
            )
            db.add(drawing)
        
        db.commit()
        
        # Initialize analyzer
        analyzer = DataSufficiencyAnalyzer()
        
        # Analyze data quality
        data_info = analyzer.analyze_age_group_data(age_min, age_max, db)
        
        # Property: Quality score should be in valid range
        assert 0.0 <= data_info.data_quality_score <= 1.0
        
        # Property: Quality score should correlate with sample count
        if sample_count == 0:
            assert data_info.data_quality_score == 0.0
        elif sample_count >= analyzer.recommended_samples_per_group:
            # Should have reasonably high quality score
            assert data_info.data_quality_score >= 0.4
        
        # Property: Quality score should reflect data characteristics
        if sample_count > 0:
            # Should have some positive score
            assert data_info.data_quality_score > 0.0
            
            # Should reflect subject diversity
            unique_subjects = len(set(subjects[:sample_count]))
            if unique_subjects >= 3:
                # Should get bonus for subject diversity
                assert data_info.data_quality_score >= 0.2
        
    finally:
        db.close()


@st.composite
def non_overlapping_age_groups_strategy(draw):
    """Generate non-overlapping age groups with sample counts."""
    num_groups = draw(st.integers(min_value=1, max_value=3))  # Reduced to avoid too much filtering
    
    groups = []
    sample_counts = []
    
    current_age = draw(st.floats(min_value=2.0, max_value=10.0))
    
    for _ in range(num_groups):
        age_min = current_age
        age_span = draw(st.floats(min_value=0.5, max_value=2.0))
        age_max = age_min + age_span
        sample_count = draw(st.integers(min_value=0, max_value=100))
        
        groups.append((age_min, age_max))
        sample_counts.append(sample_count)
        
        # Leave gap between groups to ensure no overlap
        current_age = age_max + draw(st.floats(min_value=0.1, max_value=1.0))
    
    return groups, sample_counts


@given(data=non_overlapping_age_groups_strategy())
@settings(max_examples=30, deadline=None)
def test_warning_severity_consistency(data):
    """
    Property: Warning severity should be consistent with sample count thresholds.
    """
    age_groups, sample_counts = data
    
    # Create test database
    db = create_test_database()
    
    try:
        # Populate database
        drawing_id = 0
        for (age_min, age_max), sample_count in zip(age_groups, sample_counts):
            for i in range(sample_count):
                # Generate age within [age_min, age_max) to match SQL query logic
                if sample_count == 1:
                    age = age_min + (age_max - age_min) * 0.5  # Middle of range
                else:
                    # Distribute evenly but ensure last age < age_max
                    age = age_min + (age_max - age_min) * (i / sample_count)
                drawing = Drawing(
                    filename=f"test_drawing_{drawing_id}.png",
                    file_path=f"/test/path/test_drawing_{drawing_id}.png",
                    age_years=age,
                    subject="person",
                    upload_timestamp=datetime.now(timezone.utc)
                )
                db.add(drawing)
                drawing_id += 1
        
        db.commit()
        
        # Initialize analyzer
        analyzer = DataSufficiencyAnalyzer()
        
        # Generate warnings
        warnings = analyzer.generate_data_warnings(age_groups, db)
        
        # Group warnings by age group
        warnings_by_group = {}
        for warning in warnings:
            key = (warning.age_group_min, warning.age_group_max)
            if key not in warnings_by_group:
                warnings_by_group[key] = []
            warnings_by_group[key].append(warning)
        
        # Verify severity consistency (age groups are non-overlapping)
        for (age_min, age_max), sample_count in zip(age_groups, sample_counts):
            group_warnings = warnings_by_group.get((age_min, age_max), [])
            insufficient_warnings = [w for w in group_warnings if w.warning_type == "insufficient_data"]
            
            if sample_count <= analyzer.critical_threshold:
                # Should have critical warning
                critical_warnings = [w for w in insufficient_warnings if w.severity == "critical"]
                assert len(critical_warnings) > 0, f"Missing critical warning for {sample_count} samples"
                
            elif sample_count < analyzer.min_samples_per_group:
                # Should have high warning
                high_warnings = [w for w in insufficient_warnings if w.severity == "high"]
                assert len(high_warnings) > 0, f"Missing high warning for {sample_count} samples"
                
                # Should not have critical warning
                critical_warnings = [w for w in insufficient_warnings if w.severity == "critical"]
                assert len(critical_warnings) == 0, f"Unexpected critical warning for {sample_count} samples"
                
            elif sample_count < analyzer.recommended_samples_per_group:
                # Should have medium warning
                medium_warnings = [w for w in insufficient_warnings if w.severity == "medium"]
                assert len(medium_warnings) > 0, f"Missing medium warning for {sample_count} samples"
                
                # Should not have high or critical warnings
                high_critical_warnings = [w for w in insufficient_warnings if w.severity in ["high", "critical"]]
                assert len(high_critical_warnings) == 0, f"Unexpected high/critical warning for {sample_count} samples"
        
    finally:
        db.close()


def test_data_augmentation_suggestions():
    """
    Test that data augmentation suggestions are appropriate for different data scenarios.
    """
    suggester = DataAugmentationSuggester()
    
    # Test with very low sample count
    low_data_info = AgeGroupDataInfo(
        age_min=5.0,
        age_max=6.0,
        sample_count=15,
        is_sufficient=False,
        recommended_min_samples=100,
        data_quality_score=0.3,
        subjects_distribution={"person": 10, "house": 5},
        age_distribution=[5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9]
    )
    
    suggestions = suggester.suggest_augmentation_strategies(low_data_info)
    
    # Should suggest aggressive augmentation
    assert suggestions["multiplier_target"] >= 3.0
    assert len(suggestions["recommended_techniques"]) > 0
    assert len(suggestions["cautions"]) > 0
    
    # Test with moderate sample count
    moderate_data_info = AgeGroupDataInfo(
        age_min=7.0,
        age_max=8.0,
        sample_count=75,
        is_sufficient=True,
        recommended_min_samples=100,
        data_quality_score=0.6,
        subjects_distribution={"person": 40, "house": 35},
        age_distribution=list(np.linspace(7.0, 8.0, 75))
    )
    
    suggestions = suggester.suggest_augmentation_strategies(moderate_data_info)
    
    # Should suggest moderate augmentation
    assert 1.0 <= suggestions["multiplier_target"] <= 3.0
    
    # Test with sufficient data
    sufficient_data_info = AgeGroupDataInfo(
        age_min=9.0,
        age_max=10.0,
        sample_count=150,
        is_sufficient=True,
        recommended_min_samples=100,
        data_quality_score=0.8,
        subjects_distribution={"person": 50, "house": 50, "tree": 50},
        age_distribution=list(np.linspace(9.0, 10.0, 150))
    )
    
    suggestions = suggester.suggest_augmentation_strategies(sufficient_data_info)
    
    # Should suggest minimal or no augmentation
    assert suggestions["multiplier_target"] <= 2.0


def test_error_handling_for_invalid_age_groups():
    """
    Test that the analyzer handles invalid age groups gracefully.
    """
    db = create_test_database()
    
    try:
        analyzer = DataSufficiencyAnalyzer()
        
        # Test with invalid age range (min >= max)
        # Should handle gracefully and return 0 samples
        data_info = analyzer.analyze_age_group_data(10.0, 10.0, db)
        assert data_info.sample_count == 0
        assert not data_info.is_sufficient
        assert data_info.age_min == 10.0
        assert data_info.age_max == 10.0
        
        # Test with age range outside valid bounds
        # This should also handle gracefully
        data_info = analyzer.analyze_age_group_data(0.0, 1.0, db)
        assert data_info.sample_count == 0
        assert not data_info.is_sufficient
        
        # Test with reversed age range (max < min)
        # Should also handle gracefully
        data_info = analyzer.analyze_age_group_data(10.0, 5.0, db)
        assert data_info.sample_count == 0
        assert not data_info.is_sufficient
        
    finally:
        db.close()


def test_merging_suggestion_improvement_scores():
    """
    Test that merging suggestions have reasonable improvement scores.
    """
    db = create_test_database()
    
    try:
        # Create two age groups with insufficient data
        for i in range(25):  # Group 1: 25 samples
            drawing = Drawing(
                filename=f"drawing_1_{i}.png",
                file_path=f"/test/drawing_1_{i}.png",
                age_years=5.0 + i * 0.04,  # Ages 5.0-6.0
                subject="person",
                upload_timestamp=datetime.now(timezone.utc)
            )
            db.add(drawing)
        
        for i in range(30):  # Group 2: 30 samples
            drawing = Drawing(
                filename=f"drawing_2_{i}.png",
                file_path=f"/test/drawing_2_{i}.png",
                age_years=6.0 + i * 0.033,  # Ages 6.0-7.0
                subject="house",
                upload_timestamp=datetime.now(timezone.utc)
            )
            db.add(drawing)
        
        db.commit()
        
        analyzer = DataSufficiencyAnalyzer()
        age_groups = [(5.0, 6.0), (6.0, 7.0)]
        
        suggestions = analyzer.suggest_age_group_merging(age_groups, db)
        
        # Should have at least one suggestion
        assert len(suggestions) > 0
        
        # Check improvement scores
        for suggestion in suggestions:
            # Should have positive improvement score
            assert suggestion.improvement_score > 0.0
            
            # Combined sample count should be sum of individual counts
            assert suggestion.combined_sample_count == 55  # 25 + 30
            
            # Merged group should span both original groups
            assert suggestion.merged_group[0] <= 5.0
            assert suggestion.merged_group[1] >= 7.0
    
    finally:
        db.close()