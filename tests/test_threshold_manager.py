"""
Tests for threshold management and threshold-based anomaly detection.

**Feature: children-drawing-anomaly-detection, Property 9: Threshold-Based Anomaly Detection**
**Validates: Requirements 5.3**

**Feature: children-drawing-anomaly-detection, Property 10: Threshold Recalculation Accuracy**
**Validates: Requirements 5.1, 5.5**

**Feature: children-drawing-anomaly-detection, Property 11: Dynamic Threshold Updates**
**Validates: Requirements 5.2, 5.4**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pickle

# Import test utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.threshold_manager import ThresholdManager, ThresholdConfig, ThresholdCalculationError
from app.models.database import Base, AgeGroupModel, Drawing, DrawingEmbedding
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def test_db():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def threshold_manager():
    """Create a threshold manager for testing."""
    config = ThresholdConfig(
        default_percentile=95.0,
        min_samples_for_calculation=10,  # Lower for testing
        confidence_levels=[90.0, 95.0, 99.0]
    )
    return ThresholdManager(config)


def create_test_age_group_model(db, age_min: float, age_max: float, threshold: float = 0.5):
    """Create a test age group model in the database."""
    model = AgeGroupModel(
        age_min=age_min,
        age_max=age_max,
        model_type="autoencoder",
        vision_model="vit",
        parameters='{"test": "parameters"}',
        sample_count=50,
        threshold=threshold,
        is_active=True
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def create_test_drawings_with_embeddings(db, age_min: float, age_max: float, num_samples: int):
    """Create test drawings and embeddings in the database."""
    drawings = []
    for i in range(num_samples):
        age = age_min + (age_max - age_min) * np.random.random()
        drawing = Drawing(
            filename=f"test_drawing_{i}.png",
            file_path=f"/test/path/drawing_{i}.png",
            age_years=age,
            subject="test_subject"
        )
        db.add(drawing)
        db.flush()  # Get the ID
        
        # Create embedding
        embedding_data = np.random.randn(64).astype(np.float32)
        from app.utils.embedding_serialization import serialize_embedding_for_db
        embedding_record = DrawingEmbedding(
            drawing_id=drawing.id,
            model_type="vit",
            embedding_vector=serialize_embedding_for_db(embedding_data),
            vector_dimension=64
        )
        db.add(embedding_record)
        drawings.append(drawing)
    
    db.commit()
    return drawings


@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100
    ),
    threshold=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=100, deadline=None)
def test_threshold_based_anomaly_detection(scores, threshold):
    """
    **Feature: children-drawing-anomaly-detection, Property 9: Threshold-Based Anomaly Detection**
    **Validates: Requirements 5.3**
    
    Property: For any drawing with computed anomaly score, the drawing should be flagged 
    as anomalous if and only if its score exceeds the configured threshold.
    """
    threshold_manager = ThresholdManager()
    
    # Test each score against the threshold
    for score in scores:
        # Mock database and age group model
        mock_db = Mock()
        mock_model = Mock()
        mock_model.threshold = threshold
        mock_model.age_min = 5.0
        mock_model.age_max = 6.0
        mock_model.id = 1
        
        # Mock the age group manager to return our test model
        with patch('app.services.age_group_manager.get_age_group_manager') as mock_age_manager:
            mock_age_manager.return_value.find_appropriate_model.return_value = mock_model
            
            # Test the is_anomaly method
            is_anomaly, used_threshold, model_info = threshold_manager.is_anomaly(score, 5.5, mock_db)
            
            # Verify the threshold-based decision is correct
            expected_anomaly = score > threshold
            assert is_anomaly == expected_anomaly, \
                f"Score {score} with threshold {threshold}: expected {expected_anomaly}, got {is_anomaly}"
            
            # Verify the threshold used matches what we set
            assert used_threshold == threshold, \
                f"Expected threshold {threshold}, got {used_threshold}"
            
            # Verify model info is provided
            assert model_info is not None
            assert "Age group" in model_info


@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=200
    ),
    percentile=st.floats(min_value=50.0, max_value=99.9, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50)
def test_threshold_recalculation_accuracy(scores, percentile):
    """
    **Feature: children-drawing-anomaly-detection, Property 10: Threshold Recalculation Accuracy**
    **Validates: Requirements 5.1, 5.5**
    
    Property: For any set of validation scores, the 95th percentile threshold should be 
    mathematically correct and properly applied to subsequent analyses.
    """
    threshold_manager = ThresholdManager()
    scores_array = np.array(scores)
    
    # Calculate threshold using our method
    calculated_threshold = threshold_manager.calculate_percentile_threshold(scores_array, percentile)
    
    # Calculate expected threshold using numpy
    expected_threshold = np.percentile(scores_array, percentile)
    
    # Verify the calculated threshold matches the expected value
    assert np.isclose(calculated_threshold, expected_threshold, rtol=1e-10), \
        f"Calculated threshold {calculated_threshold} should match expected {expected_threshold}"
    
    # Verify the threshold is finite and non-negative
    assert np.isfinite(calculated_threshold), "Calculated threshold should be finite"
    assert calculated_threshold >= 0, "Calculated threshold should be non-negative"
    
    # Verify the percentile property: the threshold should be reasonable relative to the data
    # The key property is that our calculation matches numpy's calculation exactly
    # Additional verification: threshold should be within the range of the data
    min_score = np.min(scores_array)
    max_score = np.max(scores_array)
    
    assert min_score <= calculated_threshold <= max_score, \
        f"Threshold {calculated_threshold} should be within data range [{min_score}, {max_score}]"
    
    # For percentiles >= 50, threshold should be >= median
    if percentile >= 50.0:
        median = np.median(scores_array)
        assert calculated_threshold >= median, \
            f"Threshold {calculated_threshold} should be >= median {median} for percentile {percentile}"


def test_dynamic_threshold_updates(test_db, threshold_manager):
    """
    **Feature: children-drawing-anomaly-detection, Property 11: Dynamic Threshold Updates**
    **Validates: Requirements 5.2, 5.4**
    
    Property: For any threshold configuration change, subsequent analyses should use 
    the new threshold without requiring system restart.
    """
    # Create a test age group model
    model = create_test_age_group_model(test_db, 5.0, 6.0, threshold=0.5)
    
    # Initial threshold check
    initial_threshold = model.threshold
    assert initial_threshold == 0.5
    
    # Test threshold update
    new_threshold = 0.8
    success = threshold_manager.update_model_threshold(model.id, new_threshold, test_db)
    assert success, "Threshold update should succeed"
    
    # Verify the threshold was updated in the database
    test_db.refresh(model)
    assert model.threshold == new_threshold, \
        f"Model threshold should be updated to {new_threshold}, got {model.threshold}"
    
    # Test that subsequent analyses use the new threshold
    mock_age_manager = Mock()
    mock_age_manager.find_appropriate_model.return_value = model
    
    with patch('app.services.age_group_manager.get_age_group_manager', return_value=mock_age_manager):
        # Test score below new threshold
        is_anomaly_low, used_threshold, _ = threshold_manager.is_anomaly(0.7, 5.5, test_db)
        assert not is_anomaly_low, "Score 0.7 should not be anomaly with threshold 0.8"
        assert used_threshold == new_threshold, "Should use updated threshold"
        
        # Test score above new threshold
        is_anomaly_high, used_threshold, _ = threshold_manager.is_anomaly(0.9, 5.5, test_db)
        assert is_anomaly_high, "Score 0.9 should be anomaly with threshold 0.8"
        assert used_threshold == new_threshold, "Should use updated threshold"
    
    # Test another threshold update
    newer_threshold = 0.3
    success = threshold_manager.update_model_threshold(model.id, newer_threshold, test_db)
    assert success, "Second threshold update should succeed"
    
    # Verify the second update
    test_db.refresh(model)
    assert model.threshold == newer_threshold, \
        f"Model threshold should be updated to {newer_threshold}, got {model.threshold}"
    
    # Test that the cache is properly cleared and new threshold is used
    with patch('app.services.age_group_manager.get_age_group_manager', return_value=mock_age_manager):
        is_anomaly, used_threshold, _ = threshold_manager.is_anomaly(0.5, 5.5, test_db)
        assert is_anomaly, "Score 0.5 should be anomaly with threshold 0.3"
        assert used_threshold == newer_threshold, "Should use newest threshold"


def test_percentile_threshold_calculation_edge_cases(threshold_manager):
    """Test edge cases for percentile threshold calculation."""
    
    # Test with empty array
    with pytest.raises(ThresholdCalculationError, match="empty scores array"):
        threshold_manager.calculate_percentile_threshold(np.array([]), 95.0)
    
    # Test with invalid percentile
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ThresholdCalculationError, match="Percentile must be between 0 and 100"):
        threshold_manager.calculate_percentile_threshold(scores, -5.0)
    
    with pytest.raises(ThresholdCalculationError, match="Percentile must be between 0 and 100"):
        threshold_manager.calculate_percentile_threshold(scores, 105.0)
    
    # Test with array containing NaN/inf values
    scores_with_nan = np.array([1.0, 2.0, np.nan, 4.0, np.inf, 5.0])
    threshold = threshold_manager.calculate_percentile_threshold(scores_with_nan, 95.0)
    assert np.isfinite(threshold), "Should handle NaN/inf values and return finite threshold"
    
    # Test with all NaN values
    all_nan_scores = np.array([np.nan, np.nan, np.nan])
    with pytest.raises(ThresholdCalculationError, match="No valid.*finite.*scores"):
        threshold_manager.calculate_percentile_threshold(all_nan_scores, 95.0)


def test_threshold_update_validation(test_db, threshold_manager):
    """Test validation of threshold updates."""
    model = create_test_age_group_model(test_db, 5.0, 6.0, threshold=0.5)
    
    # Test invalid threshold values
    with pytest.raises(ThresholdCalculationError, match="must be finite"):
        threshold_manager.update_model_threshold(model.id, np.nan, test_db)
    
    with pytest.raises(ThresholdCalculationError, match="must be finite"):
        threshold_manager.update_model_threshold(model.id, np.inf, test_db)
    
    with pytest.raises(ThresholdCalculationError, match="must be non-negative"):
        threshold_manager.update_model_threshold(model.id, -0.5, test_db)
    
    # Test with non-existent model
    with pytest.raises(ThresholdCalculationError, match="not found"):
        threshold_manager.update_model_threshold(99999, 0.5, test_db)


def test_threshold_statistics(test_db, threshold_manager):
    """Test threshold statistics calculation."""
    # Create multiple models with different thresholds
    model1 = create_test_age_group_model(test_db, 3.0, 4.0, threshold=0.3)
    model2 = create_test_age_group_model(test_db, 5.0, 6.0, threshold=0.5)
    model3 = create_test_age_group_model(test_db, 7.0, 8.0, threshold=0.7)
    
    stats = threshold_manager.get_threshold_statistics(test_db)
    
    assert stats["total_models"] == 3
    assert stats["threshold_range"] == (0.3, 0.7)
    assert np.isclose(stats["mean_threshold"], 0.5)
    assert len(stats["model_thresholds"]) == 3
    
    # Verify individual model information
    model_info = {info["model_id"]: info for info in stats["model_thresholds"]}
    assert model_info[model1.id]["threshold"] == 0.3
    assert model_info[model2.id]["threshold"] == 0.5
    assert model_info[model3.id]["threshold"] == 0.7


def test_no_appropriate_model_handling(threshold_manager):
    """Test handling when no appropriate model is found for an age."""
    mock_db = Mock()
    
    # Mock age group manager to return None (no model found)
    with patch('app.services.age_group_manager.get_age_group_manager') as mock_age_manager:
        mock_age_manager.return_value.find_appropriate_model.return_value = None
        
        # Test is_anomaly with no model
        is_anomaly, threshold, model_info = threshold_manager.is_anomaly(0.8, 15.0, mock_db)
        
        assert not is_anomaly, "Should not flag as anomaly when no model available"
        assert threshold == 0.0, "Should return 0.0 threshold when no model available"
        assert "No appropriate model found" in model_info
        
        # Test get_threshold_for_age with no model
        threshold = threshold_manager.get_threshold_for_age(15.0, mock_db)
        assert threshold is None, "Should return None when no model available"


# Additional tests for score normalization

from app.services.score_normalizer import ScoreNormalizer, NormalizationConfig, ScoreNormalizationError


@given(
    raw_scores=st.lists(
        st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50
    ),
    normalization_method=st.sampled_from(["z_score", "min_max", "robust"])
)
@settings(max_examples=50)
def test_score_normalization_consistency(raw_scores, normalization_method):
    """
    **Feature: children-drawing-anomaly-detection, Property 8: Score Normalization Consistency**
    **Validates: Requirements 4.2**
    
    Property: For any set of anomaly scores across different age groups, normalized scores 
    should be comparable and fall within expected ranges.
    """
    # Create score normalizer with the specified method
    config = NormalizationConfig(normalization_method=normalization_method)
    normalizer = ScoreNormalizer(config)
    
    # Mock database and age group model
    mock_db = Mock()
    mock_model_id = 1
    
    # Create mock analysis objects with anomaly scores
    mock_analyses = []
    for score in raw_scores:
        mock_analysis = Mock()
        mock_analysis.anomaly_score = score
        mock_analyses.append(mock_analysis)
    
    # Mock the database query to return our mock analyses
    mock_query = Mock()
    mock_query.filter.return_value.all.return_value = mock_analyses
    mock_db.query.return_value = mock_query
    
    # Test normalization for each score
    normalized_scores = []
    for raw_score in raw_scores:
        normalized_score = normalizer.normalize_score(raw_score, mock_model_id, mock_db)
        normalized_scores.append(normalized_score)
        
        # Verify normalized score is finite
        assert np.isfinite(normalized_score), f"Normalized score should be finite, got {normalized_score}"
    
    normalized_array = np.array(normalized_scores)
    
    # Test general properties of percentile-based normalization
    # All normalized scores should be in [0, 100] range
    assert np.all(normalized_array >= 0.0), "Normalized scores should be >= 0"
    assert np.all(normalized_array <= 100.0), "Normalized scores should be <= 100"
    
    # Test consistency: same input should give same output
    if len(raw_scores) > 0:
        test_score = raw_scores[0]
        normalized_1 = normalizer.normalize_score(test_score, mock_model_id, mock_db)
        normalized_2 = normalizer.normalize_score(test_score, mock_model_id, mock_db)
        assert abs(normalized_1 - normalized_2) < 1e-10, "Same input should give same output"
    
    # Test monotonicity for distinct values
    raw_scores_array = np.array(raw_scores)
    if len(np.unique(raw_scores)) > 1:
        # Higher raw scores should generally have higher normalized scores
        # (this is a general property of percentile-based normalization)
        min_raw = np.min(raw_scores_array)
        max_raw = np.max(raw_scores_array)
        
        if max_raw > min_raw:
            min_normalized = normalizer.normalize_score(min_raw, mock_model_id, mock_db)
            max_normalized = normalizer.normalize_score(max_raw, mock_model_id, mock_db)
            assert max_normalized >= min_normalized, \
                f"Higher raw score should have higher normalized score: {max_raw}→{max_normalized} vs {min_raw}→{min_normalized}"
    sorted_indices = np.argsort(raw_scores)
    sorted_normalized = [normalized_scores[i] for i in sorted_indices]
    
    for i in range(len(sorted_normalized) - 1):
        # Allow small tolerance for floating point precision
        assert sorted_normalized[i] <= sorted_normalized[i + 1] + 1e-10, \
            f"Normalization should preserve order: {sorted_normalized[i]} > {sorted_normalized[i + 1]}"


def test_score_normalization_edge_cases():
    """Test edge cases for score normalization."""
    config = NormalizationConfig(normalization_method="z_score")
    normalizer = ScoreNormalizer(config)
    mock_db = Mock()
    mock_model_id = 1
    
    # Create mock analysis objects with identical scores
    identical_scores = [5.0] * 10
    mock_analyses = []
    for score in identical_scores:
        mock_analysis = Mock()
        mock_analysis.anomaly_score = score
        mock_analyses.append(mock_analysis)
    
    # Mock the database query to return our mock analyses
    mock_query = Mock()
    mock_query.filter.return_value.all.return_value = mock_analyses
    mock_db.query.return_value = mock_query
    
    # Score equal to mean should normalize to around 50 (percentile-based)
    norm_score = normalizer.normalize_score(5.0, mock_model_id, mock_db)
    assert 40.0 <= norm_score <= 60.0, f"Score equal to mean should normalize to around 50, got {norm_score}"
    
    # Score higher than all existing scores should normalize to 100
    norm_score = normalizer.normalize_score(6.0, mock_model_id, mock_db)
    assert norm_score == 100.0, f"Score higher than all existing should normalize to 100, got {norm_score}"
    
    # Test invalid inputs
    with pytest.raises(ScoreNormalizationError, match="must be finite"):
        normalizer.normalize_score(np.nan, mock_model_id, mock_db)
    
    with pytest.raises(ScoreNormalizationError, match="must be finite"):
        normalizer.normalize_score(np.inf, mock_model_id, mock_db)


def test_confidence_calculation_consistency():
    """Test confidence calculation consistency."""
    config = NormalizationConfig(confidence_method="statistical")
    normalizer = ScoreNormalizer(config)
    mock_db = Mock()
    mock_model_id = 1
    
    # Create mock analysis objects with a range of scores
    test_scores = [5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0] * 12  # 96 scores
    mock_analyses = []
    for score in test_scores:
        mock_analysis = Mock()
        mock_analysis.anomaly_score = score
        mock_analyses.append(mock_analysis)
    
    # Mock the database query to return our mock analyses
    mock_query = Mock()
    mock_query.filter.return_value.all.return_value = mock_analyses
    mock_db.query.return_value = mock_query
    
    # Test confidence for different scores
    test_cases = [
        10.0,   # Mean score
        16.0,   # High score
        15.0,   # Medium-high score
        14.0,   # Medium score
        13.0,   # Medium score
        12.0,   # Medium score
    ]
    
    for raw_score in test_cases:
        normalized_score = normalizer.normalize_score(raw_score, mock_model_id, mock_db)
        confidence = normalizer.calculate_confidence(raw_score, normalized_score, mock_model_id, mock_db)
        
        # Verify confidence is in valid range
        assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1], got {confidence}"
        
        # Verify normalized score is in valid range
        assert 0.0 <= normalized_score <= 100.0, f"Normalized score should be in [0,100], got {normalized_score}"


def test_score_comparison_ranking():
    """Test score comparison and ranking functionality."""
    normalizer = ScoreNormalizer()
    mock_db = Mock()
    
    # Create mock age group manager
    mock_age_manager = Mock()
    mock_model = Mock()
    mock_model.id = 1
    mock_model.age_min = 5.0
    mock_model.age_max = 6.0
    mock_age_manager.find_appropriate_model.return_value = mock_model
    
    # Create mock analyses for the database query
    mock_analyses = []
    for score in [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0]:
        mock_analysis = Mock()
        mock_analysis.anomaly_score = score
        mock_analyses.append(mock_analysis)
    
    # Mock the database query to return our mock analyses
    mock_db.query.return_value.filter.return_value.all.return_value = mock_analyses
    
    # Mock statistics
    mock_stats = {
        "mean": 10.0,
        "std": 2.0,
        "min": 5.0,
        "max": 20.0,
        "median": 10.0,
        "q25": 8.0,
        "q75": 12.0,
        "sample_count": 50,
        "age_range": (5.0, 6.0)
    }
    normalizer._cached_stats[1] = mock_stats
    
    # Test scores and ages
    scores_and_ages = [
        (15.0, 5.5),  # High score
        (10.0, 5.3),  # Medium score
        (8.0, 5.7),   # Low score
        (18.0, 5.1),  # Very high score
    ]
    
    with patch('app.services.age_group_manager.get_age_group_manager', return_value=mock_age_manager):
        results = normalizer.compare_scores(scores_and_ages, mock_db)
    
    # Verify results structure
    assert len(results) == 4, "Should return results for all input scores"
    
    # Verify ranking (should be sorted by normalized score, descending)
    for i in range(len(results) - 1):
        assert results[i]["normalized_score"] >= results[i + 1]["normalized_score"], \
            "Results should be sorted by normalized score (descending)"
        assert results[i]["rank"] < results[i + 1]["rank"], \
            "Ranks should be ascending"
    
    # Verify all results have required fields
    required_fields = ["index", "raw_score", "age", "normalized_score", "confidence", 
                      "age_group_model_id", "age_range", "comparable", "rank", "percentile_rank"]
    for result in results:
        for field in required_fields:
            assert field in result, f"Result should contain field '{field}'"
        
        # Verify field types and ranges
        assert isinstance(result["rank"], int) and result["rank"] >= 1
        assert 0 <= result["percentile_rank"] <= 100
        assert 0 <= result["confidence"] <= 1
        assert result["comparable"] is True  # All should be comparable in this test