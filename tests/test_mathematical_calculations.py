"""
Unit tests for mathematical calculations and algorithms.

This module tests the mathematical correctness of calculations used throughout
the system, including threshold calculations, score normalization, and
statistical computations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.threshold_manager import ThresholdManager, ThresholdCalculationError
from app.services.score_normalizer import ScoreNormalizer, ScoreNormalizationError
from app.services.model_manager import ModelManager, AutoencoderTrainingError
from app.services.interpretability_engine import InterpretabilityPipeline


class TestThresholdCalculations:
    """Test threshold calculation algorithms."""
    
    def test_percentile_threshold_calculation(self):
        """Test percentile threshold calculation with known data."""
        threshold_manager = ThresholdManager()
        
        # Test with known data
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        # Test 50th percentile (median)
        threshold_50 = threshold_manager.calculate_percentile_threshold(scores, 50.0)
        assert abs(threshold_50 - 5.5) < 0.1, f"50th percentile should be ~5.5, got {threshold_50}"
        
        # Test 95th percentile
        threshold_95 = threshold_manager.calculate_percentile_threshold(scores, 95.0)
        assert threshold_95 > 9.0, f"95th percentile should be > 9.0, got {threshold_95}"
        
        # Test 5th percentile
        threshold_5 = threshold_manager.calculate_percentile_threshold(scores, 5.0)
        assert threshold_5 < 2.0, f"5th percentile should be < 2.0, got {threshold_5}"
    
    def test_percentile_edge_cases(self):
        """Test percentile calculation with edge cases."""
        threshold_manager = ThresholdManager()
        
        # Test with single value
        single_score = np.array([5.0])
        threshold = threshold_manager.calculate_percentile_threshold(single_score, 95.0)
        assert threshold == 5.0, f"Single value threshold should be 5.0, got {threshold}"
        
        # Test with identical values
        identical_scores = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        threshold = threshold_manager.calculate_percentile_threshold(identical_scores, 95.0)
        assert threshold == 3.0, f"Identical values threshold should be 3.0, got {threshold}"
        
        # Test with extreme percentiles
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        threshold_0 = threshold_manager.calculate_percentile_threshold(scores, 0.0)
        threshold_100 = threshold_manager.calculate_percentile_threshold(scores, 100.0)
        assert threshold_0 <= min(scores), "0th percentile should be <= minimum"
        assert threshold_100 >= max(scores), "100th percentile should be >= maximum"
    
    def test_invalid_percentile_values(self):
        """Test that invalid percentile values raise appropriate errors."""
        threshold_manager = ThresholdManager()
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test negative percentile
        with pytest.raises(ThresholdCalculationError):
            threshold_manager.calculate_percentile_threshold(scores, -10.0)
        
        # Test percentile > 100
        with pytest.raises(ThresholdCalculationError):
            threshold_manager.calculate_percentile_threshold(scores, 110.0)
        
        # Test empty array
        with pytest.raises(ThresholdCalculationError):
            threshold_manager.calculate_percentile_threshold(np.array([]), 95.0)


class TestScoreNormalization:
    """Test score normalization algorithms."""
    
    def test_z_score_normalization(self):
        """Test percentile-based normalization with known data."""
        from app.services.score_normalizer import NormalizationConfig
        config = NormalizationConfig(normalization_method="percentile")
        normalizer = ScoreNormalizer(config)
        
        # Create mock database with known scores
        mock_db = Mock()
        mock_model_id = 1
        
        # Create mock analysis objects with known scores
        test_scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mock_analyses = []
        for score in test_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Test score equal to median (5.0)
        normalized = normalizer.normalize_score(5.0, mock_model_id, mock_db)
        assert 40.0 <= normalized <= 60.0, f"Median score should normalize to ~50, got {normalized}"
        
        # Test score above median
        normalized = normalizer.normalize_score(8.0, mock_model_id, mock_db)
        assert normalized > 70.0, f"Score above median should normalize > 70, got {normalized}"
        
        # Test score below median
        normalized = normalizer.normalize_score(2.0, mock_model_id, mock_db)
        assert normalized < 30.0, f"Score below median should normalize < 30, got {normalized}"
    
    def test_min_max_normalization_fallback(self):
        """Test normalization when all scores are identical."""
        normalizer = ScoreNormalizer()
        
        # Create mock database with identical scores
        mock_db = Mock()
        mock_model_id = 1
        
        # Create mock analysis objects with identical scores
        identical_scores = [5.0] * 10
        mock_analyses = []
        for score in identical_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Should handle identical scores gracefully
        normalized = normalizer.normalize_score(5.0, mock_model_id, mock_db)
        assert 0.0 <= normalized <= 100.0, f"Normalized score should be in [0,100], got {normalized}"
    
    def test_confidence_calculation(self):
        """Test confidence calculation based on normalized score."""
        normalizer = ScoreNormalizer()
        
        # Create mock database
        mock_db = Mock()
        mock_model_id = 1
        
        # Create mock analysis objects with a range of scores
        test_scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mock_analyses = []
        for score in test_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Test confidence calculation
        normalized_score = normalizer.normalize_score(8.0, mock_model_id, mock_db)
        confidence = normalizer.calculate_confidence(8.0, normalized_score, mock_model_id, mock_db)
        
        # Confidence should be in valid range
        assert 0.0 <= confidence <= 1.0, f"Confidence should be in [0,1], got {confidence}"
    
    def test_normalization_bounds(self):
        """Test that normalized scores are properly bounded."""
        normalizer = ScoreNormalizer()
        
        # Create mock database with normal range of scores
        mock_db = Mock()
        mock_model_id = 1
        
        # Create mock analysis objects with normal scores
        test_scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        mock_analyses = []
        for score in test_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Test extreme values
        extreme_high = normalizer.normalize_score(100.0, mock_model_id, mock_db)
        extreme_low = normalizer.normalize_score(-100.0, mock_model_id, mock_db)
        
        assert 0.0 <= extreme_high <= 100.0, f"Extreme high should be bounded [0,100], got {extreme_high}"
        assert 0.0 <= extreme_low <= 100.0, f"Extreme low should be bounded [0,100], got {extreme_low}"


class TestModelManagerCalculations:
    """Test model manager mathematical calculations."""
    
    def test_reconstruction_loss_calculation(self):
        """Test reconstruction loss calculation."""
        model_manager = ModelManager()
        
        # Mock model and data
        original_embedding = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        reconstructed_embedding = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        
        with patch.object(model_manager, 'model') as mock_model:
            mock_model.predict.return_value = reconstructed_embedding.reshape(1, -1)
            
            # Calculate loss
            loss = model_manager._calculate_reconstruction_loss(
                original_embedding, reconstructed_embedding
            )
            
            # Should be mean squared error
            expected_mse = np.mean((original_embedding - reconstructed_embedding) ** 2)
            assert abs(loss - expected_mse) < 1e-6, f"Loss should be MSE, expected {expected_mse}, got {loss}"
    
    def test_training_metrics_calculation(self):
        """Test training metrics calculation."""
        model_manager = ModelManager()
        
        # Mock training data
        embeddings = np.random.rand(100, 10)  # 100 samples, 10 features
        
        with patch.object(model_manager, 'model') as mock_model:
            # Mock model predictions
            mock_model.predict.return_value = embeddings + np.random.normal(0, 0.1, embeddings.shape)
            
            metrics = model_manager._calculate_metrics(embeddings)
            
            # Check that metrics are calculated
            assert 'mean_loss' in metrics
            assert 'std_loss' in metrics
            assert 'min_loss' in metrics
            assert 'max_loss' in metrics
            
            # Check that values are reasonable
            assert metrics['mean_loss'] >= 0, "Mean loss should be non-negative"
            assert metrics['std_loss'] >= 0, "Std loss should be non-negative"
            assert metrics['min_loss'] <= metrics['mean_loss'], "Min should be <= mean"
            assert metrics['max_loss'] >= metrics['mean_loss'], "Max should be >= mean"


class TestInterpretabilityCalculations:
    """Test interpretability engine calculations."""
    
    def test_edge_density_calculation(self):
        """Test edge density calculation."""
        pipeline = InterpretabilityPipeline()
        
        # Create test image with known edge characteristics
        # High edge density image (checkerboard pattern)
        high_edge_image = np.zeros((100, 100), dtype=np.uint8)
        high_edge_image[::2, ::2] = 255
        high_edge_image[1::2, 1::2] = 255
        
        # Low edge density image (solid color)
        low_edge_image = np.full((100, 100), 128, dtype=np.uint8)
        
        high_density = pipeline._calculate_edge_density(high_edge_image)
        low_density = pipeline._calculate_edge_density(low_edge_image)
        
        assert high_density > low_density, f"High edge image should have higher density: {high_density} vs {low_density}"
        assert 0.0 <= high_density <= 1.0, f"Edge density should be in [0,1], got {high_density}"
        assert 0.0 <= low_density <= 1.0, f"Edge density should be in [0,1], got {low_density}"
    
    def test_complexity_calculation(self):
        """Test drawing complexity calculation."""
        pipeline = InterpretabilityPipeline()
        
        # Create images with different complexity levels
        # Simple image (few colors, low variance)
        simple_image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        
        # Complex image (many colors, high variance)
        complex_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        simple_complexity = pipeline._calculate_complexity(simple_image)
        complex_complexity = pipeline._calculate_complexity(complex_image)
        
        assert complex_complexity > simple_complexity, \
            f"Complex image should have higher complexity: {complex_complexity} vs {simple_complexity}"
        assert simple_complexity >= 0, f"Complexity should be non-negative, got {simple_complexity}"
        assert complex_complexity >= 0, f"Complexity should be non-negative, got {complex_complexity}"
    
    def test_severity_determination(self):
        """Test anomaly severity determination."""
        pipeline = InterpretabilityPipeline()
        
        # Test severity levels
        low_score = 0.3
        medium_score = 0.6
        high_score = 0.9
        
        low_severity = pipeline._determine_severity(low_score)
        medium_severity = pipeline._determine_severity(medium_score)
        high_severity = pipeline._determine_severity(high_score)
        
        assert low_severity in ["low", "mild", "normal"], f"Low score should give low severity, got {low_severity}"
        assert medium_severity in ["medium", "moderate"], f"Medium score should give medium severity, got {medium_severity}"
        assert high_severity in ["high", "severe"], f"High score should give high severity, got {high_severity}"


class TestStatisticalValidation:
    """Test statistical validation of calculations."""
    
    def test_percentile_consistency(self):
        """Test that percentile calculations are consistent."""
        threshold_manager = ThresholdManager()
        
        # Generate test data
        np.random.seed(42)  # For reproducibility
        scores = np.random.normal(5.0, 2.0, 1000)
        
        # Calculate multiple percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        thresholds = []
        
        for p in percentiles:
            threshold = threshold_manager.calculate_percentile_threshold(scores, float(p))
            thresholds.append(threshold)
        
        # Verify monotonicity (higher percentiles should give higher thresholds)
        for i in range(1, len(thresholds)):
            assert thresholds[i] >= thresholds[i-1], \
                f"Percentile {percentiles[i]} threshold {thresholds[i]} should be >= " \
                f"percentile {percentiles[i-1]} threshold {thresholds[i-1]}"
    
    def test_normalization_distribution(self):
        """Test that normalization produces expected distribution."""
        normalizer = ScoreNormalizer()
        
        # Generate test scores with known distribution
        np.random.seed(42)
        raw_scores = np.random.normal(10.0, 3.0, 100)  # Smaller sample for testing
        
        # Create mock database with the generated scores
        mock_db = Mock()
        mock_model_id = 1
        
        # Create mock analysis objects
        mock_analyses = []
        for score in raw_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Test a subset of scores
        test_scores = raw_scores[:10]  # Test with first 10 scores
        normalized_scores = []
        
        for score in test_scores:
            normalized = normalizer.normalize_score(score, mock_model_id, mock_db)
            normalized_scores.append(normalized)
        
        normalized_scores = np.array(normalized_scores)
        
        # Check distribution properties
        assert np.all((normalized_scores >= 0) & (normalized_scores <= 100)), \
            "All normalized scores should be in [0,100]"
    
    def test_mathematical_properties(self):
        """Test mathematical properties of calculations."""
        threshold_manager = ThresholdManager()
        
        # Test linearity property: percentile of (a*X + b) = a*percentile(X) + b
        base_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        a, b = 2.0, 3.0
        scaled_scores = a * base_scores + b
        
        base_p95 = threshold_manager.calculate_percentile_threshold(base_scores, 95.0)
        scaled_p95 = threshold_manager.calculate_percentile_threshold(scaled_scores, 95.0)
        expected_scaled_p95 = a * base_p95 + b
        
        assert abs(scaled_p95 - expected_scaled_p95) < 1e-10, \
            f"Percentile should be linear: expected {expected_scaled_p95}, got {scaled_p95}"


class TestNumericalStability:
    """Test numerical stability of calculations."""
    
    def test_large_number_handling(self):
        """Test handling of large numbers."""
        threshold_manager = ThresholdManager()
        
        # Test with very large numbers
        large_scores = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
        threshold = threshold_manager.calculate_percentile_threshold(large_scores, 95.0)
        
        assert np.isfinite(threshold), "Threshold should be finite for large numbers"
        assert threshold > 4e10, "Threshold should be reasonable for large numbers"
    
    def test_small_number_handling(self):
        """Test handling of very small numbers."""
        threshold_manager = ThresholdManager()
        
        # Test with very small numbers
        small_scores = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        threshold = threshold_manager.calculate_percentile_threshold(small_scores, 95.0)
        
        assert np.isfinite(threshold), "Threshold should be finite for small numbers"
        assert threshold > 4e-10, "Threshold should be reasonable for small numbers"
    
    def test_precision_handling(self):
        """Test numerical precision in calculations."""
        normalizer = ScoreNormalizer()
        
        # Create mock database with precision-sensitive scores
        mock_db = Mock()
        mock_model_id = 1
        
        # Test with numbers that might cause precision issues
        precision_scores = [1.0, 1.0000000001, 1.0000000002]
        mock_analyses = []
        for score in precision_scores:
            mock_analysis = Mock()
            mock_analysis.anomaly_score = score
            mock_analyses.append(mock_analysis)
        
        # Mock the database query
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_analyses
        mock_db.query.return_value = mock_query
        
        # Should handle precision gracefully
        normalized = normalizer.normalize_score(1.0000000001, mock_model_id, mock_db)
        assert np.isfinite(normalized), "Should handle precision issues gracefully"
        assert 0.0 <= normalized <= 100.0, "Should maintain bounds despite precision issues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])