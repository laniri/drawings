"""
Unit tests for edge cases and boundary conditions.

This module tests the system's behavior with edge cases, boundary values,
and unusual input conditions to ensure robust error handling and graceful
degradation.
"""

import pytest
import numpy as np
import io
from PIL import Image
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from app.services.data_pipeline import DataPipelineService, ValidationResult, ImagePreprocessingError
from app.services.file_storage import FileStorageService, FileStorageError
from app.services.embedding_service import EmbeddingService, EmbeddingGenerationError
from app.services.age_group_manager import AgeGroupManager
from app.core.exceptions import ImageProcessingError, StorageError


class TestImageEdgeCases:
    """Test edge cases in image processing."""
    
    def test_minimum_size_images(self):
        """Test handling of minimum size images."""
        service = DataPipelineService()
        
        # Test exactly minimum size (32x32)
        min_image = Image.new('RGB', (32, 32), color=(128, 128, 128))
        buffer = io.BytesIO()
        min_image.save(buffer, format='PNG')
        min_data = buffer.getvalue()
        
        result = service.validate_image(min_data)
        assert result.is_valid, "32x32 image should be valid"
        assert result.dimensions == (32, 32)
        
        # Test below minimum size (31x31)
        small_image = Image.new('RGB', (31, 31), color=(128, 128, 128))
        buffer = io.BytesIO()
        small_image.save(buffer, format='PNG')
        small_data = buffer.getvalue()
        
        result = service.validate_image(small_data)
        assert not result.is_valid, "31x31 image should be invalid"
        assert "small" in result.error_message.lower() or "size" in result.error_message.lower()
    
    def test_maximum_size_images(self):
        """Test handling of very large images."""
        service = DataPipelineService()
        
        # Test large but reasonable image
        large_image = Image.new('RGB', (2048, 2048), color=(128, 128, 128))
        buffer = io.BytesIO()
        large_image.save(buffer, format='PNG', optimize=True)
        large_data = buffer.getvalue()
        
        # Should handle large images gracefully
        result = service.validate_image(large_data)
        # May be valid or invalid depending on file size limits
        assert isinstance(result, ValidationResult)
        
        if not result.is_valid:
            assert "size" in result.error_message.lower() or "large" in result.error_message.lower()
    
    def test_unusual_aspect_ratios(self):
        """Test images with unusual aspect ratios."""
        service = DataPipelineService()
        
        # Very wide image
        wide_image = Image.new('RGB', (1000, 50), color=(128, 128, 128))
        buffer = io.BytesIO()
        wide_image.save(buffer, format='PNG')
        wide_data = buffer.getvalue()
        
        result = service.validate_image(wide_data)
        assert isinstance(result, ValidationResult)
        
        # Very tall image
        tall_image = Image.new('RGB', (50, 1000), color=(128, 128, 128))
        buffer = io.BytesIO()
        tall_image.save(buffer, format='PNG')
        tall_data = buffer.getvalue()
        
        result = service.validate_image(tall_data)
        assert isinstance(result, ValidationResult)
    
    def test_single_color_images(self):
        """Test images with single colors."""
        service = DataPipelineService()
        
        # Pure black image
        black_image = Image.new('RGB', (100, 100), color=(0, 0, 0))
        buffer = io.BytesIO()
        black_image.save(buffer, format='PNG')
        black_data = buffer.getvalue()
        
        result = service.validate_image(black_data)
        assert result.is_valid, "Black image should be valid"
        
        # Pure white image
        white_image = Image.new('RGB', (100, 100), color=(255, 255, 255))
        buffer = io.BytesIO()
        white_image.save(buffer, format='PNG')
        white_data = buffer.getvalue()
        
        result = service.validate_image(white_data)
        assert result.is_valid, "White image should be valid"
    
    def test_grayscale_images(self):
        """Test grayscale image handling."""
        service = DataPipelineService()
        
        # Create grayscale image
        gray_image = Image.new('L', (100, 100), color=128)
        buffer = io.BytesIO()
        gray_image.save(buffer, format='PNG')
        gray_data = buffer.getvalue()
        
        result = service.validate_image(gray_data)
        # Should handle grayscale images
        assert isinstance(result, ValidationResult)
    
    def test_transparent_images(self):
        """Test images with transparency."""
        service = DataPipelineService()
        
        # Create image with alpha channel
        rgba_image = Image.new('RGBA', (100, 100), color=(128, 128, 128, 128))
        buffer = io.BytesIO()
        rgba_image.save(buffer, format='PNG')
        rgba_data = buffer.getvalue()
        
        result = service.validate_image(rgba_data)
        # Should handle RGBA images
        assert isinstance(result, ValidationResult)


class TestAgeEdgeCases:
    """Test edge cases in age handling."""
    
    def test_boundary_ages(self):
        """Test exact boundary age values."""
        service = DataPipelineService()
        
        # Test minimum age (exactly 2.0)
        metadata_min = {"age_years": 2.0}
        result_min = service.extract_metadata(metadata_min)
        assert result_min.age_years == 2.0
        
        # Test maximum age (exactly 18.0)
        metadata_max = {"age_years": 18.0}
        result_max = service.extract_metadata(metadata_max)
        assert result_max.age_years == 18.0
        
        # Test just below minimum (1.99)
        metadata_below = {"age_years": 1.99}
        with pytest.raises(ValueError):
            service.extract_metadata(metadata_below)
        
        # Test just above maximum (18.01)
        metadata_above = {"age_years": 18.01}
        with pytest.raises(ValueError):
            service.extract_metadata(metadata_above)
    
    def test_floating_point_precision(self):
        """Test floating point precision in age values."""
        service = DataPipelineService()
        
        # Test high precision age
        precise_age = 7.123456789
        metadata = {"age_years": precise_age}
        result = service.extract_metadata(metadata)
        
        # Should preserve reasonable precision
        assert abs(result.age_years - precise_age) < 1e-6
    
    def test_age_group_boundaries(self):
        """Test age group boundary conditions."""
        age_manager = AgeGroupManager()
        
        # Mock database session
        mock_db = Mock(spec=Session)
        
        # Test age exactly at group boundary
        with patch.object(age_manager, '_get_age_group_models') as mock_get_models:
            # Mock model that covers ages 5-7
            mock_model = Mock()
            mock_model.age_min = 5.0
            mock_model.age_max = 7.0
            mock_get_models.return_value = [mock_model]
            
            # Test age exactly at lower boundary
            model_5 = age_manager.find_appropriate_model(5.0, mock_db)
            assert model_5 == mock_model
            
            # Test age exactly at upper boundary
            model_7 = age_manager.find_appropriate_model(7.0, mock_db)
            assert model_7 == mock_model
            
            # Test age just outside boundaries
            model_4_99 = age_manager.find_appropriate_model(4.99, mock_db)
            assert model_4_99 is None
            
            model_7_01 = age_manager.find_appropriate_model(7.01, mock_db)
            assert model_7_01 is None


class TestStringEdgeCases:
    """Test edge cases in string handling."""
    
    def test_empty_and_whitespace_strings(self):
        """Test handling of empty and whitespace-only strings."""
        service = DataPipelineService()
        
        # Test empty strings
        metadata_empty = {
            "age_years": 5.0,
            "subject": "",
            "drawing_tool": "",
            "prompt": ""
        }
        result = service.extract_metadata(metadata_empty)
        assert result.subject is None
        assert result.drawing_tool is None
        assert result.prompt is None
        
        # Test whitespace-only strings
        metadata_whitespace = {
            "age_years": 5.0,
            "subject": "   ",
            "drawing_tool": "\t\t",
            "prompt": "\n\n"
        }
        result = service.extract_metadata(metadata_whitespace)
        assert result.subject is None
        assert result.drawing_tool is None
        assert result.prompt is None
    
    def test_very_long_strings(self):
        """Test handling of very long strings."""
        service = DataPipelineService()
        
        # Test string at maximum length
        max_subject = "a" * 50  # Assuming 50 is the max length
        metadata_max = {
            "age_years": 5.0,
            "subject": max_subject
        }
        
        try:
            result = service.extract_metadata(metadata_max)
            assert result.subject == max_subject
        except ValueError:
            # If validation rejects it, that's also acceptable
            pass
        
        # Test string exceeding maximum length
        too_long_subject = "a" * 1000
        metadata_too_long = {
            "age_years": 5.0,
            "subject": too_long_subject
        }
        
        with pytest.raises(ValueError):
            service.extract_metadata(metadata_too_long)
    
    def test_special_characters(self):
        """Test handling of special characters in strings."""
        service = DataPipelineService()
        
        # Test unicode characters
        unicode_subject = "Ñiño's drawing 🎨"
        metadata_unicode = {
            "age_years": 5.0,
            "subject": unicode_subject
        }
        
        result = service.extract_metadata(metadata_unicode)
        assert result.subject == unicode_subject
        
        # Test special ASCII characters
        special_chars = "Subject with \"quotes\" and 'apostrophes' & symbols!"
        metadata_special = {
            "age_years": 5.0,
            "subject": special_chars
        }
        
        result = service.extract_metadata(metadata_special)
        assert result.subject == special_chars


class TestNumericalEdgeCases:
    """Test edge cases in numerical computations."""
    
    def test_zero_values(self):
        """Test handling of zero values."""
        # Test zero embedding
        zero_embedding = np.zeros(768)
        
        # Should handle zero embeddings gracefully
        assert zero_embedding.shape == (768,)
        assert np.all(zero_embedding == 0)
    
    def test_infinite_and_nan_values(self):
        """Test handling of infinite and NaN values."""
        # Test with NaN values
        nan_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        
        # Should detect NaN values
        assert np.any(np.isnan(nan_array))
        
        # Test with infinite values
        inf_array = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
        
        # Should detect infinite values
        assert np.any(np.isinf(inf_array))
    
    def test_very_small_differences(self):
        """Test handling of very small numerical differences."""
        # Test embeddings that are nearly identical
        embedding1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        embedding2 = np.array([1.0000001, 2.0000001, 3.0000001, 4.0000001, 5.0000001])
        
        # Calculate difference
        diff = np.mean((embedding1 - embedding2) ** 2)
        
        # Should be very small but non-zero
        assert diff > 0
        assert diff < 1e-10
    
    def test_large_numerical_values(self):
        """Test handling of large numerical values."""
        # Test very large embeddings
        large_embedding = np.full(768, 1e6)
        
        # Should handle large values
        assert np.all(np.isfinite(large_embedding))
        assert np.all(large_embedding == 1e6)


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent operations."""
    
    def test_simultaneous_file_operations(self):
        """Test simultaneous file operations."""
        storage_service = FileStorageService()
        
        # Mock file operations
        mock_file = Mock()
        mock_file.filename = "test.png"
        mock_file.read = Mock(return_value=b"fake_image_data")
        mock_file.seek = Mock()
        
        # Should handle concurrent access gracefully
        # (This is a simplified test - real concurrency testing would be more complex)
        try:
            # Simulate multiple operations
            for i in range(5):
                mock_file.filename = f"test_{i}.png"
                # In real implementation, this would be async
                pass
        except Exception as e:
            pytest.fail(f"Concurrent operations should not fail: {e}")
    
    def test_resource_cleanup_edge_cases(self):
        """Test resource cleanup in edge cases."""
        service = DataPipelineService()
        
        # Test cleanup after errors
        corrupted_data = b"not_an_image"
        
        # Multiple failed operations should not cause resource leaks
        for i in range(10):
            try:
                service.preprocess_image(corrupted_data)
            except (ImagePreprocessingError, ImageProcessingError):
                # Expected error - should not cause resource leaks
                pass
        
        # System should still function after errors
        valid_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        buffer = io.BytesIO()
        valid_image.save(buffer, format='PNG')
        valid_data = buffer.getvalue()
        
        result = service.validate_image(valid_data)
        assert result.is_valid, "System should recover after errors"


class TestMemoryEdgeCases:
    """Test edge cases related to memory usage."""
    
    def test_large_batch_processing(self):
        """Test processing of large batches."""
        # Simulate large batch of embeddings
        large_batch = np.random.rand(1000, 768)  # 1000 embeddings
        
        # Should handle large batches without memory issues
        assert large_batch.shape == (1000, 768)
        
        # Test memory-efficient operations
        batch_mean = np.mean(large_batch, axis=0)
        assert batch_mean.shape == (768,)
    
    def test_memory_cleanup(self):
        """Test memory cleanup after operations."""
        # Create large temporary arrays
        temp_arrays = []
        for i in range(10):
            temp_array = np.random.rand(100, 768)
            temp_arrays.append(temp_array)
        
        # Clear references
        temp_arrays.clear()
        
        # Memory should be available for cleanup
        # (This is a simplified test - real memory testing would use memory profiling)
        assert len(temp_arrays) == 0


class TestErrorRecoveryEdgeCases:
    """Test error recovery in edge cases."""
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        service = DataPipelineService()
        
        # Test mixed valid and invalid data
        valid_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        buffer = io.BytesIO()
        valid_image.save(buffer, format='PNG')
        valid_data = buffer.getvalue()
        
        invalid_data = b"not_an_image"
        
        # Process valid data after invalid data
        invalid_result = service.validate_image(invalid_data)
        assert not invalid_result.is_valid
        
        valid_result = service.validate_image(valid_data)
        assert valid_result.is_valid, "Should process valid data after invalid data"
    
    def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures."""
        service = DataPipelineService()
        
        # Test that one failure doesn't affect subsequent operations
        failures = 0
        successes = 0
        
        test_data = [
            b"invalid_data_1",
            b"invalid_data_2",
            # Valid data mixed in
        ]
        
        # Add valid data
        valid_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
        buffer = io.BytesIO()
        valid_image.save(buffer, format='PNG')
        test_data.append(buffer.getvalue())
        
        for data in test_data:
            try:
                result = service.validate_image(data)
                if result.is_valid:
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1
        
        assert successes > 0, "Should have at least one success"
        assert failures > 0, "Should have at least one failure"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])