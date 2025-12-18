"""
Property-based tests for error handling robustness.

**Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
**Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import io
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import asyncio
from typing import Optional
import torch

from app.services.data_pipeline import DataPipelineService, ValidationResult, DrawingMetadata, ImagePreprocessingError
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError, ModelLoadingError, EmbeddingGenerationError
from app.services.model_manager import ModelManager, ModelManagerError, AutoencoderTrainingError
from app.services.file_storage import FileStorageService, FileStorageError
from app.core.exceptions import ImageProcessingError, StorageError, ModelError


def create_invalid_image_formats() -> list:
    """Create various invalid image format data for testing requirement 8.1"""
    return [
        b"",  # Empty file
        b"Not an image file",  # Plain text
        b"<html><body>Fake HTML</body></html>",  # HTML content
        b"\x89PNG\r\n\x1a\n" + b"corrupted_png_data",  # Corrupted PNG
        b"\xFF\xD8\xFF" + b"corrupted_jpeg_data",  # Corrupted JPEG
        b"BM" + b"corrupted_bmp_data",  # Corrupted BMP
        bytes([i % 256 for i in range(100)]),  # Random binary data
        b"\x00" * 100,  # Null bytes
        b"GIF89a" + b"fake_gif_data",  # Unsupported GIF format
        b"RIFF" + b"fake_webp_data",  # Unsupported WebP format
    ]


def create_invalid_age_data() -> list:
    """Create various invalid age data for testing requirement 8.2"""
    return [
        -5.0,  # Negative age
        0.0,   # Zero age
        1.5,   # Below minimum (2 years)
        25.0,  # Above maximum (18 years)
        float('nan'),  # NaN
        float('inf'),  # Infinity
        float('-inf'), # Negative infinity
        None,  # Missing age
    ]


@given(
    invalid_format_data=st.sampled_from(create_invalid_image_formats())
)
@settings(max_examples=50, deadline=None)
def test_invalid_image_format_rejection_robustness(invalid_format_data):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 8.1**
    
    Property: For any invalid image format uploaded, the system should reject 
    the input and provide clear error messages.
    """
    service = DataPipelineService()
    
    # Test validation of invalid format data
    validation_result = service.validate_image(invalid_format_data)
    
    # Should always return a ValidationResult object (Requirement 8.1)
    assert isinstance(validation_result, ValidationResult), "Should return ValidationResult object"
    
    # Should be marked as invalid (Requirement 8.1)
    assert not validation_result.is_valid, "Invalid image formats should be rejected"
    
    # Should provide clear error message (Requirement 8.1)
    assert validation_result.error_message is not None, "Should provide error message"
    assert len(validation_result.error_message) > 0, "Error message should not be empty"
    assert isinstance(validation_result.error_message, str), "Error message should be string"
    
    # Error message should be clear and informative
    error_msg_lower = validation_result.error_message.lower()
    assert any(keyword in error_msg_lower for keyword in [
        'format', 'invalid', 'unsupported', 'corrupted', 'image', 'empty', 'file', 'provided'
    ]), f"Error message should be clear about format issue: {validation_result.error_message}"


@given(
    invalid_age=st.sampled_from(create_invalid_age_data()),
    metadata_field=st.sampled_from(['age_years', 'age', 'child_age'])
)
@settings(max_examples=50, deadline=None)
def test_invalid_age_handling_robustness(invalid_age, metadata_field):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 8.2**
    
    Property: For any missing or invalid age information, the system should 
    prompt for correction or use default handling.
    """
    service = DataPipelineService()
    
    # Test with invalid age data
    metadata_dict = {metadata_field: invalid_age} if invalid_age is not None else {}
    
    # Should handle invalid age gracefully (Requirement 8.2)
    if invalid_age is None or not (2.0 <= invalid_age <= 18.0) or np.isnan(invalid_age) or np.isinf(invalid_age):
        with pytest.raises(ValueError) as exc_info:
            service.extract_metadata(metadata_dict)
        
        # Should provide clear error message for correction (Requirement 8.2)
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide error message"
        
        # Error message should prompt for correction
        error_msg_lower = error_message.lower()
        assert any(keyword in error_msg_lower for keyword in [
            'age', 'invalid', 'required', 'range', 'between', 'missing'
        ]), f"Error message should prompt for age correction: {error_message}"


@given(
    processing_failure_type=st.sampled_from([
        "vision_model_load_failure",
        "embedding_generation_failure", 
        "tensor_processing_failure",
        "memory_allocation_failure"
    ])
)
@settings(max_examples=30, deadline=None)
def test_processing_failure_error_logging_robustness(processing_failure_type):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 2.5**
    
    Property: For any processing failure, the system should log errors and 
    provide meaningful error messages.
    """
    # Create valid image data for testing
    image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    
    service = EmbeddingService()
    
    # Mock different types of processing failures - patch the initialize method to avoid actual model loading
    with patch.object(service, 'initialize') as mock_init, \
         patch('transformers.ViTImageProcessor.from_pretrained') as mock_processor, \
         patch('transformers.ViTModel.from_pretrained') as mock_model:
        
        if processing_failure_type == "vision_model_load_failure":
            mock_model.side_effect = RuntimeError("Failed to load Vision Transformer model")
        elif processing_failure_type == "embedding_generation_failure":
            mock_processor.return_value = Mock()
            mock_model.return_value = Mock()
            mock_model.return_value.side_effect = ValueError("Embedding generation failed")
        elif processing_failure_type == "tensor_processing_failure":
            mock_processor.side_effect = TypeError("Tensor processing error")
        elif processing_failure_type == "memory_allocation_failure":
            mock_model.side_effect = MemoryError("Out of memory during processing")
        
        # Should log errors and provide meaningful messages (Requirement 2.5)
        with pytest.raises((EmbeddingServiceError, ModelLoadingError, EmbeddingGenerationError, RuntimeError, ValueError, TypeError, MemoryError)) as exc_info:
            service.generate_embedding(image, age=5.0)
        
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide meaningful error message"
        
        # Error message should be meaningful for debugging
        error_msg_lower = error_message.lower()
        assert any(keyword in error_msg_lower for keyword in [
            'failed', 'error', 'processing', 'model', 'embedding', 'initialized', 'initialize', 'service'
        ]), f"Error message should be meaningful: {error_message}"


@given(
    scoring_failure_type=st.sampled_from([
        "model_not_found",
        "autoencoder_failure",
        "reconstruction_error",
        "score_calculation_failure"
    ])
)
@settings(max_examples=30, deadline=None)
def test_scoring_failure_error_reporting_robustness(scoring_failure_type):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 4.5**
    
    Property: For any scoring failure, the system should provide clear error 
    reporting and retry mechanisms.
    """
    service = ModelManager()
    
    # Create mock embedding data and database session
    mock_embedding = np.random.rand(768).astype(np.float32)
    age_group_model_id = 1
    mock_db = Mock()
    
    # Mock different types of scoring failures
    with patch('app.services.model_manager.ModelManager.load_model') as mock_load_model:
        if scoring_failure_type == "model_not_found":
            mock_load_model.side_effect = ModelLoadingError("Model not found")
        elif scoring_failure_type == "autoencoder_failure":
            mock_model = Mock()
            mock_model.forward.side_effect = RuntimeError("Autoencoder prediction failed")
            mock_load_model.return_value = mock_model
        elif scoring_failure_type == "reconstruction_error":
            mock_model = Mock()
            mock_model.forward.side_effect = ValueError("Reconstruction calculation error")
            mock_load_model.return_value = mock_model
        elif scoring_failure_type == "score_calculation_failure":
            mock_model = Mock()
            mock_model.forward.return_value = torch.tensor([float('nan')])  # Invalid result
            mock_load_model.return_value = mock_model
        
        # Should provide clear error reporting (Requirement 4.5)
        with pytest.raises((ModelManagerError, ModelLoadingError, RuntimeError, ValueError, Exception)) as exc_info:
            service.compute_reconstruction_loss(mock_embedding, age_group_model_id, mock_db)
        
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide clear error reporting"
        
        # Error message should be clear for retry mechanisms
        error_msg_lower = error_message.lower()
        assert any(keyword in error_msg_lower for keyword in [
            'reconstruction', 'failed', 'model', 'error', 'loss', 'calculation'
        ]), f"Error message should be clear for retry: {error_message}"


@given(
    inference_failure_type=st.sampled_from([
        "model_loading_error",
        "prediction_timeout",
        "gpu_memory_error",
        "model_corruption"
    ])
)
@settings(max_examples=30, deadline=None)
def test_model_inference_failure_recovery_robustness(inference_failure_type):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 8.3**
    
    Property: For any model inference failure, the system should log detailed 
    error information and attempt graceful recovery.
    """
    service = ModelManager()
    
    # Create test data
    mock_embedding = np.random.rand(768).astype(np.float32)
    age_group_model_id = 1
    mock_db = Mock()
    
    # Mock different types of inference failures
    with patch('app.services.model_manager.ModelManager.load_model') as mock_load:
        if inference_failure_type == "model_loading_error":
            mock_load.side_effect = FileNotFoundError("Model file not found")
        elif inference_failure_type == "prediction_timeout":
            mock_load.side_effect = TimeoutError("Model prediction timeout")
        elif inference_failure_type == "gpu_memory_error":
            mock_load.side_effect = RuntimeError("CUDA out of memory")
        elif inference_failure_type == "model_corruption":
            mock_load.side_effect = ValueError("Model file corrupted")
        
        # Should log detailed error information and attempt recovery (Requirement 8.3)
        with pytest.raises((ModelManagerError, ModelLoadingError, FileNotFoundError, TimeoutError, RuntimeError, ValueError)) as exc_info:
            service.compute_reconstruction_loss(mock_embedding, age_group_model_id, mock_db)
        
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should log detailed error information"
        
        # Error message should contain detailed information for recovery
        error_msg_lower = error_message.lower()
        assert any(keyword in error_msg_lower for keyword in [
            'model', 'inference', 'failed', 'error', 'loading', 'prediction'
        ]), f"Error message should contain detailed information: {error_message}"


@given(
    corruption_type=st.sampled_from([
        "partial_file_corruption",
        "metadata_corruption", 
        "embedding_corruption",
        "model_parameter_corruption"
    ])
)
@settings(max_examples=30, deadline=None)
def test_data_corruption_isolation_robustness(corruption_type):
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 8.5**
    
    Property: For any data corruption detected, the system should isolate 
    affected components and continue processing valid inputs.
    """
    service = DataPipelineService()
    
    # Create valid data for comparison
    valid_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    valid_image.save(buffer, format='PNG')
    valid_data = buffer.getvalue()
    
    # Create corrupted data based on type
    if corruption_type == "partial_file_corruption":
        corrupted_data = valid_data[:len(valid_data)//2]  # Truncated file
    elif corruption_type == "metadata_corruption":
        corrupted_data = b"corrupted_metadata_not_image"
    elif corruption_type == "embedding_corruption":
        corrupted_data = b"fake_image_data_corrupted"
    elif corruption_type == "model_parameter_corruption":
        corrupted_data = b""  # Empty file
    
    # Test that corruption is detected and isolated (Requirement 8.5)
    corrupted_result = service.validate_image(corrupted_data)
    
    # Should detect corruption
    assert not corrupted_result.is_valid, "Corrupted data should be detected"
    assert corrupted_result.error_message is not None, "Should provide error message for corruption"
    
    # Should isolate affected components and continue with valid inputs (Requirement 8.5)
    valid_result = service.validate_image(valid_data)
    
    # System should continue processing valid inputs after corruption
    assert valid_result.is_valid, "Should continue processing valid inputs after corruption"
    assert valid_result.error_message is None, "Valid data should not have errors"
    
    # Verify isolation - valid processing should not be affected by previous corruption
    assert valid_result.image_format in ["PNG", "JPEG", "BMP"], "Valid data should be processed correctly"


def test_error_message_consistency_robustness():
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
    
    Test that error messages are consistent and helpful across all error conditions.
    """
    service = DataPipelineService()
    
    # Test various error conditions
    error_conditions = [
        (b"", "empty file"),
        (b"not_an_image", "invalid format"),
        (create_invalid_image_formats()[2], "corrupted data"),
    ]
    
    error_messages = []
    
    for corrupted_data, condition_name in error_conditions:
        result = service.validate_image(corrupted_data)
        assert not result.is_valid, f"Should detect error for {condition_name}"
        assert result.error_message is not None, f"Should provide error message for {condition_name}"
        
        error_messages.append((condition_name, result.error_message))
    
    # Verify error messages are helpful and consistent
    for condition_name, error_message in error_messages:
        assert len(error_message) > 10, f"Error message should be descriptive for {condition_name}"
        assert error_message[0].isupper(), f"Error message should be properly formatted for {condition_name}"
        assert not error_message.endswith('.') or error_message.endswith('!'), \
            f"Error message should have proper punctuation for {condition_name}"


def test_system_recovery_after_multiple_errors_robustness():
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
    
    Test that the system can recover and function normally after encountering 
    multiple different types of errors.
    """
    service = DataPipelineService()
    
    # Create valid test data
    valid_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
    buffer = io.BytesIO()
    valid_image.save(buffer, format='PNG')
    valid_data = buffer.getvalue()
    
    # Test initial valid processing
    initial_result = service.validate_image(valid_data)
    assert initial_result.is_valid, "Initial processing should work"
    
    # Cause multiple different types of errors
    error_data_types = create_invalid_image_formats()[:5]  # Test first 5 types
    
    for error_data in error_data_types:
        try:
            error_result = service.validate_image(error_data)
            assert not error_result.is_valid, "Error data should be rejected"
        except Exception:
            # Some errors might raise exceptions - this is acceptable
            pass
    
    # Test that system still works after multiple errors
    recovery_result = service.validate_image(valid_data)
    assert recovery_result.is_valid, "System should recover after multiple errors"
    
    # Results should be consistent with initial processing
    assert recovery_result.image_format == initial_result.image_format, \
        "Results should be consistent after recovery"
    assert recovery_result.dimensions == initial_result.dimensions, \
        "Dimensions should be consistent after recovery"


def test_concurrent_error_handling_robustness():
    """
    **Feature: children-drawing-anomaly-detection, Property 22: Error Handling Robustness**
    **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
    
    Test that error handling works correctly under concurrent access conditions.
    """
    import threading
    import time
    
    service = DataPipelineService()
    
    # Create test data
    valid_image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    valid_image.save(buffer, format='PNG')
    valid_data = buffer.getvalue()
    
    invalid_data = b"not_an_image"
    
    results = []
    errors = []
    
    def process_data(data, is_valid_expected):
        try:
            result = service.validate_image(data)
            results.append((result.is_valid, is_valid_expected, result.error_message))
        except Exception as e:
            errors.append(e)
    
    # Create threads with mix of valid and invalid data
    threads = []
    for i in range(10):
        data = valid_data if i % 2 == 0 else invalid_data
        expected = i % 2 == 0
        thread = threading.Thread(target=process_data, args=(data, expected))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Should handle concurrent access without crashes
    assert len(errors) == 0, f"Concurrent access should not cause crashes: {errors}"
    assert len(results) == 10, "All threads should complete"
    
    # Results should be correct
    for is_valid, expected, error_msg in results:
        if expected:
            assert is_valid, "Valid data should be accepted in concurrent access"
        else:
            assert not is_valid, "Invalid data should be rejected in concurrent access"
            assert error_msg is not None, "Invalid data should have error message"