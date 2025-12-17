"""
Property-based tests for error handling robustness.

**Feature: children-drawing-anomaly-detection, Property 13: Error Handling Robustness**
**Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import io
import tempfile
import os
from unittest.mock import Mock, patch

from app.services.data_pipeline import DataPipelineService, ValidationResult, DrawingMetadata, ImagePreprocessingError
from app.services.file_storage import FileStorageService, FileStorageError
from app.core.exceptions import ImageProcessingError, StorageError


def create_corrupted_image_data() -> bytes:
    """Create various types of corrupted image data for testing"""
    corrupted_data_types = [
        b"",  # Empty file
        b"Not an image file",  # Plain text
        b"\x89PNG\r\n\x1a\n" + b"corrupted_data",  # Corrupted PNG header
        b"\xFF\xD8\xFF" + b"corrupted_jpeg_data",  # Corrupted JPEG header
        b"BM" + b"corrupted_bmp_data",  # Corrupted BMP header
        bytes([i % 256 for i in range(1000)]),  # Random binary data
        b"<html><body>Not an image</body></html>",  # HTML content
        b"\x00" * 1000,  # Null bytes
    ]
    return corrupted_data_types


def create_oversized_image_data(size_mb: int) -> bytes:
    """Create image data that exceeds size limits"""
    # Create a large image that would exceed memory/size limits
    try:
        # Create a very wide image to simulate large file
        width = min(10000, size_mb * 100)  # Reasonable width to avoid memory issues in tests
        height = min(1000, size_mb * 10)
        image = Image.new('RGB', (width, height), color=(255, 0, 0))
        
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=False)
        return buffer.getvalue()
    except Exception:
        # If we can't create the large image, return a large byte array
        return b"fake_large_image_data" * (size_mb * 1024 * 50)  # Approximate size


@given(
    corrupted_data=st.sampled_from(create_corrupted_image_data())
)
@settings(max_examples=50, deadline=None)
def test_corrupted_image_handling_robustness(corrupted_data):
    """
    **Feature: children-drawing-anomaly-detection, Property 13: Error Handling Robustness**
    **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**
    
    Property: For any corrupted or invalid image data, the system should provide 
    appropriate error messages and continue processing valid inputs.
    """
    service = DataPipelineService()
    
    # Test validation of corrupted data
    validation_result = service.validate_image(corrupted_data)
    
    # Should always return a ValidationResult object
    assert isinstance(validation_result, ValidationResult), "Should return ValidationResult object"
    
    # Should be marked as invalid
    assert not validation_result.is_valid, "Corrupted data should be marked as invalid"
    
    # Should have a meaningful error message
    assert validation_result.error_message is not None, "Should provide error message"
    assert len(validation_result.error_message) > 0, "Error message should not be empty"
    assert isinstance(validation_result.error_message, str), "Error message should be string"
    
    # Should not raise unhandled exceptions
    try:
        # Attempt preprocessing (should fail gracefully)
        with pytest.raises((ImagePreprocessingError, ImageProcessingError)):
            service.preprocess_image(corrupted_data)
    except (ImagePreprocessingError, ImageProcessingError):
        # This is expected - the error should be caught and wrapped
        pass
    except Exception as e:
        pytest.fail(f"Unexpected exception type: {type(e).__name__}: {str(e)}")


@given(
    invalid_age=st.one_of(
        st.floats(min_value=-100.0, max_value=1.99, allow_nan=False, allow_infinity=False),
        st.floats(min_value=18.01, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.just(float('nan')),
        st.just(float('inf')),
        st.just(float('-inf'))
    ),
    invalid_string=st.one_of(
        st.text(min_size=1000, max_size=2000),  # Too long
        st.just(""),  # Empty (should be converted to None)
        st.just("   "),  # Whitespace only
    )
)
@settings(max_examples=50, deadline=None)
def test_invalid_metadata_handling_robustness(invalid_age, invalid_string):
    """
    Property: For any invalid metadata values, the system should provide 
    clear error messages and reject the input gracefully.
    """
    service = DataPipelineService()
    
    # Test invalid age handling
    if not (2.0 <= invalid_age <= 18.0) and not np.isnan(invalid_age) and not np.isinf(invalid_age):
        metadata_dict = {"age_years": invalid_age}
        
        with pytest.raises(ValueError) as exc_info:
            service.extract_metadata(metadata_dict)
        
        # Should provide meaningful error message
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide error message"
        assert "age" in error_message.lower() or "invalid" in error_message.lower(), \
            f"Error message should mention age or validation issue: {error_message}"
    
    # Test handling of NaN and infinity values
    if np.isnan(invalid_age) or np.isinf(invalid_age):
        metadata_dict = {"age_years": invalid_age}
        
        with pytest.raises(ValueError):
            service.extract_metadata(metadata_dict)


@given(
    file_operation_error=st.sampled_from([
        "permission_denied",
        "disk_full", 
        "path_not_found",
        "file_locked"
    ])
)
@settings(max_examples=20, deadline=None)
def test_file_storage_error_handling_robustness(file_operation_error):
    """
    Property: For any file system error condition, the storage service should 
    handle errors gracefully and provide meaningful error messages.
    """
    service = FileStorageService()
    
    # Create a mock file-like object
    mock_file = Mock()
    mock_file.filename = "test.png"
    mock_file.read = Mock(return_value=b"fake_image_data")
    mock_file.seek = Mock()
    
    # Simulate different file system errors
    with patch('aiofiles.open') as mock_open:
        if file_operation_error == "permission_denied":
            mock_open.side_effect = PermissionError("Permission denied")
        elif file_operation_error == "disk_full":
            mock_open.side_effect = OSError("No space left on device")
        elif file_operation_error == "path_not_found":
            mock_open.side_effect = FileNotFoundError("Path not found")
        elif file_operation_error == "file_locked":
            mock_open.side_effect = OSError("File is locked")
        
        # Should raise StorageError with meaningful message
        with pytest.raises((FileStorageError, StorageError)) as exc_info:
            import asyncio
            asyncio.run(service.save_uploaded_file(mock_file))
        
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide error message"
        assert "failed" in error_message.lower(), f"Error message should indicate failure: {error_message}"


@given(
    memory_limit_mb=st.integers(min_value=100, max_value=500)
)
@settings(max_examples=10, deadline=None)
def test_large_file_handling_robustness(memory_limit_mb):
    """
    Property: For any file that exceeds reasonable size limits, the system should 
    reject it gracefully without consuming excessive resources.
    """
    service = DataPipelineService()
    
    # Create oversized data
    oversized_data = create_oversized_image_data(memory_limit_mb)
    
    # Should handle large files gracefully
    validation_result = service.validate_image(oversized_data)
    
    # Should either reject due to size or handle gracefully
    if len(oversized_data) > service.MAX_FILE_SIZE:
        assert not validation_result.is_valid, "Oversized files should be rejected"
        assert "size" in validation_result.error_message.lower() or \
               "large" in validation_result.error_message.lower() or \
               "exceeds" in validation_result.error_message.lower(), \
               f"Error should mention size issue: {validation_result.error_message}"


@given(
    processing_error_type=st.sampled_from([
        "memory_error",
        "value_error", 
        "type_error",
        "runtime_error"
    ])
)
@settings(max_examples=20, deadline=None)
def test_processing_error_recovery_robustness(processing_error_type):
    """
    Property: For any processing error during image preprocessing, the system should 
    catch the error, wrap it appropriately, and continue functioning.
    """
    service = DataPipelineService()
    
    # Create valid image data
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    valid_image_data = buffer.getvalue()
    
    # Mock PIL operations to simulate different errors
    with patch('PIL.Image.open') as mock_open:
        if processing_error_type == "memory_error":
            mock_open.side_effect = MemoryError("Out of memory")
        elif processing_error_type == "value_error":
            mock_open.side_effect = ValueError("Invalid image data")
        elif processing_error_type == "type_error":
            mock_open.side_effect = TypeError("Wrong data type")
        elif processing_error_type == "runtime_error":
            mock_open.side_effect = RuntimeError("Processing failed")
        
        # Should wrap the error in ImageProcessingError
        with pytest.raises((ImagePreprocessingError, ImageProcessingError)) as exc_info:
            service.preprocess_image(valid_image_data)
        
        error_message = str(exc_info.value)
        assert len(error_message) > 0, "Should provide error message"
        assert "preprocessing" in error_message.lower() or "failed" in error_message.lower(), \
            f"Error message should indicate preprocessing failure: {error_message}"


def test_concurrent_access_robustness():
    """
    Test that the system handles concurrent access gracefully.
    """
    import threading
    import time
    
    service = DataPipelineService()
    
    # Create test image
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    
    results = []
    errors = []
    
    def process_image():
        try:
            result = service.validate_image(image_data)
            results.append(result)
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads to test concurrent access
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=process_image)
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Should handle concurrent access without errors
    assert len(errors) == 0, f"Concurrent access should not cause errors: {errors}"
    assert len(results) == 5, "All threads should complete successfully"
    
    # All results should be consistent
    for result in results:
        assert result.is_valid, "All validation results should be consistent"
        assert result.image_format == "PNG", "All results should have same format"


def test_resource_cleanup_robustness():
    """
    Test that resources are properly cleaned up even when errors occur.
    """
    service = DataPipelineService()
    
    # Test with corrupted data that will cause errors
    corrupted_data = b"not_an_image"
    
    # Multiple attempts should not cause resource leaks
    for _ in range(10):
        try:
            service.preprocess_image(corrupted_data)
        except (ImagePreprocessingError, ImageProcessingError):
            # Expected error - should not cause resource leaks
            pass
    
    # System should still function normally after errors
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    valid_data = buffer.getvalue()
    
    # Should still work after previous errors
    result = service.validate_image(valid_data)
    assert result.is_valid, "System should recover from previous errors"


@given(
    edge_case_dimensions=st.tuples(
        st.integers(min_value=1, max_value=31),  # Too small
        st.integers(min_value=1, max_value=31)
    )
)
@settings(max_examples=20, deadline=None)
def test_edge_case_dimension_handling_robustness(edge_case_dimensions):
    """
    Property: For any edge case image dimensions, the system should handle 
    them gracefully with appropriate error messages.
    """
    width, height = edge_case_dimensions
    
    # Create image with edge case dimensions
    image = Image.new('RGB', (width, height), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    
    service = DataPipelineService()
    
    # Should handle edge cases gracefully
    validation_result = service.validate_image(image_data)
    
    if width < 32 or height < 32:
        # Should be rejected for being too small
        assert not validation_result.is_valid, "Small images should be rejected"
        assert "small" in validation_result.error_message.lower() or \
               "size" in validation_result.error_message.lower(), \
               f"Error should mention size issue: {validation_result.error_message}"
    else:
        # Should be accepted
        assert validation_result.is_valid, "Valid sized images should be accepted"


def test_system_state_consistency_after_errors():
    """
    Test that the system maintains consistent state even after encountering errors.
    """
    service = DataPipelineService()
    
    # Process some valid data first
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    valid_data = buffer.getvalue()
    
    result1 = service.validate_image(valid_data)
    assert result1.is_valid, "Initial validation should succeed"
    
    # Cause some errors
    for corrupted_data in create_corrupted_image_data()[:3]:
        try:
            service.validate_image(corrupted_data)
            service.preprocess_image(corrupted_data)
        except (ImagePreprocessingError, Exception):
            # Errors are expected
            pass
    
    # System should still work correctly after errors
    result2 = service.validate_image(valid_data)
    assert result2.is_valid, "Validation should still work after errors"
    
    # Results should be consistent
    assert result1.image_format == result2.image_format, "Results should be consistent"
    assert result1.dimensions == result2.dimensions, "Dimensions should be consistent"