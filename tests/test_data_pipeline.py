"""
Property-based tests for data pipeline service.

**Feature: children-drawing-anomaly-detection, Property 2: Image Preprocessing Uniformity**
**Validates: Requirements 1.2**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import io
import tempfile
import os

from app.services.data_pipeline import DataPipelineService, ValidationResult, DrawingMetadata, ImagePreprocessingError


def create_test_image(width: int, height: int, mode: str = 'RGB', format: str = 'PNG') -> bytes:
    """Create a test image with specified dimensions and format"""
    # Create a simple test image with random colors
    if mode == 'RGB':
        image = Image.new(mode, (width, height), color=(128, 128, 128))
    elif mode == 'RGBA':
        image = Image.new(mode, (width, height), color=(128, 128, 128, 255))
    elif mode == 'L':
        image = Image.new(mode, (width, height), color=128)
    else:
        image = Image.new('RGB', (width, height), color=(128, 128, 128))
    
    # Add some pattern to make it more realistic
    pixels = image.load()
    for i in range(0, width, 10):
        for j in range(0, height, 10):
            if mode == 'RGB':
                pixels[i, j] = (255, 0, 0)
            elif mode == 'RGBA':
                pixels[i, j] = (255, 0, 0, 255)
            elif mode == 'L':
                pixels[i, j] = 255
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


# Hypothesis strategies
valid_dimensions_strategy = st.tuples(
    st.integers(min_value=32, max_value=2048),  # width
    st.integers(min_value=32, max_value=2048)   # height
)

image_mode_strategy = st.sampled_from(['RGB', 'RGBA', 'L'])
image_format_strategy = st.sampled_from(['PNG', 'JPEG', 'BMP'])

target_size_strategy = st.tuples(
    st.integers(min_value=64, max_value=512),  # width
    st.integers(min_value=64, max_value=512)   # height
)


@given(
    dimensions=valid_dimensions_strategy,
    mode=image_mode_strategy,
    format=image_format_strategy,
    target_size=target_size_strategy
)
@settings(max_examples=100, deadline=None)
def test_image_preprocessing_uniformity(dimensions, mode, format, target_size):
    """
    **Feature: children-drawing-anomaly-detection, Property 2: Image Preprocessing Uniformity**
    **Validates: Requirements 1.2**
    
    Property: For any valid input image, the preprocessing pipeline should produce 
    output tensors with identical dimensions and normalized pixel values in the expected range.
    """
    width, height = dimensions
    
    # Skip JPEG with RGBA mode as it's not supported
    assume(not (format == 'JPEG' and mode == 'RGBA'))
    
    # Create test image
    image_data = create_test_image(width, height, mode, format)
    
    # Initialize service with target size
    service = DataPipelineService(target_size=target_size)
    
    # Preprocess the image
    try:
        preprocessed = service.preprocess_image(image_data, target_size)
        
        # Verify output dimensions are consistent with target size
        expected_height, expected_width = target_size[1], target_size[0]  # PIL uses (W, H), numpy uses (H, W, C)
        assert preprocessed.shape == (expected_height, expected_width, 3), \
            f"Expected shape ({expected_height}, {expected_width}, 3), got {preprocessed.shape}"
        
        # Verify pixel values are normalized to [0, 1]
        assert preprocessed.dtype == np.float32, f"Expected dtype float32, got {preprocessed.dtype}"
        assert np.all(preprocessed >= 0.0), "All pixel values should be >= 0.0"
        assert np.all(preprocessed <= 1.0), "All pixel values should be <= 1.0"
        
        # Verify the image has 3 channels (RGB)
        assert preprocessed.shape[2] == 3, f"Expected 3 channels, got {preprocessed.shape[2]}"
        
        # Verify the array is finite (no NaN or inf values)
        assert np.all(np.isfinite(preprocessed)), "All values should be finite"
        
    except ImagePreprocessingError as e:
        # If preprocessing fails, it should be due to a valid reason
        pytest.fail(f"Preprocessing should not fail for valid input: {e}")


@given(
    dimensions1=valid_dimensions_strategy,
    dimensions2=valid_dimensions_strategy,
    target_size=target_size_strategy
)
@settings(max_examples=50, deadline=None)
def test_consistent_output_dimensions(dimensions1, dimensions2, target_size):
    """
    Property: For any two different input images processed with the same target size,
    the output dimensions should be identical.
    """
    width1, height1 = dimensions1
    width2, height2 = dimensions2
    
    # Create two different test images
    image_data1 = create_test_image(width1, height1, 'RGB', 'PNG')
    image_data2 = create_test_image(width2, height2, 'RGB', 'PNG')
    
    service = DataPipelineService(target_size=target_size)
    
    # Preprocess both images
    preprocessed1 = service.preprocess_image(image_data1, target_size)
    preprocessed2 = service.preprocess_image(image_data2, target_size)
    
    # Verify both have identical dimensions
    assert preprocessed1.shape == preprocessed2.shape, \
        f"Output shapes should be identical: {preprocessed1.shape} vs {preprocessed2.shape}"
    
    # Verify both have the expected target dimensions
    expected_height, expected_width = target_size[1], target_size[0]
    assert preprocessed1.shape == (expected_height, expected_width, 3)
    assert preprocessed2.shape == (expected_height, expected_width, 3)


@given(
    dimensions=valid_dimensions_strategy,
    mode=image_mode_strategy
)
@settings(max_examples=50, deadline=None)
def test_mode_conversion_consistency(dimensions, mode):
    """
    Property: For any input image mode (RGB, RGBA, L), the output should always be RGB with 3 channels.
    """
    width, height = dimensions
    
    # Create test image with specified mode
    image_data = create_test_image(width, height, mode, 'PNG')
    
    service = DataPipelineService()
    
    # Preprocess the image
    preprocessed = service.preprocess_image(image_data)
    
    # Verify output is always RGB (3 channels)
    assert preprocessed.shape[2] == 3, f"Expected 3 channels for RGB output, got {preprocessed.shape[2]}"
    
    # Verify pixel values are in valid range
    assert np.all(preprocessed >= 0.0) and np.all(preprocessed <= 1.0), \
        "Pixel values should be in range [0, 1]"


@given(
    width=st.integers(min_value=1, max_value=31),
    height=st.integers(min_value=1, max_value=31)
)
@settings(max_examples=30, deadline=None)
def test_small_image_rejection(width, height):
    """
    Property: For any image smaller than 32x32 pixels, the validation should fail.
    """
    # Create a small test image
    image_data = create_test_image(width, height, 'RGB', 'PNG')
    
    service = DataPipelineService()
    
    # Validation should fail for small images
    validation_result = service.validate_image(image_data)
    assert not validation_result.is_valid, "Small images should be rejected"
    assert "too small" in validation_result.error_message.lower(), \
        f"Error message should mention size issue: {validation_result.error_message}"


@given(
    age=st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False),
    subject=st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
    expert_label=st.one_of(st.none(), st.sampled_from(['normal', 'concern', 'severe'])),
    drawing_tool=st.one_of(st.none(), st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
    prompt=st.one_of(st.none(), st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
)
@settings(max_examples=50, deadline=None)
def test_metadata_extraction_consistency(age, subject, expert_label, drawing_tool, prompt):
    """
    Property: For any valid metadata dictionary, the extraction should produce 
    a consistent DrawingMetadata object with all fields properly validated.
    """
    service = DataPipelineService()
    
    # Create metadata dictionary
    metadata_dict = {
        'age_years': age,
        'subject': subject,
        'expert_label': expert_label,
        'drawing_tool': drawing_tool,
        'prompt': prompt
    }
    
    # Extract metadata
    metadata = service.extract_metadata(metadata_dict)
    
    # Verify all fields are preserved correctly
    assert metadata.age_years == age, f"Age should be preserved: expected {age}, got {metadata.age_years}"
    assert 2.0 <= metadata.age_years <= 18.0, "Age should be within valid range"
    
    assert metadata.subject == subject, f"Subject should be preserved: expected {subject}, got {metadata.subject}"
    assert metadata.expert_label == expert_label, f"Expert label should be preserved: expected {expert_label}, got {metadata.expert_label}"
    assert metadata.drawing_tool == drawing_tool, f"Drawing tool should be preserved: expected {drawing_tool}, got {metadata.drawing_tool}"
    assert metadata.prompt == prompt, f"Prompt should be preserved: expected {prompt}, got {metadata.prompt}"


@given(
    dimensions=valid_dimensions_strategy,
    age=st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30, deadline=None)
def test_combined_validation_preprocessing_consistency(dimensions, age):
    """
    Property: For any valid image and metadata, the combined validation and preprocessing 
    should produce consistent results.
    """
    width, height = dimensions
    
    # Create test image and metadata
    image_data = create_test_image(width, height, 'RGB', 'PNG')
    metadata_dict = {'age_years': age}
    
    service = DataPipelineService()
    
    # Run combined validation and preprocessing
    preprocessed_image, validated_metadata = service.validate_and_preprocess(image_data, metadata_dict)
    
    # Verify image preprocessing results
    assert isinstance(preprocessed_image, np.ndarray), "Should return numpy array"
    assert preprocessed_image.shape == (224, 224, 3), "Should have default target size"
    assert preprocessed_image.dtype == np.float32, "Should be float32"
    assert np.all(preprocessed_image >= 0.0) and np.all(preprocessed_image <= 1.0), "Values should be normalized"
    
    # Verify metadata validation results
    assert isinstance(validated_metadata, DrawingMetadata), "Should return DrawingMetadata object"
    assert validated_metadata.age_years == age, "Age should be preserved"


def test_corrupted_image_handling():
    """
    Test that corrupted image data is properly handled.
    """
    service = DataPipelineService()
    
    # Test with completely invalid data
    invalid_data = b"This is not an image"
    validation_result = service.validate_image(invalid_data)
    assert not validation_result.is_valid
    assert "corrupted" in validation_result.error_message.lower() or "invalid" in validation_result.error_message.lower()
    
    # Test with empty data
    empty_data = b""
    validation_result = service.validate_image(empty_data)
    assert not validation_result.is_valid
    assert "empty" in validation_result.error_message.lower()


def test_unsupported_format_handling():
    """
    Test that unsupported image formats are properly rejected.
    """
    service = DataPipelineService()
    
    # Create a TIFF image (unsupported format)
    image = Image.new('RGB', (100, 100), color=(128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format='TIFF')
    tiff_data = buffer.getvalue()
    
    validation_result = service.validate_image(tiff_data)
    assert not validation_result.is_valid
    assert "unsupported" in validation_result.error_message.lower()
    assert "TIFF" in validation_result.error_message or "tiff" in validation_result.error_message.lower()