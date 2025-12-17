"""
Property-based tests for input validation consistency.

**Feature: children-drawing-anomaly-detection, Property 1: Image Format Validation Consistency**
**Validates: Requirements 1.1, 1.3**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError
import io
from PIL import Image
import tempfile
import os

from app.schemas import (
    DrawingUploadRequest, 
    ExpertLabel, 
    ImageFormat, 
    ImageValidationRequest,
    DrawingFilterRequest,
    BatchAnalysisRequest
)


# Hypothesis strategies for generating test data
valid_age_strategy = st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)
invalid_age_strategy = st.one_of(
    st.floats(min_value=-100.0, max_value=1.99, allow_nan=False, allow_infinity=False),
    st.floats(min_value=18.01, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.just(float('nan')),
    st.just(float('inf')),
    st.just(float('-inf'))
)

valid_string_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters=' -_'))
)

valid_drawing_tool_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters=' -_'))
)

invalid_string_strategy = st.text(min_size=51, max_size=200)

expert_label_strategy = st.one_of(st.none(), st.sampled_from(list(ExpertLabel)))

image_format_strategy = st.sampled_from(list(ImageFormat))


@given(
    age=valid_age_strategy,
    subject=valid_string_strategy,
    expert_label=expert_label_strategy,
    drawing_tool=valid_drawing_tool_strategy,
    prompt=st.one_of(st.none(), st.text(min_size=1, max_size=500))
)
@settings(max_examples=100, deadline=None)
def test_valid_drawing_upload_acceptance(age, subject, expert_label, drawing_tool, prompt):
    """
    **Feature: children-drawing-anomaly-detection, Property 1: Image Format Validation Consistency**
    **Validates: Requirements 1.1, 1.3**
    
    Property: For any uploaded file with valid age metadata in the range 2-18 years 
    and valid optional metadata, the system should accept it.
    """
    # Create a valid drawing upload request
    request = DrawingUploadRequest(
        age_years=age,
        subject=subject,
        expert_label=expert_label,
        drawing_tool=drawing_tool,
        prompt=prompt
    )
    
    # Verify the request was created successfully
    assert request.age_years == age, "Age should be preserved"
    assert 2.0 <= request.age_years <= 18.0, "Age should be within valid range"
    
    # Verify optional fields are handled correctly
    if subject is not None and subject.strip():
        assert request.subject == subject, "Non-empty subject should be preserved"
    else:
        # Empty strings should be converted to None by validator
        assert request.subject is None or request.subject == subject, "Empty/None subject should be handled correctly"
    
    assert request.expert_label == expert_label, "Expert label should be preserved"
    
    if drawing_tool is not None and drawing_tool.strip():
        assert request.drawing_tool == drawing_tool, "Non-empty drawing tool should be preserved"
    else:
        assert request.drawing_tool is None or request.drawing_tool == drawing_tool, "Empty/None drawing tool should be handled correctly"
    
    if prompt is not None and prompt.strip():
        assert request.prompt == prompt, "Non-empty prompt should be preserved"
    else:
        assert request.prompt is None or request.prompt == prompt, "Empty/None prompt should be handled correctly"


@given(age=invalid_age_strategy)
@settings(max_examples=50, deadline=None)
def test_invalid_age_rejection(age):
    """
    Property: For any age value outside the range 2-18 years or invalid float values,
    the system should reject the upload request.
    """
    assume(not (2.0 <= age <= 18.0))  # Only test invalid ages
    
    with pytest.raises(ValidationError) as exc_info:
        DrawingUploadRequest(age_years=age)
    
    # Verify that the validation error is related to age constraints
    errors = exc_info.value.errors()
    assert len(errors) > 0, "Should have validation errors"
    
    # Check that at least one error is about the age field
    age_errors = [error for error in errors if 'age_years' in str(error.get('loc', []))]
    assert len(age_errors) > 0, "Should have age-related validation error"


@given(
    subject=invalid_string_strategy,
    drawing_tool=invalid_string_strategy,
    prompt=st.text(min_size=501, max_size=1000)
)
@settings(max_examples=50, deadline=None)
def test_string_length_validation(subject, drawing_tool, prompt):
    """
    Property: For any string fields that exceed maximum length constraints,
    the system should reject the request.
    """
    # Test subject length validation
    with pytest.raises(ValidationError):
        DrawingUploadRequest(
            age_years=5.0,
            subject=subject  # Too long
        )
    
    # Test drawing_tool length validation
    with pytest.raises(ValidationError):
        DrawingUploadRequest(
            age_years=5.0,
            drawing_tool=drawing_tool  # Too long
        )
    
    # Test prompt length validation
    with pytest.raises(ValidationError):
        DrawingUploadRequest(
            age_years=5.0,
            prompt=prompt  # Too long
        )


@given(
    age_min=valid_age_strategy,
    age_max=valid_age_strategy
)
@settings(max_examples=50, deadline=None)
def test_age_range_filter_validation(age_min, age_max):
    """
    Property: For any age range filter where age_max <= age_min,
    the system should reject the filter request.
    """
    assume(age_max <= age_min)  # Only test invalid ranges
    
    with pytest.raises(ValidationError) as exc_info:
        DrawingFilterRequest(
            age_min=age_min,
            age_max=age_max
        )
    
    # Verify the error is about age range
    errors = exc_info.value.errors()
    assert any('age_max must be greater than age_min' in str(error.get('msg', '')) for error in errors)


@given(
    base_ids=st.lists(
        st.integers(min_value=1, max_value=100), 
        min_size=1, 
        max_size=5,
        unique=True
    ),
    duplicate_id=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50, deadline=None)
def test_duplicate_drawing_ids_rejection(base_ids, duplicate_id):
    """
    Property: For any list of drawing IDs containing duplicates,
    the system should reject the batch analysis request.
    """
    # Create a list with guaranteed duplicates by adding duplicate_id twice
    drawing_ids = base_ids + [duplicate_id, duplicate_id]
    
    with pytest.raises(ValidationError) as exc_info:
        BatchAnalysisRequest(drawing_ids=drawing_ids)
    
    # Verify the error is about duplicate IDs
    errors = exc_info.value.errors()
    assert any('Drawing IDs must be unique' in str(error.get('msg', '')) for error in errors)


@given(
    drawing_ids=st.lists(
        st.integers(min_value=-1000, max_value=0), 
        min_size=1, 
        max_size=10
    )
)
@settings(max_examples=50, deadline=None)
def test_negative_drawing_ids_rejection(drawing_ids):
    """
    Property: For any list of drawing IDs containing non-positive values,
    the system should reject the batch analysis request.
    """
    assume(any(id <= 0 for id in drawing_ids))  # Ensure negative/zero IDs exist
    
    with pytest.raises(ValidationError) as exc_info:
        BatchAnalysisRequest(drawing_ids=drawing_ids)
    
    # Verify the error is about positive IDs
    errors = exc_info.value.errors()
    assert any('All drawing IDs must be positive' in str(error.get('msg', '')) for error in errors)


@given(
    max_file_size=st.integers(min_value=1, max_value=100 * 1024 * 1024),
    allowed_formats=st.lists(image_format_strategy, min_size=1, max_size=4, unique=True),
    min_width=st.integers(min_value=1, max_value=1024),
    max_width=st.integers(min_value=1025, max_value=8192),
    min_height=st.integers(min_value=1, max_value=1024),
    max_height=st.integers(min_value=1025, max_value=8192)
)
@settings(max_examples=50, deadline=None)
def test_image_validation_consistency(max_file_size, allowed_formats, min_width, max_width, min_height, max_height):
    """
    Property: For any valid image validation parameters,
    the system should create a consistent validation configuration.
    """
    validation_config = ImageValidationRequest(
        max_file_size=max_file_size,
        allowed_formats=allowed_formats,
        min_width=min_width,
        max_width=max_width,
        min_height=min_height,
        max_height=max_height
    )
    
    # Verify all parameters are preserved
    assert validation_config.max_file_size == max_file_size
    assert validation_config.allowed_formats == allowed_formats
    assert validation_config.min_width == min_width
    assert validation_config.max_width == max_width
    assert validation_config.min_height == min_height
    assert validation_config.max_height == max_height
    
    # Verify logical constraints
    assert validation_config.max_width >= validation_config.min_width
    assert validation_config.max_height >= validation_config.min_height
    assert validation_config.max_file_size > 0
    assert len(validation_config.allowed_formats) > 0


@given(whitespace_string=st.sampled_from(["   ", "\t", "\n", " \t \n "]))
@settings(max_examples=20, deadline=None)
def test_empty_string_normalization(whitespace_string):
    """
    Property: For any empty or whitespace-only strings in optional fields,
    the system should normalize them to None.
    """
    # Test empty string
    request1 = DrawingUploadRequest(
        age_years=5.0,
        subject="",
        drawing_tool="",
        prompt=""
    )
    
    assert request1.subject is None, "Empty string should be converted to None"
    assert request1.drawing_tool is None, "Empty string should be converted to None"
    assert request1.prompt is None, "Empty string should be converted to None"
    
    # Test whitespace-only string
    request2 = DrawingUploadRequest(
        age_years=5.0,
        subject=whitespace_string,
        drawing_tool=whitespace_string,
        prompt=whitespace_string
    )
    
    assert request2.subject is None, "Whitespace-only string should be converted to None"
    assert request2.drawing_tool is None, "Whitespace-only string should be converted to None"
    assert request2.prompt is None, "Whitespace-only string should be converted to None"