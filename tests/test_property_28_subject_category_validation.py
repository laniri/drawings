"""
Property-based tests for subject category validation.

**Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
**Validates: Requirements 1.4**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from pydantic import ValidationError

from app.schemas.drawings import (
    DrawingUploadRequest, 
    SubjectCategory,
    DrawingFilterRequest,
    ExpertLabel
)


# Hypothesis strategies for generating test data
valid_age_strategy = st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)

# Strategy for valid subject categories (all enum values)
valid_subject_category_strategy = st.one_of(
    st.none(),
    st.sampled_from(list(SubjectCategory))
)

# Strategy for invalid subject categories (strings not in enum)
invalid_subject_category_strategy = st.text(
    min_size=1, 
    max_size=50,
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters=' -_')
).filter(lambda x: x not in [category.value for category in SubjectCategory])

# Strategy for valid optional fields
valid_drawing_tool_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters=' -_'))
)

expert_label_strategy = st.one_of(st.none(), st.sampled_from(list(ExpertLabel)))


@given(
    age=valid_age_strategy,
    subject=valid_subject_category_strategy,
    expert_label=expert_label_strategy,
    drawing_tool=valid_drawing_tool_strategy,
    prompt=st.one_of(st.none(), st.text(min_size=1, max_size=500))
)
@settings(max_examples=100, deadline=None)
def test_valid_subject_category_acceptance(age, subject, expert_label, drawing_tool, prompt):
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Property: For any subject metadata that matches supported subject categories,
    the system should accept it during drawing upload.
    """
    # Create a valid drawing upload request with valid subject category
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
    
    # Verify subject category is handled correctly
    if subject is not None:
        assert request.subject == subject, "Valid subject category should be preserved"
        assert isinstance(request.subject, SubjectCategory), "Subject should be a SubjectCategory enum"
        # Verify it's one of the supported categories
        assert request.subject in list(SubjectCategory), "Subject should be in supported categories"
    else:
        assert request.subject is None, "None subject should remain None"
    
    # Verify other fields are preserved
    assert request.expert_label == expert_label, "Expert label should be preserved"
    assert request.drawing_tool == drawing_tool, "Drawing tool should be preserved"
    assert request.prompt == prompt, "Prompt should be preserved"


@given(
    age=valid_age_strategy,
    subject=invalid_subject_category_strategy,
    expert_label=expert_label_strategy,
    drawing_tool=valid_drawing_tool_strategy,
    prompt=st.one_of(st.none(), st.text(min_size=1, max_size=500))
)
@settings(max_examples=100, deadline=None)
def test_invalid_subject_category_rejection(age, subject, expert_label, drawing_tool, prompt):
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Property: For any subject metadata that does NOT match supported subject categories,
    the system should reject it during drawing upload.
    """
    # Attempt to create a drawing upload request with invalid subject category
    with pytest.raises(ValidationError) as exc_info:
        DrawingUploadRequest(
            age_years=age,
            subject=subject,  # This should be invalid
            expert_label=expert_label,
            drawing_tool=drawing_tool,
            prompt=prompt
        )
    
    # Verify that the validation error is related to the subject field
    errors = exc_info.value.errors()
    subject_errors = [error for error in errors if 'subject' in str(error.get('loc', []))]
    assert len(subject_errors) > 0, "Should have validation error for invalid subject category"
    
    # Verify the error indicates an invalid enum value
    for error in subject_errors:
        assert 'enum' in error.get('type', '').lower() or 'literal' in error.get('type', '').lower(), \
            "Error should indicate invalid enum/literal value"


@given(
    age_min=st.one_of(st.none(), valid_age_strategy),
    age_max=st.one_of(st.none(), valid_age_strategy),
    subject=valid_subject_category_strategy,
    expert_label=expert_label_strategy,
    page=st.integers(min_value=1, max_value=100),
    page_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100, deadline=None)
def test_subject_category_filtering_acceptance(age_min, age_max, subject, expert_label, page, page_size):
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Property: For any valid subject category used in filtering,
    the system should accept it in filter requests.
    """
    # Ensure age_max > age_min if both are provided, and within valid range
    if age_min is not None and age_max is not None and age_max <= age_min:
        # If age_min is too close to the upper limit, adjust age_min instead
        if age_min >= 17.0:
            age_min = age_max - 1.0
            if age_min < 2.0:
                age_min = 2.0
                age_max = 3.0
        else:
            age_max = min(age_min + 1.0, 18.0)
    
    # Create a valid drawing filter request with valid subject category
    filter_request = DrawingFilterRequest(
        age_min=age_min,
        age_max=age_max,
        subject=subject,
        expert_label=expert_label,
        page=page,
        page_size=page_size
    )
    
    # Verify the request was created successfully
    assert filter_request.page == page, "Page should be preserved"
    assert filter_request.page_size == page_size, "Page size should be preserved"
    
    # Verify subject category is handled correctly
    if subject is not None:
        assert filter_request.subject == subject, "Valid subject category should be preserved"
        assert isinstance(filter_request.subject, SubjectCategory), "Subject should be a SubjectCategory enum"
        assert filter_request.subject in list(SubjectCategory), "Subject should be in supported categories"
    else:
        assert filter_request.subject is None, "None subject should remain None"
    
    # Verify age range validation
    if age_min is not None:
        assert filter_request.age_min == age_min, "Age min should be preserved"
    if age_max is not None:
        assert filter_request.age_max == age_max, "Age max should be preserved"
    
    assert filter_request.expert_label == expert_label, "Expert label should be preserved"


@given(
    age_min=st.one_of(st.none(), valid_age_strategy),
    age_max=st.one_of(st.none(), valid_age_strategy),
    subject=invalid_subject_category_strategy,
    expert_label=expert_label_strategy,
    page=st.integers(min_value=1, max_value=100),
    page_size=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100, deadline=None)
def test_invalid_subject_category_filtering_rejection(age_min, age_max, subject, expert_label, page, page_size):
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Property: For any invalid subject category used in filtering,
    the system should reject it in filter requests.
    """
    # Ensure age_max > age_min if both are provided, and within valid range
    if age_min is not None and age_max is not None and age_max <= age_min:
        # If age_min is too close to the upper limit, adjust age_min instead
        if age_min >= 17.0:
            age_min = age_max - 1.0
            if age_min < 2.0:
                age_min = 2.0
                age_max = 3.0
        else:
            age_max = min(age_min + 1.0, 18.0)
    
    # Attempt to create a drawing filter request with invalid subject category
    with pytest.raises(ValidationError) as exc_info:
        DrawingFilterRequest(
            age_min=age_min,
            age_max=age_max,
            subject=subject,  # This should be invalid
            expert_label=expert_label,
            page=page,
            page_size=page_size
        )
    
    # Verify that the validation error is related to the subject field
    errors = exc_info.value.errors()
    subject_errors = [error for error in errors if 'subject' in str(error.get('loc', []))]
    assert len(subject_errors) > 0, "Should have validation error for invalid subject category"
    
    # Verify the error indicates an invalid enum value
    for error in subject_errors:
        assert 'enum' in error.get('type', '').lower() or 'literal' in error.get('type', '').lower(), \
            "Error should indicate invalid enum/literal value"


def test_all_subject_categories_supported():
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Test that all defined subject categories are properly supported.
    """
    # Test that all enum values can be used in upload requests
    for category in SubjectCategory:
        request = DrawingUploadRequest(
            age_years=5.0,
            subject=category
        )
        assert request.subject == category, f"Category {category} should be supported"
    
    # Test that all enum values can be used in filter requests
    for category in SubjectCategory:
        filter_request = DrawingFilterRequest(
            subject=category
        )
        assert filter_request.subject == category, f"Category {category} should be supported in filters"


def test_unspecified_default_category():
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Test that "unspecified" is available as the default category.
    """
    # Test that UNSPECIFIED category exists
    assert SubjectCategory.UNSPECIFIED == "unspecified", "UNSPECIFIED should have value 'unspecified'"
    
    # Test that it can be used in requests
    request = DrawingUploadRequest(
        age_years=5.0,
        subject=SubjectCategory.UNSPECIFIED
    )
    assert request.subject == SubjectCategory.UNSPECIFIED, "UNSPECIFIED category should be usable"
    
    # Test that it can be used in filters
    filter_request = DrawingFilterRequest(
        subject=SubjectCategory.UNSPECIFIED
    )
    assert filter_request.subject == SubjectCategory.UNSPECIFIED, "UNSPECIFIED category should be usable in filters"


def test_subject_category_count_within_limits():
    """
    **Feature: children-drawing-anomaly-detection, Property 28: Subject Category Validation**
    **Validates: Requirements 1.4**
    
    Test that the number of subject categories is within the 64-category limit
    for one-hot encoding as specified in the design.
    """
    categories = list(SubjectCategory)
    assert len(categories) <= 64, f"Should have at most 64 categories for one-hot encoding, but found {len(categories)}"
    assert len(categories) > 0, "Should have at least one category"
    
    # Verify UNSPECIFIED is included
    category_values = [cat.value for cat in categories]
    assert "unspecified" in category_values, "Should include 'unspecified' as default category"