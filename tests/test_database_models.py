"""
Property-based tests for database model consistency.

**Feature: children-drawing-anomaly-detection, Property 3: Metadata Persistence**
**Validates: Requirements 1.4, 1.5**
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from datetime import datetime

from app.models.database import Drawing


# Hypothesis strategies for generating test data
drawing_metadata_strategy = st.fixed_dictionaries({
    'filename': st.text(min_size=1, max_size=100, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters='.-_')),
    'file_path': st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'), whitelist_characters='.-_/')),
    'age_years': st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False),
    'subject': st.one_of(st.none(), st.text(min_size=1, max_size=50)),
    'expert_label': st.one_of(st.none(), st.sampled_from(['normal', 'concern', 'severe'])),
    'drawing_tool': st.one_of(st.none(), st.text(min_size=1, max_size=30)),
    'prompt': st.one_of(st.none(), st.text(min_size=1, max_size=500))
})


@given(metadata=drawing_metadata_strategy)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_metadata_persistence_property(db_session, metadata):
    """
    **Feature: children-drawing-anomaly-detection, Property 3: Metadata Persistence**
    **Validates: Requirements 1.4, 1.5**
    
    Property: For any drawing upload with optional metadata (subject, expert labels), 
    the provided metadata should be stored and retrievable during analysis.
    """
    # Create a drawing with the generated metadata
    drawing = Drawing(
        filename=metadata['filename'],
        file_path=metadata['file_path'],
        age_years=metadata['age_years'],
        subject=metadata['subject'],
        expert_label=metadata['expert_label'],
        drawing_tool=metadata['drawing_tool'],
        prompt=metadata['prompt']
    )
    
    # Store the drawing
    db_session.add(drawing)
    db_session.commit()
    
    # Retrieve the drawing from database
    retrieved_drawing = db_session.query(Drawing).filter_by(id=drawing.id).first()
    
    # Verify all metadata is preserved
    assert retrieved_drawing is not None, "Drawing should be retrievable from database"
    assert retrieved_drawing.filename == metadata['filename'], "Filename should be preserved"
    assert retrieved_drawing.file_path == metadata['file_path'], "File path should be preserved"
    assert retrieved_drawing.age_years == metadata['age_years'], "Age should be preserved"
    assert retrieved_drawing.subject == metadata['subject'], "Subject should be preserved"
    assert retrieved_drawing.expert_label == metadata['expert_label'], "Expert label should be preserved"
    assert retrieved_drawing.drawing_tool == metadata['drawing_tool'], "Drawing tool should be preserved"
    assert retrieved_drawing.prompt == metadata['prompt'], "Prompt should be preserved"
    
    # Verify timestamp was automatically set
    assert retrieved_drawing.upload_timestamp is not None, "Upload timestamp should be automatically set"
    assert isinstance(retrieved_drawing.upload_timestamp, datetime), "Upload timestamp should be a datetime object"


@given(
    metadata1=drawing_metadata_strategy,
    metadata2=drawing_metadata_strategy
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_multiple_drawings_metadata_independence(db_session, metadata1, metadata2):
    """
    Test that metadata for different drawings is stored independently.
    
    This ensures that storing one drawing doesn't affect the metadata of another.
    """
    # Create two drawings with different metadata
    drawing1 = Drawing(
        filename=metadata1['filename'],
        file_path=metadata1['file_path'],
        age_years=metadata1['age_years'],
        subject=metadata1['subject'],
        expert_label=metadata1['expert_label'],
        drawing_tool=metadata1['drawing_tool'],
        prompt=metadata1['prompt']
    )
    
    drawing2 = Drawing(
        filename=metadata2['filename'],
        file_path=metadata2['file_path'],
        age_years=metadata2['age_years'],
        subject=metadata2['subject'],
        expert_label=metadata2['expert_label'],
        drawing_tool=metadata2['drawing_tool'],
        prompt=metadata2['prompt']
    )
    
    # Store both drawings
    db_session.add(drawing1)
    db_session.add(drawing2)
    db_session.commit()
    
    # Retrieve both drawings
    retrieved1 = db_session.query(Drawing).filter_by(id=drawing1.id).first()
    retrieved2 = db_session.query(Drawing).filter_by(id=drawing2.id).first()
    
    # Verify each drawing maintains its own metadata
    assert retrieved1.filename == metadata1['filename']
    assert retrieved2.filename == metadata2['filename']
    assert retrieved1.subject == metadata1['subject']
    assert retrieved2.subject == metadata2['subject']
    assert retrieved1.expert_label == metadata1['expert_label']
    assert retrieved2.expert_label == metadata2['expert_label']


@given(metadata=drawing_metadata_strategy)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_optional_metadata_handling(db_session, metadata):
    """
    Test that optional metadata fields (None values) are handled correctly.
    
    This verifies that the system properly stores and retrieves None values
    for optional fields without corruption.
    """
    # Create drawing with the metadata (which may contain None values)
    drawing = Drawing(
        filename=metadata['filename'],
        file_path=metadata['file_path'],
        age_years=metadata['age_years'],
        subject=metadata['subject'],
        expert_label=metadata['expert_label'],
        drawing_tool=metadata['drawing_tool'],
        prompt=metadata['prompt']
    )
    
    db_session.add(drawing)
    db_session.commit()
    
    # Retrieve and verify
    retrieved = db_session.query(Drawing).filter_by(id=drawing.id).first()
    
    # Specifically test that None values are preserved as None
    if metadata['subject'] is None:
        assert retrieved.subject is None, "None subject should remain None"
    if metadata['expert_label'] is None:
        assert retrieved.expert_label is None, "None expert_label should remain None"
    if metadata['drawing_tool'] is None:
        assert retrieved.drawing_tool is None, "None drawing_tool should remain None"
    if metadata['prompt'] is None:
        assert retrieved.prompt is None, "None prompt should remain None"