"""
Property-based test for subject encoding consistency.

**Feature: children-drawing-anomaly-detection, Property 29: Subject Encoding Consistency**
**Validates: Requirements 2.4**

This test verifies that subject category encoding is consistent and reversible.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

from app.services.embedding_service import SubjectEncoder, SubjectEncodingError
from app.schemas.drawings import SubjectCategory


class TestSubjectEncodingConsistency:
    """Test subject encoding consistency properties."""
    
    @given(st.sampled_from(list(SubjectCategory)))
    @settings(max_examples=100)
    def test_subject_encoding_round_trip(self, subject_category):
        """
        Property 29: Subject Encoding Consistency
        
        For any valid subject category, encoding then decoding should return the original category.
        This ensures the encoding is bijective and consistent.
        
        **Validates: Requirements 2.4**
        """
        # Encode the subject category
        encoding = SubjectEncoder.encode_subject_category(subject_category)
        
        # Verify encoding properties
        assert encoding.shape == (64,), f"Encoding should be 64-dimensional, got {encoding.shape}"
        assert encoding.dtype == np.float32, f"Encoding should be float32, got {encoding.dtype}"
        
        # Verify it's a valid one-hot encoding
        assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0, got {np.sum(encoding)}"
        assert np.sum(encoding == 1.0) == 1, f"Exactly one position should be 1.0, got {np.sum(encoding == 1.0)}"
        assert np.sum(encoding == 0.0) == 63, f"Exactly 63 positions should be 0.0, got {np.sum(encoding == 0.0)}"
        
        # Decode back to subject category
        decoded_category = SubjectEncoder.decode_subject_encoding(encoding)
        
        # Verify round-trip consistency
        assert decoded_category == subject_category, (
            f"Round-trip failed: {subject_category} -> {encoding} -> {decoded_category}"
        )
    
    @given(st.text())
    @settings(max_examples=100)
    def test_unknown_subject_fallback_consistency(self, unknown_subject):
        """
        Property 29: Subject Encoding Consistency (Fallback Handling)
        
        For any unknown subject string, the encoder should consistently fall back to "unspecified"
        and the encoding should be valid and decodable.
        
        **Validates: Requirements 2.4**
        """
        # Skip if the string happens to be a valid subject category
        valid_subjects = {cat.value for cat in SubjectCategory}
        assume(unknown_subject.lower() not in valid_subjects)
        assume(unknown_subject.strip() != "")  # Empty strings are handled separately
        
        # Encode the unknown subject (should fall back to unspecified)
        encoding = SubjectEncoder.encode_subject_category(unknown_subject)
        
        # Verify encoding properties
        assert encoding.shape == (64,), f"Encoding should be 64-dimensional, got {encoding.shape}"
        assert encoding.dtype == np.float32, f"Encoding should be float32, got {encoding.dtype}"
        
        # Verify it's a valid one-hot encoding
        assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0, got {np.sum(encoding)}"
        assert np.sum(encoding == 1.0) == 1, f"Exactly one position should be 1.0, got {np.sum(encoding == 1.0)}"
        
        # Decode should return unspecified
        decoded_category = SubjectEncoder.decode_subject_encoding(encoding)
        assert decoded_category == SubjectCategory.UNSPECIFIED, (
            f"Unknown subject '{unknown_subject}' should decode to UNSPECIFIED, got {decoded_category}"
        )
    
    @given(st.none() | st.just("") | st.text().filter(lambda x: x.strip() == ""))
    @settings(max_examples=50)
    def test_none_and_empty_subject_consistency(self, empty_subject):
        """
        Property 29: Subject Encoding Consistency (None/Empty Handling)
        
        For None or empty subject values, the encoder should consistently use "unspecified"
        and produce identical encodings.
        
        **Validates: Requirements 2.4**
        """
        # Encode the empty/None subject
        encoding = SubjectEncoder.encode_subject_category(empty_subject)
        
        # Verify encoding properties
        assert encoding.shape == (64,), f"Encoding should be 64-dimensional, got {encoding.shape}"
        assert encoding.dtype == np.float32, f"Encoding should be float32, got {encoding.dtype}"
        
        # Verify it's a valid one-hot encoding
        assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0, got {np.sum(encoding)}"
        
        # Decode should return unspecified
        decoded_category = SubjectEncoder.decode_subject_encoding(encoding)
        assert decoded_category == SubjectCategory.UNSPECIFIED, (
            f"Empty subject '{empty_subject}' should decode to UNSPECIFIED, got {decoded_category}"
        )
        
        # Should be identical to explicit unspecified encoding
        explicit_unspecified = SubjectEncoder.encode_subject_category(SubjectCategory.UNSPECIFIED)
        assert np.array_equal(encoding, explicit_unspecified), (
            f"Empty subject encoding should match explicit UNSPECIFIED encoding"
        )
    
    def test_subject_index_uniqueness(self):
        """
        Verify that each subject category maps to a unique index.
        This is a structural property that ensures no collisions.
        """
        indices = set()
        categories = list(SubjectCategory)
        
        for category in categories:
            encoding = SubjectEncoder.encode_subject_category(category)
            active_index = np.where(encoding == 1.0)[0][0]
            
            assert active_index not in indices, (
                f"Index collision: {category} maps to index {active_index} which is already used"
            )
            indices.add(active_index)
        
        # Verify we're using indices efficiently (no gaps in low range)
        max_index = max(indices)
        assert max_index < 64, f"Maximum index {max_index} should be less than 64"
        assert len(indices) == len(categories), f"Should have {len(categories)} unique indices, got {len(indices)}"
    
    def test_encoding_dimension_consistency(self):
        """
        Verify that all encodings have consistent dimensions regardless of subject.
        """
        expected_dim = SubjectEncoder.ENCODING_DIMENSION
        
        for category in SubjectCategory:
            encoding = SubjectEncoder.encode_subject_category(category)
            assert encoding.shape == (expected_dim,), (
                f"Category {category} encoding has shape {encoding.shape}, expected ({expected_dim},)"
            )
    
    def test_invalid_encoding_rejection(self):
        """
        Test that invalid encodings are properly rejected during decoding.
        """
        # Test wrong dimension
        with pytest.raises(SubjectEncodingError, match="Invalid encoding shape"):
            SubjectEncoder.decode_subject_encoding(np.array([1.0, 0.0]))  # Wrong size
        
        # Test multiple active positions
        invalid_encoding = np.zeros(64, dtype=np.float32)
        invalid_encoding[0] = 1.0
        invalid_encoding[1] = 1.0  # Two active positions
        
        with pytest.raises(SubjectEncodingError, match="Invalid one-hot encoding"):
            SubjectEncoder.decode_subject_encoding(invalid_encoding)
        
        # Test no active positions
        zero_encoding = np.zeros(64, dtype=np.float32)
        
        with pytest.raises(SubjectEncodingError, match="Invalid one-hot encoding"):
            SubjectEncoder.decode_subject_encoding(zero_encoding)
    
    def test_supported_categories_completeness(self):
        """
        Verify that all supported categories are properly mapped and accessible.
        """
        supported = SubjectEncoder.get_supported_categories()
        all_categories = list(SubjectCategory)
        
        assert len(supported) == len(all_categories), (
            f"Supported categories count {len(supported)} != all categories count {len(all_categories)}"
        )
        
        for category in all_categories:
            assert category in supported, f"Category {category} not in supported list"
            assert SubjectEncoder.validate_subject_category(category), (
                f"Category {category} failed validation"
            )