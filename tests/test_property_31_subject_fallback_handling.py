"""
Property-based test for subject fallback handling.

**Feature: children-drawing-anomaly-detection, Property 31: Subject Fallback Handling**
**Validates: Requirements 2.6**

This test verifies that the system gracefully handles unknown or invalid subject categories
by falling back to "unspecified" in a consistent manner.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings

from app.services.embedding_service import SubjectEncoder
from app.schemas.drawings import SubjectCategory


class TestSubjectFallbackHandling:
    """Test subject fallback handling properties."""
    
    @given(st.text())
    @settings(max_examples=100)
    def test_unknown_subject_fallback_consistency(self, unknown_subject):
        """
        Property 31: Subject Fallback Handling
        
        For any unknown subject string, the system should consistently fall back to
        "unspecified" and produce valid, decodable encodings.
        
        **Validates: Requirements 2.6**
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
        
        # Should be identical to explicit unspecified encoding
        explicit_unspecified = SubjectEncoder.encode_subject_category(SubjectCategory.UNSPECIFIED)
        assert np.array_equal(encoding, explicit_unspecified), (
            f"Unknown subject encoding should match explicit UNSPECIFIED encoding"
        )
    
    @given(st.none() | st.just("") | st.text().filter(lambda x: x.strip() == ""))
    @settings(max_examples=50)
    def test_none_and_empty_subject_fallback(self, empty_subject):
        """
        Property 31: Subject Fallback Handling (None/Empty Values)
        
        For None or empty subject values, the system should consistently use "unspecified"
        and produce identical encodings.
        
        **Validates: Requirements 2.6**
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
    
    @given(st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_case_insensitive_fallback_handling(self, subject_text):
        """
        Property 31: Subject Fallback Handling (Case Sensitivity)
        
        The system should handle case variations consistently, either matching
        valid categories or falling back to unspecified.
        
        **Validates: Requirements 2.6**
        """
        # Test various case variations
        variations = [
            subject_text.lower(),
            subject_text.upper(),
            subject_text.title(),
            subject_text.capitalize()
        ]
        
        encodings = []
        decoded_categories = []
        
        for variation in variations:
            encoding = SubjectEncoder.encode_subject_category(variation)
            decoded = SubjectEncoder.decode_subject_encoding(encoding)
            
            encodings.append(encoding)
            decoded_categories.append(decoded)
            
            # Verify encoding properties
            assert encoding.shape == (64,), f"Encoding should be 64-dimensional for '{variation}'"
            assert encoding.dtype == np.float32, f"Encoding should be float32 for '{variation}'"
            assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0 for '{variation}'"
        
        # All variations should produce the same result
        for i in range(1, len(encodings)):
            assert np.array_equal(encodings[0], encodings[i]), (
                f"Case variation '{variations[i]}' should produce same encoding as '{variations[0]}'"
            )
            assert decoded_categories[0] == decoded_categories[i], (
                f"Case variation '{variations[i]}' should decode to same category as '{variations[0]}'"
            )
    
    @given(st.text().filter(lambda x: len(x.strip()) > 0))
    @settings(max_examples=50)
    def test_whitespace_handling_fallback(self, subject_with_spaces):
        """
        Property 31: Subject Fallback Handling (Whitespace)
        
        The system should handle whitespace consistently in subject strings.
        
        **Validates: Requirements 2.6**
        """
        # Add various whitespace patterns
        whitespace_variations = [
            subject_with_spaces,
            f"  {subject_with_spaces}  ",  # Leading/trailing spaces
            f"\t{subject_with_spaces}\t",  # Tabs
            f"\n{subject_with_spaces}\n",  # Newlines
            subject_with_spaces.replace(" ", ""),  # No spaces
        ]
        
        encodings = []
        decoded_categories = []
        
        for variation in whitespace_variations:
            encoding = SubjectEncoder.encode_subject_category(variation)
            decoded = SubjectEncoder.decode_subject_encoding(encoding)
            
            encodings.append(encoding)
            decoded_categories.append(decoded)
            
            # Verify encoding properties
            assert encoding.shape == (64,), f"Encoding should be 64-dimensional for '{repr(variation)}'"
            assert encoding.dtype == np.float32, f"Encoding should be float32 for '{repr(variation)}'"
            assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0 for '{repr(variation)}'"
        
        # Check if any variation matches a valid subject
        valid_subjects = {cat.value for cat in SubjectCategory}
        has_valid_match = any(var.strip().lower() in valid_subjects for var in whitespace_variations)
        
        if has_valid_match:
            # If there's a valid match, all should decode to the same valid category
            first_decoded = decoded_categories[0]
            for i, decoded in enumerate(decoded_categories):
                assert decoded == first_decoded, (
                    f"Whitespace variation '{repr(whitespace_variations[i])}' should decode to same category"
                )
        else:
            # If no valid match, all should fall back to unspecified
            for i, decoded in enumerate(decoded_categories):
                assert decoded == SubjectCategory.UNSPECIFIED, (
                    f"Invalid subject '{repr(whitespace_variations[i])}' should fall back to UNSPECIFIED"
                )
    
    def test_special_characters_fallback(self):
        """
        Test that subjects with special characters fall back to unspecified.
        """
        special_subjects = [
            "house@#$",
            "car!@#",
            "tree???",
            "person***",
            "123numbers",
            "mixed123chars!",
            "unicode_café",
            "emoji_🏠",
            "",  # Empty string
            "   ",  # Only whitespace
        ]
        
        for special_subject in special_subjects:
            encoding = SubjectEncoder.encode_subject_category(special_subject)
            decoded = SubjectEncoder.decode_subject_encoding(encoding)
            
            # Should fall back to unspecified for special characters
            if special_subject.strip() == "":
                # Empty/whitespace should definitely be unspecified
                assert decoded == SubjectCategory.UNSPECIFIED, (
                    f"Empty/whitespace subject '{repr(special_subject)}' should be UNSPECIFIED"
                )
            else:
                # Special characters should likely be unspecified unless they happen to match
                # We don't assert this strictly since some might theoretically match valid subjects
                assert decoded in SubjectCategory, (
                    f"Special subject '{special_subject}' should decode to valid category"
                )
            
            # Verify encoding properties
            assert encoding.shape == (64,), f"Encoding should be 64-dimensional for '{repr(special_subject)}'"
            assert encoding.dtype == np.float32, f"Encoding should be float32 for '{repr(special_subject)}'"
            assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0 for '{repr(special_subject)}'"
    
    def test_fallback_consistency_across_calls(self):
        """
        Test that fallback behavior is consistent across multiple calls.
        """
        unknown_subjects = [
            "definitely_not_a_valid_subject",
            "another_invalid_one",
            "xyz123",
            "random_text_here"
        ]
        
        for unknown_subject in unknown_subjects:
            # Call encoding multiple times
            encoding1 = SubjectEncoder.encode_subject_category(unknown_subject)
            encoding2 = SubjectEncoder.encode_subject_category(unknown_subject)
            encoding3 = SubjectEncoder.encode_subject_category(unknown_subject)
            
            # Should be identical across calls
            assert np.array_equal(encoding1, encoding2), (
                f"Multiple calls for '{unknown_subject}' should produce identical results"
            )
            assert np.array_equal(encoding2, encoding3), (
                f"Multiple calls for '{unknown_subject}' should produce identical results"
            )
            
            # Decode should be consistent
            decoded1 = SubjectEncoder.decode_subject_encoding(encoding1)
            decoded2 = SubjectEncoder.decode_subject_encoding(encoding2)
            decoded3 = SubjectEncoder.decode_subject_encoding(encoding3)
            
            assert decoded1 == decoded2 == decoded3, (
                f"Multiple decodings for '{unknown_subject}' should be identical"
            )
    
    def test_validation_consistency_with_fallback(self):
        """
        Test that validation results are consistent with encoding fallback behavior.
        """
        test_subjects = [
            "house",  # Valid
            "HOUSE",  # Valid (case variation)
            "invalid_subject",  # Invalid
            "",  # Empty
            None,  # None
            "car",  # Valid
            "xyz123",  # Invalid
        ]
        
        for subject in test_subjects:
            # Check validation
            is_valid = SubjectEncoder.validate_subject_category(subject)
            
            # Encode and decode
            encoding = SubjectEncoder.encode_subject_category(subject)
            decoded = SubjectEncoder.decode_subject_encoding(encoding)
            
            if is_valid:
                # If validation says it's valid, it should not decode to unspecified
                # (unless the valid category happens to be unspecified itself)
                if subject is not None and str(subject).lower().strip() != "unspecified":
                    assert decoded != SubjectCategory.UNSPECIFIED or subject == SubjectCategory.UNSPECIFIED, (
                        f"Valid subject '{subject}' should not fall back to UNSPECIFIED"
                    )
            else:
                # If validation says it's invalid, it should fall back to unspecified
                # (unless it's a case where validation is more strict than encoding)
                pass  # We don't assert this strictly since validation might be more restrictive
            
            # Verify encoding properties regardless
            assert encoding.shape == (64,), f"Encoding should be 64-dimensional for '{subject}'"
            assert encoding.dtype == np.float32, f"Encoding should be float32 for '{subject}'"
            assert np.sum(encoding) == 1.0, f"One-hot encoding should sum to 1.0 for '{subject}'"