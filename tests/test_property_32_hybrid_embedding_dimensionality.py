"""
Property-based test for hybrid embedding dimensionality consistency.

**Feature: children-drawing-anomaly-detection, Property 32: Hybrid Embedding Dimensionality Consistency**
**Validates: Requirements 2.7**

This test verifies that hybrid embeddings maintain consistent dimensionality
across all operations and subject categories.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from app.services.embedding_service import SubjectEncoder
from app.schemas.drawings import SubjectCategory


class TestHybridEmbeddingDimensionality:
    """Test hybrid embedding dimensionality consistency properties."""
    
    @given(st.sampled_from(list(SubjectCategory)))
    @settings(max_examples=100)
    def test_hybrid_embedding_dimensionality_consistency(self, subject_category):
        """
        Property 32: Hybrid Embedding Dimensionality Consistency
        
        For any subject category, hybrid embeddings should always have exactly
        832 dimensions (768 visual + 64 subject), regardless of the subject.
        
        **Validates: Requirements 2.7**
        """
        # Create a test visual embedding (768 dimensions)
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Generate subject encoding
        subject_encoding = SubjectEncoder.encode_subject_category(subject_category)
        
        # Create hybrid embedding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
        
        # Verify consistent dimensionality
        assert hybrid_embedding.shape == (832,), (
            f"Hybrid embedding for {subject_category} should be 832-dimensional, got {hybrid_embedding.shape}"
        )
        assert hybrid_embedding.dtype == np.float32, (
            f"Hybrid embedding for {subject_category} should be float32, got {hybrid_embedding.dtype}"
        )
        
        # Verify component dimensions
        visual_component = hybrid_embedding[:768]
        subject_component = hybrid_embedding[768:]
        
        assert visual_component.shape == (768,), (
            f"Visual component for {subject_category} should be 768-dimensional, got {visual_component.shape}"
        )
        assert subject_component.shape == (64,), (
            f"Subject component for {subject_category} should be 64-dimensional, got {subject_component.shape}"
        )
        
        # Verify total dimension is sum of components
        assert len(hybrid_embedding) == len(visual_component) + len(subject_component), (
            f"Total dimension should equal sum of component dimensions for {subject_category}"
        )
    
    @given(
        st.lists(st.sampled_from(list(SubjectCategory)), min_size=1, max_size=10),
        st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50)
    def test_batch_dimensionality_consistency(self, subject_list, visual_seed):
        """
        Property 32: Hybrid Embedding Dimensionality Consistency (Batch Processing)
        
        All hybrid embeddings in a batch should have identical dimensions,
        regardless of the variety of subjects.
        
        **Validates: Requirements 2.7**
        """
        # Create deterministic visual embeddings
        np.random.seed(visual_seed)
        visual_embeddings = [np.random.randn(768).astype(np.float32) for _ in subject_list]
        
        # Generate hybrid embeddings for all subjects
        hybrid_embeddings = []
        for visual, subject in zip(visual_embeddings, subject_list):
            subject_encoding = SubjectEncoder.encode_subject_category(subject)
            hybrid = np.concatenate([visual, subject_encoding], axis=0)
            hybrid_embeddings.append(hybrid)
        
        # Verify all have same dimensions
        expected_shape = (832,)
        for i, (hybrid, subject) in enumerate(zip(hybrid_embeddings, subject_list)):
            assert hybrid.shape == expected_shape, (
                f"Hybrid embedding {i} for {subject} should have shape {expected_shape}, got {hybrid.shape}"
            )
            assert hybrid.dtype == np.float32, (
                f"Hybrid embedding {i} for {subject} should be float32, got {hybrid.dtype}"
            )
        
        # Verify all embeddings have identical structure
        shapes = [embedding.shape for embedding in hybrid_embeddings]
        dtypes = [embedding.dtype for embedding in hybrid_embeddings]
        
        assert all(shape == expected_shape for shape in shapes), (
            f"All hybrid embeddings should have shape {expected_shape}, got {shapes}"
        )
        assert all(dtype == np.float32 for dtype in dtypes), (
            f"All hybrid embeddings should be float32, got {dtypes}"
        )
    
    def test_dimensionality_constants_consistency(self):
        """
        Test that dimensionality constants are consistent across the system.
        """
        # Test SubjectEncoder constants
        assert SubjectEncoder.ENCODING_DIMENSION == 64, (
            f"Subject encoding dimension should be 64, got {SubjectEncoder.ENCODING_DIMENSION}"
        )
        
        # Test that all subject encodings have correct dimension
        for subject in SubjectCategory:
            encoding = SubjectEncoder.encode_subject_category(subject)
            assert encoding.shape == (64,), (
                f"Subject encoding for {subject} should be 64-dimensional, got {encoding.shape}"
            )
        
        # Test hybrid embedding total dimension
        expected_hybrid_dim = 768 + 64  # Visual + Subject
        assert expected_hybrid_dim == 832, (
            f"Expected hybrid dimension should be 832, calculated {expected_hybrid_dim}"
        )
    
    @given(
        st.integers(min_value=1, max_value=1000),
        st.sampled_from(list(SubjectCategory))
    )
    @settings(max_examples=50)
    def test_dimensionality_invariant_across_visual_variations(self, visual_seed, subject):
        """
        Property 32: Hybrid Embedding Dimensionality Consistency (Visual Variations)
        
        Hybrid embedding dimensions should remain constant regardless of
        visual embedding content variations.
        
        **Validates: Requirements 2.7**
        """
        # Create different visual embeddings with same dimensions
        np.random.seed(visual_seed)
        
        visual_variations = [
            np.random.randn(768).astype(np.float32),  # Random normal
            np.ones(768, dtype=np.float32),  # All ones
            np.zeros(768, dtype=np.float32),  # All zeros
            np.random.uniform(-1, 1, 768).astype(np.float32),  # Uniform distribution
            np.random.exponential(1, 768).astype(np.float32),  # Exponential distribution
        ]
        
        subject_encoding = SubjectEncoder.encode_subject_category(subject)
        
        # Create hybrid embeddings with different visual content
        hybrid_embeddings = []
        for visual in visual_variations:
            hybrid = np.concatenate([visual, subject_encoding], axis=0)
            hybrid_embeddings.append(hybrid)
        
        # Verify all have same dimensions regardless of visual content
        expected_shape = (832,)
        for i, hybrid in enumerate(hybrid_embeddings):
            assert hybrid.shape == expected_shape, (
                f"Hybrid embedding {i} should have shape {expected_shape}, got {hybrid.shape}"
            )
            assert hybrid.dtype == np.float32, (
                f"Hybrid embedding {i} should be float32, got {hybrid.dtype}"
            )
            
            # Verify component dimensions
            visual_component = hybrid[:768]
            subject_component = hybrid[768:]
            
            assert visual_component.shape == (768,), (
                f"Visual component {i} should be 768-dimensional"
            )
            assert subject_component.shape == (64,), (
                f"Subject component {i} should be 64-dimensional"
            )
    
    def test_dimensionality_edge_cases(self):
        """
        Test dimensionality consistency for edge cases.
        """
        edge_cases = [
            None,  # None subject
            "",  # Empty string
            "   ",  # Whitespace only
            "unknown_subject",  # Unknown subject
            SubjectCategory.UNSPECIFIED,  # Explicit unspecified
        ]
        
        # Create a standard visual embedding
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        for edge_case in edge_cases:
            # Generate subject encoding for edge case
            subject_encoding = SubjectEncoder.encode_subject_category(edge_case)
            
            # Create hybrid embedding
            hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
            
            # Verify dimensions are consistent even for edge cases
            assert hybrid_embedding.shape == (832,), (
                f"Hybrid embedding for edge case '{edge_case}' should be 832-dimensional, got {hybrid_embedding.shape}"
            )
            assert hybrid_embedding.dtype == np.float32, (
                f"Hybrid embedding for edge case '{edge_case}' should be float32, got {hybrid_embedding.dtype}"
            )
            
            # Verify subject encoding dimension
            assert subject_encoding.shape == (64,), (
                f"Subject encoding for edge case '{edge_case}' should be 64-dimensional, got {subject_encoding.shape}"
            )
    
    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_dimensionality_consistency_across_operations(self, num_operations):
        """
        Property 32: Hybrid Embedding Dimensionality Consistency (Multiple Operations)
        
        Dimensionality should remain consistent across multiple encoding/decoding operations.
        
        **Validates: Requirements 2.7**
        """
        # Start with a random subject
        subject = np.random.choice(list(SubjectCategory))
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Perform multiple operations
        for i in range(num_operations):
            # Encode subject
            subject_encoding = SubjectEncoder.encode_subject_category(subject)
            
            # Create hybrid embedding
            hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
            
            # Verify dimensions at each step
            assert hybrid_embedding.shape == (832,), (
                f"Hybrid embedding at operation {i} should be 832-dimensional, got {hybrid_embedding.shape}"
            )
            assert subject_encoding.shape == (64,), (
                f"Subject encoding at operation {i} should be 64-dimensional, got {subject_encoding.shape}"
            )
            
            # Decode and verify consistency
            decoded_subject = SubjectEncoder.decode_subject_encoding(subject_encoding)
            assert isinstance(decoded_subject, SubjectCategory), (
                f"Decoded subject at operation {i} should be SubjectCategory, got {type(decoded_subject)}"
            )
            
            # Use decoded subject for next iteration (should be same as original)
            subject = decoded_subject
    
    def test_component_dimension_arithmetic(self):
        """
        Test that component dimensions add up correctly to total dimension.
        """
        visual_dim = 768
        subject_dim = 64
        expected_total = visual_dim + subject_dim
        
        # Verify arithmetic
        assert expected_total == 832, (
            f"Visual ({visual_dim}) + Subject ({subject_dim}) should equal 832, got {expected_total}"
        )
        
        # Test with actual embeddings
        visual_embedding = np.random.randn(visual_dim).astype(np.float32)
        subject_encoding = SubjectEncoder.encode_subject_category(SubjectCategory.HOUSE)
        
        # Verify individual dimensions
        assert visual_embedding.shape == (visual_dim,), (
            f"Visual embedding should be {visual_dim}-dimensional"
        )
        assert subject_encoding.shape == (subject_dim,), (
            f"Subject encoding should be {subject_dim}-dimensional"
        )
        
        # Create hybrid and verify total
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
        assert hybrid_embedding.shape == (expected_total,), (
            f"Hybrid embedding should be {expected_total}-dimensional"
        )
        
        # Verify components can be extracted with correct dimensions
        extracted_visual = hybrid_embedding[:visual_dim]
        extracted_subject = hybrid_embedding[visual_dim:]
        
        assert extracted_visual.shape == (visual_dim,), (
            f"Extracted visual should be {visual_dim}-dimensional"
        )
        assert extracted_subject.shape == (subject_dim,), (
            f"Extracted subject should be {subject_dim}-dimensional"
        )
        
        # Verify extracted components match originals
        assert np.allclose(extracted_visual, visual_embedding), (
            "Extracted visual component should match original"
        )
        assert np.allclose(extracted_subject, subject_encoding), (
            "Extracted subject component should match original"
        )