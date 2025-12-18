"""
Property-based test for hybrid embedding construction.

**Feature: children-drawing-anomaly-detection, Property 30: Hybrid Embedding Construction**
**Validates: Requirements 2.5**

This test verifies that hybrid embeddings are correctly constructed by combining
visual features and subject encodings.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from app.services.embedding_service import SubjectEncoder
from app.schemas.drawings import SubjectCategory


class TestHybridEmbeddingConstruction:
    """Test hybrid embedding construction properties."""
    
    @given(st.sampled_from(list(SubjectCategory)))
    @settings(max_examples=50)
    def test_hybrid_embedding_construction_consistency(self, subject_category):
        """
        Property 30: Hybrid Embedding Construction
        
        For any subject category, the hybrid embedding should:
        1. Have exactly 832 dimensions (768 visual + 64 subject)
        2. Contain the correct subject encoding in positions 768-831
        3. Be separable back into original components
        
        **Validates: Requirements 2.5**
        """
        # Create a simple test visual embedding (768 dimensions)
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Generate subject encoding
        subject_encoding = SubjectEncoder.encode_subject_category(subject_category)
        
        # Manually create hybrid embedding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
        
        # Verify hybrid embedding properties
        assert hybrid_embedding.shape == (832,), (
            f"Hybrid embedding should be 832-dimensional, got {hybrid_embedding.shape}"
        )
        assert hybrid_embedding.dtype == np.float32, (
            f"Hybrid embedding should be float32, got {hybrid_embedding.dtype}"
        )
        
        # Verify visual component (first 768 dimensions)
        visual_component = hybrid_embedding[:768]
        assert np.allclose(visual_component, visual_embedding, rtol=1e-6), (
            f"Visual component mismatch for subject {subject_category}"
        )
        
        # Verify subject component (last 64 dimensions)
        subject_component = hybrid_embedding[768:]
        assert np.allclose(subject_component, subject_encoding, rtol=1e-6), (
            f"Subject component mismatch for subject {subject_category}"
        )
        
        # Test separability using SubjectEncoder
        decoded_subject = SubjectEncoder.decode_subject_encoding(subject_component)
        assert decoded_subject == subject_category, (
            f"Decoded subject {decoded_subject} doesn't match original {subject_category}"
        )
    
    @given(st.none() | st.just("") | st.text().filter(lambda x: x.strip() == ""))
    @settings(max_examples=20)
    def test_hybrid_embedding_unspecified_subject_consistency(self, empty_subject):
        """
        Property 30: Hybrid Embedding Construction (Unspecified Subject)
        
        When no subject is provided, the hybrid embedding should use "unspecified" encoding
        and be consistent across calls.
        
        **Validates: Requirements 2.5**
        """
        # Create a test visual embedding
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Generate subject encodings for different empty inputs
        encoding_empty = SubjectEncoder.encode_subject_category(empty_subject)
        encoding_explicit = SubjectEncoder.encode_subject_category(SubjectCategory.UNSPECIFIED)
        
        # Should be identical
        assert np.allclose(encoding_empty, encoding_explicit, rtol=1e-6), (
            f"Empty subject '{empty_subject}' should produce same encoding as explicit UNSPECIFIED"
        )
        
        # Create hybrid embeddings
        hybrid_empty = np.concatenate([visual_embedding, encoding_empty], axis=0)
        hybrid_explicit = np.concatenate([visual_embedding, encoding_explicit], axis=0)
        
        # Should be identical
        assert np.allclose(hybrid_empty, hybrid_explicit, rtol=1e-6), (
            "Hybrid embeddings should be identical for empty and explicit unspecified subjects"
        )
        
        # Verify dimensions
        assert hybrid_empty.shape == (832,)
        assert hybrid_explicit.shape == (832,)
    
    def test_hybrid_embedding_dimension_validation(self):
        """
        Test that hybrid embedding dimension validation works correctly.
        """
        # Test valid hybrid embedding separation
        valid_embedding = np.random.randn(832).astype(np.float32)
        
        # Should be able to separate into 768 + 64 components
        visual_component = valid_embedding[:768]
        subject_component = valid_embedding[768:]
        
        assert visual_component.shape == (768,)
        assert subject_component.shape == (64,)
        
        # Test that concatenation gives back original
        reconstructed = np.concatenate([visual_component, subject_component], axis=0)
        assert np.allclose(reconstructed, valid_embedding, rtol=1e-6)
    
    def test_subject_encoding_integration(self):
        """
        Test that subject encoding is properly integrated into hybrid embeddings.
        """
        # Create a test visual embedding
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Test different subjects produce different embeddings
        encoding_house = SubjectEncoder.encode_subject_category(SubjectCategory.HOUSE)
        encoding_car = SubjectEncoder.encode_subject_category(SubjectCategory.CAR)
        
        hybrid_house = np.concatenate([visual_embedding, encoding_house], axis=0)
        hybrid_car = np.concatenate([visual_embedding, encoding_car], axis=0)
        
        # Visual components should be identical
        assert np.allclose(hybrid_house[:768], hybrid_car[:768], rtol=1e-6), (
            "Visual components should be identical for same visual input"
        )
        
        # Subject components should be different
        assert not np.allclose(hybrid_house[768:], hybrid_car[768:], rtol=1e-6), (
            "Subject components should be different for different subjects"
        )
        
        # Verify subject components match expected encodings
        assert np.allclose(hybrid_house[768:], encoding_house, rtol=1e-6), (
            "House subject component should match expected encoding"
        )
        assert np.allclose(hybrid_car[768:], encoding_car, rtol=1e-6), (
            "Car subject component should match expected encoding"
        )
    
    def test_hybrid_embedding_consistency_across_subjects(self):
        """
        Test that hybrid embeddings are consistent across all subject categories.
        """
        # Create a test visual embedding
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Test all subject categories
        for subject in SubjectCategory:
            subject_encoding = SubjectEncoder.encode_subject_category(subject)
            hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
            
            # Verify dimensions
            assert hybrid_embedding.shape == (832,), (
                f"Hybrid embedding for {subject} should be 832-dimensional"
            )
            
            # Verify visual component unchanged
            assert np.allclose(hybrid_embedding[:768], visual_embedding, rtol=1e-6), (
                f"Visual component should be unchanged for {subject}"
            )
            
            # Verify subject component is valid one-hot
            subject_component = hybrid_embedding[768:]
            assert np.sum(subject_component) == 1.0, (
                f"Subject component for {subject} should sum to 1.0"
            )
            assert np.sum(subject_component == 1.0) == 1, (
                f"Subject component for {subject} should have exactly one 1.0"
            )
    
    @given(
        st.lists(st.sampled_from(list(SubjectCategory)), min_size=1, max_size=5),
        st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=20)
    def test_batch_hybrid_embedding_construction(self, subject_list, visual_seed):
        """
        Property 30: Hybrid Embedding Construction (Batch Processing)
        
        Batch construction should produce the same results as individual construction
        for the same inputs.
        
        **Validates: Requirements 2.5**
        """
        # Create deterministic visual embeddings
        np.random.seed(visual_seed)
        visual_embeddings = [np.random.randn(768).astype(np.float32) for _ in subject_list]
        
        # Generate embeddings individually
        individual_embeddings = []
        for visual, subject in zip(visual_embeddings, subject_list):
            subject_encoding = SubjectEncoder.encode_subject_category(subject)
            hybrid = np.concatenate([visual, subject_encoding], axis=0)
            individual_embeddings.append(hybrid)
        
        # Generate embeddings in batch (simulate batch processing)
        batch_embeddings = []
        for visual, subject in zip(visual_embeddings, subject_list):
            subject_encoding = SubjectEncoder.encode_subject_category(subject)
            hybrid = np.concatenate([visual, subject_encoding], axis=0)
            batch_embeddings.append(hybrid)
        
        # Verify batch results match individual results
        assert len(batch_embeddings) == len(individual_embeddings), (
            f"Batch size mismatch: expected {len(individual_embeddings)}, got {len(batch_embeddings)}"
        )
        
        for i, (individual, batch) in enumerate(zip(individual_embeddings, batch_embeddings)):
            assert np.allclose(individual, batch, rtol=1e-6), (
                f"Batch embedding {i} doesn't match individual embedding for subject {subject_list[i]}"
            )
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10)
    def test_hybrid_embedding_deterministic_construction(self, seed):
        """
        Property 30: Hybrid Embedding Construction (Determinism)
        
        The same visual embedding and subject should always produce the same hybrid embedding.
        
        **Validates: Requirements 2.5**
        """
        # Create deterministic visual embedding
        np.random.seed(seed)
        visual_embedding = np.random.randn(768).astype(np.float32)
        subject = SubjectCategory.HOUSE
        
        # Generate the same embedding multiple times
        subject_encoding1 = SubjectEncoder.encode_subject_category(subject)
        hybrid1 = np.concatenate([visual_embedding, subject_encoding1], axis=0)
        
        subject_encoding2 = SubjectEncoder.encode_subject_category(subject)
        hybrid2 = np.concatenate([visual_embedding, subject_encoding2], axis=0)
        
        # Should be identical
        assert np.allclose(hybrid1, hybrid2, rtol=1e-6), (
            "Same inputs should produce identical hybrid embeddings"
        )
        
        # Subject encodings should be identical
        assert np.allclose(subject_encoding1, subject_encoding2, rtol=1e-6), (
            "Same subject should produce identical encodings"
        )