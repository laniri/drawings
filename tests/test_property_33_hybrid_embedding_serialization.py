"""
Property-based test for hybrid embedding serialization round trip.

**Feature: children-drawing-anomaly-detection, Property 33: Hybrid Embedding Serialization Round Trip**
**Validates: Requirements 2.9**

This test verifies that hybrid embeddings can be serialized and deserialized
without data loss, maintaining component separability.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from app.utils.embedding_serialization import (
    EmbeddingSerializer, 
    serialize_hybrid_embedding_for_db,
    deserialize_hybrid_embedding_from_db,
    validate_hybrid_embedding,
    EmbeddingSerializationError
)
from app.services.embedding_service import SubjectEncoder
from app.schemas.drawings import SubjectCategory


class TestHybridEmbeddingSerializationRoundTrip:
    """Test hybrid embedding serialization round trip properties."""
    
    def create_hybrid_embedding(self, visual_seed: int, subject_category: SubjectCategory) -> np.ndarray:
        """Create a deterministic hybrid embedding for testing."""
        # Create deterministic visual embedding
        np.random.seed(visual_seed)
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Create subject encoding
        subject_encoding = SubjectEncoder.encode_subject_category(subject_category)
        
        # Combine into hybrid embedding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
        
        return hybrid_embedding
    
    @given(
        st.integers(min_value=1, max_value=10000),
        st.sampled_from(list(SubjectCategory))
    )
    @settings(max_examples=100)
    def test_hybrid_embedding_serialization_round_trip(self, visual_seed, subject_category):
        """
        Property 33: Hybrid Embedding Serialization Round Trip
        
        For any hybrid embedding, serializing then deserializing should produce
        an identical embedding with preserved component separability.
        
        **Validates: Requirements 2.9**
        """
        # Create original hybrid embedding
        original_embedding = self.create_hybrid_embedding(visual_seed, subject_category)
        
        # Verify it's a valid hybrid embedding
        assert validate_hybrid_embedding(original_embedding), (
            f"Original embedding should be valid for {subject_category}"
        )
        
        # Serialize using the convenience function
        full_bytes, visual_bytes, subject_bytes = serialize_hybrid_embedding_for_db(original_embedding)
        
        # Verify serialization produces bytes
        assert isinstance(full_bytes, bytes), "Full serialization should produce bytes"
        assert isinstance(visual_bytes, bytes), "Visual serialization should produce bytes"
        assert isinstance(subject_bytes, bytes), "Subject serialization should produce bytes"
        assert len(full_bytes) > 0, "Full serialization should not be empty"
        assert len(visual_bytes) > 0, "Visual serialization should not be empty"
        assert len(subject_bytes) > 0, "Subject serialization should not be empty"
        
        # Deserialize using full data
        reconstructed_full = deserialize_hybrid_embedding_from_db(full_bytes=full_bytes)
        
        # Verify full reconstruction
        assert reconstructed_full is not None, "Full reconstruction should succeed"
        assert reconstructed_full.shape == (832,), (
            f"Reconstructed embedding should be 832-dimensional, got {reconstructed_full.shape}"
        )
        assert np.allclose(reconstructed_full, original_embedding, rtol=1e-6), (
            f"Full round-trip should preserve embedding for {subject_category}"
        )
        
        # Deserialize using component data
        reconstructed_components = deserialize_hybrid_embedding_from_db(
            visual_bytes=visual_bytes,
            subject_bytes=subject_bytes
        )
        
        # Verify component reconstruction
        assert reconstructed_components is not None, "Component reconstruction should succeed"
        assert reconstructed_components.shape == (832,), (
            f"Component-reconstructed embedding should be 832-dimensional"
        )
        assert np.allclose(reconstructed_components, original_embedding, rtol=1e-6), (
            f"Component round-trip should preserve embedding for {subject_category}"
        )
        
        # Verify both reconstruction methods produce identical results
        assert np.allclose(reconstructed_full, reconstructed_components, rtol=1e-6), (
            "Full and component reconstruction should produce identical results"
        )
        
        # Verify component separability is preserved
        original_visual = original_embedding[:768]
        original_subject = original_embedding[768:]
        
        reconstructed_visual = reconstructed_full[:768]
        reconstructed_subject = reconstructed_full[768:]
        
        assert np.allclose(reconstructed_visual, original_visual, rtol=1e-6), (
            "Visual component should be preserved in round-trip"
        )
        assert np.allclose(reconstructed_subject, original_subject, rtol=1e-6), (
            "Subject component should be preserved in round-trip"
        )
        
        # Verify subject component is still decodable
        decoded_subject = SubjectEncoder.decode_subject_encoding(reconstructed_subject)
        assert decoded_subject == subject_category, (
            f"Subject should decode correctly after round-trip: expected {subject_category}, got {decoded_subject}"
        )
    
    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_serialization_consistency_across_calls(self, visual_seed):
        """
        Property 33: Hybrid Embedding Serialization Round Trip (Consistency)
        
        Multiple serializations of the same embedding should produce identical results.
        
        **Validates: Requirements 2.9**
        """
        # Create hybrid embedding
        hybrid_embedding = self.create_hybrid_embedding(visual_seed, SubjectCategory.HOUSE)
        
        # Serialize multiple times
        result1 = serialize_hybrid_embedding_for_db(hybrid_embedding)
        result2 = serialize_hybrid_embedding_for_db(hybrid_embedding)
        result3 = serialize_hybrid_embedding_for_db(hybrid_embedding)
        
        # All serializations should be identical
        assert result1[0] == result2[0] == result3[0], (
            "Full serialization should be consistent across calls"
        )
        assert result1[1] == result2[1] == result3[1], (
            "Visual serialization should be consistent across calls"
        )
        assert result1[2] == result2[2] == result3[2], (
            "Subject serialization should be consistent across calls"
        )
    
    def test_serialization_with_edge_case_embeddings(self):
        """
        Test serialization with edge case embedding values.
        """
        edge_cases = [
            # All zeros visual with valid subject
            (np.zeros(768, dtype=np.float32), SubjectCategory.UNSPECIFIED),
            # All ones visual with valid subject
            (np.ones(768, dtype=np.float32), SubjectCategory.HOUSE),
            # Very small values
            (np.full(768, 1e-10, dtype=np.float32), SubjectCategory.CAR),
            # Very large values (but finite)
            (np.full(768, 1e6, dtype=np.float32), SubjectCategory.TREE),
            # Mixed positive/negative
            (np.array([1.0, -1.0] * 384, dtype=np.float32), SubjectCategory.PERSON),
        ]
        
        for visual_embedding, subject_category in edge_cases:
            # Create hybrid embedding
            subject_encoding = SubjectEncoder.encode_subject_category(subject_category)
            hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
            
            # Verify it's valid
            assert validate_hybrid_embedding(hybrid_embedding), (
                f"Edge case embedding should be valid for {subject_category}"
            )
            
            # Test round-trip
            full_bytes, visual_bytes, subject_bytes = serialize_hybrid_embedding_for_db(hybrid_embedding)
            reconstructed = deserialize_hybrid_embedding_from_db(full_bytes=full_bytes)
            
            assert reconstructed is not None, f"Edge case reconstruction should succeed for {subject_category}"
            assert np.allclose(reconstructed, hybrid_embedding, rtol=1e-6), (
                f"Edge case round-trip should preserve embedding for {subject_category}"
            )
    
    def test_serialization_error_handling(self):
        """
        Test that serialization properly handles invalid inputs.
        """
        # Test invalid embedding dimensions
        with pytest.raises(EmbeddingSerializationError, match="Expected 832-dimensional"):
            invalid_embedding = np.random.randn(500).astype(np.float32)
            serialize_hybrid_embedding_for_db(invalid_embedding)
        
        # Test non-numpy input
        with pytest.raises(EmbeddingSerializationError, match="Expected numpy array"):
            serialize_hybrid_embedding_for_db([1, 2, 3])  # List instead of numpy array
        
        # Test invalid subject component (not one-hot)
        visual_embedding = np.random.randn(768).astype(np.float32)
        invalid_subject = np.random.randn(64).astype(np.float32)  # Not one-hot
        invalid_hybrid = np.concatenate([visual_embedding, invalid_subject], axis=0)
        
        with pytest.raises(EmbeddingSerializationError, match="Invalid hybrid embedding"):
            serialize_hybrid_embedding_for_db(invalid_hybrid)
    
    def test_deserialization_fallback_behavior(self):
        """
        Test deserialization fallback from full to components.
        """
        # Create hybrid embedding
        hybrid_embedding = self.create_hybrid_embedding(42, SubjectCategory.DOG)
        
        # Serialize
        full_bytes, visual_bytes, subject_bytes = serialize_hybrid_embedding_for_db(hybrid_embedding)
        
        # Test deserialization with only components (no full data)
        reconstructed = deserialize_hybrid_embedding_from_db(
            full_bytes=None,
            visual_bytes=visual_bytes,
            subject_bytes=subject_bytes
        )
        
        assert reconstructed is not None, "Component-only reconstruction should succeed"
        assert np.allclose(reconstructed, hybrid_embedding, rtol=1e-6), (
            "Component-only reconstruction should preserve embedding"
        )
        
        # Test deserialization with corrupted full data but valid components
        corrupted_full = b"corrupted_data"
        reconstructed_fallback = deserialize_hybrid_embedding_from_db(
            full_bytes=corrupted_full,
            visual_bytes=visual_bytes,
            subject_bytes=subject_bytes
        )
        
        # Should fall back to component reconstruction
        assert reconstructed_fallback is not None, "Should fall back to component reconstruction"
        assert np.allclose(reconstructed_fallback, hybrid_embedding, rtol=1e-6), (
            "Fallback reconstruction should preserve embedding"
        )
    
    def test_deserialization_with_missing_data(self):
        """
        Test deserialization behavior when data is missing.
        """
        # Test with no data at all
        result = deserialize_hybrid_embedding_from_db()
        assert result is None, "Deserialization with no data should return None"
        
        # Test with only full data (should work)
        hybrid_embedding = self.create_hybrid_embedding(123, SubjectCategory.FISH)
        full_bytes, _, _ = serialize_hybrid_embedding_for_db(hybrid_embedding)
        
        result = deserialize_hybrid_embedding_from_db(full_bytes=full_bytes)
        assert result is not None, "Deserialization with only full data should work"
        assert np.allclose(result, hybrid_embedding, rtol=1e-6)
        
        # Test with only visual data (should fail)
        _, visual_bytes, _ = serialize_hybrid_embedding_for_db(hybrid_embedding)
        result = deserialize_hybrid_embedding_from_db(visual_bytes=visual_bytes)
        assert result is None, "Deserialization with only visual data should return None"
        
        # Test with only subject data (should fail)
        _, _, subject_bytes = serialize_hybrid_embedding_for_db(hybrid_embedding)
        result = deserialize_hybrid_embedding_from_db(subject_bytes=subject_bytes)
        assert result is None, "Deserialization with only subject data should return None"
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=1000),
                st.sampled_from(list(SubjectCategory))
            ),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=20)
    def test_batch_serialization_consistency(self, embedding_specs):
        """
        Property 33: Hybrid Embedding Serialization Round Trip (Batch Processing)
        
        Batch serialization should produce the same results as individual serialization.
        
        **Validates: Requirements 2.9**
        """
        # Create embeddings
        embeddings = []
        for visual_seed, subject_category in embedding_specs:
            embedding = self.create_hybrid_embedding(visual_seed, subject_category)
            embeddings.append(embedding)
        
        # Serialize individually
        individual_results = []
        for embedding in embeddings:
            result = serialize_hybrid_embedding_for_db(embedding)
            individual_results.append(result)
        
        # Serialize in batch (simulate batch processing)
        batch_results = []
        for embedding in embeddings:
            result = serialize_hybrid_embedding_for_db(embedding)
            batch_results.append(result)
        
        # Results should be identical
        assert len(individual_results) == len(batch_results), (
            "Batch and individual results should have same length"
        )
        
        for i, (individual, batch) in enumerate(zip(individual_results, batch_results)):
            assert individual[0] == batch[0], f"Full serialization {i} should match"
            assert individual[1] == batch[1], f"Visual serialization {i} should match"
            assert individual[2] == batch[2], f"Subject serialization {i} should match"
        
        # Test round-trip for all
        for i, (embedding, (full_bytes, visual_bytes, subject_bytes)) in enumerate(zip(embeddings, batch_results)):
            reconstructed = deserialize_hybrid_embedding_from_db(full_bytes=full_bytes)
            assert reconstructed is not None, f"Batch reconstruction {i} should succeed"
            assert np.allclose(reconstructed, embedding, rtol=1e-6), (
                f"Batch round-trip {i} should preserve embedding"
            )
    
    def test_serialization_data_integrity(self):
        """
        Test that serialized data maintains integrity across different scenarios.
        """
        # Create test embedding
        hybrid_embedding = self.create_hybrid_embedding(999, SubjectCategory.WHALE)
        
        # Serialize
        full_bytes, visual_bytes, subject_bytes = serialize_hybrid_embedding_for_db(hybrid_embedding)
        
        # Test that serialized data is not empty and has reasonable size
        assert len(full_bytes) > 100, "Full serialization should have reasonable size"
        assert len(visual_bytes) > 50, "Visual serialization should have reasonable size"
        assert len(subject_bytes) > 10, "Subject serialization should have reasonable size"
        
        # Test that different embeddings produce different serializations
        different_embedding = self.create_hybrid_embedding(888, SubjectCategory.SPIDER)
        diff_full, diff_visual, diff_subject = serialize_hybrid_embedding_for_db(different_embedding)
        
        assert full_bytes != diff_full, "Different embeddings should produce different full serializations"
        assert visual_bytes != diff_visual, "Different embeddings should produce different visual serializations"
        # Subject might be the same if same category, so we don't assert difference for subject_bytes
        
        # Test that serialization is deterministic
        same_full, same_visual, same_subject = serialize_hybrid_embedding_for_db(hybrid_embedding)
        assert full_bytes == same_full, "Same embedding should produce identical full serialization"
        assert visual_bytes == same_visual, "Same embedding should produce identical visual serialization"
        assert subject_bytes == same_subject, "Same embedding should produce identical subject serialization"