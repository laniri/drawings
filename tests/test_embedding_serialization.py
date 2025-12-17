"""
Property-based tests for embedding serialization system.

**Feature: children-drawing-anomaly-detection, Property 6: Embedding Serialization Round Trip**
*For any* generated embedding, serializing then deserializing should produce an equivalent embedding vector
**Validates: Requirements 2.6**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays

from app.utils.embedding_serialization import (
    EmbeddingSerializer,
    EmbeddingCache,
    EmbeddingStorage,
    serialize_embedding_for_db,
    deserialize_embedding_from_db,
    EmbeddingSerializationError
)


class TestEmbeddingSerializer:
    """Test the EmbeddingSerializer class."""
    
    @given(
        embedding=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=2048),
            elements=st.floats(
                min_value=-100.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_embedding_serialization_round_trip(self, embedding):
        """
        **Feature: children-drawing-anomaly-detection, Property 6: Embedding Serialization Round Trip**
        *For any* generated embedding, serializing then deserializing should produce an equivalent embedding vector
        **Validates: Requirements 2.6**
        """
        # Serialize the embedding
        serialized_data = EmbeddingSerializer.serialize_embedding(embedding)
        
        # Verify serialized data is bytes
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        
        # Deserialize the embedding
        deserialized_embedding = EmbeddingSerializer.deserialize_embedding(serialized_data)
        
        # Verify the round trip preserves the embedding
        assert isinstance(deserialized_embedding, np.ndarray)
        assert deserialized_embedding.dtype == np.float32
        assert deserialized_embedding.shape == embedding.shape
        
        # Check that values are equivalent (allowing for small floating point differences)
        np.testing.assert_allclose(
            deserialized_embedding, 
            embedding.astype(np.float32), 
            rtol=1e-6, 
            atol=1e-6
        )
    
    @given(
        embedding=arrays(
            dtype=np.float64,  # Test with different dtype
            shape=st.integers(min_value=1, max_value=1024),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=64
            )
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_serialization_dtype_conversion(self, embedding):
        """Test that serialization converts to float32 consistently."""
        # Serialize and deserialize
        serialized_data = EmbeddingSerializer.serialize_embedding(embedding)
        deserialized_embedding = EmbeddingSerializer.deserialize_embedding(serialized_data)
        
        # Should be converted to float32
        assert deserialized_embedding.dtype == np.float32
        
        # Values should be equivalent within float32 precision
        np.testing.assert_allclose(
            deserialized_embedding, 
            embedding.astype(np.float32), 
            rtol=1e-6, 
            atol=1e-6
        )
    
    def test_serialization_error_handling(self):
        """Test error handling in serialization."""
        # Test with invalid input types
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.serialize_embedding("not an array")
        
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.serialize_embedding([1, 2, 3])
        
        # Test deserialization with invalid data
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.deserialize_embedding(b"invalid pickle data")
        
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.deserialize_embedding(b"")
        
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.deserialize_embedding("not bytes")
    
    @given(
        embedding=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=512),
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32
            )
        ),
        expected_dim=st.integers(min_value=1, max_value=512)
    )
    @settings(max_examples=50, deadline=None)
    def test_embedding_validation(self, embedding, expected_dim):
        """Test embedding validation logic."""
        # Valid embedding should pass validation
        is_valid = EmbeddingSerializer.validate_embedding(embedding)
        assert is_valid == True
        
        # Test dimension validation
        if len(embedding) == expected_dim:
            is_valid_dim = EmbeddingSerializer.validate_embedding(embedding, expected_dim)
            assert is_valid_dim == True
        else:
            is_valid_dim = EmbeddingSerializer.validate_embedding(embedding, expected_dim)
            assert is_valid_dim == False


class TestEmbeddingCache:
    """Test the EmbeddingCache class."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        
        # Test cache miss
        result = cache.get(drawing_id=1, model_type="vit")
        assert result is None
        
        # Test cache put and hit
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(drawing_id=1, model_type="vit", embedding=embedding)
        
        cached_embedding = cache.get(drawing_id=1, model_type="vit")
        assert cached_embedding is not None
        np.testing.assert_array_equal(cached_embedding, embedding)
        
        # Test that we get a copy (not the same object)
        assert cached_embedding is not embedding
    
    @given(
        drawing_ids=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20, unique=True),
        embeddings=st.lists(
            arrays(
                dtype=np.float32,
                shape=st.integers(min_value=10, max_value=100),
                elements=st.floats(
                    min_value=-1.0,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                    width=32
                )
            ),
            min_size=1,
            max_size=20
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_cache_consistency(self, drawing_ids, embeddings):
        """Test that cache consistently returns the same data that was stored."""
        # Ensure lists have same length and unique drawing IDs
        min_len = min(len(drawing_ids), len(embeddings))
        drawing_ids = drawing_ids[:min_len]
        embeddings = embeddings[:min_len]
        
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)
        
        # Store all embeddings
        for drawing_id, embedding in zip(drawing_ids, embeddings):
            cache.put(drawing_id=drawing_id, model_type="vit", embedding=embedding)
        
        # Retrieve and verify all embeddings
        for drawing_id, original_embedding in zip(drawing_ids, embeddings):
            cached_embedding = cache.get(drawing_id=drawing_id, model_type="vit")
            
            if cached_embedding is not None:  # May be evicted due to size limits
                assert isinstance(cached_embedding, np.ndarray)
                assert cached_embedding.dtype == np.float32
                np.testing.assert_array_equal(cached_embedding, original_embedding)
    
    def test_cache_eviction(self):
        """Test cache eviction policies."""
        cache = EmbeddingCache(max_size=3, ttl_seconds=3600)
        
        # Fill cache to capacity
        for i in range(3):
            embedding = np.array([float(i)], dtype=np.float32)
            cache.put(drawing_id=i, model_type="vit", embedding=embedding)
        
        # Verify all are cached
        for i in range(3):
            cached = cache.get(drawing_id=i, model_type="vit")
            assert cached is not None
        
        # Add one more (should evict LRU)
        new_embedding = np.array([99.0], dtype=np.float32)
        cache.put(drawing_id=99, model_type="vit", embedding=new_embedding)
        
        # First item should be evicted
        evicted = cache.get(drawing_id=0, model_type="vit")
        assert evicted is None
        
        # New item should be present
        new_cached = cache.get(drawing_id=99, model_type="vit")
        assert new_cached is not None
        np.testing.assert_array_equal(new_cached, new_embedding)


class TestEmbeddingStorage:
    """Test the EmbeddingStorage high-level interface."""
    
    @given(
        drawing_id=st.integers(min_value=1, max_value=1000),
        embedding=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=10, max_value=1024),
            elements=st.floats(
                min_value=-10.0,
                max_value=10.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32
            )
        ),
        age=st.one_of(st.none(), st.floats(min_value=2.0, max_value=18.0))
    )
    @settings(max_examples=50, deadline=None)
    def test_storage_round_trip(self, drawing_id, embedding, age):
        """Test complete storage and retrieval round trip."""
        storage = EmbeddingStorage(cache_size=100, cache_ttl=3600)
        
        # Store embedding
        serialized_data, dimension = storage.store_embedding(
            drawing_id=drawing_id,
            model_type="vit",
            embedding=embedding,
            age=age,
            use_cache=True
        )
        
        # Verify storage result
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        assert dimension == len(embedding)
        
        # Retrieve from cache (should hit cache)
        cached_embedding = storage.retrieve_embedding(
            drawing_id=drawing_id,
            model_type="vit",
            age=age,
            use_cache=True
        )
        
        assert cached_embedding is not None
        np.testing.assert_allclose(cached_embedding, embedding, rtol=1e-6, atol=1e-6)
        
        # Retrieve from serialized data (simulating database retrieval)
        db_embedding = storage.retrieve_embedding(
            drawing_id=drawing_id,
            model_type="vit",
            serialized_data=serialized_data,
            age=age,
            use_cache=False
        )
        
        assert db_embedding is not None
        np.testing.assert_allclose(db_embedding, embedding, rtol=1e-6, atol=1e-6)


class TestConvenienceFunctions:
    """Test the convenience functions for serialization."""
    
    @given(
        embedding=arrays(
            dtype=np.float32,
            shape=st.integers(min_value=1, max_value=512),
            elements=st.floats(
                min_value=-5.0,
                max_value=5.0,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32
            )
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_convenience_functions_round_trip(self, embedding):
        """
        Test convenience functions for database serialization.
        
        This is another validation of Property 6: Embedding Serialization Round Trip
        using the convenience functions that will be used throughout the codebase.
        """
        # Serialize using convenience function
        serialized_data = serialize_embedding_for_db(embedding)
        
        # Verify it's bytes
        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0
        
        # Deserialize using convenience function
        deserialized_embedding = deserialize_embedding_from_db(serialized_data)
        
        # Verify round trip
        assert isinstance(deserialized_embedding, np.ndarray)
        assert deserialized_embedding.dtype == np.float32
        assert deserialized_embedding.shape == embedding.shape
        
        np.testing.assert_allclose(
            deserialized_embedding, 
            embedding.astype(np.float32), 
            rtol=1e-6, 
            atol=1e-6
        )


# Unit tests for edge cases and error conditions
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_embedding(self):
        """Test handling of empty embeddings."""
        with pytest.raises(EmbeddingSerializationError):
            EmbeddingSerializer.serialize_embedding(np.array([]))
    
    def test_multidimensional_embedding(self):
        """Test handling of multidimensional arrays."""
        # 2D array should fail validation
        embedding_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)
        is_valid = EmbeddingSerializer.validate_embedding(embedding_2d)
        assert is_valid == False
    
    def test_nan_infinity_handling(self):
        """Test handling of NaN and infinity values."""
        # NaN values should fail validation
        embedding_nan = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        is_valid = EmbeddingSerializer.validate_embedding(embedding_nan)
        assert is_valid == False
        
        # Infinity values should fail validation
        embedding_inf = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        is_valid = EmbeddingSerializer.validate_embedding(embedding_inf)
        assert is_valid == False
    
    def test_cache_with_age_information(self):
        """Test cache behavior with age information."""
        cache = EmbeddingCache(max_size=10, ttl_seconds=3600)
        
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        # Store with age
        cache.put(drawing_id=1, model_type="vit", embedding=embedding, age=5.5)
        
        # Should not find without age
        result_no_age = cache.get(drawing_id=1, model_type="vit")
        assert result_no_age is None
        
        # Should find with correct age
        result_with_age = cache.get(drawing_id=1, model_type="vit", age=5.5)
        assert result_with_age is not None
        np.testing.assert_array_equal(result_with_age, embedding)
        
        # Should not find with different age
        result_wrong_age = cache.get(drawing_id=1, model_type="vit", age=6.0)
        assert result_wrong_age is None