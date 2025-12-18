"""
Embedding serialization utilities for numpy array storage and caching.

This module provides utilities for serializing and deserializing embedding vectors
for database storage, with support for caching mechanisms and data integrity validation.
"""

import logging
import pickle
import hashlib
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
import sqlite3
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import StorageError

logger = logging.getLogger(__name__)


class EmbeddingSerializationError(Exception):
    """Base exception for embedding serialization errors."""
    pass


class EmbeddingSerializer:
    """Handles serialization and deserialization of embedding vectors."""
    
    @staticmethod
    def serialize_hybrid_embedding(hybrid_embedding: np.ndarray) -> Tuple[bytes, bytes, bytes]:
        """
        Serialize a hybrid embedding with component separation.
        
        Args:
            hybrid_embedding: 832-dimensional hybrid embedding (768 visual + 64 subject)
            
        Returns:
            Tuple of (full_hybrid_bytes, visual_component_bytes, subject_component_bytes)
            
        Raises:
            EmbeddingSerializationError: If serialization fails
        """
        try:
            if not isinstance(hybrid_embedding, np.ndarray):
                raise EmbeddingSerializationError(f"Expected numpy array, got {type(hybrid_embedding)}")
            
            if hybrid_embedding.shape != (832,):
                raise EmbeddingSerializationError(
                    f"Expected 832-dimensional hybrid embedding, got shape {hybrid_embedding.shape}"
                )
            
            # Validate hybrid embedding before serialization
            if not EmbeddingSerializer.validate_hybrid_embedding(hybrid_embedding):
                raise EmbeddingSerializationError("Invalid hybrid embedding provided")
            
            # Ensure float32 for consistency
            if hybrid_embedding.dtype != np.float32:
                hybrid_embedding = hybrid_embedding.astype(np.float32)
            
            # Separate components
            visual_component = hybrid_embedding[:768]
            subject_component = hybrid_embedding[768:]
            
            # Serialize all components
            full_bytes = pickle.dumps(hybrid_embedding, protocol=pickle.HIGHEST_PROTOCOL)
            visual_bytes = pickle.dumps(visual_component, protocol=pickle.HIGHEST_PROTOCOL)
            subject_bytes = pickle.dumps(subject_component, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Serialized hybrid embedding: full_size={len(full_bytes)}, "
                        f"visual_size={len(visual_bytes)}, subject_size={len(subject_bytes)}")
            
            return full_bytes, visual_bytes, subject_bytes
            
        except Exception as e:
            if isinstance(e, EmbeddingSerializationError):
                raise
            raise EmbeddingSerializationError(f"Failed to serialize hybrid embedding: {str(e)}")
    
    @staticmethod
    def deserialize_hybrid_embedding(full_bytes: Optional[bytes] = None,
                                   visual_bytes: Optional[bytes] = None,
                                   subject_bytes: Optional[bytes] = None) -> Optional[np.ndarray]:
        """
        Deserialize a hybrid embedding from components or full data.
        
        Args:
            full_bytes: Serialized full hybrid embedding (preferred)
            visual_bytes: Serialized visual component (768-dim)
            subject_bytes: Serialized subject component (64-dim)
            
        Returns:
            832-dimensional hybrid embedding or None if reconstruction fails
        """
        # Try full embedding first
        if full_bytes is not None:
            try:
                hybrid_embedding = pickle.loads(full_bytes)
                if isinstance(hybrid_embedding, np.ndarray) and hybrid_embedding.shape == (832,):
                    return hybrid_embedding.astype(np.float32)
            except Exception as e:
                logger.debug(f"Failed to deserialize full hybrid embedding: {str(e)}")
        
        # Try to reconstruct from components
        if visual_bytes is not None and subject_bytes is not None:
            try:
                visual_component = pickle.loads(visual_bytes)
                subject_component = pickle.loads(subject_bytes)
                
                if (isinstance(visual_component, np.ndarray) and visual_component.shape == (768,) and
                    isinstance(subject_component, np.ndarray) and subject_component.shape == (64,)):
                    
                    # Reconstruct hybrid embedding
                    hybrid_embedding = np.concatenate([
                        visual_component.astype(np.float32),
                        subject_component.astype(np.float32)
                    ], axis=0)
                    
                    logger.debug("Reconstructed hybrid embedding from components")
                    return hybrid_embedding
            except Exception as e:
                logger.debug(f"Failed to reconstruct from components: {str(e)}")
        
        logger.debug("Could not deserialize hybrid embedding from available data")
        return None
    
    @staticmethod
    def validate_hybrid_embedding(embedding: np.ndarray) -> bool:
        """
        Validate a hybrid embedding for correctness.
        
        Args:
            embedding: Numpy array to validate (should be 832-dimensional)
            
        Returns:
            True if valid hybrid embedding, False otherwise
        """
        try:
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Invalid embedding type: {type(embedding)}")
                return False
            
            if embedding.shape != (832,):
                logger.warning(f"Invalid hybrid embedding shape: {embedding.shape} (expected (832,))")
                return False
            
            if not np.isfinite(embedding).all():
                logger.warning("Hybrid embedding contains non-finite values")
                return False
            
            # Validate subject component (last 64 dimensions should be one-hot)
            subject_component = embedding[768:]
            if not (np.sum(subject_component) == 1.0 and np.sum(subject_component == 1.0) == 1):
                logger.warning("Invalid subject component in hybrid embedding (not one-hot)")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Hybrid embedding validation failed: {str(e)}")
            return False

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """
        Serialize a numpy embedding array to bytes for database storage.
        
        Args:
            embedding: Numpy array containing the embedding vector
            
        Returns:
            Serialized bytes representation
            
        Raises:
            EmbeddingSerializationError: If serialization fails
        """
        try:
            if not isinstance(embedding, np.ndarray):
                raise EmbeddingSerializationError(f"Expected numpy array, got {type(embedding)}")
            
            if len(embedding) == 0:
                raise EmbeddingSerializationError("Cannot serialize empty embedding array")
            
            # Ensure float32 for consistency and space efficiency
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            
            # Use pickle for serialization (more efficient than JSON for numpy arrays)
            serialized_data = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.debug(f"Serialized embedding: shape={embedding.shape}, dtype={embedding.dtype}, size={len(serialized_data)} bytes")
            return serialized_data
            
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to serialize embedding: {str(e)}")
    
    @staticmethod
    def deserialize_embedding(serialized_data: bytes) -> np.ndarray:
        """
        Deserialize bytes back to numpy embedding array.
        
        Args:
            serialized_data: Serialized bytes from database
            
        Returns:
            Numpy array containing the embedding vector
            
        Raises:
            EmbeddingSerializationError: If deserialization fails
        """
        try:
            if not isinstance(serialized_data, bytes):
                raise EmbeddingSerializationError(f"Expected bytes, got {type(serialized_data)}")
            
            if len(serialized_data) == 0:
                raise EmbeddingSerializationError("Empty serialized data")
            
            # Deserialize using pickle
            embedding = pickle.loads(serialized_data)
            
            if not isinstance(embedding, np.ndarray):
                raise EmbeddingSerializationError(f"Deserialized data is not numpy array: {type(embedding)}")
            
            # Ensure float32 for consistency
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            
            logger.debug(f"Deserialized embedding: shape={embedding.shape}, dtype={embedding.dtype}")
            return embedding
            
        except pickle.PickleError as e:
            raise EmbeddingSerializationError(f"Failed to deserialize embedding (pickle error): {str(e)}")
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to deserialize embedding: {str(e)}")
    
    @staticmethod
    def validate_embedding(embedding: np.ndarray, expected_dimension: Optional[int] = None) -> bool:
        """
        Validate an embedding array for correctness.
        
        Args:
            embedding: Numpy array to validate
            expected_dimension: Expected embedding dimension (optional)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(embedding, np.ndarray):
                logger.warning(f"Invalid embedding type: {type(embedding)}")
                return False
            
            if embedding.ndim != 1:
                logger.warning(f"Invalid embedding dimensions: {embedding.ndim} (expected 1)")
                return False
            
            if len(embedding) == 0:
                logger.warning("Empty embedding array")
                return False
            
            if expected_dimension is not None and len(embedding) != expected_dimension:
                logger.warning(f"Invalid embedding size: {len(embedding)} (expected {expected_dimension})")
                return False
            
            if not np.isfinite(embedding).all():
                logger.warning("Embedding contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Embedding validation failed: {str(e)}")
            return False
    
    @staticmethod
    def compute_embedding_hash(embedding: np.ndarray) -> str:
        """
        Compute a hash for an embedding for caching and deduplication.
        
        Args:
            embedding: Numpy array to hash
            
        Returns:
            Hexadecimal hash string
        """
        try:
            # Convert to bytes and compute hash
            embedding_bytes = embedding.astype(np.float32).tobytes()
            hash_obj = hashlib.sha256(embedding_bytes)
            return hash_obj.hexdigest()[:16]  # Use first 16 characters for efficiency
            
        except Exception as e:
            logger.warning(f"Failed to compute embedding hash: {str(e)}")
            return ""


class EmbeddingCache:
    """In-memory cache for embedding vectors with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_order: List[str] = []
        
        logger.info(f"EmbeddingCache initialized: max_size={max_size}, ttl={ttl_seconds}s")
    
    def _generate_cache_key(self, drawing_id: int, model_type: str, age: Optional[float] = None) -> str:
        """Generate a cache key for an embedding."""
        age_str = f"_{age}" if age is not None else ""
        return f"{drawing_id}_{model_type}{age_str}"
    
    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, entry in self._cache.items():
            if (current_time - entry['timestamp']).total_seconds() > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
        
        if expired_keys:
            logger.debug(f"Evicted {len(expired_keys)} expired cache entries")
    
    def _evict_lru(self) -> None:
        """Remove least recently used entries to maintain max_size."""
        while len(self._cache) >= self.max_size and self._access_order:
            lru_key = self._access_order[0]
            self._remove_entry(lru_key)
            logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU tracking."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def get(self, drawing_id: int, model_type: str, age: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Get an embedding from cache.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            age: Optional age information
            
        Returns:
            Cached embedding array or None if not found
        """
        try:
            self._evict_expired()
            
            cache_key = self._generate_cache_key(drawing_id, model_type, age)
            
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                self._update_access_order(cache_key)
                
                logger.debug(f"Cache hit for key: {cache_key}")
                return entry['embedding'].copy()  # Return copy to prevent modification
            
            logger.debug(f"Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            logger.warning(f"Cache get failed for drawing {drawing_id}: {str(e)}")
            return None
    
    def put(self, drawing_id: int, model_type: str, embedding: np.ndarray, age: Optional[float] = None) -> None:
        """
        Store an embedding in cache.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            embedding: Embedding array to cache
            age: Optional age information
        """
        try:
            if not EmbeddingSerializer.validate_embedding(embedding):
                logger.warning(f"Invalid embedding not cached for drawing {drawing_id}")
                return
            
            self._evict_expired()
            self._evict_lru()
            
            cache_key = self._generate_cache_key(drawing_id, model_type, age)
            
            entry = {
                'embedding': embedding.copy(),  # Store copy to prevent external modification
                'timestamp': datetime.now(timezone.utc),
                'drawing_id': drawing_id,
                'model_type': model_type,
                'age': age,
                'dimension': len(embedding)
            }
            
            self._cache[cache_key] = entry
            self._update_access_order(cache_key)
            
            logger.debug(f"Cached embedding for key: {cache_key}, dimension: {len(embedding)}")
            
        except Exception as e:
            logger.warning(f"Cache put failed for drawing {drawing_id}: {str(e)}")
    
    def remove(self, drawing_id: int, model_type: str, age: Optional[float] = None) -> bool:
        """
        Remove an embedding from cache.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            age: Optional age information
            
        Returns:
            True if removed, False if not found
        """
        try:
            cache_key = self._generate_cache_key(drawing_id, model_type, age)
            
            if cache_key in self._cache:
                self._remove_entry(cache_key)
                logger.debug(f"Removed cache entry: {cache_key}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Cache remove failed for drawing {drawing_id}: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = datetime.now(timezone.utc)
        
        # Count expired entries
        expired_count = 0
        for entry in self._cache.values():
            if (current_time - entry['timestamp']).total_seconds() > self.ttl_seconds:
                expired_count += 1
        
        return {
            'total_entries': len(self._cache),
            'expired_entries': expired_count,
            'active_entries': len(self._cache) - expired_count,
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'memory_usage_estimate_mb': len(self._cache) * 0.003  # Rough estimate
        }


class EmbeddingStorage:
    """High-level interface for embedding storage with caching."""
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize embedding storage.
        
        Args:
            cache_size: Maximum number of embeddings to cache
            cache_ttl: Cache time-to-live in seconds
        """
        self.serializer = EmbeddingSerializer()
        self.cache = EmbeddingCache(max_size=cache_size, ttl_seconds=cache_ttl)
        
        logger.info("EmbeddingStorage initialized")
    
    def store_hybrid_embedding(self,
                              drawing_id: int,
                              model_type: str,
                              hybrid_embedding: np.ndarray,
                              age: Optional[float] = None,
                              use_cache: bool = True) -> Tuple[bytes, int, bytes, bytes]:
        """
        Store a hybrid embedding with component separation.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            hybrid_embedding: 832-dimensional hybrid embedding
            age: Optional age information
            use_cache: Whether to cache the embedding
            
        Returns:
            Tuple of (full_hybrid_bytes, dimension, visual_bytes, subject_bytes)
            
        Raises:
            EmbeddingSerializationError: If storage fails
        """
        try:
            # Validate hybrid embedding
            if not self.serializer.validate_hybrid_embedding(hybrid_embedding):
                raise EmbeddingSerializationError("Invalid hybrid embedding provided")
            
            # Serialize with component separation
            full_bytes, visual_bytes, subject_bytes = self.serializer.serialize_hybrid_embedding(hybrid_embedding)
            
            # Cache if requested
            if use_cache:
                self.cache.put(drawing_id, model_type, hybrid_embedding, age)
            
            logger.debug(f"Stored hybrid embedding for drawing {drawing_id}: dimension={len(hybrid_embedding)}")
            return full_bytes, len(hybrid_embedding), visual_bytes, subject_bytes
            
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to store hybrid embedding: {str(e)}")
    
    def retrieve_hybrid_embedding(self,
                                 drawing_id: int,
                                 model_type: str,
                                 full_data: Optional[bytes] = None,
                                 visual_data: Optional[bytes] = None,
                                 subject_data: Optional[bytes] = None,
                                 age: Optional[float] = None,
                                 use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Retrieve a hybrid embedding with component reconstruction support.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            full_data: Serialized full hybrid embedding data
            visual_data: Serialized visual component data
            subject_data: Serialized subject component data
            age: Optional age information
            use_cache: Whether to use cache
            
        Returns:
            832-dimensional hybrid embedding or None if not found
            
        Raises:
            EmbeddingSerializationError: If retrieval fails
        """
        try:
            # Try cache first if enabled
            if use_cache:
                cached_embedding = self.cache.get(drawing_id, model_type, age)
                if cached_embedding is not None and cached_embedding.shape == (832,):
                    logger.debug(f"Retrieved hybrid embedding from cache for drawing {drawing_id}")
                    return cached_embedding
            
            # Deserialize from database components
            hybrid_embedding = self.serializer.deserialize_hybrid_embedding(
                full_bytes=full_data,
                visual_bytes=visual_data,
                subject_bytes=subject_data
            )
            
            if hybrid_embedding is not None:
                # Cache for future use
                if use_cache:
                    self.cache.put(drawing_id, model_type, hybrid_embedding, age)
                
                logger.debug(f"Retrieved hybrid embedding from database for drawing {drawing_id}")
                return hybrid_embedding
            
            logger.debug(f"No hybrid embedding found for drawing {drawing_id}")
            return None
            
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to retrieve hybrid embedding: {str(e)}")

    def store_embedding(self, 
                       drawing_id: int, 
                       model_type: str, 
                       embedding: np.ndarray,
                       age: Optional[float] = None,
                       use_cache: bool = True) -> Tuple[bytes, int]:
        """
        Store an embedding with serialization and caching.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            embedding: Embedding array to store
            age: Optional age information
            use_cache: Whether to cache the embedding
            
        Returns:
            Tuple of (serialized_bytes, dimension)
            
        Raises:
            EmbeddingSerializationError: If storage fails
        """
        try:
            # Validate embedding
            if not self.serializer.validate_embedding(embedding):
                raise EmbeddingSerializationError("Invalid embedding provided")
            
            # Serialize for database storage
            serialized_data = self.serializer.serialize_embedding(embedding)
            
            # Cache if requested
            if use_cache:
                self.cache.put(drawing_id, model_type, embedding, age)
            
            logger.debug(f"Stored embedding for drawing {drawing_id}: dimension={len(embedding)}")
            return serialized_data, len(embedding)
            
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to store embedding: {str(e)}")
    
    def retrieve_embedding(self, 
                          drawing_id: int, 
                          model_type: str,
                          serialized_data: Optional[bytes] = None,
                          age: Optional[float] = None,
                          use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Retrieve an embedding with caching support.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            serialized_data: Serialized embedding data from database (if not using cache)
            age: Optional age information
            use_cache: Whether to use cache
            
        Returns:
            Embedding array or None if not found
            
        Raises:
            EmbeddingSerializationError: If retrieval fails
        """
        try:
            # Try cache first if enabled
            if use_cache:
                cached_embedding = self.cache.get(drawing_id, model_type, age)
                if cached_embedding is not None:
                    logger.debug(f"Retrieved embedding from cache for drawing {drawing_id}")
                    return cached_embedding
            
            # Deserialize from database if provided
            if serialized_data is not None:
                embedding = self.serializer.deserialize_embedding(serialized_data)
                
                # Cache for future use
                if use_cache:
                    self.cache.put(drawing_id, model_type, embedding, age)
                
                logger.debug(f"Retrieved embedding from database for drawing {drawing_id}")
                return embedding
            
            logger.debug(f"No embedding found for drawing {drawing_id}")
            return None
            
        except Exception as e:
            raise EmbeddingSerializationError(f"Failed to retrieve embedding: {str(e)}")
    
    def batch_store_embeddings(self, 
                              embeddings_data: List[Dict[str, Any]],
                              use_cache: bool = True) -> List[Tuple[bytes, int]]:
        """
        Store multiple embeddings in batch.
        
        Args:
            embeddings_data: List of dicts with keys: drawing_id, model_type, embedding, age (optional)
            use_cache: Whether to cache embeddings
            
        Returns:
            List of (serialized_bytes, dimension) tuples
        """
        results = []
        
        for data in embeddings_data:
            try:
                result = self.store_embedding(
                    drawing_id=data['drawing_id'],
                    model_type=data['model_type'],
                    embedding=data['embedding'],
                    age=data.get('age'),
                    use_cache=use_cache
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to store embedding for drawing {data.get('drawing_id')}: {str(e)}")
                # Add placeholder for failed embedding
                results.append((b'', 0))
        
        logger.info(f"Batch stored {len(results)} embeddings")
        return results
    
    def invalidate_cache(self, drawing_id: int, model_type: str, age: Optional[float] = None) -> bool:
        """
        Invalidate cached embedding for a specific drawing.
        
        Args:
            drawing_id: Database ID of the drawing
            model_type: Type of model used for embedding
            age: Optional age information
            
        Returns:
            True if cache entry was removed
        """
        return self.cache.remove(drawing_id, model_type, age)
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage and cache statistics."""
        return {
            'cache_stats': self.cache.get_stats(),
            'serializer_type': 'pickle',
            'storage_ready': True
        }


# Global storage instance
_embedding_storage = None


def get_embedding_storage() -> EmbeddingStorage:
    """Get the global embedding storage instance."""
    global _embedding_storage
    if _embedding_storage is None:
        _embedding_storage = EmbeddingStorage(
            cache_size=getattr(settings, 'EMBEDDING_CACHE_SIZE', 1000),
            cache_ttl=getattr(settings, 'EMBEDDING_CACHE_TTL', 3600)
        )
    return _embedding_storage


def serialize_embedding_for_db(embedding: np.ndarray) -> bytes:
    """
    Convenience function to serialize an embedding for database storage.
    
    Args:
        embedding: Numpy array to serialize
        
    Returns:
        Serialized bytes
    """
    return EmbeddingSerializer.serialize_embedding(embedding)


def deserialize_embedding_from_db(serialized_data: bytes) -> np.ndarray:
    """
    Convenience function to deserialize an embedding from database.
    
    Args:
        serialized_data: Serialized bytes from database
        
    Returns:
        Numpy array
    """
    return EmbeddingSerializer.deserialize_embedding(serialized_data)


def serialize_hybrid_embedding_for_db(hybrid_embedding: np.ndarray) -> Tuple[bytes, bytes, bytes]:
    """
    Convenience function to serialize a hybrid embedding for database storage.
    
    Args:
        hybrid_embedding: 832-dimensional hybrid embedding array
        
    Returns:
        Tuple of (full_hybrid_bytes, visual_component_bytes, subject_component_bytes)
    """
    return EmbeddingSerializer.serialize_hybrid_embedding(hybrid_embedding)


def deserialize_hybrid_embedding_from_db(full_bytes: Optional[bytes] = None,
                                        visual_bytes: Optional[bytes] = None,
                                        subject_bytes: Optional[bytes] = None) -> Optional[np.ndarray]:
    """
    Convenience function to deserialize a hybrid embedding from database.
    
    Args:
        full_bytes: Serialized full hybrid embedding
        visual_bytes: Serialized visual component
        subject_bytes: Serialized subject component
        
    Returns:
        832-dimensional hybrid embedding or None if reconstruction fails
    """
    return EmbeddingSerializer.deserialize_hybrid_embedding(full_bytes, visual_bytes, subject_bytes)


def validate_hybrid_embedding(embedding: np.ndarray) -> bool:
    """
    Convenience function to validate a hybrid embedding.
    
    Args:
        embedding: Numpy array to validate
        
    Returns:
        True if valid hybrid embedding, False otherwise
    """
    return EmbeddingSerializer.validate_hybrid_embedding(embedding)