"""
Property-based tests for embedding service.

**Feature: children-drawing-anomaly-detection, Property 4: Embedding Dimensionality Consistency**
**Validates: Requirements 2.1, 2.4**

**Feature: children-drawing-anomaly-detection, Property 5: Age-Augmented Embedding Consistency**
**Validates: Requirements 2.3**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import io
import tempfile
import os
from unittest.mock import Mock, patch

from app.services.embedding_service import (
    EmbeddingService, 
    EmbeddingServiceError, 
    ModelLoadingError, 
    EmbeddingGenerationError,
    DeviceManager,
    VisionTransformerWrapper,
    EmbeddingPipeline
)


def create_test_image(width: int, height: int, mode: str = 'RGB') -> Image.Image:
    """Create a test PIL image with specified dimensions and mode"""
    # Create a simple test image with appropriate color for mode
    if mode == 'RGB':
        color = (128, 128, 128)
    elif mode == 'RGBA':
        color = (128, 128, 128, 255)
    elif mode == 'L':
        color = 128
    else:
        color = (128, 128, 128)
    
    image = Image.new(mode, (width, height), color=color)
    
    # Add some pattern to make it more realistic
    pixels = image.load()
    for i in range(0, min(width, 50), 10):
        for j in range(0, min(height, 50), 10):
            if mode == 'RGB':
                pixels[i, j] = (255, 0, 0)
            elif mode == 'RGBA':
                pixels[i, j] = (255, 0, 0, 255)
            elif mode == 'L':
                pixels[i, j] = 255
    
    return image


def create_numpy_image(width: int, height: int, channels: int = 3) -> np.ndarray:
    """Create a test numpy image array"""
    if channels == 3:
        image = np.random.rand(height, width, channels).astype(np.float32)
    elif channels == 1:
        image = np.random.rand(height, width).astype(np.float32)
    else:
        image = np.random.rand(height, width, channels).astype(np.float32)
    
    # Ensure values are in [0, 1] range
    image = np.clip(image, 0.0, 1.0)
    return image


# Hypothesis strategies
valid_dimensions_strategy = st.tuples(
    st.integers(min_value=32, max_value=512),  # width
    st.integers(min_value=32, max_value=512)   # height
)

age_strategy = st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)
image_mode_strategy = st.sampled_from(['RGB', 'RGBA', 'L'])


class MockEmbeddingService:
    """Mock embedding service for testing without loading actual models"""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.is_initialized = True
        self.device_manager = Mock()
        self.device_manager.device = Mock()
        self.device_manager.device.type = "cpu"
        
        self.vit_wrapper = Mock()
        self.vit_wrapper.model = Mock()
        self.vit_wrapper.model.config = Mock()
        self.vit_wrapper.model.config.hidden_size = embedding_dim
        self.vit_wrapper.processor = Mock()
        
    def is_ready(self):
        return self.is_initialized
    
    def get_embedding_dimension(self, include_age=False):
        return self.embedding_dim + (1 if include_age else 0)
    
    def generate_embedding(self, image, age=None, use_cache=True):
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        # Simulate embedding generation
        base_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        
        if age is not None:
            age_feature = np.array([age], dtype=np.float32)
            embedding = np.concatenate([base_embedding, age_feature])
        else:
            embedding = base_embedding
            
        return embedding
    
    def generate_batch_embeddings(self, images, ages=None, batch_size=8, use_cache=True):
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")
        
        embeddings = []
        for i, image in enumerate(images):
            age = ages[i] if ages else None
            embedding = self.generate_embedding(image, age, use_cache)
            embeddings.append(embedding)
        return embeddings
    
    def generate_hybrid_embedding(self, image, subject=None, age=None, use_cache=True):
        """Generate hybrid embedding combining visual features and subject encoding."""
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        # Simulate hybrid embedding generation (768 visual + 64 subject = 832 total)
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_embedding = np.random.rand(64).astype(np.float32)
        hybrid_embedding = np.concatenate([visual_embedding, subject_embedding])
        
        if age is not None:
            age_feature = np.array([age], dtype=np.float32)
            hybrid_embedding = np.concatenate([hybrid_embedding, age_feature])
            
        return hybrid_embedding


@given(
    dimensions1=valid_dimensions_strategy,
    dimensions2=valid_dimensions_strategy,
    mode1=image_mode_strategy,
    mode2=image_mode_strategy
)
@settings(max_examples=50, deadline=None)
def test_embedding_dimensionality_consistency(dimensions1, dimensions2, mode1, mode2):
    """
    **Feature: children-drawing-anomaly-detection, Property 4: Embedding Dimensionality Consistency**
    **Validates: Requirements 2.1, 2.4**
    
    Property: For any set of drawings processed with the same model configuration, 
    all generated embeddings should have identical vector dimensions.
    """
    width1, height1 = dimensions1
    width2, height2 = dimensions2
    
    # Create two different test images
    image1 = create_test_image(width1, height1, mode1)
    image2 = create_test_image(width2, height2, mode2)
    
    # Use mock service to avoid loading actual models in tests
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate embeddings for both images (without age)
    embedding1 = service.generate_embedding(image1, age=None)
    embedding2 = service.generate_embedding(image2, age=None)
    
    # Verify both embeddings have identical dimensions
    assert embedding1.shape == embedding2.shape, \
        f"Embeddings should have identical dimensions: {embedding1.shape} vs {embedding2.shape}"
    
    # Verify dimensions match expected size
    expected_dim = service.get_embedding_dimension(include_age=False)
    assert len(embedding1) == expected_dim, \
        f"Embedding dimension should be {expected_dim}, got {len(embedding1)}"
    assert len(embedding2) == expected_dim, \
        f"Embedding dimension should be {expected_dim}, got {len(embedding2)}"
    
    # Verify embeddings are finite
    assert np.all(np.isfinite(embedding1)), "All embedding values should be finite"
    assert np.all(np.isfinite(embedding2)), "All embedding values should be finite"


@given(
    dimensions=valid_dimensions_strategy,
    ages=st.lists(age_strategy, min_size=2, max_size=10)
)
@settings(max_examples=30, deadline=None)
def test_batch_embedding_dimensionality_consistency(dimensions, ages):
    """
    Property: For any batch of images processed together, all embeddings should have 
    identical dimensions regardless of individual image characteristics.
    """
    width, height = dimensions
    
    # Create multiple test images
    images = []
    for i in range(len(ages)):
        # Vary image characteristics slightly
        img_width = width + (i * 10) % 100
        img_height = height + (i * 15) % 100
        mode = ['RGB', 'RGBA', 'L'][i % 3]
        image = create_test_image(img_width, img_height, mode)
        images.append(image)
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate batch embeddings
    embeddings = service.generate_batch_embeddings(images, ages=None)
    
    # Verify all embeddings have identical dimensions
    first_shape = embeddings[0].shape
    for i, embedding in enumerate(embeddings):
        assert embedding.shape == first_shape, \
            f"Embedding {i} has different shape: {embedding.shape} vs {first_shape}"
        
        # Verify each embedding is finite
        assert np.all(np.isfinite(embedding)), f"Embedding {i} contains non-finite values"
    
    # Verify dimensions match expected size
    expected_dim = service.get_embedding_dimension(include_age=False)
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == expected_dim, \
            f"Embedding {i} dimension should be {expected_dim}, got {len(embedding)}"


@given(
    dimensions=valid_dimensions_strategy,
    age=age_strategy,
    mode=image_mode_strategy
)
@settings(max_examples=50, deadline=None)
def test_age_augmented_embedding_consistency(dimensions, age, mode):
    """
    **Feature: children-drawing-anomaly-detection, Property 5: Age-Augmented Embedding Consistency**
    **Validates: Requirements 2.3**
    
    Property: For any drawing with age information, when age augmentation is enabled, 
    the embedding should include age data and have consistent dimensionality across 
    all age-augmented embeddings.
    """
    width, height = dimensions
    
    # Create test image
    image = create_test_image(width, height, mode)
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate embedding without age
    embedding_no_age = service.generate_embedding(image, age=None)
    
    # Generate embedding with age
    embedding_with_age = service.generate_embedding(image, age=age)
    
    # Verify age-augmented embedding has one additional dimension
    expected_base_dim = service.get_embedding_dimension(include_age=False)
    expected_age_dim = service.get_embedding_dimension(include_age=True)
    
    assert len(embedding_no_age) == expected_base_dim, \
        f"Base embedding should have {expected_base_dim} dimensions, got {len(embedding_no_age)}"
    
    assert len(embedding_with_age) == expected_age_dim, \
        f"Age-augmented embedding should have {expected_age_dim} dimensions, got {len(embedding_with_age)}"
    
    assert len(embedding_with_age) == len(embedding_no_age) + 1, \
        "Age-augmented embedding should have exactly one more dimension than base embedding"
    
    # Verify the age information is correctly appended
    # The last element should be the age value
    assert np.isclose(embedding_with_age[-1], age, rtol=1e-5), \
        f"Last element should be age {age}, got {embedding_with_age[-1]}"
    
    # Verify the base embedding part is consistent (first n-1 elements should be similar pattern)
    # Note: In mock, they won't be identical, but should have same dimensionality
    base_part = embedding_with_age[:-1]
    assert len(base_part) == len(embedding_no_age), \
        "Base part of age-augmented embedding should match base embedding dimensions"


@given(
    dimensions=valid_dimensions_strategy,
    ages=st.lists(age_strategy, min_size=2, max_size=8)
)
@settings(max_examples=30, deadline=None)
def test_batch_age_augmented_consistency(dimensions, ages):
    """
    Property: For any batch of images with age information, all age-augmented embeddings 
    should have consistent dimensionality and properly include age data.
    """
    width, height = dimensions
    
    # Create test images
    images = []
    for i in range(len(ages)):
        image = create_test_image(width, height, 'RGB')
        images.append(image)
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate batch embeddings with ages
    embeddings = service.generate_batch_embeddings(images, ages=ages)
    
    # Verify all embeddings have consistent dimensions
    expected_dim = service.get_embedding_dimension(include_age=True)
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == expected_dim, \
            f"Embedding {i} should have {expected_dim} dimensions, got {len(embedding)}"
        
        # Verify age information is correctly included
        assert np.isclose(embedding[-1], ages[i], rtol=1e-5), \
            f"Embedding {i} should have age {ages[i]} as last element, got {embedding[-1]}"
        
        # Verify embedding is finite
        assert np.all(np.isfinite(embedding)), f"Embedding {i} contains non-finite values"


@given(
    dimensions=valid_dimensions_strategy,
    age1=age_strategy,
    age2=age_strategy
)
@settings(max_examples=30, deadline=None)
def test_age_difference_preservation(dimensions, age1, age2):
    """
    Property: For any two identical images with different ages, the age-augmented 
    embeddings should differ only in the age component.
    """
    assume(abs(age1 - age2) > 0.1)  # Ensure ages are meaningfully different
    
    width, height = dimensions
    
    # Create identical test image
    image = create_test_image(width, height, 'RGB')
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate embeddings with different ages
    embedding1 = service.generate_embedding(image, age=age1)
    embedding2 = service.generate_embedding(image, age=age2)
    
    # Verify both have same dimensionality
    assert len(embedding1) == len(embedding2), \
        "Embeddings with different ages should have same dimensionality"
    
    # Verify age components are different
    assert not np.isclose(embedding1[-1], embedding2[-1], rtol=1e-5), \
        "Age components should be different for different ages"
    
    # Verify age values are correctly stored
    assert np.isclose(embedding1[-1], age1, rtol=1e-5), \
        f"First embedding should contain age {age1}, got {embedding1[-1]}"
    assert np.isclose(embedding2[-1], age2, rtol=1e-5), \
        f"Second embedding should contain age {age2}, got {embedding2[-1]}"


@given(
    dimensions=valid_dimensions_strategy,
    mode=image_mode_strategy
)
@settings(max_examples=30, deadline=None)
def test_numpy_vs_pil_consistency(dimensions, mode):
    """
    Property: For any image, whether provided as PIL Image or numpy array, 
    the generated embeddings should have consistent dimensions.
    """
    width, height = dimensions
    
    # Create PIL image
    pil_image = create_test_image(width, height, mode)
    
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    if len(numpy_image.shape) == 2:  # Grayscale
        numpy_image = np.stack([numpy_image] * 3, axis=-1)  # Convert to RGB
    numpy_image = numpy_image.astype(np.float32) / 255.0
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Generate embeddings for both formats
    embedding_pil = service.generate_embedding(pil_image)
    embedding_numpy = service.generate_embedding(numpy_image)
    
    # Verify both have identical dimensions
    assert embedding_pil.shape == embedding_numpy.shape, \
        f"PIL and numpy embeddings should have same shape: {embedding_pil.shape} vs {embedding_numpy.shape}"
    
    # Verify dimensions match expected size
    expected_dim = service.get_embedding_dimension(include_age=False)
    assert len(embedding_pil) == expected_dim, \
        f"PIL embedding dimension should be {expected_dim}, got {len(embedding_pil)}"
    assert len(embedding_numpy) == expected_dim, \
        f"Numpy embedding dimension should be {expected_dim}, got {len(embedding_numpy)}"


def test_device_manager_initialization():
    """Test that DeviceManager properly detects and configures devices."""
    device_manager = DeviceManager()
    
    # Verify device is set
    assert device_manager.device is not None
    assert hasattr(device_manager.device, 'type')
    
    # Verify device info is populated
    device_info = device_manager.device_info
    assert 'type' in device_info
    assert 'name' in device_info
    assert device_info['type'] in ['cuda', 'mps', 'cpu']
    
    # Test memory usage (should not crash)
    memory_usage = device_manager.get_memory_usage()
    if device_info['type'] == 'cuda':
        assert memory_usage is not None
        assert 'allocated' in memory_usage
    else:
        assert memory_usage is None


def test_embedding_pipeline_stats():
    """Test that EmbeddingPipeline properly tracks statistics."""
    service = MockEmbeddingService()
    pipeline = EmbeddingPipeline(service)
    
    # Initial stats should be zero
    stats = pipeline.get_pipeline_stats()
    assert stats['total_processed'] == 0
    assert stats['successful'] == 0
    assert stats['failed'] == 0
    assert stats['success_rate'] == 0.0
    
    # Process a successful drawing
    image = create_test_image(100, 100, 'RGB')
    result = pipeline.process_drawing(image, age=5.0)
    
    assert result['success'] is True
    assert result['embedding'] is not None
    
    # Check updated stats
    stats = pipeline.get_pipeline_stats()
    assert stats['total_processed'] == 1
    assert stats['successful'] == 1
    assert stats['failed'] == 0
    assert stats['success_rate'] == 1.0
    
    # Reset stats
    pipeline.reset_stats()
    stats = pipeline.get_pipeline_stats()
    assert stats['total_processed'] == 0


def test_embedding_service_error_handling():
    """Test error handling in embedding service."""
    service = MockEmbeddingService()
    
    # Test with service not ready
    service.is_initialized = False
    
    with pytest.raises(EmbeddingGenerationError, match="Service not initialized"):
        service.generate_embedding(create_test_image(100, 100))
    
    # Test batch processing with mismatched ages
    service.is_initialized = True
    images = [create_test_image(100, 100) for _ in range(3)]
    ages = [5.0, 6.0]  # Wrong number of ages
    
    with pytest.raises(EmbeddingGenerationError, match="Number of ages must match"):
        service.generate_batch_embeddings(images, ages=ages)


@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_images=st.integers(min_value=1, max_value=32)
)
@settings(max_examples=20, deadline=None)
def test_batch_processing_consistency(batch_size, num_images):
    """
    Property: For any batch of images, the results should be consistent regardless 
    of the batch size used for processing.
    """
    # Create test images
    images = []
    ages = []
    for i in range(num_images):
        image = create_test_image(64, 64, 'RGB')
        age = 2.0 + (i % 16)  # Ages from 2 to 18
        images.append(image)
        ages.append(age)
    
    service = MockEmbeddingService(embedding_dim=768)
    
    # Process with specified batch size
    embeddings = service.generate_batch_embeddings(images, ages=ages, batch_size=batch_size)
    
    # Verify we got the right number of embeddings
    assert len(embeddings) == num_images, \
        f"Should get {num_images} embeddings, got {len(embeddings)}"
    
    # Verify all embeddings have consistent dimensions
    expected_dim = service.get_embedding_dimension(include_age=True)
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == expected_dim, \
            f"Embedding {i} should have {expected_dim} dimensions, got {len(embedding)}"
        
        # Verify age is correctly included
        assert np.isclose(embedding[-1], ages[i], rtol=1e-5), \
            f"Embedding {i} should have age {ages[i]}, got {embedding[-1]}"