"""
Embedding Service for generating feature vectors from children's drawings.

This service handles Vision Transformer model loading, caching, and embedding generation
with support for age-augmented embeddings and batch processing.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib

import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel
import numpy as np
from PIL import Image

from app.core.config import settings
from app.utils.embedding_serialization import get_embedding_storage, EmbeddingStorage

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass


class ModelLoadingError(EmbeddingServiceError):
    """Raised when model loading fails."""
    pass


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embedding generation fails."""
    pass


class DeviceManager:
    """Manages GPU/CPU device detection and selection."""
    
    def __init__(self):
        self._device = None
        self._device_info = None
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Detect and configure the best available device."""
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._device_info = {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0),
                "memory": torch.cuda.get_device_properties(0).total_memory,
                "count": torch.cuda.device_count()
            }
            logger.info(f"Using CUDA device: {self._device_info['name']}")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._device_info = {
                "type": "mps",
                "name": "Apple Metal Performance Shaders",
                "memory": None,
                "count": 1
            }
            logger.info("Using MPS (Apple Silicon) device")
        else:
            self._device = torch.device("cpu")
            self._device_info = {
                "type": "cpu",
                "name": "CPU",
                "memory": None,
                "count": 1
            }
            logger.info("Using CPU device")
    
    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device
    
    @property
    def device_info(self) -> Dict:
        """Get device information."""
        return self._device_info.copy()
    
    def get_memory_usage(self) -> Optional[Dict]:
        """Get current memory usage if available."""
        if self._device.type == "cuda":
            return {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated()
            }
        return None


class VisionTransformerWrapper:
    """Wrapper for Vision Transformer model with caching and optimization."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device_manager: DeviceManager = None):
        self.model_name = model_name
        self.device_manager = device_manager or DeviceManager()
        self.model = None
        self.processor = None
        self._model_hash = None
        self._cache_dir = Path("static/models")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_model_hash(self) -> str:
        """Generate a hash for the model configuration."""
        config_str = f"{self.model_name}_{self.device_manager.device.type}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_cache_path(self) -> Path:
        """Get the cache path for the model."""
        return self._cache_dir / f"vit_model_{self._get_model_hash()}.pkl"
    
    def load_model(self, use_cache: bool = True) -> None:
        """Load the Vision Transformer model with optional caching."""
        try:
            self._model_hash = self._get_model_hash()
            cache_path = self._get_cache_path()
            
            # Try to load from cache first
            if use_cache and cache_path.exists():
                try:
                    logger.info(f"Loading cached model from {cache_path}")
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        self.model = cached_data['model']
                        self.processor = cached_data['processor']
                        self.model.to(self.device_manager.device)
                        logger.info("Successfully loaded cached model")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}, loading fresh model")
            
            # Load fresh model from Hugging Face
            logger.info(f"Loading Vision Transformer model: {self.model_name}")
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.model = ViTModel.from_pretrained(self.model_name)
            
            # Move to device and set to evaluation mode
            self.model.to(self.device_manager.device)
            self.model.eval()
            
            # Cache the model if requested
            if use_cache:
                try:
                    logger.info(f"Caching model to {cache_path}")
                    # Move to CPU for caching to save space
                    cpu_model = self.model.cpu()
                    with open(cache_path, 'wb') as f:
                        pickle.dump({
                            'model': cpu_model,
                            'processor': self.processor
                        }, f)
                    # Move back to device
                    self.model.to(self.device_manager.device)
                    logger.info("Model cached successfully")
                except Exception as e:
                    logger.warning(f"Failed to cache model: {e}")
            
            logger.info(f"Model loaded successfully on {self.device_manager.device}")
            
        except Exception as e:
            raise ModelLoadingError(f"Failed to load Vision Transformer model: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None and self.processor is not None
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "device": str(self.device_manager.device),
            "embedding_dim": self.model.config.hidden_size,
            "patch_size": self.model.config.patch_size,
            "image_size": self.model.config.image_size,
            "num_attention_heads": self.model.config.num_attention_heads,
            "num_hidden_layers": self.model.config.num_hidden_layers
        }


class EmbeddingService:
    """Service for generating embeddings from children's drawings using Vision Transformers."""
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.device_manager = DeviceManager()
        self.vit_wrapper = VisionTransformerWrapper(model_name, self.device_manager)
        self._embedding_cache = {}
        self._cache_size_limit = 1000  # Maximum number of cached embeddings
        self.embedding_storage = get_embedding_storage()
        
    def initialize(self, use_cache: bool = True) -> None:
        """Initialize the embedding service by loading the model."""
        logger.info("Initializing Embedding Service")
        self.vit_wrapper.load_model(use_cache=use_cache)
        logger.info("Embedding Service initialized successfully")
    
    def is_ready(self) -> bool:
        """Check if the service is ready to generate embeddings."""
        return self.vit_wrapper.is_loaded()
    
    def get_service_info(self) -> Dict:
        """Get comprehensive service information."""
        return {
            "ready": self.is_ready(),
            "device_info": self.device_manager.device_info,
            "model_info": self.vit_wrapper.get_model_info(),
            "memory_usage": self.device_manager.get_memory_usage(),
            "cache_size": len(self._embedding_cache)
        }
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for Vision Transformer input."""
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to expected size (224x224 for ViT)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Process with ViT processor
            inputs = self.vit_wrapper.processor(
                images=image, 
                return_tensors="pt"
            )
            return inputs['pixel_values'].to(self.device_manager.device)
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to preprocess image: {str(e)}")
    
    def _generate_cache_key(self, image_tensor: torch.Tensor, age: Optional[float] = None) -> str:
        """Generate a cache key for the embedding."""
        # Use tensor hash and age for caching
        tensor_hash = hashlib.md5(image_tensor.cpu().numpy().tobytes()).hexdigest()[:16]
        age_str = f"_{age}" if age is not None else ""
        return f"{tensor_hash}{age_str}"
    
    def _manage_cache(self) -> None:
        """Manage embedding cache size."""
        if len(self._embedding_cache) > self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[:-self._cache_size_limit//2]
            for key in keys_to_remove:
                del self._embedding_cache[key]
    
    async def generate_embedding_from_file(self, 
                                         file_path: str, 
                                         age: Optional[float] = None,
                                         use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for an image file.
        
        Args:
            file_path: Path to the image file
            age: Optional age information to concatenate
            use_cache: Whether to use caching for embeddings
            
        Returns:
            numpy array containing the embedding vector
        """
        try:
            # Load image from file
            image = Image.open(file_path).convert('RGB')
            # Generate embedding using the existing method
            return self.generate_embedding(image, age, use_cache)
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embedding from file {file_path}: {str(e)}")

    def generate_embedding(self, 
                         image: Union[Image.Image, np.ndarray], 
                         age: Optional[float] = None,
                         use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image or numpy array
            age: Optional age information to concatenate
            use_cache: Whether to use caching for embeddings
            
        Returns:
            numpy array containing the embedding vector
        """
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Check cache
            cache_key = self._generate_cache_key(image_tensor, age) if use_cache else None
            if cache_key and cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key].copy()
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.vit_wrapper.model(pixel_values=image_tensor)
                # Use the [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Concatenate age information if provided
                if age is not None:
                    age_feature = np.array([[age]], dtype=np.float32)
                    embedding = np.concatenate([embedding, age_feature], axis=1)
                
                # Squeeze to remove batch dimension
                embedding = embedding.squeeze(0)
            
            # Cache the result
            if cache_key:
                self._embedding_cache[cache_key] = embedding.copy()
                self._manage_cache()
            
            return embedding
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embedding: {str(e)}")
    
    def generate_batch_embeddings(self, 
                                images: List[Union[Image.Image, np.ndarray]], 
                                ages: Optional[List[float]] = None,
                                batch_size: int = 8,
                                use_cache: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            images: List of PIL Images or numpy arrays
            ages: Optional list of ages corresponding to images
            batch_size: Number of images to process in each batch
            use_cache: Whether to use caching for embeddings
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")
        
        embeddings = []
        
        try:
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                batch_ages = ages[i:i + batch_size] if ages else [None] * len(batch_images)
                
                # Process each image in the batch
                batch_embeddings = []
                for image, age in zip(batch_images, batch_ages):
                    embedding = self.generate_embedding(image, age, use_cache)
                    batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
                
                # Log progress for large batches
                if len(images) > 10:
                    logger.info(f"Processed {min(i + batch_size, len(images))}/{len(images)} images")
            
            return embeddings
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate batch embeddings: {str(e)}")
    
    def get_embedding_dimension(self, include_age: bool = False) -> int:
        """Get the dimension of embeddings generated by this service."""
        if not self.is_ready():
            raise EmbeddingGenerationError("Service not initialized. Call initialize() first.")
        
        base_dim = self.vit_wrapper.model.config.hidden_size
        return base_dim + (1 if include_age else 0)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.embedding_storage.clear_cache()
        logger.info("Embedding cache cleared")
    
    def store_embedding_for_db(self, 
                              drawing_id: int, 
                              embedding: np.ndarray,
                              age: Optional[float] = None,
                              model_type: str = "vit") -> Tuple[bytes, int]:
        """
        Store an embedding for database persistence with serialization.
        
        Args:
            drawing_id: Database ID of the drawing
            embedding: Embedding array to store
            age: Optional age information
            model_type: Type of model used for embedding
            
        Returns:
            Tuple of (serialized_bytes, dimension)
        """
        return self.embedding_storage.store_embedding(
            drawing_id=drawing_id,
            model_type=model_type,
            embedding=embedding,
            age=age,
            use_cache=True
        )
    
    def retrieve_embedding_from_db(self, 
                                  drawing_id: int,
                                  serialized_data: Optional[bytes] = None,
                                  age: Optional[float] = None,
                                  model_type: str = "vit") -> Optional[np.ndarray]:
        """
        Retrieve an embedding from database with caching support.
        
        Args:
            drawing_id: Database ID of the drawing
            serialized_data: Serialized embedding data from database
            age: Optional age information
            model_type: Type of model used for embedding
            
        Returns:
            Embedding array or None if not found
        """
        return self.embedding_storage.retrieve_embedding(
            drawing_id=drawing_id,
            model_type=model_type,
            serialized_data=serialized_data,
            age=age,
            use_cache=True
        )
    
    def invalidate_embedding_cache(self, 
                                  drawing_id: int,
                                  age: Optional[float] = None,
                                  model_type: str = "vit") -> bool:
        """
        Invalidate cached embedding for a specific drawing.
        
        Args:
            drawing_id: Database ID of the drawing
            age: Optional age information
            model_type: Type of model used for embedding
            
        Returns:
            True if cache entry was removed
        """
        return self.embedding_storage.invalidate_cache(drawing_id, model_type, age)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get embedding storage and cache statistics."""
        service_stats = self.get_service_info()
        storage_stats = self.embedding_storage.get_storage_stats()
        
        return {
            **service_stats,
            'storage': storage_stats
        }


# Global service instance
_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def initialize_embedding_service(use_cache: bool = True) -> None:
    """Initialize the global embedding service."""
    service = get_embedding_service()
    service.initialize(use_cache=use_cache)


class EmbeddingPipeline:
    """High-level pipeline for processing drawings and generating embeddings."""
    
    def __init__(self, embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or get_embedding_service()
        self._pipeline_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0
        }
    
    def process_drawing(self, 
                       image: Union[Image.Image, np.ndarray], 
                       age: Optional[float] = None,
                       drawing_id: Optional[int] = None,
                       use_cache: bool = True) -> Dict:
        """
        Process a single drawing through the complete embedding pipeline.
        
        Args:
            image: PIL Image or numpy array
            age: Optional age information
            drawing_id: Optional database ID for tracking
            use_cache: Whether to use embedding cache
            
        Returns:
            Dictionary containing embedding and metadata
        """
        try:
            self._pipeline_stats["total_processed"] += 1
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(
                image=image, 
                age=age, 
                use_cache=use_cache
            )
            
            # Prepare result
            result = {
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "age_augmented": age is not None,
                "drawing_id": drawing_id,
                "model_info": self.embedding_service.vit_wrapper.get_model_info(),
                "success": True,
                "error": None
            }
            
            self._pipeline_stats["successful"] += 1
            return result
            
        except Exception as e:
            self._pipeline_stats["failed"] += 1
            logger.error(f"Pipeline processing failed for drawing {drawing_id}: {str(e)}")
            
            return {
                "embedding": None,
                "embedding_dimension": 0,
                "age_augmented": False,
                "drawing_id": drawing_id,
                "model_info": None,
                "success": False,
                "error": str(e)
            }
    
    def process_batch(self, 
                     images: List[Union[Image.Image, np.ndarray]], 
                     ages: Optional[List[float]] = None,
                     drawing_ids: Optional[List[int]] = None,
                     batch_size: int = 8,
                     use_cache: bool = True) -> List[Dict]:
        """
        Process multiple drawings through the embedding pipeline.
        
        Args:
            images: List of PIL Images or numpy arrays
            ages: Optional list of ages
            drawing_ids: Optional list of database IDs
            batch_size: Batch size for processing
            use_cache: Whether to use embedding cache
            
        Returns:
            List of result dictionaries
        """
        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")
        
        if drawing_ids is not None and len(drawing_ids) != len(images):
            raise EmbeddingGenerationError("Number of drawing IDs must match number of images")
        
        results = []
        
        try:
            # Generate embeddings in batches
            embeddings = self.embedding_service.generate_batch_embeddings(
                images=images,
                ages=ages,
                batch_size=batch_size,
                use_cache=use_cache
            )
            
            # Create result objects
            for i, embedding in enumerate(embeddings):
                age = ages[i] if ages else None
                drawing_id = drawing_ids[i] if drawing_ids else None
                
                result = {
                    "embedding": embedding,
                    "embedding_dimension": len(embedding),
                    "age_augmented": age is not None,
                    "drawing_id": drawing_id,
                    "model_info": self.embedding_service.vit_wrapper.get_model_info(),
                    "success": True,
                    "error": None
                }
                results.append(result)
                self._pipeline_stats["successful"] += 1
            
            self._pipeline_stats["total_processed"] += len(images)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            
            # Create error results for all images
            for i in range(len(images)):
                age = ages[i] if ages else None
                drawing_id = drawing_ids[i] if drawing_ids else None
                
                result = {
                    "embedding": None,
                    "embedding_dimension": 0,
                    "age_augmented": False,
                    "drawing_id": drawing_id,
                    "model_info": None,
                    "success": False,
                    "error": str(e)
                }
                results.append(result)
                self._pipeline_stats["failed"] += 1
            
            self._pipeline_stats["total_processed"] += len(images)
        
        return results
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline processing statistics."""
        stats = self._pipeline_stats.copy()
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful"] / stats["total_processed"]
            stats["failure_rate"] = stats["failed"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self._pipeline_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cache_hits": 0
        }


# Global pipeline instance
_embedding_pipeline = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get the global embedding pipeline instance."""
    global _embedding_pipeline
    if _embedding_pipeline is None:
        _embedding_pipeline = EmbeddingPipeline()
    return _embedding_pipeline