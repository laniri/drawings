"""
Embedding Service for generating feature vectors from children's drawings.

This service handles Vision Transformer model loading, caching, and embedding generation
with support for age-augmented embeddings and batch processing.
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from app.core.config import settings
from app.schemas.drawings import SubjectCategory
from app.utils.embedding_serialization import EmbeddingStorage, get_embedding_storage

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


class SubjectEncodingError(EmbeddingServiceError):
    """Raised when subject encoding fails."""

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
                "count": torch.cuda.device_count(),
            }
            logger.info(f"Using CUDA device: {self._device_info['name']}")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self._device_info = {
                "type": "mps",
                "name": "Apple Metal Performance Shaders",
                "memory": None,
                "count": 1,
            }
            logger.info("Using MPS (Apple Silicon) device")
        else:
            self._device = torch.device("cpu")
            self._device_info = {
                "type": "cpu",
                "name": "CPU",
                "memory": None,
                "count": 1,
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
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        return None


class SubjectEncoder:
    """Handles subject category encoding for hybrid embeddings."""

    # Subject category to index mapping (64-dimensional one-hot encoding)
    SUBJECT_TO_INDEX = {
        # Default category (position 0)
        SubjectCategory.UNSPECIFIED: 0,
        # Objects (positions 1-27)
        SubjectCategory.TV: 1,
        SubjectCategory.AIRPLANE: 2,
        SubjectCategory.APPLE: 3,
        SubjectCategory.BED: 4,
        SubjectCategory.BIKE: 5,
        SubjectCategory.BOAT: 6,
        SubjectCategory.BOOK: 7,
        SubjectCategory.BOTTLE: 8,
        SubjectCategory.BOWL: 9,
        SubjectCategory.CACTUS: 10,
        SubjectCategory.CAR: 11,
        SubjectCategory.CHAIR: 12,
        SubjectCategory.CLOCK: 13,
        SubjectCategory.COUCH: 14,
        SubjectCategory.CUP: 15,
        SubjectCategory.HAT: 16,
        SubjectCategory.HOUSE: 17,
        SubjectCategory.ICE_CREAM: 18,
        SubjectCategory.KEY: 19,
        SubjectCategory.LAMP: 20,
        SubjectCategory.MUSHROOM: 21,
        SubjectCategory.PHONE: 22,
        SubjectCategory.PIANO: 23,
        SubjectCategory.SCISSORS: 24,
        SubjectCategory.TRAIN: 25,
        SubjectCategory.TREE: 26,
        SubjectCategory.WATCH: 27,
        # Animals (positions 28-45)
        SubjectCategory.BEAR: 28,
        SubjectCategory.BEE: 29,
        SubjectCategory.BIRD: 30,
        SubjectCategory.CAMEL: 31,
        SubjectCategory.CAT: 32,
        SubjectCategory.COW: 33,
        SubjectCategory.DOG: 34,
        SubjectCategory.ELEPHANT: 35,
        SubjectCategory.FISH: 36,
        SubjectCategory.FROG: 37,
        SubjectCategory.HORSE: 38,
        SubjectCategory.OCTOPUS: 39,
        SubjectCategory.RABBIT: 40,
        SubjectCategory.SHEEP: 41,
        SubjectCategory.SNAIL: 42,
        SubjectCategory.SPIDER: 43,
        SubjectCategory.TIGER: 44,
        SubjectCategory.WHALE: 45,
        # People and body parts (positions 46-48)
        SubjectCategory.FACE: 46,
        SubjectCategory.HAND: 47,
        SubjectCategory.PERSON: 48,
        # Abstract/other categories (positions 49-51)
        SubjectCategory.FAMILY: 49,
        SubjectCategory.ABSTRACT: 50,
        SubjectCategory.OTHER: 51,
        # Reserved positions 52-63 for future expansion
    }

    # Reverse mapping for decoding
    INDEX_TO_SUBJECT = {v: k for k, v in SUBJECT_TO_INDEX.items()}

    # Encoding dimension (supports up to 64 categories)
    ENCODING_DIMENSION = 64

    @classmethod
    def encode_subject_category(
        cls, subject: Optional[Union[str, SubjectCategory]]
    ) -> np.ndarray:
        """
        Encode a subject category into a 64-dimensional one-hot vector.

        Args:
            subject: Subject category string, SubjectCategory enum, or None

        Returns:
            64-dimensional one-hot encoded numpy array

        Raises:
            SubjectEncodingError: If subject category is not supported
        """
        try:
            # Handle None or empty string - use default "unspecified"
            if subject is None or (isinstance(subject, str) and subject.strip() == ""):
                subject = SubjectCategory.UNSPECIFIED

            # Convert string to SubjectCategory enum if needed
            if isinstance(subject, str):
                try:
                    # Strip whitespace and try to find matching enum by value
                    subject_stripped = subject.strip()
                    for category in SubjectCategory:
                        if category.value.lower() == subject_stripped.lower():
                            subject = category
                            break
                    else:
                        # If no match found, use unspecified
                        logger.warning(
                            f"Unknown subject category '{subject}', using 'unspecified'"
                        )
                        subject = SubjectCategory.UNSPECIFIED
                except Exception:
                    # If any error occurs, use unspecified
                    logger.warning(
                        f"Error processing subject category '{subject}', using 'unspecified'"
                    )
                    subject = SubjectCategory.UNSPECIFIED

            # At this point, subject should be a SubjectCategory enum
            if not isinstance(subject, SubjectCategory):
                logger.warning(
                    f"Invalid subject type '{type(subject)}', using 'unspecified'"
                )
                subject = SubjectCategory.UNSPECIFIED

            # Get the index for this subject
            if subject not in cls.SUBJECT_TO_INDEX:
                raise SubjectEncodingError(
                    f"Subject category '{subject}' not found in mapping"
                )

            subject_index = cls.SUBJECT_TO_INDEX[subject]

            # Create one-hot encoding
            encoding = np.zeros(cls.ENCODING_DIMENSION, dtype=np.float32)
            encoding[subject_index] = 1.0

            return encoding

        except Exception as e:
            if isinstance(e, SubjectEncodingError):
                raise
            raise SubjectEncodingError(
                f"Failed to encode subject category '{subject}': {str(e)}"
            )

    @classmethod
    def decode_subject_encoding(cls, encoding: np.ndarray) -> SubjectCategory:
        """
        Decode a one-hot encoded vector back to a subject category.

        Args:
            encoding: 64-dimensional one-hot encoded numpy array

        Returns:
            SubjectCategory enum value

        Raises:
            SubjectEncodingError: If encoding is invalid
        """
        try:
            if encoding.shape != (cls.ENCODING_DIMENSION,):
                raise SubjectEncodingError(
                    f"Invalid encoding shape {encoding.shape}, expected ({cls.ENCODING_DIMENSION},)"
                )

            # Find the index with value 1.0
            active_indices = np.where(encoding == 1.0)[0]

            if len(active_indices) != 1:
                raise SubjectEncodingError(
                    f"Invalid one-hot encoding: found {len(active_indices)} active positions, expected 1"
                )

            subject_index = active_indices[0]

            if subject_index not in cls.INDEX_TO_SUBJECT:
                raise SubjectEncodingError(
                    f"Subject index {subject_index} not found in mapping"
                )

            return cls.INDEX_TO_SUBJECT[subject_index]

        except Exception as e:
            if isinstance(e, SubjectEncodingError):
                raise
            raise SubjectEncodingError(f"Failed to decode subject encoding: {str(e)}")

    @classmethod
    def get_supported_categories(cls) -> List[SubjectCategory]:
        """Get list of all supported subject categories."""
        return list(cls.SUBJECT_TO_INDEX.keys())

    @classmethod
    def get_category_count(cls) -> int:
        """Get the number of supported subject categories."""
        return len(cls.SUBJECT_TO_INDEX)

    @classmethod
    def validate_subject_category(cls, subject: Union[str, SubjectCategory]) -> bool:
        """
        Validate if a subject category is supported.

        Args:
            subject: Subject category to validate

        Returns:
            True if supported, False otherwise
        """
        try:
            if isinstance(subject, SubjectCategory):
                return subject in cls.SUBJECT_TO_INDEX
            elif isinstance(subject, str):
                # Try to find matching enum by value
                for category in SubjectCategory:
                    if category.value.lower() == subject.lower():
                        return category in cls.SUBJECT_TO_INDEX
                return False
            else:
                return False
        except (ValueError, TypeError):
            return False


class VisionTransformerWrapper:
    """Wrapper for Vision Transformer model with caching and optimization."""

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        device_manager: DeviceManager = None,
    ):
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
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                        self.model = cached_data["model"]
                        self.processor = cached_data["processor"]
                        self.model.to(self.device_manager.device)
                        logger.info("Successfully loaded cached model")
                        return
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached model: {e}, loading fresh model"
                    )

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
                    with open(cache_path, "wb") as f:
                        pickle.dump(
                            {"model": cpu_model, "processor": self.processor}, f
                        )
                    # Move back to device
                    self.model.to(self.device_manager.device)
                    logger.info("Model cached successfully")
                except Exception as e:
                    logger.warning(f"Failed to cache model: {e}")

            logger.info(f"Model loaded successfully on {self.device_manager.device}")

        except Exception as e:
            raise ModelLoadingError(
                f"Failed to load Vision Transformer model: {str(e)}"
            )

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
            "num_hidden_layers": self.model.config.num_hidden_layers,
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
        info = {
            "ready": self.is_ready(),
            "device_info": self.device_manager.device_info,
            "model_info": self.vit_wrapper.get_model_info(),
            "memory_usage": self.device_manager.get_memory_usage(),
            "cache_size": len(self._embedding_cache),
        }

        # Add hybrid embedding information
        if self.is_ready():
            info.update(
                {
                    "hybrid_embedding_dimension": self.get_embedding_dimension(
                        hybrid=True
                    ),
                    "visual_embedding_dimension": self.get_visual_embedding_dimension(),
                    "subject_encoding_dimension": self.get_subject_encoding_dimension(),
                    "supported_subject_categories": len(
                        SubjectEncoder.get_supported_categories()
                    ),
                    "subject_categories": [
                        cat.value for cat in SubjectEncoder.get_supported_categories()
                    ],
                }
            )

        return info

    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for Vision Transformer input."""
        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)

            # Ensure RGB format
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image to expected size (224x224 for ViT)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)

            # Process with ViT processor
            inputs = self.vit_wrapper.processor(images=image, return_tensors="pt")
            return inputs["pixel_values"].to(self.device_manager.device)

        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to preprocess image: {str(e)}")

    def _generate_cache_key(
        self,
        image_tensor: torch.Tensor,
        age: Optional[float] = None,
        subject: Optional[Union[str, SubjectCategory]] = None,
    ) -> str:
        """Generate a cache key for the embedding."""
        # Use tensor hash, age, and subject for caching
        tensor_hash = hashlib.md5(image_tensor.cpu().numpy().tobytes()).hexdigest()[:16]
        age_str = f"_{age}" if age is not None else ""
        subject_str = f"_{subject}" if subject is not None else ""
        return f"{tensor_hash}{age_str}{subject_str}"

    def _manage_cache(self) -> None:
        """Manage embedding cache size."""
        if len(self._embedding_cache) > self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._embedding_cache.keys())[
                : -self._cache_size_limit // 2
            ]
            for key in keys_to_remove:
                del self._embedding_cache[key]

    async def generate_embedding_from_file(
        self, file_path: str, age: Optional[float] = None, use_cache: bool = True
    ) -> np.ndarray:
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
            image = Image.open(file_path).convert("RGB")
            # Generate embedding using the existing method
            return self.generate_embedding(image, age, use_cache)
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate embedding from file {file_path}: {str(e)}"
            )

    def generate_hybrid_embedding(
        self,
        image: Union[Image.Image, np.ndarray],
        subject: Optional[Union[str, SubjectCategory]] = None,
        age: Optional[float] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate hybrid embedding combining visual features and subject encoding.

        Args:
            image: PIL Image or numpy array
            subject: Subject category (string, enum, or None for "unspecified")
            age: Optional age information (used for model selection, not embedding)
            use_cache: Whether to use caching for embeddings

        Returns:
            832-dimensional hybrid embedding (768 visual + 64 subject)
        """
        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        try:
            # Generate visual embedding (768 dimensions)
            visual_embedding = self._generate_visual_embedding(image, use_cache)

            # Generate subject encoding (64 dimensions)
            subject_encoding = SubjectEncoder.encode_subject_category(subject)

            # Combine into hybrid embedding (832 dimensions total)
            hybrid_embedding = np.concatenate(
                [visual_embedding, subject_encoding], axis=0
            )

            return hybrid_embedding

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate hybrid embedding: {str(e)}"
            )

    def _generate_visual_embedding(
        self, image: Union[Image.Image, np.ndarray], use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate visual embedding using Vision Transformer.

        Args:
            image: PIL Image or numpy array
            use_cache: Whether to use caching for embeddings

        Returns:
            768-dimensional visual embedding
        """
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)

            # Check cache (using image tensor hash)
            cache_key = self._generate_cache_key(image_tensor) if use_cache else None
            if cache_key and cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key].copy()

            # Generate visual embedding
            with torch.no_grad():
                outputs = self.vit_wrapper.model(pixel_values=image_tensor)
                # Use the [CLS] token embedding (first token)
                visual_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Squeeze to remove batch dimension
                visual_embedding = visual_embedding.squeeze(0)

            # Cache the result
            if cache_key:
                self._embedding_cache[cache_key] = visual_embedding.copy()
                self._manage_cache()

            return visual_embedding

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate visual embedding: {str(e)}"
            )

    def separate_embedding_components(
        self, hybrid_embedding: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate hybrid embedding into visual and subject components.

        Args:
            hybrid_embedding: 832-dimensional hybrid embedding

        Returns:
            Tuple of (visual_component, subject_component)

        Raises:
            EmbeddingGenerationError: If embedding has wrong dimensions
        """
        try:
            if hybrid_embedding.shape != (832,):
                raise EmbeddingGenerationError(
                    f"Invalid hybrid embedding shape {hybrid_embedding.shape}, expected (832,)"
                )

            # Split into components
            visual_component = hybrid_embedding[:768]  # First 768 dimensions
            subject_component = hybrid_embedding[768:]  # Last 64 dimensions

            return visual_component, subject_component

        except Exception as e:
            if isinstance(e, EmbeddingGenerationError):
                raise
            raise EmbeddingGenerationError(
                f"Failed to separate embedding components: {str(e)}"
            )

    def generate_embedding(
        self,
        image: Union[Image.Image, np.ndarray],
        age: Optional[float] = None,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Generate embedding for a single image (legacy method).

        DEPRECATED: Use generate_hybrid_embedding() for new implementations.
        This method is maintained for backward compatibility.

        Args:
            image: PIL Image or numpy array
            age: Optional age information to concatenate
            use_cache: Whether to use caching for embeddings

        Returns:
            numpy array containing the embedding vector
        """
        logger.warning(
            "generate_embedding() is deprecated. Use generate_hybrid_embedding() instead."
        )

        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)

            # Check cache
            cache_key = (
                self._generate_cache_key(image_tensor, age) if use_cache else None
            )
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

    def batch_embed(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        subjects: Optional[List[Union[str, SubjectCategory]]] = None,
        ages: Optional[List[float]] = None,
        batch_size: int = 8,
        use_cache: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate hybrid embeddings for multiple images in batches.

        Args:
            images: List of PIL Images or numpy arrays
            subjects: Optional list of subject categories corresponding to images
            ages: Optional list of ages (used for model selection, not embedding)
            batch_size: Number of images to process in each batch
            use_cache: Whether to use caching for embeddings

        Returns:
            List of 832-dimensional hybrid embedding arrays
        """
        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        if subjects is not None and len(subjects) != len(images):
            raise EmbeddingGenerationError(
                "Number of subjects must match number of images"
            )

        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")

        embeddings = []

        try:
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_subjects = (
                    subjects[i : i + batch_size]
                    if subjects
                    else [None] * len(batch_images)
                )
                batch_ages = (
                    ages[i : i + batch_size] if ages else [None] * len(batch_images)
                )

                # Process each image in the batch
                batch_embeddings = []
                for image, subject, age in zip(
                    batch_images, batch_subjects, batch_ages
                ):
                    embedding = self.generate_hybrid_embedding(
                        image, subject, age, use_cache
                    )
                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Log progress for large batches
                if len(images) > 10:
                    logger.info(
                        f"Processed {min(i + batch_size, len(images))}/{len(images)} images"
                    )

            return embeddings

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate batch hybrid embeddings: {str(e)}"
            )

    def generate_batch_embeddings(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        ages: Optional[List[float]] = None,
        batch_size: int = 8,
        use_cache: bool = True,
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images in batches (legacy method).

        DEPRECATED: Use batch_embed() for new implementations.
        This method is maintained for backward compatibility.

        Args:
            images: List of PIL Images or numpy arrays
            ages: Optional list of ages corresponding to images
            batch_size: Number of images to process in each batch
            use_cache: Whether to use caching for embeddings

        Returns:
            List of numpy arrays containing embedding vectors
        """
        logger.warning(
            "generate_batch_embeddings() is deprecated. Use batch_embed() instead."
        )

        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")

        embeddings = []

        try:
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_ages = (
                    ages[i : i + batch_size] if ages else [None] * len(batch_images)
                )

                # Process each image in the batch
                batch_embeddings = []
                for image, age in zip(batch_images, batch_ages):
                    embedding = self.generate_embedding(image, age, use_cache)
                    batch_embeddings.append(embedding)

                embeddings.extend(batch_embeddings)

                # Log progress for large batches
                if len(images) > 10:
                    logger.info(
                        f"Processed {min(i + batch_size, len(images))}/{len(images)} images"
                    )

            return embeddings

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to generate batch embeddings: {str(e)}"
            )

    def get_embedding_dimension(
        self, include_age: bool = False, hybrid: bool = True
    ) -> int:
        """
        Get the dimension of embeddings generated by this service.

        Args:
            include_age: Whether to include age dimension (legacy parameter)
            hybrid: Whether to return hybrid embedding dimension (default: True)

        Returns:
            Embedding dimension size
        """
        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        if hybrid:
            # Hybrid embeddings: 768 (visual) + 64 (subject) = 832
            return 832
        else:
            # Legacy embeddings: base dimension + optional age
            base_dim = self.vit_wrapper.model.config.hidden_size
            return base_dim + (1 if include_age else 0)

    def get_visual_embedding_dimension(self) -> int:
        """Get the dimension of visual embeddings (768 for ViT)."""
        if not self.is_ready():
            raise EmbeddingGenerationError(
                "Service not initialized. Call initialize() first."
            )

        return self.vit_wrapper.model.config.hidden_size

    def get_subject_encoding_dimension(self) -> int:
        """Get the dimension of subject encodings (64)."""
        return SubjectEncoder.ENCODING_DIMENSION

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self.embedding_storage.clear_cache()
        logger.info("Embedding cache cleared")

    def store_hybrid_embedding_for_db(
        self,
        drawing_id: int,
        hybrid_embedding: np.ndarray,
        subject: Optional[Union[str, SubjectCategory]] = None,
        age: Optional[float] = None,
        model_type: str = "vit",
    ) -> Tuple[bytes, int, bytes, bytes]:
        """
        Store a hybrid embedding for database persistence with component separation.

        Args:
            drawing_id: Database ID of the drawing
            hybrid_embedding: 832-dimensional hybrid embedding array
            subject: Subject category used for encoding
            age: Optional age information
            model_type: Type of model used for embedding

        Returns:
            Tuple of (serialized_hybrid_bytes, dimension, visual_component_bytes, subject_component_bytes)
        """
        try:
            # Validate hybrid embedding
            if hybrid_embedding.shape != (832,):
                raise EmbeddingGenerationError(
                    f"Invalid hybrid embedding shape {hybrid_embedding.shape}, expected (832,)"
                )

            # Separate components
            visual_component, subject_component = self.separate_embedding_components(
                hybrid_embedding
            )

            # Serialize all components
            hybrid_bytes = self.embedding_storage.serialize_embedding(hybrid_embedding)
            visual_bytes = self.embedding_storage.serialize_embedding(visual_component)
            subject_bytes = self.embedding_storage.serialize_embedding(
                subject_component
            )

            return hybrid_bytes, 832, visual_bytes, subject_bytes

        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to store hybrid embedding: {str(e)}"
            )

    def store_embedding_for_db(
        self,
        drawing_id: int,
        embedding: np.ndarray,
        age: Optional[float] = None,
        model_type: str = "vit",
    ) -> Tuple[bytes, int]:
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
            use_cache=True,
        )

    def retrieve_hybrid_embedding_from_db(
        self,
        drawing_id: int,
        serialized_data: Optional[bytes] = None,
        visual_component_data: Optional[bytes] = None,
        subject_component_data: Optional[bytes] = None,
        age: Optional[float] = None,
        model_type: str = "vit",
    ) -> Optional[np.ndarray]:
        """
        Retrieve a hybrid embedding from database with component reconstruction.

        Args:
            drawing_id: Database ID of the drawing
            serialized_data: Serialized hybrid embedding data from database
            visual_component_data: Serialized visual component data
            subject_component_data: Serialized subject component data
            age: Optional age information
            model_type: Type of model used for embedding

        Returns:
            832-dimensional hybrid embedding array or None if not found
        """
        try:
            # Try to retrieve full hybrid embedding first
            if serialized_data is not None:
                hybrid_embedding = self.embedding_storage.deserialize_embedding(
                    serialized_data
                )
                if hybrid_embedding is not None and hybrid_embedding.shape == (832,):
                    return hybrid_embedding

            # If full embedding not available, try to reconstruct from components
            if visual_component_data is not None and subject_component_data is not None:
                visual_component = self.embedding_storage.deserialize_embedding(
                    visual_component_data
                )
                subject_component = self.embedding_storage.deserialize_embedding(
                    subject_component_data
                )

                if (
                    visual_component is not None
                    and visual_component.shape == (768,)
                    and subject_component is not None
                    and subject_component.shape == (64,)
                ):
                    # Reconstruct hybrid embedding
                    hybrid_embedding = np.concatenate(
                        [visual_component, subject_component], axis=0
                    )
                    return hybrid_embedding

            return None

        except Exception as e:
            logger.error(
                f"Failed to retrieve hybrid embedding for drawing {drawing_id}: {str(e)}"
            )
            return None

    def retrieve_embedding_from_db(
        self,
        drawing_id: int,
        serialized_data: Optional[bytes] = None,
        age: Optional[float] = None,
        model_type: str = "vit",
    ) -> Optional[np.ndarray]:
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
            use_cache=True,
        )

    def invalidate_embedding_cache(
        self, drawing_id: int, age: Optional[float] = None, model_type: str = "vit"
    ) -> bool:
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

        return {**service_stats, "storage": storage_stats}


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
            "cache_hits": 0,
        }

    def process_drawing(
        self,
        image: Union[Image.Image, np.ndarray],
        age: Optional[float] = None,
        subject: Optional[Union[str, SubjectCategory]] = None,
        drawing_id: Optional[int] = None,
        use_cache: bool = True,
        use_hybrid: bool = True,
    ) -> Dict:
        """
        Process a single drawing through the complete embedding pipeline.

        Args:
            image: PIL Image or numpy array
            age: Optional age information (used for model selection)
            subject: Optional subject category for hybrid embeddings
            drawing_id: Optional database ID for tracking
            use_cache: Whether to use embedding cache
            use_hybrid: Whether to generate hybrid embeddings (default: True)

        Returns:
            Dictionary containing embedding and metadata
        """
        try:
            self._pipeline_stats["total_processed"] += 1

            # Generate embedding (hybrid by default)
            if use_hybrid:
                embedding = self.embedding_service.generate_hybrid_embedding(
                    image=image, subject=subject, age=age, use_cache=use_cache
                )
                embedding_type = "hybrid"
            else:
                embedding = self.embedding_service.generate_embedding(
                    image=image, age=age, use_cache=use_cache
                )
                embedding_type = "legacy"

            # Prepare result
            result = {
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "embedding_type": embedding_type,
                "age_augmented": age is not None,
                "subject_category": subject,
                "drawing_id": drawing_id,
                "model_info": self.embedding_service.vit_wrapper.get_model_info(),
                "success": True,
                "error": None,
            }

            self._pipeline_stats["successful"] += 1
            return result

        except Exception as e:
            self._pipeline_stats["failed"] += 1
            logger.error(
                f"Pipeline processing failed for drawing {drawing_id}: {str(e)}"
            )

            return {
                "embedding": None,
                "embedding_dimension": 0,
                "embedding_type": "unknown",
                "age_augmented": False,
                "subject_category": None,
                "drawing_id": drawing_id,
                "model_info": None,
                "success": False,
                "error": str(e),
            }

    def process_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        ages: Optional[List[float]] = None,
        subjects: Optional[List[Union[str, SubjectCategory]]] = None,
        drawing_ids: Optional[List[int]] = None,
        batch_size: int = 8,
        use_cache: bool = True,
        use_hybrid: bool = True,
    ) -> List[Dict]:
        """
        Process multiple drawings through the embedding pipeline.

        Args:
            images: List of PIL Images or numpy arrays
            ages: Optional list of ages (used for model selection)
            subjects: Optional list of subject categories for hybrid embeddings
            drawing_ids: Optional list of database IDs
            batch_size: Batch size for processing
            use_cache: Whether to use embedding cache
            use_hybrid: Whether to generate hybrid embeddings (default: True)

        Returns:
            List of result dictionaries
        """
        if ages is not None and len(ages) != len(images):
            raise EmbeddingGenerationError("Number of ages must match number of images")

        if subjects is not None and len(subjects) != len(images):
            raise EmbeddingGenerationError(
                "Number of subjects must match number of images"
            )

        if drawing_ids is not None and len(drawing_ids) != len(images):
            raise EmbeddingGenerationError(
                "Number of drawing IDs must match number of images"
            )

        results = []

        try:
            # Generate embeddings in batches
            if use_hybrid:
                embeddings = self.embedding_service.batch_embed(
                    images=images,
                    subjects=subjects,
                    ages=ages,
                    batch_size=batch_size,
                    use_cache=use_cache,
                )
                embedding_type = "hybrid"
            else:
                embeddings = self.embedding_service.generate_batch_embeddings(
                    images=images, ages=ages, batch_size=batch_size, use_cache=use_cache
                )
                embedding_type = "legacy"

            # Create result objects
            for i, embedding in enumerate(embeddings):
                age = ages[i] if ages else None
                subject = subjects[i] if subjects else None
                drawing_id = drawing_ids[i] if drawing_ids else None

                result = {
                    "embedding": embedding,
                    "embedding_dimension": len(embedding),
                    "embedding_type": embedding_type,
                    "age_augmented": age is not None,
                    "subject_category": subject,
                    "drawing_id": drawing_id,
                    "model_info": self.embedding_service.vit_wrapper.get_model_info(),
                    "success": True,
                    "error": None,
                }
                results.append(result)
                self._pipeline_stats["successful"] += 1

            self._pipeline_stats["total_processed"] += len(images)

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")

            # Create error results for all images
            for i in range(len(images)):
                age = ages[i] if ages else None
                subject = subjects[i] if subjects else None
                drawing_id = drawing_ids[i] if drawing_ids else None

                result = {
                    "embedding": None,
                    "embedding_dimension": 0,
                    "embedding_type": "unknown",
                    "age_augmented": False,
                    "subject_category": subject,
                    "drawing_id": drawing_id,
                    "model_info": None,
                    "success": False,
                    "error": str(e),
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
            "cache_hits": 0,
        }


# Global pipeline instance
_embedding_pipeline = None


def get_embedding_pipeline() -> EmbeddingPipeline:
    """Get the global embedding pipeline instance."""
    global _embedding_pipeline
    if _embedding_pipeline is None:
        _embedding_pipeline = EmbeddingPipeline()
    return _embedding_pipeline
