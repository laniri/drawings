"""
Interpretability Engine for generating explanations and saliency maps.

This service provides attention visualization, saliency map generation, and explanation
capabilities for Vision Transformer models used in children's drawing anomaly detection.
"""

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional OpenCV import for advanced image processing
try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    cv2 = None
except Exception as e:
    # Handle other OpenCV-related errors (e.g., missing system libraries)
    import logging

    logging.getLogger(__name__).warning(f"OpenCV import failed: {e}")
    HAS_OPENCV = False
    cv2 = None
except Exception as e:
    # Handle other OpenCV-related errors (e.g., missing system libraries)
    logger.warning(f"OpenCV import failed: {e}")
    HAS_OPENCV = False
    cv2 = None

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import ViTImageProcessor, ViTModel

from app.core.config import settings
from app.services.embedding_service import (
    VisionTransformerWrapper,
    get_embedding_service,
)

logger = logging.getLogger(__name__)


def _resize_with_pil(
    image_array: np.ndarray, target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Fallback image resizing using PIL when OpenCV is not available.

    Args:
        image_array: Input image as numpy array
        target_size: Target size as (width, height)

    Returns:
        Resized image as numpy array
    """
    # Convert numpy array to PIL Image
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    pil_image = Image.fromarray(image_array)
    resized_pil = pil_image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(resized_pil)


def _rgb_to_grayscale(image_array: np.ndarray) -> np.ndarray:
    """
    Fallback RGB to grayscale conversion when OpenCV is not available.

    Args:
        image_array: RGB image as numpy array

    Returns:
        Grayscale image as numpy array
    """
    if len(image_array.shape) == 3:
        # Use standard RGB to grayscale weights
        return np.dot(image_array[..., :3], [0.299, 0.587, 0.114]).astype(
            image_array.dtype
        )
    return image_array


def _simple_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Simple edge detection fallback when OpenCV Canny is not available.

    Args:
        gray_image: Grayscale image as numpy array

    Returns:
        Binary edge map
    """
    # Simple Sobel-like edge detection
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Pad image
    padded = np.pad(gray_image, 1, mode="edge")

    # Apply kernels
    edges_x = np.zeros_like(gray_image)
    edges_y = np.zeros_like(gray_image)

    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            patch = padded[i : i + 3, j : j + 3]
            edges_x[i, j] = np.sum(patch * kernel_x)
            edges_y[i, j] = np.sum(patch * kernel_y)

    # Combine edges
    edges = np.sqrt(edges_x**2 + edges_y**2)

    # Threshold (simple approach)
    threshold = np.mean(edges) + np.std(edges)
    return (edges > threshold).astype(np.uint8) * 255


def _safe_resize(image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Safe image resizing with OpenCV fallback to PIL.

    Args:
        image_array: Input image as numpy array
        target_size: Target size as (width, height)

    Returns:
        Resized image as numpy array
    """
    if HAS_OPENCV and cv2 is not None:
        return cv2.resize(image_array, target_size, interpolation=cv2.INTER_CUBIC)
    else:
        return _resize_with_pil(image_array, target_size)


def _safe_rgb_to_gray(image_array: np.ndarray) -> np.ndarray:
    """
    Safe RGB to grayscale conversion with OpenCV fallback.

    Args:
        image_array: RGB image as numpy array

    Returns:
        Grayscale image as numpy array
    """
    if HAS_OPENCV and cv2 is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        return _rgb_to_grayscale(image_array)


def _safe_edge_detection(gray_image: np.ndarray) -> np.ndarray:
    """
    Safe edge detection with OpenCV fallback.

    Args:
        gray_image: Grayscale image as numpy array

    Returns:
        Binary edge map
    """
    if HAS_OPENCV and cv2 is not None:
        return cv2.Canny(gray_image, 50, 150)
    else:
        return _simple_edge_detection(gray_image)


class InterpretabilityError(Exception):
    """Base exception for interpretability engine errors."""

    pass


class SaliencyGenerationError(InterpretabilityError):
    """Raised when saliency map generation fails."""

    pass


class AttentionVisualizationError(InterpretabilityError):
    """Raised when attention visualization fails."""

    pass


class AttentionRollout:
    """
    Attention rollout technique for Vision Transformers.

    This class implements the attention rollout method to compute attention
    maps that show which patches the model focuses on for its predictions.
    """

    def __init__(self, model: ViTModel, discard_ratio: float = 0.9):
        """
        Initialize attention rollout.

        Args:
            model: Vision Transformer model
            discard_ratio: Ratio of attention to discard (keep top 1-discard_ratio)
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self.attention_maps = []

        # Register hooks to capture attention weights
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""

        def hook_fn(module, input, output):
            # ViT attention output is (attention_weights, context_layer)
            if hasattr(output, "attentions") or isinstance(output, tuple):
                if isinstance(output, tuple) and len(output) >= 2:
                    attention_weights = output[0]  # First element is attention weights
                    self.attention_maps.append(attention_weights.detach())

        # Register hooks on all attention layers
        for name, module in self.model.named_modules():
            if "attention" in name and hasattr(module, "attention"):
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_rollout(
        self, input_tensor: torch.Tensor, start_layer: int = 0
    ) -> torch.Tensor:
        """
        Generate attention rollout for input tensor.

        Args:
            input_tensor: Input tensor [1, 3, H, W]
            start_layer: Layer to start rollout from

        Returns:
            Attention rollout tensor [H_patches, W_patches]
        """
        try:
            self.attention_maps = []

            # Forward pass to collect attention maps
            with torch.no_grad():
                _ = self.model(input_tensor, output_attentions=True)

            if not self.attention_maps:
                # Fallback: try to get attention from model output
                outputs = self.model(input_tensor, output_attentions=True)
                if hasattr(outputs, "attentions") and outputs.attentions:
                    self.attention_maps = [att.detach() for att in outputs.attentions]

            if not self.attention_maps:
                raise AttentionVisualizationError("No attention maps captured")

            # Process attention maps
            rollout = self._compute_rollout(start_layer)

            return rollout

        except Exception as e:
            logger.error(f"Attention rollout generation failed: {str(e)}")
            raise AttentionVisualizationError(f"Rollout generation failed: {str(e)}")
        finally:
            self.attention_maps = []

    def _compute_rollout(self, start_layer: int = 0) -> torch.Tensor:
        """Compute attention rollout from captured attention maps."""
        if not self.attention_maps:
            raise AttentionVisualizationError("No attention maps available")

        # Start with identity matrix
        num_tokens = self.attention_maps[0].shape[-1]
        rollout = torch.eye(num_tokens)

        # Apply attention rollout through layers
        for attention in self.attention_maps[start_layer:]:
            # Average over attention heads: [batch, heads, tokens, tokens] -> [tokens, tokens]
            attention_heads_fused = attention.squeeze(0).mean(dim=0)

            # Apply discard ratio to keep only top attention weights
            flat_attention = attention_heads_fused.view(-1)
            _, indices = torch.topk(
                flat_attention, int(flat_attention.shape[0] * (1 - self.discard_ratio))
            )

            # Create mask for top attention weights
            mask = torch.zeros_like(flat_attention)
            mask[indices] = 1
            attention_heads_fused = attention_heads_fused * mask.view(
                attention_heads_fused.shape
            )

            # Normalize
            attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(
                dim=-1, keepdim=True
            )

            # Add residual connection (identity matrix)
            attention_heads_fused = 0.5 * attention_heads_fused + 0.5 * torch.eye(
                num_tokens
            )

            # Matrix multiplication for rollout
            rollout = torch.matmul(attention_heads_fused, rollout)

        # Extract attention for CLS token (first token) to all patches
        cls_attention = rollout[0, 1:]  # Exclude CLS token itself

        return cls_attention

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self._remove_hooks()


class GradCAMViT:
    """
    Gradient-based Class Activation Mapping for Vision Transformers.

    This class implements Grad-CAM specifically adapted for Vision Transformers
    to generate saliency maps showing important regions for anomaly detection.
    """

    def __init__(self, model: ViTModel, target_layer: str = None):
        """
        Initialize Grad-CAM for ViT.

        Args:
            model: Vision Transformer model
            target_layer: Name of target layer (default: last encoder layer)
        """
        self.model = model
        self.target_layer = target_layer or self._get_last_encoder_layer()
        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _get_last_encoder_layer(self) -> str:
        """Get the name of the last encoder layer."""
        # Find the last encoder layer in ViT
        encoder_layers = []
        for name, _ in self.model.named_modules():
            if "encoder.layer" in name and "output" in name:
                encoder_layers.append(name)

        if encoder_layers:
            return encoder_layers[-1]
        else:
            # Fallback to a common ViT layer name
            return "encoder.layer.11.output"

    def _register_hooks(self):
        """Register hooks to capture gradients and activations."""

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_cam(
        self, input_tensor: torch.Tensor, reconstruction_loss: float
    ) -> torch.Tensor:
        """
        Generate Class Activation Map using gradients.

        Args:
            input_tensor: Input tensor [1, 3, H, W]
            reconstruction_loss: Reconstruction loss to compute gradients for

        Returns:
            CAM tensor [H_patches, W_patches]
        """
        try:
            self.model.eval()
            input_tensor.requires_grad_(True)

            # Forward pass
            outputs = self.model(input_tensor)

            # Use the CLS token output for gradient computation
            cls_output = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]

            # Compute a proxy loss (we'll use the norm of CLS token as proxy)
            proxy_loss = torch.norm(cls_output, dim=1).mean()

            # Backward pass
            self.model.zero_grad()
            proxy_loss.backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise SaliencyGenerationError(
                    "Failed to capture gradients or activations"
                )

            # Compute CAM
            # Gradients: [batch, seq_len, hidden_size]
            # Activations: [batch, seq_len, hidden_size]

            # Global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(0, 2))  # [seq_len]

            # Weighted combination of activations
            cam = torch.zeros(self.activations.shape[1])  # [seq_len]
            for i, w in enumerate(weights):
                cam[i] = w * torch.mean(self.activations[0, i, :])

            # Apply ReLU and normalize
            cam = F.relu(cam)
            if cam.max() > 0:
                cam = cam / cam.max()

            # Remove CLS token (first token)
            cam = cam[1:]

            return cam

        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {str(e)}")
            raise SaliencyGenerationError(f"CAM generation failed: {str(e)}")
        finally:
            # Reset gradients and activations
            self.gradients = None
            self.activations = None

    def __del__(self):
        """Cleanup hooks when object is destroyed."""
        self._remove_hooks()


class PatchImportanceScorer:
    """
    Patch-level importance scoring for Vision Transformers.

    This class provides methods to compute importance scores for individual
    patches in the input image based on various techniques.
    """

    def __init__(self, model: ViTModel, patch_size: int = 16):
        """
        Initialize patch importance scorer.

        Args:
            model: Vision Transformer model
            patch_size: Size of patches (default: 16 for ViT-Base)
        """
        self.model = model
        self.patch_size = patch_size

    def compute_attention_importance(
        self, input_tensor: torch.Tensor, method: str = "rollout"
    ) -> torch.Tensor:
        """
        Compute patch importance using attention mechanisms.

        Args:
            input_tensor: Input tensor [1, 3, H, W]
            method: Method to use ("rollout" or "last_layer")

        Returns:
            Importance scores for each patch
        """
        try:
            if method == "rollout":
                rollout = AttentionRollout(self.model)
                importance = rollout.generate_rollout(input_tensor)
            elif method == "last_layer":
                # Use attention from last layer only
                with torch.no_grad():
                    outputs = self.model(input_tensor, output_attentions=True)
                    if hasattr(outputs, "attentions") and outputs.attentions:
                        last_attention = outputs.attentions[-1]  # Last layer
                        # Average over heads and get CLS attention to patches
                        importance = last_attention.squeeze(0).mean(dim=0)[0, 1:]
                    else:
                        raise AttentionVisualizationError(
                            "No attention weights available"
                        )
            else:
                raise ValueError(f"Unknown attention method: {method}")

            return importance

        except Exception as e:
            logger.error(f"Attention importance computation failed: {str(e)}")
            raise AttentionVisualizationError(
                f"Importance computation failed: {str(e)}"
            )

    def compute_gradient_importance(
        self, input_tensor: torch.Tensor, reconstruction_loss: float
    ) -> torch.Tensor:
        """
        Compute patch importance using gradient-based methods.

        Args:
            input_tensor: Input tensor [1, 3, H, W]
            reconstruction_loss: Reconstruction loss for gradient computation

        Returns:
            Importance scores for each patch
        """
        try:
            grad_cam = GradCAMViT(self.model)
            importance = grad_cam.generate_cam(input_tensor, reconstruction_loss)
            return importance

        except Exception as e:
            logger.error(f"Gradient importance computation failed: {str(e)}")
            raise SaliencyGenerationError(f"Gradient importance failed: {str(e)}")

    def reshape_to_spatial(
        self, importance_scores: torch.Tensor, image_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Reshape 1D importance scores to 2D spatial map.

        Args:
            importance_scores: 1D tensor of patch importance scores
            image_size: Original image size (H, W)

        Returns:
            2D spatial importance map
        """
        # Calculate number of patches per dimension
        patches_per_dim = image_size[0] // self.patch_size

        # Ensure we have the right number of patches
        expected_patches = patches_per_dim * patches_per_dim
        if len(importance_scores) != expected_patches:
            logger.warning(
                f"Expected {expected_patches} patches, got {len(importance_scores)}"
            )
            # Pad or truncate as needed
            if len(importance_scores) < expected_patches:
                padding = expected_patches - len(importance_scores)
                importance_scores = torch.cat([importance_scores, torch.zeros(padding)])
            else:
                importance_scores = importance_scores[:expected_patches]

        # Reshape to 2D
        spatial_map = importance_scores.view(patches_per_dim, patches_per_dim)

        return spatial_map


class SaliencyMapGenerator:
    """
    Main class for generating saliency maps from Vision Transformer models.

    This class combines various techniques to create comprehensive saliency maps
    that highlight important regions in children's drawings for anomaly detection.
    """

    def __init__(self, embedding_service=None):
        """
        Initialize saliency map generator.

        Args:
            embedding_service: Embedding service instance (optional)
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.output_dir = Path("static/saliency_maps")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure embedding service is initialized
        if not self.embedding_service.is_ready():
            logger.warning("Embedding service not ready, initializing...")
            self.embedding_service.initialize()

    def generate_saliency_map(
        self,
        image: Union[Image.Image, np.ndarray],
        reconstruction_loss: float,
        method: str = "attention_rollout",
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate saliency map for an image.

        Args:
            image: Input image (PIL Image or numpy array)
            reconstruction_loss: Reconstruction loss from autoencoder
            method: Saliency method ("attention_rollout", "grad_cam", "combined")
            save_path: Optional path to save the saliency map

        Returns:
            Dictionary containing saliency map and metadata
        """
        try:
            # Preprocess image
            image_tensor = self.embedding_service._preprocess_image(image)

            # Get the Vision Transformer model
            vit_model = self.embedding_service.vit_wrapper.model

            if method == "attention_rollout":
                importance_scores = self._generate_attention_saliency(
                    image_tensor, vit_model
                )
            elif method == "grad_cam":
                importance_scores = self._generate_gradient_saliency(
                    image_tensor, vit_model, reconstruction_loss
                )
            elif method == "combined":
                # Combine attention and gradient methods
                attention_scores = self._generate_attention_saliency(
                    image_tensor, vit_model
                )
                gradient_scores = self._generate_gradient_saliency(
                    image_tensor, vit_model, reconstruction_loss
                )
                # Weighted combination
                importance_scores = 0.6 * attention_scores + 0.4 * gradient_scores
            else:
                raise ValueError(f"Unknown saliency method: {method}")

            # Convert to spatial map
            patch_scorer = PatchImportanceScorer(vit_model)
            spatial_map = patch_scorer.reshape_to_spatial(importance_scores)

            # Convert to numpy for further processing
            saliency_map = spatial_map.cpu().numpy()

            # Generate visualization
            visualization_path = None
            if save_path:
                visualization_path = self._create_saliency_visualization(
                    image, saliency_map, save_path
                )

            result = {
                "saliency_map": saliency_map,
                "importance_scores": importance_scores.cpu().numpy(),
                "method": method,
                "reconstruction_loss": reconstruction_loss,
                "visualization_path": visualization_path,
                "map_shape": saliency_map.shape,
                "max_importance": float(saliency_map.max()),
                "mean_importance": float(saliency_map.mean()),
            }

            logger.info(f"Generated saliency map using {method} method")
            return result

        except Exception as e:
            logger.error(f"Saliency map generation failed: {str(e)}")
            raise SaliencyGenerationError(f"Generation failed: {str(e)}")

    def generate_attribution_aware_saliency(
        self,
        image: Union[Image.Image, np.ndarray],
        reconstruction_loss: float,
        attribution_info: Dict,
        method: str = "combined",
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Generate saliency map with attribution-aware highlighting.

        Args:
            image: Input image (PIL Image or numpy array)
            reconstruction_loss: Reconstruction loss from autoencoder
            attribution_info: Attribution information including component scores
            method: Saliency method to use
            save_path: Optional path to save the saliency map

        Returns:
            Dictionary containing attribution-aware saliency map and metadata
        """
        try:
            # Generate base saliency map
            base_result = self.generate_saliency_map(
                image, reconstruction_loss, method, save_path=None
            )

            # Get attribution information
            attribution = attribution_info.get("anomaly_attribution", "unknown")
            visual_score = attribution_info.get("visual_anomaly_score", 0)
            subject_score = attribution_info.get("subject_anomaly_score", 0)

            # Modify saliency map based on attribution
            saliency_map = base_result["saliency_map"]

            if attribution == "visual":
                # Enhance visual regions - boost overall saliency
                enhanced_saliency = saliency_map * 1.2
                enhanced_saliency = np.clip(enhanced_saliency, 0, 1)
                attribution_note = "Enhanced highlighting for visual anomalies"

            elif attribution == "subject":
                # Focus on central regions where subject is typically depicted
                center_mask = self._create_center_focus_mask(saliency_map.shape)
                enhanced_saliency = saliency_map * (1 + 0.3 * center_mask)
                enhanced_saliency = np.clip(enhanced_saliency, 0, 1)
                attribution_note = (
                    "Enhanced highlighting for subject-specific anomalies"
                )

            elif attribution == "both":
                # Balanced enhancement
                center_mask = self._create_center_focus_mask(saliency_map.shape)
                enhanced_saliency = saliency_map * (1.1 + 0.2 * center_mask)
                enhanced_saliency = np.clip(enhanced_saliency, 0, 1)
                attribution_note = "Balanced highlighting for combined anomalies"

            else:
                # Default - no modification
                enhanced_saliency = saliency_map
                attribution_note = "Standard saliency highlighting"

            # Create attribution-aware visualization
            visualization_path = None
            if save_path:
                visualization_path = self._create_attribution_visualization(
                    image, enhanced_saliency, attribution_info, save_path
                )

            # Combine results
            result = base_result.copy()
            result.update(
                {
                    "attribution_enhanced_map": enhanced_saliency,
                    "attribution_type": attribution,
                    "attribution_note": attribution_note,
                    "component_scores": {
                        "visual": float(visual_score),
                        "subject": float(subject_score),
                    },
                    "attribution_visualization_path": visualization_path,
                }
            )

            logger.info(
                f"Generated attribution-aware saliency map for {attribution} anomaly"
            )
            return result

        except Exception as e:
            logger.error(f"Attribution-aware saliency generation failed: {str(e)}")
            raise SaliencyGenerationError(
                f"Attribution-aware generation failed: {str(e)}"
            )

    def _create_center_focus_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create a mask that emphasizes central regions."""
        height, width = shape
        center_y, center_x = height // 2, width // 2

        # Create distance from center
        y, x = np.ogrid[:height, :width]
        distances = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Normalize distances and invert (closer to center = higher value)
        max_distance = np.sqrt(center_y**2 + center_x**2)
        center_mask = 1 - (distances / max_distance)

        # Apply gaussian-like falloff
        center_mask = np.exp(-2 * (distances / max_distance) ** 2)

        return center_mask

    def _create_attribution_visualization(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        attribution_info: Dict,
        save_path: str,
    ) -> str:
        """
        Create attribution-aware saliency visualization.

        Args:
            original_image: Original input image
            saliency_map: Attribution-enhanced saliency map
            attribution_info: Attribution information
            save_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = original_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize saliency map
            if saliency_resized.max() > 0:
                saliency_resized = saliency_resized / saliency_resized.max()

            # Choose colormap based on attribution
            attribution = attribution_info.get("anomaly_attribution", "unknown")
            if attribution == "visual":
                colormap = cm.plasma  # Purple-pink for visual
            elif attribution == "subject":
                colormap = cm.viridis  # Green-blue for subject
            elif attribution == "both":
                colormap = cm.inferno  # Red-orange for both
            else:
                colormap = cm.jet  # Default rainbow

            # Create heatmap
            heatmap = colormap(saliency_resized)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)

            # Convert original image to numpy
            original_np = np.array(original_image)

            # Create overlay
            alpha = 0.4
            overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)

            # Add attribution text overlay
            overlay_image = Image.fromarray(overlay)
            draw = ImageDraw.Draw(overlay_image)

            # Add attribution label
            attribution_text = f"Attribution: {attribution}"
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            # Draw text background
            text_bbox = draw.textbbox((10, 10), attribution_text, font=font)
            draw.rectangle(text_bbox, fill=(0, 0, 0, 128))

            # Draw text
            draw.text((10, 10), attribution_text, fill=(255, 255, 255), font=font)

            # Save visualization
            overlay_image.save(save_path)

            logger.info(
                f"Attribution-aware saliency visualization saved to {save_path}"
            )
            return save_path

        except Exception as e:
            logger.error(f"Failed to create attribution visualization: {str(e)}")
            raise SaliencyGenerationError(
                f"Attribution visualization creation failed: {str(e)}"
            )

    def _generate_attention_saliency(
        self, image_tensor: torch.Tensor, model: ViTModel
    ) -> torch.Tensor:
        """Generate saliency using attention rollout."""
        patch_scorer = PatchImportanceScorer(model)
        return patch_scorer.compute_attention_importance(image_tensor, method="rollout")

    def _generate_gradient_saliency(
        self, image_tensor: torch.Tensor, model: ViTModel, reconstruction_loss: float
    ) -> torch.Tensor:
        """Generate saliency using gradient-based methods."""
        patch_scorer = PatchImportanceScorer(model)
        return patch_scorer.compute_gradient_importance(
            image_tensor, reconstruction_loss
        )

    def _create_saliency_visualization(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        save_path: str,
    ) -> str:
        """
        Create and save saliency map visualization.

        Args:
            original_image: Original input image
            saliency_map: 2D saliency map
            save_path: Path to save visualization

        Returns:
            Path to saved visualization
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = original_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize saliency map
            if saliency_resized.max() > 0:
                saliency_resized = saliency_resized / saliency_resized.max()

            # Create heatmap
            heatmap = cm.jet(saliency_resized)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)

            # Convert original image to numpy
            original_np = np.array(original_image)

            # Create overlay
            alpha = 0.4
            overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)

            # Save visualization
            overlay_image = Image.fromarray(overlay)
            overlay_image.save(save_path)

            logger.info(f"Saliency visualization saved to {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Failed to create saliency visualization: {str(e)}")
            raise SaliencyGenerationError(f"Visualization creation failed: {str(e)}")


# Global saliency generator instance
_saliency_generator = None


def get_saliency_generator() -> SaliencyMapGenerator:
    """Get the global saliency map generator instance."""
    global _saliency_generator
    if _saliency_generator is None:
        _saliency_generator = SaliencyMapGenerator()
    return _saliency_generator


class VisualFeatureIdentifier:
    """
    Identifies and describes visual features in children's drawings.

    This class analyzes saliency maps and original images to identify
    specific visual features that contribute to anomaly detection.
    """

    def __init__(self):
        """Initialize visual feature identifier."""
        self.feature_templates = self._load_feature_templates()

    def _load_feature_templates(self) -> Dict:
        """Load templates for common visual features in children's drawings."""
        return {
            "shapes": {
                "circle": "circular or round shapes",
                "square": "square or rectangular shapes",
                "triangle": "triangular shapes",
                "line": "straight or curved lines",
                "scribble": "scribbled or chaotic patterns",
            },
            "body_parts": {
                "head": "head or face region",
                "body": "body or torso area",
                "limbs": "arms or legs",
                "hands": "hand or finger details",
                "feet": "feet or foot details",
            },
            "objects": {
                "house": "house or building structure",
                "tree": "tree or plant elements",
                "sun": "sun or circular object with rays",
                "car": "vehicle or transportation object",
                "animal": "animal or creature form",
            },
            "spatial": {
                "center": "central region of the drawing",
                "edges": "edge or border areas",
                "corners": "corner regions",
                "top": "upper portion of the drawing",
                "bottom": "lower portion of the drawing",
            },
        }

    def identify_important_regions(
        self, saliency_map: np.ndarray, threshold: float = 0.7
    ) -> List[Dict]:
        """
        Identify important regions in the saliency map.

        Args:
            saliency_map: 2D saliency map
            threshold: Threshold for considering regions important

        Returns:
            List of important region descriptions
        """
        try:
            important_regions = []

            # Find regions above threshold
            high_importance = saliency_map > threshold

            if not np.any(high_importance):
                # Lower threshold if no regions found
                threshold = np.percentile(saliency_map, 80)
                high_importance = saliency_map > threshold

            # Use connected components to find distinct regions
            try:
                from scipy import ndimage

                labeled_regions, num_regions = ndimage.label(high_importance)
            except ImportError:
                # Fallback: treat each pixel as a separate region
                labeled_regions = high_importance.astype(int)
                num_regions = np.sum(high_importance)

            for region_id in range(1, num_regions + 1):
                region_mask = labeled_regions == region_id
                region_coords = np.where(region_mask)

                if len(region_coords[0]) == 0:
                    continue

                # Calculate region properties
                min_row, max_row = region_coords[0].min(), region_coords[0].max()
                min_col, max_col = region_coords[1].min(), region_coords[1].max()
                center_row = (min_row + max_row) // 2
                center_col = (min_col + max_col) // 2

                # Calculate region size and importance
                region_size = np.sum(region_mask)
                region_importance = np.mean(saliency_map[region_mask])

                # Determine spatial location
                spatial_desc = self._describe_spatial_location(
                    center_row, center_col, saliency_map.shape
                )

                region_info = {
                    "region_id": region_id,
                    "bounding_box": (min_row, min_col, max_row, max_col),
                    "center": (center_row, center_col),
                    "size": int(region_size),
                    "importance_score": float(region_importance),
                    "spatial_location": spatial_desc,
                    "relative_size": float(region_size / saliency_map.size),
                }

                important_regions.append(region_info)

            # Sort by importance score
            important_regions.sort(key=lambda x: x["importance_score"], reverse=True)

            return important_regions

        except Exception as e:
            logger.error(f"Failed to identify important regions: {str(e)}")
            return []

    def _describe_spatial_location(
        self, row: int, col: int, shape: Tuple[int, int]
    ) -> str:
        """Describe the spatial location of a region."""
        height, width = shape

        # Determine vertical position
        if row < height * 0.33:
            vertical = "top"
        elif row < height * 0.67:
            vertical = "middle"
        else:
            vertical = "bottom"

        # Determine horizontal position
        if col < width * 0.33:
            horizontal = "left"
        elif col < width * 0.67:
            horizontal = "center"
        else:
            horizontal = "right"

        if vertical == "middle" and horizontal == "center":
            return "center"
        else:
            return f"{vertical}-{horizontal}"

    def analyze_drawing_content(
        self, image: Union[Image.Image, np.ndarray], important_regions: List[Dict]
    ) -> Dict:
        """
        Analyze drawing content to identify likely visual features.

        Args:
            image: Original drawing image
            important_regions: List of important regions from saliency analysis

        Returns:
            Dictionary containing content analysis
        """
        try:
            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image.convert("RGB"))
            else:
                image_np = image

            # Basic image analysis
            analysis = {
                "image_properties": self._analyze_image_properties(image_np),
                "region_analysis": [],
                "overall_complexity": self._calculate_complexity(image_np),
                "dominant_features": [],
            }

            # Analyze each important region
            for region in important_regions:
                region_analysis = self._analyze_region_content(image_np, region)
                analysis["region_analysis"].append(region_analysis)

            # Identify dominant features
            analysis["dominant_features"] = self._identify_dominant_features(
                analysis["region_analysis"]
            )

            return analysis

        except Exception as e:
            logger.error(f"Drawing content analysis failed: {str(e)}")
            return {"error": str(e)}

    def _analyze_image_properties(self, image: np.ndarray) -> Dict:
        """Analyze basic properties of the image."""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            if HAS_OPENCV and cv2 is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = _rgb_to_grayscale(image)
        else:
            gray = image

        return {
            "dimensions": image.shape[:2],
            "mean_intensity": float(np.mean(gray)),
            "std_intensity": float(np.std(gray)),
            "edge_density": self._calculate_edge_density(gray),
            "contrast": float(np.std(gray) / np.mean(gray)) if np.mean(gray) > 0 else 0,
        }

    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in the image."""
        if HAS_OPENCV and cv2 is not None:
            edges = cv2.Canny(gray_image, 50, 150)
            return float(np.sum(edges > 0) / edges.size)
        else:
            # Fallback edge detection using PIL/numpy
            # Simple gradient-based edge detection
            from PIL import Image, ImageFilter

            # Convert numpy array to PIL Image
            if gray_image.dtype != np.uint8:
                gray_image = (gray_image * 255).astype(np.uint8)

            pil_image = Image.fromarray(gray_image, mode="L")

            # Apply edge detection filter
            edges = pil_image.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            # Calculate edge density
            return float(np.sum(edges_array > 50) / edges_array.size)

    def _calculate_complexity(self, image: np.ndarray) -> float:
        """Calculate overall drawing complexity."""
        if len(image.shape) == 3:
            gray = convert_to_grayscale(image)
        else:
            gray = image

        # Use edge density and intensity variation as complexity measures
        edge_density = self._calculate_edge_density(gray)
        intensity_variation = np.std(gray) / 255.0

        # Combine measures
        complexity = 0.6 * edge_density + 0.4 * intensity_variation
        return float(complexity)

    def _analyze_region_content(self, image: np.ndarray, region: Dict) -> Dict:
        """Analyze content of a specific region."""
        # Extract region from image
        min_row, min_col, max_row, max_col = region["bounding_box"]
        region_image = image[min_row : max_row + 1, min_col : max_col + 1]

        if region_image.size == 0:
            return {"error": "Empty region"}

        # Analyze region properties
        region_analysis = {
            "region_id": region["region_id"],
            "spatial_location": region["spatial_location"],
            "size_category": self._categorize_size(region["relative_size"]),
            "shape_characteristics": self._analyze_shape_characteristics(region_image),
            "likely_content": self._guess_content_type(region, region_image),
        }

        return region_analysis

    def _categorize_size(self, relative_size: float) -> str:
        """Categorize region size."""
        if relative_size < 0.05:
            return "small"
        elif relative_size < 0.2:
            return "medium"
        else:
            return "large"

    def _analyze_shape_characteristics(self, region_image: np.ndarray) -> Dict:
        """Analyze shape characteristics of a region."""
        if len(region_image.shape) == 3:
            if HAS_OPENCV and cv2 is not None:
                gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = _rgb_to_grayscale(region_image)
        else:
            gray = region_image

        # Basic shape analysis
        height, width = gray.shape
        aspect_ratio = width / height if height > 0 else 1.0

        # Determine shape category
        if abs(aspect_ratio - 1.0) < 0.2:
            shape_type = "square-like"
        elif aspect_ratio > 1.5:
            shape_type = "horizontal"
        elif aspect_ratio < 0.67:
            shape_type = "vertical"
        else:
            shape_type = "rectangular"

        return {
            "aspect_ratio": float(aspect_ratio),
            "shape_type": shape_type,
            "dimensions": (height, width),
        }

    def _guess_content_type(self, region: Dict, region_image: np.ndarray) -> str:
        """Make an educated guess about region content."""
        spatial_location = region["spatial_location"]
        size_category = self._categorize_size(region["relative_size"])

        # Simple heuristics based on location and size
        if "top" in spatial_location and size_category in ["small", "medium"]:
            return "likely head or upper body part"
        elif "center" in spatial_location and size_category == "large":
            return "likely main body or central object"
        elif "bottom" in spatial_location:
            return "likely legs, feet, or ground elements"
        elif size_category == "small":
            return "likely detail or feature"
        else:
            return "unidentified drawing element"

    def _identify_dominant_features(self, region_analyses: List[Dict]) -> List[str]:
        """Identify dominant features across all regions."""
        if not region_analyses:
            return []

        # Count feature types
        feature_counts = {}
        for analysis in region_analyses:
            content = analysis.get("likely_content", "unknown")
            feature_counts[content] = feature_counts.get(content, 0) + 1

        # Sort by frequency
        sorted_features = sorted(
            feature_counts.items(), key=lambda x: x[1], reverse=True
        )

        # Return top features
        return [feature for feature, count in sorted_features[:3]]


class ExplanationGenerator:
    """
    Generates human-readable explanations for anomaly detection results.

    This class combines saliency maps, visual feature analysis, and domain knowledge
    to create comprehensive explanations for why a drawing was flagged as anomalous.
    """

    def __init__(self):
        """Initialize explanation generator."""
        self.feature_identifier = VisualFeatureIdentifier()
        self.explanation_templates = self._load_explanation_templates()

    def _load_explanation_templates(self) -> Dict:
        """Load templates for generating explanations."""
        return {
            "high_anomaly": {
                "intro": "This drawing shows significant deviations from typical patterns for this age group.",
                "patterns": [
                    "The {feature} in the {location} region shows unusual characteristics.",
                    "Atypical {feature} patterns were detected in the {location} area.",
                    "The {location} region contains {feature} elements that differ from age-expected norms.",
                ],
            },
            "medium_anomaly": {
                "intro": "This drawing shows some deviations from typical patterns for this age group.",
                "patterns": [
                    "The {feature} in the {location} region shows some unusual characteristics.",
                    "Mild atypical patterns were detected in the {location} area.",
                    "The {location} region contains {feature} elements that slightly differ from norms.",
                ],
            },
            "low_anomaly": {
                "intro": "This drawing shows minor deviations from typical patterns for this age group.",
                "patterns": [
                    "The {feature} in the {location} region shows minor variations.",
                    "Subtle differences were detected in the {location} area.",
                    "The {location} region contains {feature} elements with slight variations.",
                ],
            },
            "spatial_descriptions": {
                "center": "central area",
                "top-left": "upper left region",
                "top-center": "upper region",
                "top-right": "upper right region",
                "middle-left": "left side",
                "middle-right": "right side",
                "bottom-left": "lower left region",
                "bottom-center": "lower region",
                "bottom-right": "lower right region",
            },
        }

    def generate_explanation(
        self,
        anomaly_score: float,
        normalized_score: float,
        saliency_result: Dict,
        age_group: str,
        drawing_metadata: Optional[Dict] = None,
        attribution_info: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate comprehensive explanation for anomaly detection result.

        Args:
            anomaly_score: Raw anomaly score
            normalized_score: Normalized anomaly score
            saliency_result: Result from saliency map generation
            age_group: Age group used for comparison
            drawing_metadata: Optional metadata about the drawing
            attribution_info: Optional subject-aware attribution information

        Returns:
            Dictionary containing structured explanation
        """
        try:
            # Determine anomaly severity
            severity = self._determine_severity(normalized_score)

            # Identify important regions
            important_regions = self.feature_identifier.identify_important_regions(
                saliency_result["saliency_map"]
            )

            # Generate main explanation with attribution context
            main_explanation = self._generate_main_explanation(
                severity, important_regions, age_group, attribution_info
            )

            # Generate detailed analysis with attribution
            detailed_analysis = self._generate_detailed_analysis(
                important_regions, saliency_result, attribution_info
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                severity, important_regions
            )

            explanation = {
                "summary": main_explanation,
                "severity": severity,
                "anomaly_score": float(anomaly_score),
                "normalized_score": float(normalized_score),
                "age_group": age_group,
                "important_regions": important_regions,
                "detailed_analysis": detailed_analysis,
                "recommendations": recommendations,
                "attribution_info": attribution_info,
                "technical_details": {
                    "saliency_method": saliency_result["method"],
                    "max_importance": saliency_result["max_importance"],
                    "mean_importance": saliency_result["mean_importance"],
                    "num_important_regions": len(important_regions),
                },
            }

            # Add metadata-based insights if available
            if drawing_metadata:
                explanation["metadata_insights"] = self._analyze_metadata(
                    drawing_metadata, severity
                )

            return explanation

        except Exception as e:
            logger.error(f"Explanation generation failed: {str(e)}")
            return {
                "summary": "Unable to generate detailed explanation due to processing error.",
                "error": str(e),
                "severity": "unknown",
                "anomaly_score": float(anomaly_score),
                "normalized_score": float(normalized_score),
            }

    def _determine_severity(self, normalized_score: float) -> str:
        """Determine anomaly severity based on normalized score."""
        if normalized_score >= 0.8:
            return "high"
        elif normalized_score >= 0.6:
            return "medium"
        elif normalized_score >= 0.4:
            return "low"
        else:
            return "minimal"

    def _generate_main_explanation(
        self,
        severity: str,
        important_regions: List[Dict],
        age_group: str,
        attribution_info: Optional[Dict] = None,
    ) -> str:
        """Generate the main explanation text with attribution context."""
        # Include attribution context in explanation
        attribution_context = ""
        if attribution_info:
            attribution = attribution_info.get("anomaly_attribution", "unknown")
            subject = attribution_info.get("subject_category", "unspecified")

            if attribution == "subject":
                attribution_context = f" The anomaly appears to be primarily related to how the {subject} is depicted."
            elif attribution == "visual":
                attribution_context = f" The anomaly appears to be primarily related to visual characteristics rather than subject-specific patterns."
            elif attribution == "age":
                attribution_context = f" The anomaly appears to be primarily related to age-inappropriate patterns."
            elif attribution == "both":
                attribution_context = f" The anomaly appears to involve both subject-specific and visual characteristics."

        if not important_regions:
            base_explanation = f"This drawing shows {severity} deviation from typical patterns for age group {age_group}, but specific regions of concern could not be identified."
            return base_explanation + attribution_context

        # Get template based on severity
        template_key = (
            severity
            if severity in ["high_anomaly", "medium_anomaly", "low_anomaly"]
            else "low_anomaly"
        )
        templates = self.explanation_templates.get(
            template_key, self.explanation_templates["low_anomaly"]
        )

        # Start with intro
        explanation = templates["intro"]

        # Add specific region information
        top_regions = important_regions[:2]  # Focus on top 2 regions

        for i, region in enumerate(top_regions):
            spatial_desc = self.explanation_templates["spatial_descriptions"].get(
                region["spatial_location"], region["spatial_location"]
            )

            # Use a pattern template
            pattern_idx = i % len(templates["patterns"])
            pattern = templates["patterns"][pattern_idx]

            # Fill in the pattern
            region_explanation = pattern.format(
                feature="drawing elements", location=spatial_desc
            )

            explanation += f" {region_explanation}"

        # Add attribution context
        explanation += attribution_context

        return explanation

    def _generate_detailed_analysis(
        self,
        important_regions: List[Dict],
        saliency_result: Dict,
        attribution_info: Optional[Dict] = None,
    ) -> List[str]:
        """Generate detailed analysis points."""
        analysis_points = []

        if not important_regions:
            analysis_points.append(
                "No specific regions of high importance were identified."
            )
            return analysis_points

        # Analyze each important region
        for i, region in enumerate(important_regions[:3]):  # Top 3 regions
            spatial_desc = self.explanation_templates["spatial_descriptions"].get(
                region["spatial_location"], region["spatial_location"]
            )

            importance_pct = region["importance_score"] * 100
            size_desc = region.get("size_category", "unknown size")

            point = f"Region {i+1}: {spatial_desc} shows {importance_pct:.1f}% importance with {size_desc} coverage."
            analysis_points.append(point)

        # Add overall saliency information
        max_importance = saliency_result.get("max_importance", 0) * 100
        mean_importance = saliency_result.get("mean_importance", 0) * 100

        analysis_points.append(
            f"Overall attention distribution: maximum focus at {max_importance:.1f}%, "
            f"average attention at {mean_importance:.1f}%."
        )

        # Add attribution-specific analysis
        if attribution_info:
            attribution = attribution_info.get("anomaly_attribution", "unknown")
            visual_score = attribution_info.get("visual_anomaly_score")
            subject_score = attribution_info.get("subject_anomaly_score")

            if visual_score is not None and subject_score is not None:
                analysis_points.append(
                    f"Component analysis: visual anomaly score {visual_score:.3f}, "
                    f"subject anomaly score {subject_score:.3f}."
                )

            if attribution != "unknown":
                analysis_points.append(f"Primary anomaly attribution: {attribution}.")

        return analysis_points

    def _generate_recommendations(
        self, severity: str, important_regions: List[Dict]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if severity == "high":
            recommendations.extend(
                [
                    "Consider further evaluation by a qualified professional.",
                    "Review the drawing in context of the child's developmental history.",
                    "Compare with other recent drawings from the same child.",
                ]
            )
        elif severity == "medium":
            recommendations.extend(
                [
                    "Monitor for patterns across multiple drawings.",
                    "Consider the drawing context and child's current circumstances.",
                    "Document for potential follow-up if patterns persist.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "This drawing shows minimal deviation from typical patterns.",
                    "Continue regular monitoring as part of routine assessment.",
                    "Consider this within the normal range of variation.",
                ]
            )

        # Add region-specific recommendations
        if important_regions:
            top_region = important_regions[0]
            spatial_location = top_region["spatial_location"]

            if "center" in spatial_location:
                recommendations.append(
                    "Pay attention to central elements in future drawings."
                )
            elif "top" in spatial_location:
                recommendations.append(
                    "Monitor upper region elements in subsequent drawings."
                )
            elif "bottom" in spatial_location:
                recommendations.append(
                    "Observe lower region patterns in future assessments."
                )

        return recommendations

    def _analyze_metadata(self, metadata: Dict, severity: str) -> Dict:
        """Analyze drawing metadata for additional insights."""
        insights = {}

        # Age-related insights
        if "age_years" in metadata:
            age = metadata["age_years"]
            if age < 5 and severity in ["high", "medium"]:
                insights[
                    "age_consideration"
                ] = "Young age may contribute to developmental variations."
            elif age > 12 and severity == "high":
                insights[
                    "age_consideration"
                ] = "Older age makes significant deviations more noteworthy."

        # Subject-related insights
        if "subject" in metadata and metadata["subject"]:
            subject = metadata["subject"].lower()
            if "person" in subject or "human" in subject:
                insights[
                    "subject_insight"
                ] = "Human figure drawings can reveal developmental patterns."
            elif "house" in subject:
                insights[
                    "subject_insight"
                ] = "House drawings often reflect spatial understanding."
            elif "tree" in subject:
                insights[
                    "subject_insight"
                ] = "Tree drawings can indicate emotional expression."

        # Expert label insights
        if "expert_label" in metadata and metadata["expert_label"]:
            label = metadata["expert_label"]
            if label == "concern" and severity in ["low", "minimal"]:
                insights[
                    "expert_comparison"
                ] = "AI assessment differs from expert concern - review recommended."
            elif label == "normal" and severity == "high":
                insights[
                    "expert_comparison"
                ] = "AI detected anomalies in expert-labeled normal drawing."

        return insights

    def explain_subject_aware_anomaly(
        self,
        attribution_info: Dict,
        age_group: str,
        subject: str,
        anomaly_score: float,
        normalized_score: float,
    ) -> Dict:
        """
        Generate subject-aware anomaly explanation with detailed attribution.

        Args:
            attribution_info: Dictionary containing attribution details
            age_group: Age group used for comparison
            subject: Subject category of the drawing
            anomaly_score: Raw anomaly score
            normalized_score: Normalized anomaly score

        Returns:
            Dictionary containing detailed subject-aware explanation
        """
        try:
            attribution = attribution_info.get("anomaly_attribution", "unknown")
            visual_score = attribution_info.get("visual_anomaly_score", 0)
            subject_score = attribution_info.get("subject_anomaly_score", 0)

            # Generate attribution-specific explanation
            explanation = {
                "attribution_type": attribution,
                "subject_category": subject,
                "age_group": age_group,
                "component_scores": {
                    "visual": float(visual_score),
                    "subject": float(subject_score),
                    "overall": float(anomaly_score),
                },
                "normalized_score": float(normalized_score),
            }

            # Generate detailed explanation based on attribution
            if attribution == "subject":
                explanation["primary_explanation"] = (
                    f"This drawing shows unusual patterns in how the {subject} is depicted, "
                    f"which differs from typical {subject} representations by children in the {age_group} age group."
                )
                explanation["secondary_explanation"] = (
                    f"The visual characteristics are relatively normal, but the subject-specific "
                    f"patterns deviate from expected norms for {subject} drawings."
                )

            elif attribution == "visual":
                explanation["primary_explanation"] = (
                    f"This drawing shows unusual visual characteristics that are not typical "
                    f"for the {age_group} age group, regardless of the {subject} being drawn."
                )
                explanation["secondary_explanation"] = (
                    f"The subject-specific patterns for {subject} are relatively normal, "
                    f"but the overall visual execution shows deviations."
                )

            elif attribution == "age":
                explanation["primary_explanation"] = (
                    f"This drawing appears more appropriate for a different age group than {age_group}, "
                    f"suggesting potential developmental considerations."
                )
                explanation["secondary_explanation"] = (
                    f"The {subject} representation and visual characteristics would be more "
                    f"typical for children of a different age."
                )

            elif attribution == "both":
                explanation["primary_explanation"] = (
                    f"This drawing shows anomalies in both how the {subject} is depicted "
                    f"and in the overall visual characteristics for the {age_group} age group."
                )
                explanation["secondary_explanation"] = (
                    f"Both subject-specific patterns and general visual execution "
                    f"deviate from expected norms."
                )

            else:
                explanation["primary_explanation"] = (
                    f"This drawing shows deviations from typical patterns for {age_group} "
                    f"children drawing {subject}, but the specific source is unclear."
                )
                explanation[
                    "secondary_explanation"
                ] = f"Further analysis may be needed to determine the primary source of the anomaly."

            # Add contextual information
            explanation["contextual_notes"] = self._generate_contextual_notes(
                attribution, subject, age_group, normalized_score
            )

            # Add recommendations based on attribution
            explanation[
                "attribution_recommendations"
            ] = self._generate_attribution_recommendations(
                attribution, subject, age_group
            )

            return explanation

        except Exception as e:
            logger.error(f"Subject-aware anomaly explanation failed: {str(e)}")
            return {
                "error": str(e),
                "attribution_type": "unknown",
                "subject_category": subject,
                "age_group": age_group,
            }

    def _generate_contextual_notes(
        self, attribution: str, subject: str, age_group: str, normalized_score: float
    ) -> List[str]:
        """Generate contextual notes based on attribution type."""
        notes = []

        if attribution == "subject":
            notes.append(
                f"Subject-specific anomalies in {subject} drawings may indicate:"
            )
            notes.append("- Different understanding or representation of the subject")
            notes.append(
                "- Creative or unconventional approach to depicting the subject"
            )
            notes.append("- Possible developmental variations in subject comprehension")

        elif attribution == "visual":
            notes.append(f"Visual anomalies in {age_group} drawings may indicate:")
            notes.append("- Motor skill variations or developmental differences")
            notes.append("- Different artistic style or approach")
            notes.append("- Possible attention or execution differences")

        elif attribution == "age":
            notes.append(f"Age-related anomalies may indicate:")
            notes.append("- Advanced or delayed developmental patterns")
            notes.append("- Different exposure or experience levels")
            notes.append("- Individual variation in developmental timeline")

        elif attribution == "both":
            notes.append("Combined anomalies may indicate:")
            notes.append("- Multiple developmental factors at play")
            notes.append(
                "- Complex interaction between subject understanding and execution"
            )
            notes.append("- Need for comprehensive evaluation")

        # Add severity-based notes
        if normalized_score > 0.8:
            notes.append(
                "High anomaly score suggests significant deviation requiring attention."
            )
        elif normalized_score > 0.6:
            notes.append(
                "Moderate anomaly score suggests monitoring and potential follow-up."
            )
        else:
            notes.append(
                "Lower anomaly score suggests minor variation within acceptable range."
            )

        return notes

    def _generate_attribution_recommendations(
        self, attribution: str, subject: str, age_group: str
    ) -> List[str]:
        """Generate recommendations based on attribution type."""
        recommendations = []

        if attribution == "subject":
            recommendations.extend(
                [
                    f"Collect additional {subject} drawings to confirm pattern",
                    f"Compare with other subjects to isolate subject-specific effects",
                    f"Consider child's familiarity and experience with {subject}",
                    "Evaluate subject comprehension and representation skills",
                ]
            )

        elif attribution == "visual":
            recommendations.extend(
                [
                    "Assess motor skills and drawing execution abilities",
                    "Compare visual patterns across different subjects",
                    "Consider environmental factors affecting drawing execution",
                    "Evaluate attention and focus during drawing tasks",
                ]
            )

        elif attribution == "age":
            recommendations.extend(
                [
                    "Compare with age-matched peers in similar contexts",
                    "Consider developmental history and milestones",
                    "Evaluate across multiple drawing sessions",
                    "Consider referral for developmental assessment if pattern persists",
                ]
            )

        elif attribution == "both":
            recommendations.extend(
                [
                    "Conduct comprehensive evaluation across multiple domains",
                    "Collect drawings across various subjects and contexts",
                    "Consider multidisciplinary assessment approach",
                    "Monitor patterns over time with regular follow-up",
                ]
            )

        return recommendations


class ImportanceRegionDetector:
    """
    Detects and highlights important regions in drawings based on saliency maps.

    This class provides methods to identify, bound, and describe regions
    that contribute most to anomaly detection decisions.
    """

    def __init__(self):
        """Initialize importance region detector."""
        pass

    def detect_bounding_boxes(
        self,
        saliency_map: np.ndarray,
        threshold: float = 0.7,
        min_region_size: int = 10,
    ) -> List[Dict]:
        """
        Detect bounding boxes around important regions.

        Args:
            saliency_map: 2D saliency map
            threshold: Importance threshold for region detection
            min_region_size: Minimum size for a region to be considered

        Returns:
            List of bounding box dictionaries
        """
        try:
            from scipy import ndimage

            try:
                from skimage import measure
            except ImportError:
                # Fallback to scipy for connected components if skimage not available
                measure = None

            # Threshold the saliency map
            binary_map = saliency_map > threshold

            # If no regions found, lower threshold
            if not np.any(binary_map):
                threshold = np.percentile(saliency_map, 80)
                binary_map = saliency_map > threshold

            # Find connected components
            if measure is not None:
                labeled_regions = measure.label(binary_map)
                region_props = measure.regionprops(
                    labeled_regions, intensity_image=saliency_map
                )
            else:
                # Fallback using scipy
                labeled_regions, num_labels = ndimage.label(binary_map)
                region_props = []
                for label in range(1, num_labels + 1):
                    mask = labeled_regions == label
                    coords = np.where(mask)
                    if len(coords[0]) > 0:
                        # Create a simple region properties object
                        region_prop = type(
                            "RegionProp",
                            (),
                            {
                                "label": label,
                                "area": np.sum(mask),
                                "bbox": (
                                    coords[0].min(),
                                    coords[1].min(),
                                    coords[0].max() + 1,
                                    coords[1].max() + 1,
                                ),
                                "centroid": (np.mean(coords[0]), np.mean(coords[1])),
                            },
                        )()
                        region_props.append(region_prop)

            bounding_boxes = []

            for prop in region_props:
                if prop.area < min_region_size:
                    continue

                # Get bounding box coordinates
                min_row, min_col, max_row, max_col = prop.bbox

                # Calculate region statistics
                region_mask = labeled_regions == prop.label
                region_importance = np.mean(saliency_map[region_mask])

                bbox_info = {
                    "bbox": (min_row, min_col, max_row, max_col),
                    "center": prop.centroid,
                    "area": prop.area,
                    "importance_score": float(region_importance),
                    "max_importance": float(np.max(saliency_map[region_mask])),
                    "aspect_ratio": (
                        (max_col - min_col) / (max_row - min_row)
                        if max_row > min_row
                        else 1.0
                    ),
                }

                bounding_boxes.append(bbox_info)

            # Sort by importance score
            bounding_boxes.sort(key=lambda x: x["importance_score"], reverse=True)

            return bounding_boxes

        except Exception as e:
            logger.error(f"Bounding box detection failed: {str(e)}")
            return []

    def create_region_highlights(
        self,
        original_image: Union[Image.Image, np.ndarray],
        bounding_boxes: List[Dict],
        save_path: Optional[str] = None,
    ) -> Union[Image.Image, str]:
        """
        Create image with highlighted important regions.

        Args:
            original_image: Original drawing image
            bounding_boxes: List of bounding box dictionaries
            save_path: Optional path to save highlighted image

        Returns:
            Highlighted image or path to saved image
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                image = Image.fromarray(original_image)
            else:
                image = original_image.copy()

            # Ensure RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Create drawing context
            draw = ImageDraw.Draw(image)

            # Define colors for different importance levels
            colors = [
                (255, 0, 0),  # Red for highest importance
                (255, 165, 0),  # Orange for high importance
                (255, 255, 0),  # Yellow for medium importance
                (0, 255, 0),  # Green for lower importance
            ]

            # Draw bounding boxes
            for i, bbox_info in enumerate(bounding_boxes[:4]):  # Limit to top 4 regions
                min_row, min_col, max_row, max_col = bbox_info["bbox"]

                # Scale coordinates to image size if needed
                img_height, img_width = image.size[1], image.size[0]
                saliency_height, saliency_width = max_row + 1, max_col + 1

                if saliency_height != img_height or saliency_width != img_width:
                    # Scale coordinates
                    scale_y = img_height / saliency_height
                    scale_x = img_width / saliency_width

                    min_col = int(min_col * scale_x)
                    max_col = int(max_col * scale_x)
                    min_row = int(min_row * scale_y)
                    max_row = int(max_row * scale_y)

                # Choose color based on importance rank
                color = colors[min(i, len(colors) - 1)]

                # Draw rectangle
                draw.rectangle(
                    [(min_col, min_row), (max_col, max_row)], outline=color, width=3
                )

                # Add importance score label
                importance_pct = bbox_info["importance_score"] * 100
                label = f"{importance_pct:.1f}%"

                # Try to load a font, fallback to default
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                # Draw label background
                text_bbox = draw.textbbox((min_col, min_row - 20), label, font=font)
                draw.rectangle(text_bbox, fill=color)

                # Draw label text
                draw.text(
                    (min_col, min_row - 20), label, fill=(255, 255, 255), font=font
                )

            # Save or return image
            if save_path:
                image.save(save_path)
                logger.info(f"Highlighted regions saved to {save_path}")
                return save_path
            else:
                return image

        except Exception as e:
            logger.error(f"Region highlighting failed: {str(e)}")
            if save_path:
                return save_path  # Return path even if failed
            else:
                return original_image  # Return original image if failed


# Update the global functions
def get_visual_feature_identifier() -> VisualFeatureIdentifier:
    """Get a visual feature identifier instance."""
    return VisualFeatureIdentifier()


def get_explanation_generator() -> ExplanationGenerator:
    """Get an explanation generator instance."""
    return ExplanationGenerator()


def get_importance_region_detector() -> ImportanceRegionDetector:
    """Get an importance region detector instance."""
    return ImportanceRegionDetector()


def explain_anomaly(
    drawing_data: Dict, result: Dict, attribution_info: Optional[Dict] = None
) -> Dict:
    """
    Generate comprehensive explanation for anomaly detection with attribution context.

    Args:
        drawing_data: Dictionary containing drawing information
        result: Anomaly analysis result
        attribution_info: Optional subject-aware attribution information

    Returns:
        Dictionary containing comprehensive explanation
    """
    try:
        # Extract information from inputs
        anomaly_score = result.get("anomaly_score", 0)
        normalized_score = result.get("normalized_score", 0)
        age_group = result.get("age_group", "unknown")

        # Generate saliency map
        saliency_generator = get_saliency_generator()

        # Use attribution-aware saliency if attribution info is available
        if attribution_info:
            saliency_result = saliency_generator.generate_attribution_aware_saliency(
                image=drawing_data.get("image"),
                reconstruction_loss=anomaly_score,
                attribution_info=attribution_info,
                method="combined",
            )
        else:
            saliency_result = saliency_generator.generate_saliency_map(
                image=drawing_data.get("image"),
                reconstruction_loss=anomaly_score,
                method="combined",
            )

        # Generate explanation
        explanation_generator = get_explanation_generator()
        explanation = explanation_generator.generate_explanation(
            anomaly_score=anomaly_score,
            normalized_score=normalized_score,
            saliency_result=saliency_result,
            age_group=age_group,
            drawing_metadata=drawing_data.get("metadata"),
            attribution_info=attribution_info,
        )

        # Add subject-aware explanation if attribution info is available
        if attribution_info:
            subject = attribution_info.get("subject_category", "unspecified")
            subject_aware_explanation = (
                explanation_generator.explain_subject_aware_anomaly(
                    attribution_info=attribution_info,
                    age_group=age_group,
                    subject=subject,
                    anomaly_score=anomaly_score,
                    normalized_score=normalized_score,
                )
            )
            explanation["subject_aware_explanation"] = subject_aware_explanation

        return explanation

    except Exception as e:
        logger.error(f"Anomaly explanation failed: {str(e)}")
        return {
            "error": str(e),
            "summary": "Unable to generate explanation due to processing error.",
            "anomaly_score": anomaly_score if "anomaly_score" in locals() else 0,
            "normalized_score": (
                normalized_score if "normalized_score" in locals() else 0
            ),
        }


class SaliencyOverlayGenerator:
    """
    Generates overlay visualizations combining original images with saliency maps.

    This class provides methods to create various types of visualizations that
    help users understand which parts of drawings contribute to anomaly detection.
    """

    def __init__(self):
        """Initialize saliency overlay generator."""
        self.output_dir = Path("static/overlays")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_heatmap_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Create heatmap overlay on original image.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            alpha: Transparency of overlay (0.0 to 1.0)
            colormap: Matplotlib colormap name

        Returns:
            PIL Image with heatmap overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(original_image)
            else:
                pil_image = original_image.copy()

            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = pil_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize saliency map
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            # Create heatmap using matplotlib colormap
            cmap = cm.get_cmap(colormap)
            heatmap = cmap(saliency_normalized)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)

            # Convert original image to numpy
            original_np = np.array(pil_image)

            # Create overlay
            overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)

            # Convert back to PIL Image
            overlay_image = Image.fromarray(overlay)

            return overlay_image

        except Exception as e:
            logger.error(f"Failed to create heatmap overlay: {str(e)}")
            raise SaliencyGenerationError(f"Heatmap overlay creation failed: {str(e)}")

    def create_contour_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        threshold: float = 0.5,
        contour_color: Tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
    ) -> Image.Image:
        """
        Create contour overlay showing important regions.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            threshold: Threshold for contour detection
            contour_color: RGB color for contours
            line_width: Width of contour lines

        Returns:
            PIL Image with contour overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(original_image)
            else:
                pil_image = original_image.copy()

            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = pil_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize and threshold
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            # Create binary mask
            binary_mask = (saliency_normalized > threshold).astype(np.uint8) * 255

            if HAS_OPENCV and cv2 is not None:
                # Find contours using OpenCV
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Convert PIL to OpenCV format for drawing
                image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                # Draw contours
                cv2.drawContours(
                    image_cv, contours, -1, contour_color[::-1], line_width
                )  # BGR format

                # Convert back to PIL
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                overlay_image = Image.fromarray(image_rgb)
            else:
                # Fallback contour detection using PIL
                from PIL import ImageDraw

                overlay_image = pil_image.copy()
                draw = ImageDraw.Draw(overlay_image)

                # Simple contour approximation by finding edges of binary regions
                # This is a simplified approach compared to OpenCV's contour detection
                binary_pil = Image.fromarray(binary_mask, mode="L")

                # Find approximate contours by scanning for edge pixels
                binary_array = np.array(binary_pil)
                height, width = binary_array.shape

                # Simple edge detection for contour approximation
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        if binary_array[y, x] > 0:
                            # Check if this is an edge pixel
                            neighbors = [
                                binary_array[y - 1, x],
                                binary_array[y + 1, x],
                                binary_array[y, x - 1],
                                binary_array[y, x + 1],
                            ]
                            if any(n == 0 for n in neighbors):
                                # This is an edge pixel, draw a small circle
                                draw.ellipse(
                                    [x - 1, y - 1, x + 1, y + 1],
                                    outline=contour_color,
                                    width=line_width,
                                )

            return overlay_image

        except Exception as e:
            logger.error(f"Failed to create contour overlay: {str(e)}")
            raise SaliencyGenerationError(f"Contour overlay creation failed: {str(e)}")

    def create_masked_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        threshold: float = 0.6,
        highlight_color: Tuple[int, int, int] = (255, 255, 0),
        alpha: float = 0.3,
    ) -> Image.Image:
        """
        Create masked overlay highlighting important regions.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            threshold: Threshold for highlighting
            highlight_color: RGB color for highlights
            alpha: Transparency of highlights

        Returns:
            PIL Image with masked overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(original_image)
            else:
                pil_image = original_image.copy()

            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = pil_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize saliency map
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            # Create mask for important regions
            important_mask = saliency_normalized > threshold

            # Convert original image to numpy
            original_np = np.array(pil_image)

            # Create highlight overlay
            highlight_overlay = np.zeros_like(original_np)
            highlight_overlay[important_mask] = highlight_color

            # Blend with original image
            overlay = original_np.copy()
            overlay[important_mask] = (
                alpha * highlight_overlay[important_mask]
                + (1 - alpha) * original_np[important_mask]
            ).astype(np.uint8)

            # Convert back to PIL Image
            overlay_image = Image.fromarray(overlay)

            return overlay_image

        except Exception as e:
            logger.error(f"Failed to create masked overlay: {str(e)}")
            raise SaliencyGenerationError(f"Masked overlay creation failed: {str(e)}")

    def create_side_by_side_comparison(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        overlay_type: str = "heatmap",
        **overlay_kwargs,
    ) -> Image.Image:
        """
        Create side-by-side comparison of original and overlay.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
            **overlay_kwargs: Additional arguments for overlay creation

        Returns:
            PIL Image with side-by-side comparison
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(original_image)
            else:
                pil_image = original_image.copy()

            # Ensure RGB
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Create overlay based on type
            if overlay_type == "heatmap":
                overlay_image = self.create_heatmap_overlay(
                    pil_image, saliency_map, **overlay_kwargs
                )
            elif overlay_type == "contour":
                overlay_image = self.create_contour_overlay(
                    pil_image, saliency_map, **overlay_kwargs
                )
            elif overlay_type == "masked":
                overlay_image = self.create_masked_overlay(
                    pil_image, saliency_map, **overlay_kwargs
                )
            else:
                raise ValueError(f"Unknown overlay type: {overlay_type}")

            # Create side-by-side image
            width, height = pil_image.size
            comparison_image = Image.new("RGB", (width * 2, height), (255, 255, 255))

            # Paste original and overlay
            comparison_image.paste(pil_image, (0, 0))
            comparison_image.paste(overlay_image, (width, 0))

            # Add labels
            draw = ImageDraw.Draw(comparison_image)

            # Try to load a font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Add labels
            draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
            draw.text(
                (width + 10, 10),
                f"Saliency ({overlay_type})",
                fill=(0, 0, 0),
                font=font,
            )

            return comparison_image

        except Exception as e:
            logger.error(f"Failed to create side-by-side comparison: {str(e)}")
            raise SaliencyGenerationError(
                f"Side-by-side comparison creation failed: {str(e)}"
            )

    def save_overlay(
        self, overlay_image: Image.Image, filename: str, format: str = "PNG"
    ) -> str:
        """
        Save overlay image to file.

        Args:
            overlay_image: PIL Image to save
            filename: Filename (without extension)
            format: Image format ('PNG', 'JPEG')

        Returns:
            Path to saved file
        """
        try:
            # Ensure filename has correct extension
            if not filename.lower().endswith(f".{format.lower()}"):
                filename = f"{filename}.{format.lower()}"

            # Create full path
            save_path = self.output_dir / filename

            # Save image
            overlay_image.save(save_path, format=format)

            logger.info(f"Overlay saved to {save_path}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Failed to save overlay: {str(e)}")
            raise SaliencyGenerationError(f"Overlay saving failed: {str(e)}")


class VisualizationExporter:
    """
    Exports visualizations in various formats for different use cases.

    This class provides methods to export saliency visualizations in formats
    suitable for reports, presentations, and interactive applications.
    """

    def __init__(self):
        """Initialize visualization exporter."""
        self.output_dir = Path("static/exports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_comprehensive_report(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_result: Dict,
        explanation: Dict,
        filename: str,
    ) -> str:
        """
        Export comprehensive visualization report.

        Args:
            original_image: Original drawing image
            saliency_result: Result from saliency generation
            explanation: Explanation from explanation generator
            filename: Output filename (without extension)

        Returns:
            Path to exported report
        """
        try:
            # Create overlay generator
            overlay_gen = SaliencyOverlayGenerator()

            # Create different types of overlays
            heatmap_overlay = overlay_gen.create_heatmap_overlay(
                original_image, saliency_result["saliency_map"]
            )
            contour_overlay = overlay_gen.create_contour_overlay(
                original_image, saliency_result["saliency_map"]
            )

            # Create comprehensive visualization
            report_image = self._create_report_layout(
                original_image, heatmap_overlay, contour_overlay, explanation
            )

            # Save report
            report_path = self.output_dir / f"{filename}_report.png"
            report_image.save(report_path, "PNG")

            logger.info(f"Comprehensive report exported to {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to export comprehensive report: {str(e)}")
            raise SaliencyGenerationError(f"Report export failed: {str(e)}")

    def _create_report_layout(
        self,
        original_image: Union[Image.Image, np.ndarray],
        heatmap_overlay: Image.Image,
        contour_overlay: Image.Image,
        explanation: Dict,
    ) -> Image.Image:
        """Create comprehensive report layout."""
        # Convert original to PIL if needed
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            original_pil = Image.fromarray(original_image)
        else:
            original_pil = original_image.copy()

        if original_pil.mode != "RGB":
            original_pil = original_pil.convert("RGB")

        # Resize images to consistent size
        target_size = (300, 300)
        original_resized = original_pil.resize(target_size, Image.Resampling.LANCZOS)
        heatmap_resized = heatmap_overlay.resize(target_size, Image.Resampling.LANCZOS)
        contour_resized = contour_overlay.resize(target_size, Image.Resampling.LANCZOS)

        # Create report canvas
        canvas_width = 950
        canvas_height = 700
        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

        # Paste images
        canvas.paste(original_resized, (25, 50))
        canvas.paste(heatmap_resized, (350, 50))
        canvas.paste(contour_resized, (675, 50))

        # Add text information
        draw = ImageDraw.Draw(canvas)

        # Try to load fonts
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            header_font = ImageFont.truetype("arial.ttf", 16)
            text_font = ImageFont.truetype("arial.ttf", 12)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        # Add title
        draw.text(
            (25, 10),
            "Anomaly Detection Analysis Report",
            fill=(0, 0, 0),
            font=title_font,
        )

        # Add image labels
        draw.text((25, 360), "Original Drawing", fill=(0, 0, 0), font=header_font)
        draw.text((350, 360), "Attention Heatmap", fill=(0, 0, 0), font=header_font)
        draw.text((675, 360), "Important Regions", fill=(0, 0, 0), font=header_font)

        # Add explanation summary
        y_pos = 400
        draw.text((25, y_pos), "Analysis Summary:", fill=(0, 0, 0), font=header_font)
        y_pos += 25

        # Add summary text (wrap long lines)
        summary = explanation.get("summary", "No summary available")
        wrapped_summary = self._wrap_text(summary, 120)
        for line in wrapped_summary[:4]:  # Limit to 4 lines
            draw.text((25, y_pos), line, fill=(0, 0, 0), font=text_font)
            y_pos += 15

        # Add key metrics
        y_pos += 10
        draw.text((25, y_pos), "Key Metrics:", fill=(0, 0, 0), font=header_font)
        y_pos += 20

        severity = explanation.get("severity", "unknown")
        anomaly_score = explanation.get("anomaly_score", 0)
        normalized_score = explanation.get("normalized_score", 0)

        draw.text(
            (25, y_pos), f"Severity: {severity.title()}", fill=(0, 0, 0), font=text_font
        )
        y_pos += 15
        draw.text(
            (25, y_pos),
            f"Anomaly Score: {anomaly_score:.3f}",
            fill=(0, 0, 0),
            font=text_font,
        )
        y_pos += 15
        draw.text(
            (25, y_pos),
            f"Normalized Score: {normalized_score:.3f}",
            fill=(0, 0, 0),
            font=text_font,
        )

        # Add recommendations
        y_pos += 25
        draw.text((25, y_pos), "Recommendations:", fill=(0, 0, 0), font=header_font)
        y_pos += 20

        recommendations = explanation.get("recommendations", [])
        for i, rec in enumerate(recommendations[:3]):  # Limit to 3 recommendations
            wrapped_rec = self._wrap_text(f"• {rec}", 120)
            for line in wrapped_rec[:2]:  # Limit to 2 lines per recommendation
                draw.text((25, y_pos), line, fill=(0, 0, 0), font=text_font)
                y_pos += 15

        return canvas

    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """Wrap text to specified character limit."""
        words = text.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line + " " + word) <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines

    def export_interactive_data(
        self, saliency_result: Dict, explanation: Dict, filename: str
    ) -> str:
        """
        Export data for interactive visualizations.

        Args:
            saliency_result: Result from saliency generation
            explanation: Explanation from explanation generator
            filename: Output filename (without extension)

        Returns:
            Path to exported JSON file
        """
        try:
            import json

            # Prepare data for export
            export_data = {
                "saliency_map": saliency_result["saliency_map"].tolist(),
                "importance_scores": saliency_result["importance_scores"].tolist(),
                "method": saliency_result["method"],
                "map_shape": saliency_result["map_shape"],
                "max_importance": saliency_result["max_importance"],
                "mean_importance": saliency_result["mean_importance"],
                "explanation": {
                    "summary": explanation.get("summary", ""),
                    "severity": explanation.get("severity", ""),
                    "anomaly_score": explanation.get("anomaly_score", 0),
                    "normalized_score": explanation.get("normalized_score", 0),
                    "important_regions": explanation.get("important_regions", []),
                    "recommendations": explanation.get("recommendations", []),
                },
            }

            # Save to JSON file
            json_path = self.output_dir / f"{filename}_data.json"
            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Interactive data exported to {json_path}")
            return str(json_path)

        except Exception as e:
            logger.error(f"Failed to export interactive data: {str(e)}")
            raise SaliencyGenerationError(f"Interactive data export failed: {str(e)}")

    def export_presentation_slides(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_result: Dict,
        explanation: Dict,
        filename: str,
    ) -> List[str]:
        """
        Export presentation-ready slides.

        Args:
            original_image: Original drawing image
            saliency_result: Result from saliency generation
            explanation: Explanation from explanation generator
            filename: Base filename for slides

        Returns:
            List of paths to exported slide images
        """
        try:
            slide_paths = []
            overlay_gen = SaliencyOverlayGenerator()

            # Slide 1: Original image with title
            slide1 = self._create_title_slide(original_image, explanation)
            slide1_path = self.output_dir / f"{filename}_slide1_original.png"
            slide1.save(slide1_path, "PNG")
            slide_paths.append(str(slide1_path))

            # Slide 2: Saliency heatmap
            heatmap_overlay = overlay_gen.create_heatmap_overlay(
                original_image, saliency_result["saliency_map"]
            )
            slide2 = self._create_analysis_slide(
                heatmap_overlay, "Attention Heatmap", explanation
            )
            slide2_path = self.output_dir / f"{filename}_slide2_heatmap.png"
            slide2.save(slide2_path, "PNG")
            slide_paths.append(str(slide2_path))

            # Slide 3: Important regions
            contour_overlay = overlay_gen.create_contour_overlay(
                original_image, saliency_result["saliency_map"]
            )
            slide3 = self._create_analysis_slide(
                contour_overlay, "Important Regions", explanation
            )
            slide3_path = self.output_dir / f"{filename}_slide3_regions.png"
            slide3.save(slide3_path, "PNG")
            slide_paths.append(str(slide3_path))

            # Slide 4: Summary and recommendations
            slide4 = self._create_summary_slide(explanation)
            slide4_path = self.output_dir / f"{filename}_slide4_summary.png"
            slide4.save(slide4_path, "PNG")
            slide_paths.append(str(slide4_path))

            logger.info(f"Presentation slides exported: {len(slide_paths)} slides")
            return slide_paths

        except Exception as e:
            logger.error(f"Failed to export presentation slides: {str(e)}")
            raise SaliencyGenerationError(f"Presentation export failed: {str(e)}")

    def _create_title_slide(
        self, original_image: Union[Image.Image, np.ndarray], explanation: Dict
    ) -> Image.Image:
        """Create title slide with original image."""
        # Convert to PIL if needed
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image.copy()

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Create slide canvas
        slide = Image.new("RGB", (800, 600), (255, 255, 255))

        # Resize and center image
        image_resized = pil_image.resize((400, 400), Image.Resampling.LANCZOS)
        slide.paste(image_resized, (200, 100))

        # Add text
        draw = ImageDraw.Draw(slide)

        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            subtitle_font = ImageFont.truetype("arial.ttf", 18)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()

        # Add title
        draw.text((50, 30), "Drawing Anomaly Analysis", fill=(0, 0, 0), font=title_font)

        # Add severity indicator
        severity = explanation.get("severity", "unknown")
        severity_color = {
            "high": (255, 0, 0),
            "medium": (255, 165, 0),
            "low": (255, 255, 0),
            "minimal": (0, 255, 0),
        }.get(severity, (128, 128, 128))

        draw.text(
            (50, 520),
            f"Severity: {severity.title()}",
            fill=severity_color,
            font=subtitle_font,
        )

        return slide

    def _create_analysis_slide(
        self, analysis_image: Image.Image, title: str, explanation: Dict
    ) -> Image.Image:
        """Create analysis slide with overlay image."""
        # Create slide canvas
        slide = Image.new("RGB", (800, 600), (255, 255, 255))

        # Resize and center analysis image
        image_resized = analysis_image.resize((500, 400), Image.Resampling.LANCZOS)
        slide.paste(image_resized, (150, 80))

        # Add text
        draw = ImageDraw.Draw(slide)

        try:
            title_font = ImageFont.truetype("arial.ttf", 28)
            text_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        # Add title
        draw.text((50, 30), title, fill=(0, 0, 0), font=title_font)

        # Add key metrics
        y_pos = 500
        anomaly_score = explanation.get("anomaly_score", 0)
        normalized_score = explanation.get("normalized_score", 0)

        draw.text(
            (50, y_pos),
            f"Anomaly Score: {anomaly_score:.3f}",
            fill=(0, 0, 0),
            font=text_font,
        )
        draw.text(
            (300, y_pos),
            f"Normalized Score: {normalized_score:.3f}",
            fill=(0, 0, 0),
            font=text_font,
        )

        return slide

    def _create_summary_slide(self, explanation: Dict) -> Image.Image:
        """Create summary slide with recommendations."""
        # Create slide canvas
        slide = Image.new("RGB", (800, 600), (255, 255, 255))
        draw = ImageDraw.Draw(slide)

        try:
            title_font = ImageFont.truetype("arial.ttf", 32)
            header_font = ImageFont.truetype("arial.ttf", 20)
            text_font = ImageFont.truetype("arial.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            text_font = ImageFont.load_default()

        # Add title
        draw.text((50, 30), "Analysis Summary", fill=(0, 0, 0), font=title_font)

        # Add summary
        y_pos = 100
        summary = explanation.get("summary", "No summary available")
        wrapped_summary = self._wrap_text(summary, 80)
        for line in wrapped_summary[:6]:  # Limit to 6 lines
            draw.text((50, y_pos), line, fill=(0, 0, 0), font=text_font)
            y_pos += 20

        # Add recommendations
        y_pos += 40
        draw.text((50, y_pos), "Recommendations:", fill=(0, 0, 0), font=header_font)
        y_pos += 30

        recommendations = explanation.get("recommendations", [])
        for i, rec in enumerate(recommendations[:5]):  # Limit to 5 recommendations
            wrapped_rec = self._wrap_text(f"• {rec}", 80)
            for line in wrapped_rec[:2]:  # Limit to 2 lines per recommendation
                draw.text((50, y_pos), line, fill=(0, 0, 0), font=text_font)
                y_pos += 18
            y_pos += 5  # Extra space between recommendations

        return slide


# Global instances
_saliency_overlay_generator = None
_visualization_exporter = None


def get_saliency_overlay_generator() -> SaliencyOverlayGenerator:
    """Get the global saliency overlay generator instance."""
    global _saliency_overlay_generator
    if _saliency_overlay_generator is None:
        _saliency_overlay_generator = SaliencyOverlayGenerator()
    return _saliency_overlay_generator


def get_visualization_exporter() -> VisualizationExporter:
    """Get the global visualization exporter instance."""
    global _visualization_exporter
    if _visualization_exporter is None:
        _visualization_exporter = VisualizationExporter()
    return _visualization_exporter


class InterpretabilityPipeline:
    """
    Complete interpretability pipeline combining all components.

    This class provides a high-level interface for generating comprehensive
    interpretability results including saliency maps, explanations, and visualizations.
    """

    def __init__(self):
        """Initialize interpretability pipeline."""
        self.saliency_generator = get_saliency_generator()
        self.explanation_generator = get_explanation_generator()
        self.overlay_generator = get_saliency_overlay_generator()
        self.exporter = get_visualization_exporter()
        self.region_detector = get_importance_region_detector()

    def generate_complete_analysis(
        self,
        image: Union[Image.Image, np.ndarray],
        anomaly_score: float,
        normalized_score: float,
        age_group: str,
        drawing_metadata: Optional[Dict] = None,
        export_options: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate complete interpretability analysis.

        Args:
            image: Original drawing image
            anomaly_score: Raw anomaly score from model
            normalized_score: Normalized anomaly score
            age_group: Age group used for comparison
            drawing_metadata: Optional metadata about the drawing
            export_options: Optional export configuration

        Returns:
            Dictionary containing complete analysis results
        """
        try:
            logger.info("Starting complete interpretability analysis")

            # Generate saliency map
            saliency_result = self.saliency_generator.generate_saliency_map(
                image=image,
                reconstruction_loss=anomaly_score,
                method="combined",  # Use combined method for best results
            )

            # Generate explanation
            explanation = self.explanation_generator.generate_explanation(
                anomaly_score=anomaly_score,
                normalized_score=normalized_score,
                saliency_result=saliency_result,
                age_group=age_group,
                drawing_metadata=drawing_metadata,
            )

            # Detect bounding boxes for important regions
            bounding_boxes = self.region_detector.detect_bounding_boxes(
                saliency_map=saliency_result["saliency_map"], threshold=0.6
            )

            # Create visualizations
            visualizations = {}

            # Create different overlay types
            visualizations[
                "heatmap_overlay"
            ] = self.overlay_generator.create_heatmap_overlay(
                image, saliency_result["saliency_map"]
            )

            visualizations[
                "contour_overlay"
            ] = self.overlay_generator.create_contour_overlay(
                image, saliency_result["saliency_map"]
            )

            visualizations[
                "side_by_side"
            ] = self.overlay_generator.create_side_by_side_comparison(
                image, saliency_result["saliency_map"], overlay_type="heatmap"
            )

            # Create highlighted regions
            if bounding_boxes:
                visualizations[
                    "highlighted_regions"
                ] = self.region_detector.create_region_highlights(image, bounding_boxes)

            # Export results if requested
            exported_files = {}
            if export_options:
                base_filename = export_options.get("filename", "analysis")

                if export_options.get("export_report", False):
                    exported_files[
                        "report"
                    ] = self.exporter.export_comprehensive_report(
                        image, saliency_result, explanation, base_filename
                    )

                if export_options.get("export_interactive", False):
                    exported_files[
                        "interactive_data"
                    ] = self.exporter.export_interactive_data(
                        saliency_result, explanation, base_filename
                    )

                if export_options.get("export_slides", False):
                    exported_files["slides"] = self.exporter.export_presentation_slides(
                        image, saliency_result, explanation, base_filename
                    )

            # Compile complete results
            complete_analysis = {
                "saliency_result": saliency_result,
                "explanation": explanation,
                "bounding_boxes": bounding_boxes,
                "visualizations": visualizations,
                "exported_files": exported_files,
                "metadata": {
                    "anomaly_score": anomaly_score,
                    "normalized_score": normalized_score,
                    "age_group": age_group,
                    "drawing_metadata": drawing_metadata,
                    "analysis_timestamp": str(
                        Path().cwd()
                    ),  # Placeholder for timestamp
                },
            }

            logger.info("Complete interpretability analysis finished successfully")
            return complete_analysis

        except Exception as e:
            logger.error(f"Complete interpretability analysis failed: {str(e)}")
            raise InterpretabilityError(f"Complete analysis failed: {str(e)}")

    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate edge density in the image."""
        if HAS_OPENCV and cv2 is not None:
            edges = cv2.Canny(gray_image, 50, 150)
            return float(np.sum(edges > 0) / edges.size)
        else:
            # Fallback edge detection using PIL/numpy
            # Simple gradient-based edge detection
            from PIL import Image, ImageFilter

            # Convert numpy array to PIL Image
            if gray_image.dtype != np.uint8:
                gray_image = (gray_image * 255).astype(np.uint8)

            pil_image = Image.fromarray(gray_image, mode="L")

            # Apply edge detection filter
            edges = pil_image.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            # Calculate edge density
            return float(np.sum(edges_array > 50) / edges_array.size)

    def _calculate_complexity(self, image: np.ndarray) -> float:
        """Calculate overall drawing complexity."""
        if len(image.shape) == 3:
            if HAS_OPENCV and cv2 is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                # Convert RGB to grayscale using PIL fallback
                gray = _rgb_to_grayscale(image)
        else:
            gray = image

        # Use edge density and intensity variation as complexity measures
        edge_density = self._calculate_edge_density(gray)
        intensity_variation = np.std(gray) / 255.0

        # Combine measures
        complexity = 0.6 * edge_density + 0.4 * intensity_variation
        return float(complexity)

    def _determine_severity(self, score: float) -> str:
        """Determine anomaly severity based on score."""
        if score < 0.4:
            return "low"
        elif score < 0.7:
            return "medium"
        else:
            return "high"


# Global pipeline instance
_interpretability_pipeline = None


def get_interpretability_pipeline() -> InterpretabilityPipeline:
    """Get the global interpretability pipeline instance."""
    global _interpretability_pipeline
    if _interpretability_pipeline is None:
        _interpretability_pipeline = InterpretabilityPipeline()
    return _interpretability_pipeline


class SaliencyOverlayGenerator:
    """
    Generates overlay visualizations combining original images with saliency maps.

    This class provides methods to create various types of visual overlays
    that highlight important regions identified by the interpretability engine.
    """

    def __init__(self):
        """Initialize saliency overlay generator."""
        self.output_dir = Path("static/overlays")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_heatmap_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Create heatmap overlay on original image.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            alpha: Transparency of overlay (0.0 to 1.0)
            colormap: Matplotlib colormap name

        Returns:
            PIL Image with heatmap overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = original_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize saliency map
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            # Create heatmap using matplotlib colormap
            cmap = cm.get_cmap(colormap)
            heatmap = cmap(saliency_normalized)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)

            # Convert original image to numpy
            original_np = np.array(original_image)

            # Create overlay
            overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)

            return Image.fromarray(overlay)

        except Exception as e:
            logger.error(f"Failed to create heatmap overlay: {str(e)}")
            return original_image  # Return original if overlay fails

    def create_contour_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        threshold: float = 0.7,
        contour_color: Tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
    ) -> Image.Image:
        """
        Create contour overlay showing important region boundaries.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            threshold: Threshold for contour detection
            contour_color: RGB color for contour lines
            line_width: Width of contour lines

        Returns:
            PIL Image with contour overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = original_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize and threshold
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            binary_mask = (saliency_normalized > threshold).astype(np.uint8) * 255

            if HAS_OPENCV and cv2 is not None:
                # Find contours using OpenCV
                contours, _ = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Draw contours on image
                image_with_contours = np.array(original_image)
                cv2.drawContours(
                    image_with_contours, contours, -1, contour_color, line_width
                )

                return Image.fromarray(image_with_contours)
            else:
                # Fallback contour detection using PIL
                from PIL import ImageDraw

                overlay_image = original_image.copy()
                draw = ImageDraw.Draw(overlay_image)

                # Simple contour approximation by finding edges of binary regions
                binary_array = binary_mask
                height, width = binary_array.shape

                # Simple edge detection for contour approximation
                for y in range(1, height - 1):
                    for x in range(1, width - 1):
                        if binary_array[y, x] > 0:
                            # Check if this is an edge pixel
                            neighbors = [
                                binary_array[y - 1, x],
                                binary_array[y + 1, x],
                                binary_array[y, x - 1],
                                binary_array[y, x + 1],
                            ]
                            if any(n == 0 for n in neighbors):
                                # This is an edge pixel, draw a small circle
                                draw.ellipse(
                                    [x - 1, y - 1, x + 1, y + 1],
                                    outline=contour_color,
                                    width=line_width,
                                )

                return overlay_image

        except Exception as e:
            logger.error(f"Failed to create contour overlay: {str(e)}")
            return original_image  # Return original if overlay fails

    def create_masked_overlay(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        threshold: float = 0.7,
        highlight_color: Tuple[int, int, int] = (255, 255, 0),
        alpha: float = 0.3,
    ) -> Image.Image:
        """
        Create masked overlay highlighting important regions.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            threshold: Threshold for region highlighting
            highlight_color: RGB color for highlighting
            alpha: Transparency of highlight

        Returns:
            PIL Image with masked overlay
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Resize saliency map to match image size
            image_size = original_image.size  # (W, H)
            if HAS_OPENCV and cv2 is not None:
                saliency_resized = cv2.resize(
                    saliency_map, image_size, interpolation=cv2.INTER_CUBIC
                )
            else:
                saliency_resized = _resize_with_pil(saliency_map, image_size)

            # Normalize and create mask
            if saliency_resized.max() > 0:
                saliency_normalized = saliency_resized / saliency_resized.max()
            else:
                saliency_normalized = saliency_resized

            mask = saliency_normalized > threshold

            # Create highlight overlay
            original_np = np.array(original_image)
            highlight_overlay = original_np.copy()

            # Apply highlight color to masked regions
            for i in range(3):  # RGB channels
                highlight_overlay[:, :, i] = np.where(
                    mask,
                    alpha * highlight_color[i] + (1 - alpha) * original_np[:, :, i],
                    original_np[:, :, i],
                )

            return Image.fromarray(highlight_overlay.astype(np.uint8))

        except Exception as e:
            logger.error(f"Failed to create masked overlay: {str(e)}")
            return original_image  # Return original if overlay fails

    def create_side_by_side_comparison(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        overlay_type: str = "heatmap",
    ) -> Image.Image:
        """
        Create side-by-side comparison of original and overlay.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            overlay_type: Type of overlay ('heatmap', 'contour', 'masked')

        Returns:
            PIL Image with side-by-side comparison
        """
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Create overlay based on type
            if overlay_type == "heatmap":
                overlay_image = self.create_heatmap_overlay(
                    original_image, saliency_map
                )
            elif overlay_type == "contour":
                overlay_image = self.create_contour_overlay(
                    original_image, saliency_map
                )
            elif overlay_type == "masked":
                overlay_image = self.create_masked_overlay(original_image, saliency_map)
            else:
                overlay_image = self.create_heatmap_overlay(
                    original_image, saliency_map
                )

            # Create side-by-side image
            width, height = original_image.size
            combined_image = Image.new("RGB", (width * 2 + 10, height), (255, 255, 255))

            # Paste images
            combined_image.paste(original_image, (0, 0))
            combined_image.paste(overlay_image, (width + 10, 0))

            # Add labels
            draw = ImageDraw.Draw(combined_image)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
            draw.text(
                (width + 20, 10),
                f"Saliency ({overlay_type})",
                fill=(0, 0, 0),
                font=font,
            )

            return combined_image

        except Exception as e:
            logger.error(f"Failed to create side-by-side comparison: {str(e)}")
            return original_image  # Return original if comparison fails


class VisualizationExporter:
    """
    Exports interpretability visualizations in various formats.

    This class provides methods to save visualizations in different formats
    and create comprehensive reports combining multiple visualization types.
    """

    def __init__(self):
        """Initialize visualization exporter."""
        self.output_dir = Path("static/overlays")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_visualization_set(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        bounding_boxes: List[Dict],
        explanation: Dict,
        base_filename: str,
        formats: List[str] = ["png"],
    ) -> Dict[str, str]:
        """
        Export a complete set of visualizations.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            bounding_boxes: List of bounding box dictionaries
            explanation: Explanation dictionary
            base_filename: Base filename for exports
            formats: List of formats to export ('png', 'jpg', 'pdf')

        Returns:
            Dictionary mapping visualization types to file paths
        """
        try:
            exported_files = {}
            overlay_generator = SaliencyOverlayGenerator()
            region_detector = ImportanceRegionDetector()

            # Create different visualization types
            visualizations = {
                "heatmap": overlay_generator.create_heatmap_overlay(
                    original_image, saliency_map
                ),
                "contour": overlay_generator.create_contour_overlay(
                    original_image, saliency_map
                ),
                "masked": overlay_generator.create_masked_overlay(
                    original_image, saliency_map
                ),
                "side_by_side": overlay_generator.create_side_by_side_comparison(
                    original_image, saliency_map
                ),
                "bounding_boxes": region_detector.create_region_highlights(
                    original_image, bounding_boxes
                ),
            }

            # Export each visualization in requested formats
            for viz_type, image in visualizations.items():
                if isinstance(image, str):  # If it's a path, load the image
                    if os.path.exists(image):
                        image = Image.open(image)
                    else:
                        continue

                for format_ext in formats:
                    filename = f"{base_filename}_{viz_type}.{format_ext}"
                    filepath = self.output_dir / filename

                    # Save image
                    if format_ext.lower() in ["jpg", "jpeg"]:
                        # Convert to RGB for JPEG
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image.save(filepath, "JPEG", quality=95)
                    elif format_ext.lower() == "png":
                        image.save(filepath, "PNG")
                    elif format_ext.lower() == "pdf":
                        # Convert to RGB for PDF
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image.save(filepath, "PDF")

                    exported_files[f"{viz_type}_{format_ext}"] = str(filepath)

            # Create summary report
            if "png" in formats:
                report_path = self._create_summary_report(
                    original_image,
                    saliency_map,
                    bounding_boxes,
                    explanation,
                    base_filename,
                )
                exported_files["summary_report"] = report_path

            logger.info(f"Exported {len(exported_files)} visualization files")
            return exported_files

        except Exception as e:
            logger.error(f"Failed to export visualization set: {str(e)}")
            return {}

    def _create_summary_report(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        bounding_boxes: List[Dict],
        explanation: Dict,
        base_filename: str,
    ) -> str:
        """Create a comprehensive summary report image."""
        try:
            # Convert image to PIL if needed
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            # Ensure RGB
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Create visualizations
            overlay_generator = SaliencyOverlayGenerator()
            region_detector = ImportanceRegionDetector()

            heatmap = overlay_generator.create_heatmap_overlay(
                original_image, saliency_map
            )
            bbox_image = region_detector.create_region_highlights(
                original_image, bounding_boxes
            )

            if isinstance(bbox_image, str):
                if os.path.exists(bbox_image):
                    bbox_image = Image.open(bbox_image)
                else:
                    bbox_image = original_image

            # Create layout
            img_width, img_height = original_image.size

            # Calculate layout dimensions
            margin = 20
            text_height = 200
            total_width = img_width * 2 + margin * 3
            total_height = img_height * 2 + text_height + margin * 4

            # Create summary image
            summary_image = Image.new(
                "RGB", (total_width, total_height), (255, 255, 255)
            )

            # Paste images
            summary_image.paste(original_image, (margin, margin))
            summary_image.paste(heatmap, (img_width + margin * 2, margin))
            summary_image.paste(bbox_image, (margin, img_height + margin * 2))

            # Add text information
            draw = ImageDraw.Draw(summary_image)
            try:
                title_font = ImageFont.truetype("arial.ttf", 20)
                text_font = ImageFont.truetype("arial.ttf", 14)
            except:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()

            # Add labels
            draw.text(
                (margin, margin - 25),
                "Original Drawing",
                fill=(0, 0, 0),
                font=title_font,
            )
            draw.text(
                (img_width + margin * 2, margin - 25),
                "Saliency Heatmap",
                fill=(0, 0, 0),
                font=title_font,
            )
            draw.text(
                (margin, img_height + margin * 2 - 25),
                "Important Regions",
                fill=(0, 0, 0),
                font=title_font,
            )

            # Add explanation text
            text_y = img_height * 2 + margin * 3

            # Title
            draw.text(
                (margin, text_y), "Analysis Summary", fill=(0, 0, 0), font=title_font
            )
            text_y += 30

            # Summary
            summary_text = explanation.get("summary", "No summary available")
            # Wrap text
            wrapped_summary = self._wrap_text(summary_text, 80)
            for line in wrapped_summary[:4]:  # Limit to 4 lines
                draw.text((margin, text_y), line, fill=(0, 0, 0), font=text_font)
                text_y += 18

            # Severity and scores
            text_y += 10
            severity = explanation.get("severity", "unknown")
            anomaly_score = explanation.get("anomaly_score", 0)
            normalized_score = explanation.get("normalized_score", 0)

            draw.text(
                (margin, text_y),
                f"Severity: {severity.upper()}",
                fill=(0, 0, 0),
                font=text_font,
            )
            text_y += 18
            draw.text(
                (margin, text_y),
                f"Anomaly Score: {anomaly_score:.3f}",
                fill=(0, 0, 0),
                font=text_font,
            )
            text_y += 18
            draw.text(
                (margin, text_y),
                f"Normalized Score: {normalized_score:.3f}",
                fill=(0, 0, 0),
                font=text_font,
            )

            # Important regions info
            text_x = img_width + margin * 2
            text_y = img_height * 2 + margin * 3 + 30

            num_regions = len(bounding_boxes)
            draw.text(
                (text_x, text_y),
                f"Important Regions: {num_regions}",
                fill=(0, 0, 0),
                font=text_font,
            )
            text_y += 18

            for i, bbox in enumerate(bounding_boxes[:3]):  # Show top 3 regions
                importance = bbox.get("importance_score", 0) * 100
                draw.text(
                    (text_x, text_y),
                    f"Region {i+1}: {importance:.1f}% importance",
                    fill=(0, 0, 0),
                    font=text_font,
                )
                text_y += 18

            # Save summary report
            report_filename = f"{base_filename}_summary_report.png"
            report_path = self.output_dir / report_filename
            summary_image.save(report_path, "PNG")

            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to create summary report: {str(e)}")
            return ""

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def create_interactive_html_report(
        self,
        original_image: Union[Image.Image, np.ndarray],
        saliency_map: np.ndarray,
        explanation: Dict,
        base_filename: str,
    ) -> str:
        """
        Create an interactive HTML report with visualizations.

        Args:
            original_image: Original drawing image
            saliency_map: 2D saliency map
            explanation: Explanation dictionary
            base_filename: Base filename for the report

        Returns:
            Path to the HTML report file
        """
        try:
            # Create visualizations and save as base64
            overlay_generator = SaliencyOverlayGenerator()

            # Convert original image to base64
            if isinstance(original_image, np.ndarray):
                if original_image.dtype != np.uint8:
                    original_image = (original_image * 255).astype(np.uint8)
                original_image = Image.fromarray(original_image)

            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            original_b64 = self._image_to_base64(original_image)

            # Create heatmap overlay
            heatmap = overlay_generator.create_heatmap_overlay(
                original_image, saliency_map
            )
            heatmap_b64 = self._image_to_base64(heatmap)

            # Create HTML content
            html_content = self._generate_html_template(
                original_b64, heatmap_b64, explanation, base_filename
            )

            # Save HTML file
            html_filename = f"{base_filename}_interactive_report.html"
            html_path = self.output_dir / html_filename

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"Created interactive HTML report: {html_path}")
            return str(html_path)

        except Exception as e:
            logger.error(f"Failed to create HTML report: {str(e)}")
            return ""

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        import base64
        from io import BytesIO

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _generate_html_template(
        self, original_b64: str, heatmap_b64: str, explanation: Dict, title: str
    ) -> str:
        """Generate HTML template for interactive report."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drawing Analysis Report - {title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #eee;
            padding-bottom: 20px;
        }}
        .image-container {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .image-box {{
            text-align: center;
            margin: 10px;
        }}
        .image-box img {{
            max-width: 400px;
            max-height: 400px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .image-box h3 {{
            margin-top: 10px;
            color: #333;
        }}
        .analysis-section {{
            margin-bottom: 30px;
        }}
        .analysis-section h2 {{
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .severity {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            margin: 10px 0;
        }}
        .severity.high {{ background-color: #e74c3c; }}
        .severity.medium {{ background-color: #f39c12; }}
        .severity.low {{ background-color: #f1c40f; color: #333; }}
        .severity.minimal {{ background-color: #27ae60; }}
        .metrics {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
            flex-wrap: wrap;
        }}
        .metric {{
            text-align: center;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 8px;
            margin: 5px;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .recommendations {{
            background-color: #e8f6f3;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #27ae60;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .technical-details {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .technical-details h3 {{
            margin-top: 0;
            color: #495057;
        }}
        @media (max-width: 768px) {{
            .image-container {{
                flex-direction: column;
                align-items: center;
            }}
            .metrics {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Children's Drawing Analysis Report</h1>
            <p><strong>Analysis ID:</strong> {title}</p>
        </div>
        
        <div class="image-container">
            <div class="image-box">
                <img src="{original_b64}" alt="Original Drawing">
                <h3>Original Drawing</h3>
            </div>
            <div class="image-box">
                <img src="{heatmap_b64}" alt="Saliency Heatmap">
                <h3>Attention Heatmap</h3>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>Analysis Summary</h2>
            <div class="severity {explanation.get('severity', 'unknown')}">{explanation.get('severity', 'Unknown').upper()}</div>
            <p>{explanation.get('summary', 'No summary available.')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{explanation.get('anomaly_score', 0):.3f}</div>
                <div class="metric-label">Anomaly Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{explanation.get('normalized_score', 0):.3f}</div>
                <div class="metric-label">Normalized Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(explanation.get('important_regions', []))}</div>
                <div class="metric-label">Important Regions</div>
            </div>
        </div>
        
        <div class="analysis-section">
            <h2>Detailed Analysis</h2>
            <ul>
                {''.join(f'<li>{point}</li>' for point in explanation.get('detailed_analysis', []))}
            </ul>
        </div>
        
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in explanation.get('recommendations', []))}
            </ul>
        </div>
        
        <div class="technical-details">
            <h3>Technical Details</h3>
            <p><strong>Age Group:</strong> {explanation.get('age_group', 'Unknown')}</p>
            <p><strong>Saliency Method:</strong> {explanation.get('technical_details', {}).get('saliency_method', 'Unknown')}</p>
            <p><strong>Max Importance:</strong> {explanation.get('technical_details', {}).get('max_importance', 0):.3f}</p>
            <p><strong>Mean Importance:</strong> {explanation.get('technical_details', {}).get('mean_importance', 0):.3f}</p>
        </div>
    </div>
</body>
</html>
"""


# Update the global functions
def get_saliency_overlay_generator() -> SaliencyOverlayGenerator:
    """Get a saliency overlay generator instance."""
    return SaliencyOverlayGenerator()


def get_visualization_exporter() -> VisualizationExporter:
    """Get a visualization exporter instance."""
    return VisualizationExporter()
