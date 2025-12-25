# Interpretability Engine Contract

## Overview
Service contract for Interpretability Engine (service)

**Source File**: `app/services/interpretability_engine.py`

## Interface Specification

### Classes

#### InterpretabilityError

Base exception for interpretability engine errors.

**Inherits from**: Exception

#### SaliencyGenerationError

Raised when saliency map generation fails.

**Inherits from**: InterpretabilityError

#### AttentionVisualizationError

Raised when attention visualization fails.

**Inherits from**: InterpretabilityError

#### AttentionRollout

Attention rollout technique for Vision Transformers.

This class implements the attention rollout method to compute attention
maps that show which patches the model focuses on for its predictions.

#### GradCAMViT

Gradient-based Class Activation Mapping for Vision Transformers.

This class implements Grad-CAM specifically adapted for Vision Transformers
to generate saliency maps showing important regions for anomaly detection.

#### PatchImportanceScorer

Patch-level importance scoring for Vision Transformers.

This class provides methods to compute importance scores for individual
patches in the input image based on various techniques.

#### SaliencyMapGenerator

Main class for generating saliency maps from Vision Transformer models.

This class combines various techniques to create comprehensive saliency maps
that highlight important regions in children's drawings for anomaly detection.

#### VisualFeatureIdentifier

Identifies and describes visual features in children's drawings.

This class analyzes saliency maps and original images to identify
specific visual features that contribute to anomaly detection.

#### ExplanationGenerator

Generates human-readable explanations for anomaly detection results.

This class combines saliency maps, visual feature analysis, and domain knowledge
to create comprehensive explanations for why a drawing was flagged as anomalous.

#### ImportanceRegionDetector

Detects and highlights important regions in drawings based on saliency maps.

This class provides methods to identify, bound, and describe regions
that contribute most to anomaly detection decisions.

#### SaliencyOverlayGenerator

Generates overlay visualizations combining original images with saliency maps.

This class provides methods to create various types of visualizations that
help users understand which parts of drawings contribute to anomaly detection.

#### VisualizationExporter

Exports visualizations in various formats for different use cases.

This class provides methods to export saliency visualizations in formats
suitable for reports, presentations, and interactive applications.

#### InterpretabilityPipeline

Complete interpretability pipeline combining all components.

This class provides a high-level interface for generating comprehensive
interpretability results including saliency maps, explanations, and visualizations.

#### SaliencyOverlayGenerator

Generates overlay visualizations combining original images with saliency maps.

This class provides methods to create various types of visual overlays
that highlight important regions identified by the interpretability engine.

#### VisualizationExporter

Exports interpretability visualizations in various formats.

This class provides methods to save visualizations in different formats
and create comprehensive reports combining multiple visualization types.

## Methods

### generate_rollout

Generate attention rollout for input tensor.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    start_layer: Layer to start rollout from
    
Returns:
    Attention rollout tensor [H_patches, W_patches]

**Signature**: `generate_rollout(input_tensor: torch.Tensor, start_layer: int) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `input_tensor` | `torch.Tensor` | Parameter description |
| `start_layer` | `int` | Parameter description |

**Returns**: `torch.Tensor`

### generate_cam

Generate Class Activation Map using gradients.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    reconstruction_loss: Reconstruction loss to compute gradients for
    
Returns:
    CAM tensor [H_patches, W_patches]

**Signature**: `generate_cam(input_tensor: torch.Tensor, reconstruction_loss: float) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `input_tensor` | `torch.Tensor` | Parameter description |
| `reconstruction_loss` | `float` | Parameter description |

**Returns**: `torch.Tensor`

### compute_attention_importance

Compute patch importance using attention mechanisms.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    method: Method to use ("rollout" or "last_layer")
    
Returns:
    Importance scores for each patch

**Signature**: `compute_attention_importance(input_tensor: torch.Tensor, method: str) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `input_tensor` | `torch.Tensor` | Parameter description |
| `method` | `str` | Parameter description |

**Returns**: `torch.Tensor`

### compute_gradient_importance

Compute patch importance using gradient-based methods.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    reconstruction_loss: Reconstruction loss for gradient computation
    
Returns:
    Importance scores for each patch

**Signature**: `compute_gradient_importance(input_tensor: torch.Tensor, reconstruction_loss: float) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `input_tensor` | `torch.Tensor` | Parameter description |
| `reconstruction_loss` | `float` | Parameter description |

**Returns**: `torch.Tensor`

### reshape_to_spatial

Reshape 1D importance scores to 2D spatial map.

Args:
    importance_scores: 1D tensor of patch importance scores
    image_size: Original image size (H, W)
    
Returns:
    2D spatial importance map

**Signature**: `reshape_to_spatial(importance_scores: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `importance_scores` | `torch.Tensor` | Parameter description |
| `image_size` | `Tuple[int, int]` | Parameter description |

**Returns**: `torch.Tensor`

### generate_saliency_map

Generate saliency map for an image.

Args:
    image: Input image (PIL Image or numpy array)
    reconstruction_loss: Reconstruction loss from autoencoder
    method: Saliency method ("attention_rollout", "grad_cam", "combined")
    save_path: Optional path to save the saliency map
    
Returns:
    Dictionary containing saliency map and metadata

**Signature**: `generate_saliency_map(image: Union[<ast.Attribute object at 0x110447590>, <ast.Attribute object at 0x110447150>], reconstruction_loss: float, method: str, save_path: Optional[str]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image` | `Union[<ast.Attribute object at 0x110447590>, <ast.Attribute object at 0x110447150>]` | Parameter description |
| `reconstruction_loss` | `float` | Parameter description |
| `method` | `str` | Parameter description |
| `save_path` | `Optional[str]` | Parameter description |

**Returns**: `Dict`

### identify_important_regions

Identify important regions in the saliency map.

Args:
    saliency_map: 2D saliency map
    threshold: Threshold for considering regions important
    
Returns:
    List of important region descriptions

**Signature**: `identify_important_regions(saliency_map: np.ndarray, threshold: float) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |

**Returns**: `List[Dict]`

### analyze_drawing_content

Analyze drawing content to identify likely visual features.

Args:
    image: Original drawing image
    important_regions: List of important regions from saliency analysis
    
Returns:
    Dictionary containing content analysis

**Signature**: `analyze_drawing_content(image: Union[<ast.Attribute object at 0x11055e850>, <ast.Attribute object at 0x11055e750>], important_regions: List[Dict]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image` | `Union[<ast.Attribute object at 0x11055e850>, <ast.Attribute object at 0x11055e750>]` | Parameter description |
| `important_regions` | `List[Dict]` | Parameter description |

**Returns**: `Dict`

### generate_explanation

Generate comprehensive explanation for anomaly detection result.

Args:
    anomaly_score: Raw anomaly score
    normalized_score: Normalized anomaly score
    saliency_result: Result from saliency map generation
    age_group: Age group used for comparison
    drawing_metadata: Optional metadata about the drawing
    
Returns:
    Dictionary containing structured explanation

**Signature**: `generate_explanation(anomaly_score: float, normalized_score: float, saliency_result: Dict, age_group: str, drawing_metadata: Optional[Dict]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `anomaly_score` | `float` | Parameter description |
| `normalized_score` | `float` | Parameter description |
| `saliency_result` | `Dict` | Parameter description |
| `age_group` | `str` | Parameter description |
| `drawing_metadata` | `Optional[Dict]` | Parameter description |

**Returns**: `Dict`

### detect_bounding_boxes

Detect bounding boxes around important regions.

Args:
    saliency_map: 2D saliency map
    threshold: Importance threshold for region detection
    min_region_size: Minimum size for a region to be considered
    
Returns:
    List of bounding box dictionaries

**Signature**: `detect_bounding_boxes(saliency_map: np.ndarray, threshold: float, min_region_size: int) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |
| `min_region_size` | `int` | Parameter description |

**Returns**: `List[Dict]`

### create_region_highlights

Create image with highlighted important regions.

Args:
    original_image: Original drawing image
    bounding_boxes: List of bounding box dictionaries
    save_path: Optional path to save highlighted image
    
Returns:
    Highlighted image or path to saved image

**Signature**: `create_region_highlights(original_image: Union[<ast.Attribute object at 0x11060a990>, <ast.Attribute object at 0x11060aa90>], bounding_boxes: List[Dict], save_path: Optional[str]) -> Union[<ast.Attribute object at 0x11062e310>, str]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x11060a990>, <ast.Attribute object at 0x11060aa90>]` | Parameter description |
| `bounding_boxes` | `List[Dict]` | Parameter description |
| `save_path` | `Optional[str]` | Parameter description |

**Returns**: `Union[<ast.Attribute object at 0x11062e310>, str]`

### create_heatmap_overlay

Create heatmap overlay on original image.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    alpha: Transparency of overlay (0.0 to 1.0)
    colormap: Matplotlib colormap name
    
Returns:
    PIL Image with heatmap overlay

**Signature**: `create_heatmap_overlay(original_image: Union[<ast.Attribute object at 0x110634d10>, <ast.Attribute object at 0x110634e10>], saliency_map: np.ndarray, alpha: float, colormap: str) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110634d10>, <ast.Attribute object at 0x110634e10>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `alpha` | `float` | Parameter description |
| `colormap` | `str` | Parameter description |

**Returns**: `Image.Image`

### create_contour_overlay

Create contour overlay showing important regions.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for contour detection
    contour_color: RGB color for contours
    line_width: Width of contour lines
    
Returns:
    PIL Image with contour overlay

**Signature**: `create_contour_overlay(original_image: Union[<ast.Attribute object at 0x11063f890>, <ast.Attribute object at 0x11063f990>], saliency_map: np.ndarray, threshold: float, contour_color: Tuple[int, int, int], line_width: int) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x11063f890>, <ast.Attribute object at 0x11063f990>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |
| `contour_color` | `Tuple[int, int, int]` | Parameter description |
| `line_width` | `int` | Parameter description |

**Returns**: `Image.Image`

### create_masked_overlay

Create masked overlay highlighting important regions.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for highlighting
    highlight_color: RGB color for highlights
    alpha: Transparency of highlights
    
Returns:
    PIL Image with masked overlay

**Signature**: `create_masked_overlay(original_image: Union[<ast.Attribute object at 0x110656f90>, <ast.Attribute object at 0x110657090>], saliency_map: np.ndarray, threshold: float, highlight_color: Tuple[int, int, int], alpha: float) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110656f90>, <ast.Attribute object at 0x110657090>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |
| `highlight_color` | `Tuple[int, int, int]` | Parameter description |
| `alpha` | `float` | Parameter description |

**Returns**: `Image.Image`

### create_side_by_side_comparison

Create side-by-side comparison of original and overlay.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
    **overlay_kwargs: Additional arguments for overlay creation
    
Returns:
    PIL Image with side-by-side comparison

**Signature**: `create_side_by_side_comparison(original_image: Union[<ast.Attribute object at 0x11066e1d0>, <ast.Attribute object at 0x11066e2d0>], saliency_map: np.ndarray, overlay_type: str) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x11066e1d0>, <ast.Attribute object at 0x11066e2d0>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `overlay_type` | `str` | Parameter description |

**Returns**: `Image.Image`

### save_overlay

Save overlay image to file.

Args:
    overlay_image: PIL Image to save
    filename: Filename (without extension)
    format: Image format ('PNG', 'JPEG')
    
Returns:
    Path to saved file

**Signature**: `save_overlay(overlay_image: Image.Image, filename: str, format: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `overlay_image` | `Image.Image` | Parameter description |
| `filename` | `str` | Parameter description |
| `format` | `str` | Parameter description |

**Returns**: `str`

### export_comprehensive_report

Export comprehensive visualization report.

Args:
    original_image: Original drawing image
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Output filename (without extension)
    
Returns:
    Path to exported report

**Signature**: `export_comprehensive_report(original_image: Union[<ast.Attribute object at 0x11068b490>, <ast.Attribute object at 0x11068b590>], saliency_result: Dict, explanation: Dict, filename: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x11068b490>, <ast.Attribute object at 0x11068b590>]` | Parameter description |
| `saliency_result` | `Dict` | Parameter description |
| `explanation` | `Dict` | Parameter description |
| `filename` | `str` | Parameter description |

**Returns**: `str`

### export_interactive_data

Export data for interactive visualizations.

Args:
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Output filename (without extension)
    
Returns:
    Path to exported JSON file

**Signature**: `export_interactive_data(saliency_result: Dict, explanation: Dict, filename: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `saliency_result` | `Dict` | Parameter description |
| `explanation` | `Dict` | Parameter description |
| `filename` | `str` | Parameter description |

**Returns**: `str`

### export_presentation_slides

Export presentation-ready slides.

Args:
    original_image: Original drawing image
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Base filename for slides
    
Returns:
    List of paths to exported slide images

**Signature**: `export_presentation_slides(original_image: Union[<ast.Attribute object at 0x1106e11d0>, <ast.Attribute object at 0x1106e12d0>], saliency_result: Dict, explanation: Dict, filename: str) -> List[str]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x1106e11d0>, <ast.Attribute object at 0x1106e12d0>]` | Parameter description |
| `saliency_result` | `Dict` | Parameter description |
| `explanation` | `Dict` | Parameter description |
| `filename` | `str` | Parameter description |

**Returns**: `List[str]`

### generate_complete_analysis

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

**Signature**: `generate_complete_analysis(image: Union[<ast.Attribute object at 0x11073ca10>, <ast.Attribute object at 0x11073cb10>], anomaly_score: float, normalized_score: float, age_group: str, drawing_metadata: Optional[Dict], export_options: Optional[Dict]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image` | `Union[<ast.Attribute object at 0x11073ca10>, <ast.Attribute object at 0x11073cb10>]` | Parameter description |
| `anomaly_score` | `float` | Parameter description |
| `normalized_score` | `float` | Parameter description |
| `age_group` | `str` | Parameter description |
| `drawing_metadata` | `Optional[Dict]` | Parameter description |
| `export_options` | `Optional[Dict]` | Parameter description |

**Returns**: `Dict`

### create_heatmap_overlay

Create heatmap overlay on original image.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    alpha: Transparency of overlay (0.0 to 1.0)
    colormap: Matplotlib colormap name
    
Returns:
    PIL Image with heatmap overlay

**Signature**: `create_heatmap_overlay(original_image: Union[<ast.Attribute object at 0x1107641d0>, <ast.Attribute object at 0x1107642d0>], saliency_map: np.ndarray, alpha: float, colormap: str) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x1107641d0>, <ast.Attribute object at 0x1107642d0>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `alpha` | `float` | Parameter description |
| `colormap` | `str` | Parameter description |

**Returns**: `Image.Image`

### create_contour_overlay

Create contour overlay showing important region boundaries with enhanced fallback support.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for contour detection
    contour_color: RGB color for contours
    line_width: Width of contour lines
    
Returns:
    PIL Image with contour overlay

Implementation Notes:
    - Supports both OpenCV and PIL-based contour detection
    - Automatic fallback ensures consistent functionality across environments
    - Visual output remains consistent regardless of backend implementation

**Signature**: `create_contour_overlay(original_image: Union[<ast.Attribute object at 0x110772390>, <ast.Attribute object at 0x110772490>], saliency_map: np.ndarray, threshold: float, contour_color: Tuple[int, int, int], line_width: int) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110772390>, <ast.Attribute object at 0x110772490>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |
| `contour_color` | `Tuple[int, int, int]` | Parameter description |
| `line_width` | `int` | Parameter description |

**Returns**: `Image.Image`

### create_masked_overlay

Create masked overlay highlighting important regions.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for region highlighting
    highlight_color: RGB color for highlighting
    alpha: Transparency of highlight
    
Returns:
    PIL Image with masked overlay

**Signature**: `create_masked_overlay(original_image: Union[<ast.Attribute object at 0x110788710>, <ast.Attribute object at 0x110788810>], saliency_map: np.ndarray, threshold: float, highlight_color: Tuple[int, int, int], alpha: float) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110788710>, <ast.Attribute object at 0x110788810>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `threshold` | `float` | Parameter description |
| `highlight_color` | `Tuple[int, int, int]` | Parameter description |
| `alpha` | `float` | Parameter description |

**Returns**: `Image.Image`

### create_side_by_side_comparison

Create side-by-side comparison of original and overlay.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
    
Returns:
    PIL Image with side-by-side comparison

**Signature**: `create_side_by_side_comparison(original_image: Union[<ast.Attribute object at 0x110793590>, <ast.Attribute object at 0x110793690>], saliency_map: np.ndarray, overlay_type: str) -> Image.Image`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110793590>, <ast.Attribute object at 0x110793690>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `overlay_type` | `str` | Parameter description |

**Returns**: `Image.Image`

### export_visualization_set

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

**Signature**: `export_visualization_set(original_image: Union[<ast.Attribute object at 0x1107b4a50>, <ast.Attribute object at 0x1107b4b50>], saliency_map: np.ndarray, bounding_boxes: List[Dict], explanation: Dict, base_filename: str, formats: List[str]) -> Dict[str, str]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x1107b4a50>, <ast.Attribute object at 0x1107b4b50>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `bounding_boxes` | `List[Dict]` | Parameter description |
| `explanation` | `Dict` | Parameter description |
| `base_filename` | `str` | Parameter description |
| `formats` | `List[str]` | Parameter description |

**Returns**: `Dict[str, str]`

### create_interactive_html_report

Create an interactive HTML report with visualizations.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    explanation: Explanation dictionary
    base_filename: Base filename for the report
    
Returns:
    Path to the HTML report file

**Signature**: `create_interactive_html_report(original_image: Union[<ast.Attribute object at 0x110812090>, <ast.Attribute object at 0x110812190>], saliency_map: np.ndarray, explanation: Dict, base_filename: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `original_image` | `Union[<ast.Attribute object at 0x110812090>, <ast.Attribute object at 0x110812190>]` | Parameter description |
| `saliency_map` | `np.ndarray` | Parameter description |
| `explanation` | `Dict` | Parameter description |
| `base_filename` | `str` | Parameter description |

**Returns**: `str`

## Dependencies

- `app.services.embedding_service.get_embedding_service`
- `app.services.embedding_service.VisionTransformerWrapper`

## Defined Interfaces

### AttentionRolloutInterface

**Type**: Protocol
**Implemented by**: AttentionRollout

**Methods**:

- `generate_rollout(input_tensor: torch.Tensor, start_layer: int) -> torch.Tensor`

### GradCAMViTInterface

**Type**: Protocol
**Implemented by**: GradCAMViT

**Methods**:

- `generate_cam(input_tensor: torch.Tensor, reconstruction_loss: float) -> torch.Tensor`

### PatchImportanceScorerInterface

**Type**: Protocol
**Implemented by**: PatchImportanceScorer

**Methods**:

- `compute_attention_importance(input_tensor: torch.Tensor, method: str) -> torch.Tensor`
- `compute_gradient_importance(input_tensor: torch.Tensor, reconstruction_loss: float) -> torch.Tensor`
- `reshape_to_spatial(importance_scores: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor`

### SaliencyMapGeneratorInterface

**Type**: Protocol
**Implemented by**: SaliencyMapGenerator

**Methods**:

- `generate_saliency_map(image: Union[<ast.Attribute object at 0x110447590>, <ast.Attribute object at 0x110447150>], reconstruction_loss: float, method: str, save_path: Optional[str]) -> Dict`

### VisualFeatureIdentifierInterface

**Type**: Protocol
**Implemented by**: VisualFeatureIdentifier

**Methods**:

- `identify_important_regions(saliency_map: np.ndarray, threshold: float) -> List[Dict]`
- `analyze_drawing_content(image: Union[<ast.Attribute object at 0x11055e850>, <ast.Attribute object at 0x11055e750>], important_regions: List[Dict]) -> Dict`

### ExplanationGeneratorInterface

**Type**: Protocol
**Implemented by**: ExplanationGenerator

**Methods**:

- `generate_explanation(anomaly_score: float, normalized_score: float, saliency_result: Dict, age_group: str, drawing_metadata: Optional[Dict]) -> Dict`

### ImportanceRegionDetectorInterface

**Type**: Protocol
**Implemented by**: ImportanceRegionDetector

**Methods**:

- `detect_bounding_boxes(saliency_map: np.ndarray, threshold: float, min_region_size: int) -> List[Dict]`
- `create_region_highlights(original_image: Union[<ast.Attribute object at 0x11060a990>, <ast.Attribute object at 0x11060aa90>], bounding_boxes: List[Dict], save_path: Optional[str]) -> Union[<ast.Attribute object at 0x11062e310>, str]`

### SaliencyOverlayGeneratorInterface

**Type**: Protocol
**Implemented by**: SaliencyOverlayGenerator

**Methods**:

- `create_heatmap_overlay(original_image: Union[<ast.Attribute object at 0x110634d10>, <ast.Attribute object at 0x110634e10>], saliency_map: np.ndarray, alpha: float, colormap: str) -> Image.Image`
- `create_contour_overlay(original_image: Union[<ast.Attribute object at 0x11063f890>, <ast.Attribute object at 0x11063f990>], saliency_map: np.ndarray, threshold: float, contour_color: Tuple[int, int, int], line_width: int) -> Image.Image`
- `create_masked_overlay(original_image: Union[<ast.Attribute object at 0x110656f90>, <ast.Attribute object at 0x110657090>], saliency_map: np.ndarray, threshold: float, highlight_color: Tuple[int, int, int], alpha: float) -> Image.Image`
- `create_side_by_side_comparison(original_image: Union[<ast.Attribute object at 0x11066e1d0>, <ast.Attribute object at 0x11066e2d0>], saliency_map: np.ndarray, overlay_type: str) -> Image.Image`
- `save_overlay(overlay_image: Image.Image, filename: str, format: str) -> str`

### VisualizationExporterInterface

**Type**: Protocol
**Implemented by**: VisualizationExporter

**Methods**:

- `export_comprehensive_report(original_image: Union[<ast.Attribute object at 0x11068b490>, <ast.Attribute object at 0x11068b590>], saliency_result: Dict, explanation: Dict, filename: str) -> str`
- `export_interactive_data(saliency_result: Dict, explanation: Dict, filename: str) -> str`
- `export_presentation_slides(original_image: Union[<ast.Attribute object at 0x1106e11d0>, <ast.Attribute object at 0x1106e12d0>], saliency_result: Dict, explanation: Dict, filename: str) -> List[str]`

### InterpretabilityPipelineInterface

**Type**: Protocol
**Implemented by**: InterpretabilityPipeline

**Methods**:

- `generate_complete_analysis(image: Union[<ast.Attribute object at 0x11073ca10>, <ast.Attribute object at 0x11073cb10>], anomaly_score: float, normalized_score: float, age_group: str, drawing_metadata: Optional[Dict], export_options: Optional[Dict]) -> Dict`

### SaliencyOverlayGeneratorInterface

**Type**: Protocol
**Implemented by**: SaliencyOverlayGenerator

**Methods**:

- `create_heatmap_overlay(original_image: Union[<ast.Attribute object at 0x1107641d0>, <ast.Attribute object at 0x1107642d0>], saliency_map: np.ndarray, alpha: float, colormap: str) -> Image.Image`
- `create_contour_overlay(original_image: Union[<ast.Attribute object at 0x110772390>, <ast.Attribute object at 0x110772490>], saliency_map: np.ndarray, threshold: float, contour_color: Tuple[int, int, int], line_width: int) -> Image.Image`
- `create_masked_overlay(original_image: Union[<ast.Attribute object at 0x110788710>, <ast.Attribute object at 0x110788810>], saliency_map: np.ndarray, threshold: float, highlight_color: Tuple[int, int, int], alpha: float) -> Image.Image`
- `create_side_by_side_comparison(original_image: Union[<ast.Attribute object at 0x110793590>, <ast.Attribute object at 0x110793690>], saliency_map: np.ndarray, overlay_type: str) -> Image.Image`

### VisualizationExporterInterface

**Type**: Protocol
**Implemented by**: VisualizationExporter

**Methods**:

- `export_visualization_set(original_image: Union[<ast.Attribute object at 0x1107b4a50>, <ast.Attribute object at 0x1107b4b50>], saliency_map: np.ndarray, bounding_boxes: List[Dict], explanation: Dict, base_filename: str, formats: List[str]) -> Dict[str, str]`
- `create_interactive_html_report(original_image: Union[<ast.Attribute object at 0x110812090>, <ast.Attribute object at 0x110812190>], saliency_map: np.ndarray, explanation: Dict, base_filename: str) -> str`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/interpretability_engine.py`
- Last validated: 2025-12-16 15:47:04

