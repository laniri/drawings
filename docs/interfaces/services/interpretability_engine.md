# Interpretability Engine Service

Interpretability Engine for generating explanations and saliency maps.

This service provides attention visualization, saliency map generation, and explanation
capabilities for Vision Transformer models used in children's drawing anomaly detection.

## Class: InterpretabilityError

Base exception for interpretability engine errors.

## Class: SaliencyGenerationError

Raised when saliency map generation fails.

## Class: AttentionVisualizationError

Raised when attention visualization fails.

## Class: AttentionRollout

Attention rollout technique for Vision Transformers.

This class implements the attention rollout method to compute attention
maps that show which patches the model focuses on for its predictions.

### generate_rollout

Generate attention rollout for input tensor.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    start_layer: Layer to start rollout from
    
Returns:
    Attention rollout tensor [H_patches, W_patches]

**Signature**: `generate_rollout(input_tensor, start_layer)`

## Class: GradCAMViT

Gradient-based Class Activation Mapping for Vision Transformers.

This class implements Grad-CAM specifically adapted for Vision Transformers
to generate saliency maps showing important regions for anomaly detection.

### generate_cam

Generate Class Activation Map using gradients.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    reconstruction_loss: Reconstruction loss to compute gradients for
    
Returns:
    CAM tensor [H_patches, W_patches]

**Signature**: `generate_cam(input_tensor, reconstruction_loss)`

## Class: PatchImportanceScorer

Patch-level importance scoring for Vision Transformers.

This class provides methods to compute importance scores for individual
patches in the input image based on various techniques.

### compute_attention_importance

Compute patch importance using attention mechanisms.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    method: Method to use ("rollout" or "last_layer")
    
Returns:
    Importance scores for each patch

**Signature**: `compute_attention_importance(input_tensor, method)`

### compute_gradient_importance

Compute patch importance using gradient-based methods.

Args:
    input_tensor: Input tensor [1, 3, H, W]
    reconstruction_loss: Reconstruction loss for gradient computation
    
Returns:
    Importance scores for each patch

**Signature**: `compute_gradient_importance(input_tensor, reconstruction_loss)`

### reshape_to_spatial

Reshape 1D importance scores to 2D spatial map.

Args:
    importance_scores: 1D tensor of patch importance scores
    image_size: Original image size (H, W)
    
Returns:
    2D spatial importance map

**Signature**: `reshape_to_spatial(importance_scores, image_size)`

## Class: SaliencyMapGenerator

Main class for generating saliency maps from Vision Transformer models.

This class combines various techniques to create comprehensive saliency maps
that highlight important regions in children's drawings for anomaly detection.

### generate_saliency_map

Generate saliency map for an image.

Args:
    image: Input image (PIL Image or numpy array)
    reconstruction_loss: Reconstruction loss from autoencoder
    method: Saliency method ("attention_rollout", "grad_cam", "combined")
    save_path: Optional path to save the saliency map
    
Returns:
    Dictionary containing saliency map and metadata

**Signature**: `generate_saliency_map(image, reconstruction_loss, method, save_path)`

## Class: VisualFeatureIdentifier

Identifies and describes visual features in children's drawings.

This class analyzes saliency maps and original images to identify
specific visual features that contribute to anomaly detection.

### identify_important_regions

Identify important regions in the saliency map.

Args:
    saliency_map: 2D saliency map
    threshold: Threshold for considering regions important
    
Returns:
    List of important region descriptions

**Signature**: `identify_important_regions(saliency_map, threshold)`

### analyze_drawing_content

Analyze drawing content to identify likely visual features.

Args:
    image: Original drawing image
    important_regions: List of important regions from saliency analysis
    
Returns:
    Dictionary containing content analysis

**Signature**: `analyze_drawing_content(image, important_regions)`

## Class: ExplanationGenerator

Generates human-readable explanations for anomaly detection results.

This class combines saliency maps, visual feature analysis, and domain knowledge
to create comprehensive explanations for why a drawing was flagged as anomalous.

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

**Signature**: `generate_explanation(anomaly_score, normalized_score, saliency_result, age_group, drawing_metadata)`

## Class: ImportanceRegionDetector

Detects and highlights important regions in drawings based on saliency maps.

This class provides methods to identify, bound, and describe regions
that contribute most to anomaly detection decisions.

### detect_bounding_boxes

Detect bounding boxes around important regions.

Args:
    saliency_map: 2D saliency map
    threshold: Importance threshold for region detection
    min_region_size: Minimum size for a region to be considered
    
Returns:
    List of bounding box dictionaries

**Signature**: `detect_bounding_boxes(saliency_map, threshold, min_region_size)`

### create_region_highlights

Create image with highlighted important regions.

Args:
    original_image: Original drawing image
    bounding_boxes: List of bounding box dictionaries
    save_path: Optional path to save highlighted image
    
Returns:
    Highlighted image or path to saved image

**Signature**: `create_region_highlights(original_image, bounding_boxes, save_path)`

## Class: SaliencyOverlayGenerator

Generates overlay visualizations combining original images with saliency maps.

This class provides methods to create various types of visualizations that
help users understand which parts of drawings contribute to anomaly detection.

### create_heatmap_overlay

Create heatmap overlay on original image.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    alpha: Transparency of overlay (0.0 to 1.0)
    colormap: Matplotlib colormap name
    
Returns:
    PIL Image with heatmap overlay

**Signature**: `create_heatmap_overlay(original_image, saliency_map, alpha, colormap)`

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

**Signature**: `create_contour_overlay(original_image, saliency_map, threshold, contour_color, line_width)`

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

**Signature**: `create_masked_overlay(original_image, saliency_map, threshold, highlight_color, alpha)`

### create_side_by_side_comparison

Create side-by-side comparison of original and overlay.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
    **overlay_kwargs: Additional arguments for overlay creation
    
Returns:
    PIL Image with side-by-side comparison

**Signature**: `create_side_by_side_comparison(original_image, saliency_map, overlay_type)`

### save_overlay

Save overlay image to file.

Args:
    overlay_image: PIL Image to save
    filename: Filename (without extension)
    format: Image format ('PNG', 'JPEG')
    
Returns:
    Path to saved file

**Signature**: `save_overlay(overlay_image, filename, format)`

## Class: VisualizationExporter

Exports visualizations in various formats for different use cases.

This class provides methods to export saliency visualizations in formats
suitable for reports, presentations, and interactive applications.

### export_comprehensive_report

Export comprehensive visualization report.

Args:
    original_image: Original drawing image
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Output filename (without extension)
    
Returns:
    Path to exported report

**Signature**: `export_comprehensive_report(original_image, saliency_result, explanation, filename)`

### export_interactive_data

Export data for interactive visualizations.

Args:
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Output filename (without extension)
    
Returns:
    Path to exported JSON file

**Signature**: `export_interactive_data(saliency_result, explanation, filename)`

### export_presentation_slides

Export presentation-ready slides.

Args:
    original_image: Original drawing image
    saliency_result: Result from saliency generation
    explanation: Explanation from explanation generator
    filename: Base filename for slides
    
Returns:
    List of paths to exported slide images

**Signature**: `export_presentation_slides(original_image, saliency_result, explanation, filename)`

## Class: InterpretabilityPipeline

Complete interpretability pipeline combining all components.

This class provides a high-level interface for generating comprehensive
interpretability results including saliency maps, explanations, and visualizations.

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

**Signature**: `generate_complete_analysis(image, anomaly_score, normalized_score, age_group, drawing_metadata, export_options)`

## Class: SaliencyOverlayGenerator

Generates overlay visualizations combining original images with saliency maps.

This class provides methods to create various types of visual overlays
that highlight important regions identified by the interpretability engine.

### create_heatmap_overlay

Create heatmap overlay on original image.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    alpha: Transparency of overlay (0.0 to 1.0)
    colormap: Matplotlib colormap name
    
Returns:
    PIL Image with heatmap overlay

**Signature**: `create_heatmap_overlay(original_image, saliency_map, alpha, colormap)`

### create_contour_overlay

Create contour overlay showing important region boundaries.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for contour detection
    contour_color: RGB color for contour lines
    line_width: Width of contour lines
    
Returns:
    PIL Image with contour overlay

**Signature**: `create_contour_overlay(original_image, saliency_map, threshold, contour_color, line_width)`

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

**Signature**: `create_masked_overlay(original_image, saliency_map, threshold, highlight_color, alpha)`

### create_side_by_side_comparison

Create side-by-side comparison of original and overlay.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
    
Returns:
    PIL Image with side-by-side comparison

**Signature**: `create_side_by_side_comparison(original_image, saliency_map, overlay_type)`

## Class: VisualizationExporter

Exports interpretability visualizations in various formats.

This class provides methods to save visualizations in different formats
and create comprehensive reports combining multiple visualization types.

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

**Signature**: `export_visualization_set(original_image, saliency_map, bounding_boxes, explanation, base_filename, formats)`

### create_interactive_html_report

Create an interactive HTML report with visualizations.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    explanation: Explanation dictionary
    base_filename: Base filename for the report
    
Returns:
    Path to the HTML report file

**Signature**: `create_interactive_html_report(original_image, saliency_map, explanation, base_filename)`

