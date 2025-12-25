# SaliencyOverlayGenerator Algorithm Implementation

**Source File**: `app/services/interpretability_engine.py`
**Last Updated**: 2025-12-18 23:17:04

## Overview

Generates overlay visualizations combining original images with saliency maps with robust OpenCV fallback support.

This class provides methods to create various types of visual overlays
that highlight important regions identified by the interpretability engine. The implementation includes comprehensive fallback mechanisms for OpenCV functionality, automatically using PIL-based alternatives when OpenCV is not available.

## Computational Complexity Analysis

*This section analyzes the time and space complexity characteristics of the algorithm.*

Complexity analysis not available.

## Performance Analysis

*This section provides performance benchmarks and scalability characteristics.*

### Scalability Analysis

Consider profiling with representative datasets to determine scalability characteristics.

### Optimization Recommendations

- Profile algorithm performance with representative datasets
- Consider caching frequently computed results
- Evaluate opportunities for parallel processing

## Validation Methodology

*This section describes the testing and validation approach for the algorithm.*

### Testing Methods

- Unit testing for individual method correctness
- Integration testing for algorithm workflow
- Property-based testing for edge cases

### Validation Criteria

- Correctness of algorithm output
- Robustness to input variations
- Performance within acceptable bounds

### Accuracy Metrics

- AUC-ROC
- False Positive Rate
- True Positive Rate

### Edge Cases

The following edge cases should be tested:

- Special characters in overlay_type
- Very large values for alpha
- Zero value for line_width
- Empty saliency_map
- Zero value for contour_color
- Special characters in colormap
- Negative values for highlight_color
- Empty string for colormap
- Negative values for threshold
- Negative values for contour_color
- Very large original_image
- Single-element original_image
- Very large values for threshold
- Zero value for threshold
- Very large values for line_width
- Empty original_image
- Zero value for alpha
- Very large values for contour_color
- Very large values for highlight_color
- Empty string for overlay_type
- Very large saliency_map
- Single-element saliency_map
- Zero value for highlight_color
- Negative values for alpha
- Negative values for line_width

## Implementation Details

### OpenCV Fallback Mechanisms

The interpretability engine includes comprehensive fallback support for OpenCV functionality:

- **Image Resizing**: Falls back to PIL Lanczos interpolation when OpenCV cubic interpolation is unavailable
- **Grayscale Conversion**: Uses NumPy-based RGB to grayscale conversion with standard weights (0.299, 0.587, 0.114)
- **Edge Detection**: Implements gradient-based edge detection using Sobel-like kernels when Canny edge detection is unavailable
- **Complexity Calculation**: Automatically detects OpenCV availability and uses appropriate grayscale conversion method

All fallback implementations maintain functional compatibility while providing graceful degradation when OpenCV is not available.

### Methods

#### `create_heatmap_overlay`

Create heatmap overlay on original image.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    alpha: Transparency of overlay (0.0 to 1.0)
    colormap: Matplotlib colormap name
    
Returns:
    PIL Image with heatmap overlay

**Parameters:**
- `self` (Any)
- `original_image` (Union[Image.Image, np.ndarray])
- `saliency_map` (np.ndarray)
- `alpha` (float)
- `colormap` (str)

**Returns:** Image.Image

#### `create_contour_overlay`

Create contour overlay showing important region boundaries with automatic OpenCV/PIL detection.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for contour detection
    contour_color: RGB color for contour lines
    line_width: Width of contour lines
    
Returns:
    PIL Image with contour overlay

Implementation Details:
    - Automatically detects OpenCV availability using HAS_OPENCV flag
    - OpenCV path: Uses cv2.findContours() for precise boundary detection
    - PIL fallback: Implements edge pixel scanning for contour approximation
    - Both implementations maintain visual consistency and proper overlay generation

**Parameters:**
- `self` (Any)
- `original_image` (Union[Image.Image, np.ndarray])
- `saliency_map` (np.ndarray)
- `threshold` (float)
- `contour_color` (Tuple[int, int, int])
- `line_width` (int)

**Returns:** Image.Image

#### `create_masked_overlay`

Create masked overlay highlighting important regions.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    threshold: Threshold for region highlighting
    highlight_color: RGB color for highlighting
    alpha: Transparency of highlight
    
Returns:
    PIL Image with masked overlay

**Parameters:**
- `self` (Any)
- `original_image` (Union[Image.Image, np.ndarray])
- `saliency_map` (np.ndarray)
- `threshold` (float)
- `highlight_color` (Tuple[int, int, int])
- `alpha` (float)

**Returns:** Image.Image

#### `create_side_by_side_comparison`

Create side-by-side comparison of original and overlay.

Args:
    original_image: Original drawing image
    saliency_map: 2D saliency map
    overlay_type: Type of overlay ('heatmap', 'contour', 'masked')
    
Returns:
    PIL Image with side-by-side comparison

**Parameters:**
- `self` (Any)
- `original_image` (Union[Image.Image, np.ndarray])
- `saliency_map` (np.ndarray)
- `overlay_type` (str)

**Returns:** Image.Image

## LaTeX Mathematical Notation

*For formal mathematical documentation and publication:*

```latex
\section{SaliencyOverlayGenerator Algorithm}
```

## References and Standards

- **IEEE Standard 830-1998**: Software Requirements Specifications
- **Algorithm Documentation**: Following IEEE guidelines for algorithm specification
- **Mathematical Notation**: Standard mathematical notation and LaTeX formatting

---

*This documentation was automatically generated from source code analysis.*
*Generated on: 2025-12-18 23:17:04*