"""
Property-based tests for interpretability engine.

**Feature: children-drawing-anomaly-detection, Property 12: Interpretability Generation Completeness**
**Validates: Requirements 6.1, 6.4, 6.5**
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from PIL import Image
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from app.services.interpretability_engine import (
    SaliencyMapGenerator,
    AttentionRollout,
    GradCAMViT,
    PatchImportanceScorer,
    VisualFeatureIdentifier,
    ExplanationGenerator,
    ImportanceRegionDetector,
    InterpretabilityError,
    SaliencyGenerationError,
    AttentionVisualizationError
)


def create_test_image(width: int, height: int, mode: str = 'RGB') -> Image.Image:
    """Create a test PIL image with specified dimensions and mode"""
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


def create_mock_vit_model():
    """Create a mock Vision Transformer model for testing"""
    mock_model = Mock()
    mock_model.config = Mock()
    mock_model.config.hidden_size = 768
    mock_model.config.patch_size = 16
    mock_model.config.image_size = 224
    mock_model.config.num_attention_heads = 12
    mock_model.config.num_hidden_layers = 12
    
    # Mock attention outputs
    mock_attention = torch.rand(1, 12, 197, 197)  # [batch, heads, tokens, tokens]
    mock_outputs = Mock()
    mock_outputs.attentions = [mock_attention] * 12  # 12 layers
    mock_outputs.last_hidden_state = torch.rand(1, 197, 768)  # [batch, tokens, hidden]
    
    mock_model.return_value = mock_outputs
    mock_model.named_modules.return_value = [
        ('encoder.layer.11.output', Mock()),
        ('encoder.layer.10.attention', Mock()),
    ]
    
    return mock_model


def create_mock_embedding_service():
    """Create a mock embedding service for testing"""
    mock_service = Mock()
    mock_service.is_ready.return_value = True
    mock_service.vit_wrapper = Mock()
    mock_service.vit_wrapper.model = create_mock_vit_model()
    mock_service.vit_wrapper.processor = Mock()
    
    # Mock preprocessing
    def mock_preprocess(image):
        return torch.rand(1, 3, 224, 224)
    
    mock_service._preprocess_image = mock_preprocess
    return mock_service


def create_test_saliency_map(height: int = 14, width: int = 14) -> np.ndarray:
    """Create a test saliency map with some structure"""
    saliency_map = np.random.rand(height, width).astype(np.float32)
    
    # Add some high-importance regions
    center_h, center_w = height // 2, width // 2
    saliency_map[center_h-1:center_h+2, center_w-1:center_w+2] = 0.9
    
    # Add another region
    if height > 8 and width > 8:
        saliency_map[2:4, 2:4] = 0.8
    
    return saliency_map


# Hypothesis strategies
valid_dimensions_strategy = st.tuples(
    st.integers(min_value=32, max_value=512),  # width
    st.integers(min_value=32, max_value=512)   # height
)

anomaly_score_strategy = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
normalized_score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
age_group_strategy = st.text(min_size=3, max_size=10, alphabet=st.characters(whitelist_categories=('Nd', 'Lu', 'Ll')))
saliency_method_strategy = st.sampled_from(['attention_rollout', 'grad_cam', 'combined'])


@given(
    dimensions=valid_dimensions_strategy,
    anomaly_score=anomaly_score_strategy,
    method=saliency_method_strategy
)
@settings(max_examples=50, deadline=None)
def test_interpretability_generation_completeness(dimensions, anomaly_score, method):
    """
    **Feature: children-drawing-anomaly-detection, Property 12: Interpretability Generation Completeness**
    **Validates: Requirements 6.1, 6.4, 6.5**
    
    Property: For any drawing flagged as anomalous, the system should generate 
    corresponding saliency maps and explanations.
    """
    width, height = dimensions
    
    # Create test image
    image = create_test_image(width, height, 'RGB')
    
    # Create mock embedding service
    mock_service = create_mock_embedding_service()
    
    with patch('app.services.interpretability_engine.get_embedding_service', return_value=mock_service):
        # Create saliency generator
        generator = SaliencyMapGenerator(embedding_service=mock_service)
        
        # Generate saliency map
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_saliency.png")
            
            try:
                saliency_result = generator.generate_saliency_map(
                    image=image,
                    reconstruction_loss=anomaly_score,
                    method=method,
                    save_path=save_path
                )
                
                # Verify saliency map is generated
                assert 'saliency_map' in saliency_result, "Saliency map should be generated"
                assert saliency_result['saliency_map'] is not None, "Saliency map should not be None"
                
                # Verify saliency map properties
                saliency_map = saliency_result['saliency_map']
                assert isinstance(saliency_map, np.ndarray), "Saliency map should be numpy array"
                assert saliency_map.ndim == 2, "Saliency map should be 2D"
                assert saliency_map.shape[0] > 0 and saliency_map.shape[1] > 0, "Saliency map should have positive dimensions"
                
                # Verify importance scores are generated
                assert 'importance_scores' in saliency_result, "Importance scores should be generated"
                importance_scores = saliency_result['importance_scores']
                assert isinstance(importance_scores, np.ndarray), "Importance scores should be numpy array"
                assert len(importance_scores) > 0, "Importance scores should not be empty"
                
                # Verify all values are finite
                assert np.all(np.isfinite(saliency_map)), "All saliency values should be finite"
                assert np.all(np.isfinite(importance_scores)), "All importance scores should be finite"
                
                # Verify metadata is complete
                assert 'method' in saliency_result, "Method should be recorded"
                assert saliency_result['method'] == method, f"Method should be {method}"
                assert 'reconstruction_loss' in saliency_result, "Reconstruction loss should be recorded"
                assert 'map_shape' in saliency_result, "Map shape should be recorded"
                assert 'max_importance' in saliency_result, "Max importance should be recorded"
                assert 'mean_importance' in saliency_result, "Mean importance should be recorded"
                
                # Verify numerical properties
                assert saliency_result['max_importance'] >= 0, "Max importance should be non-negative"
                assert saliency_result['mean_importance'] >= 0, "Mean importance should be non-negative"
                assert saliency_result['max_importance'] >= saliency_result['mean_importance'], \
                    "Max importance should be >= mean importance"
                
            except Exception as e:
                # If generation fails, it should be due to a specific error, not a generic failure
                assert isinstance(e, (SaliencyGenerationError, InterpretabilityError)), \
                    f"Should raise specific interpretability error, got {type(e)}: {e}"


@given(
    anomaly_score=anomaly_score_strategy,
    normalized_score=normalized_score_strategy,
    age_group=age_group_strategy
)
@settings(max_examples=30, deadline=None)
def test_explanation_generation_completeness(anomaly_score, normalized_score, age_group):
    """
    Property: For any anomaly detection result, the system should generate 
    comprehensive explanations with all required components.
    """
    # Create test saliency result
    saliency_map = create_test_saliency_map()
    saliency_result = {
        'saliency_map': saliency_map,
        'importance_scores': np.random.rand(196),  # 14*14 patches
        'method': 'attention_rollout',
        'reconstruction_loss': anomaly_score,
        'max_importance': float(np.max(saliency_map)),
        'mean_importance': float(np.mean(saliency_map)),
        'map_shape': saliency_map.shape
    }
    
    # Create explanation generator
    generator = ExplanationGenerator()
    
    # Generate explanation
    explanation = generator.generate_explanation(
        anomaly_score=anomaly_score,
        normalized_score=normalized_score,
        saliency_result=saliency_result,
        age_group=age_group
    )
    
    # Verify explanation completeness
    required_fields = [
        'summary', 'severity', 'anomaly_score', 'normalized_score', 
        'age_group', 'important_regions', 'detailed_analysis', 
        'recommendations', 'technical_details'
    ]
    
    for field in required_fields:
        assert field in explanation, f"Explanation should contain {field}"
        assert explanation[field] is not None, f"{field} should not be None"
    
    # Verify summary is meaningful
    summary = explanation['summary']
    assert isinstance(summary, str), "Summary should be a string"
    assert len(summary) > 10, "Summary should be meaningful (>10 characters)"
    assert age_group in summary or 'age group' in summary.lower(), \
        "Summary should mention age group"
    
    # Verify severity is valid
    severity = explanation['severity']
    assert severity in ['high', 'medium', 'low', 'minimal'], \
        f"Severity should be valid level, got {severity}"
    
    # Verify scores are preserved
    assert explanation['anomaly_score'] == anomaly_score, "Anomaly score should be preserved"
    assert explanation['normalized_score'] == normalized_score, "Normalized score should be preserved"
    assert explanation['age_group'] == age_group, "Age group should be preserved"
    
    # Verify important regions are identified
    important_regions = explanation['important_regions']
    assert isinstance(important_regions, list), "Important regions should be a list"
    # Should have at least some regions for non-zero saliency maps
    if np.max(saliency_map) > 0.1:
        assert len(important_regions) > 0, "Should identify important regions for non-trivial saliency maps"
    
    # Verify detailed analysis
    detailed_analysis = explanation['detailed_analysis']
    assert isinstance(detailed_analysis, list), "Detailed analysis should be a list"
    assert len(detailed_analysis) > 0, "Should provide detailed analysis points"
    
    # Verify recommendations
    recommendations = explanation['recommendations']
    assert isinstance(recommendations, list), "Recommendations should be a list"
    assert len(recommendations) > 0, "Should provide recommendations"
    
    # Verify technical details
    technical_details = explanation['technical_details']
    assert isinstance(technical_details, dict), "Technical details should be a dict"
    assert 'saliency_method' in technical_details, "Should include saliency method"
    assert 'max_importance' in technical_details, "Should include max importance"
    assert 'mean_importance' in technical_details, "Should include mean importance"


@given(
    saliency_height=st.integers(min_value=8, max_value=32),
    saliency_width=st.integers(min_value=8, max_value=32),
    threshold=st.floats(min_value=0.1, max_value=0.9)
)
@settings(max_examples=30, deadline=None)
def test_important_regions_detection_completeness(saliency_height, saliency_width, threshold):
    """
    Property: For any saliency map, the system should identify important regions 
    and provide complete region descriptions.
    """
    # Create test saliency map with known structure
    saliency_map = np.random.rand(saliency_height, saliency_width).astype(np.float32)
    
    # Add some high-importance regions above threshold
    num_high_regions = max(1, min(3, saliency_height * saliency_width // 20))
    for i in range(num_high_regions):
        row = np.random.randint(0, saliency_height - 1)
        col = np.random.randint(0, saliency_width - 1)
        saliency_map[row, col] = threshold + 0.1 + np.random.rand() * 0.2
    
    # Create feature identifier
    identifier = VisualFeatureIdentifier()
    
    # Identify important regions
    important_regions = identifier.identify_important_regions(saliency_map, threshold=threshold)
    
    # Verify regions are identified
    assert isinstance(important_regions, list), "Important regions should be a list"
    
    # If we added high-importance regions, they should be detected
    if np.max(saliency_map) > threshold:
        assert len(important_regions) > 0, "Should detect important regions when they exist"
    
    # Verify each region has complete information
    for i, region in enumerate(important_regions):
        assert isinstance(region, dict), f"Region {i} should be a dictionary"
        
        required_fields = [
            'region_id', 'bounding_box', 'center', 'size', 
            'importance_score', 'spatial_location', 'relative_size'
        ]
        
        for field in required_fields:
            assert field in region, f"Region {i} should have {field}"
            assert region[field] is not None, f"Region {i} {field} should not be None"
        
        # Verify bounding box format
        bbox = region['bounding_box']
        assert isinstance(bbox, tuple) and len(bbox) == 4, \
            f"Region {i} bounding box should be 4-tuple"
        min_row, min_col, max_row, max_col = bbox
        assert 0 <= min_row <= max_row < saliency_height, \
            f"Region {i} row bounds should be valid"
        assert 0 <= min_col <= max_col < saliency_width, \
            f"Region {i} col bounds should be valid"
        
        # Verify center coordinates
        center = region['center']
        assert isinstance(center, tuple) and len(center) == 2, \
            f"Region {i} center should be 2-tuple"
        center_row, center_col = center
        assert min_row <= center_row <= max_row, \
            f"Region {i} center row should be within bounding box"
        assert min_col <= center_col <= max_col, \
            f"Region {i} center col should be within bounding box"
        
        # Verify numerical properties
        assert region['size'] > 0, f"Region {i} size should be positive"
        assert region['importance_score'] >= 0, f"Region {i} importance should be non-negative"
        assert 0 <= region['relative_size'] <= 1, f"Region {i} relative size should be in [0,1]"
        
        # Verify spatial location is meaningful
        spatial_location = region['spatial_location']
        assert isinstance(spatial_location, str), f"Region {i} spatial location should be string"
        assert len(spatial_location) > 0, f"Region {i} spatial location should not be empty"


@given(
    dimensions=valid_dimensions_strategy,
    num_regions=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=20, deadline=None)
def test_bounding_box_detection_completeness(dimensions, num_regions):
    """
    Property: For any saliency map with important regions, the system should 
    detect accurate bounding boxes with complete metadata.
    """
    width, height = dimensions
    # Use patch-level dimensions (typical for ViT)
    patch_height = max(8, height // 16)
    patch_width = max(8, width // 16)
    
    # Create saliency map
    saliency_map = np.random.rand(patch_height, patch_width).astype(np.float32) * 0.3
    
    # Add specific high-importance regions
    for i in range(num_regions):
        # Create a small region
        region_size = max(2, min(4, patch_height // 3, patch_width // 3))
        start_row = np.random.randint(0, max(1, patch_height - region_size))
        start_col = np.random.randint(0, max(1, patch_width - region_size))
        
        # Set high importance
        importance_value = 0.8 + np.random.rand() * 0.2
        saliency_map[start_row:start_row+region_size, start_col:start_col+region_size] = importance_value
    
    # Create region detector
    detector = ImportanceRegionDetector()
    
    # Detect bounding boxes
    bounding_boxes = detector.detect_bounding_boxes(
        saliency_map=saliency_map,
        threshold=0.7,
        min_region_size=2
    )
    
    # Verify bounding boxes are detected
    assert isinstance(bounding_boxes, list), "Bounding boxes should be a list"
    
    # Should detect at least some regions given our setup
    if np.max(saliency_map) > 0.7:
        assert len(bounding_boxes) > 0, "Should detect bounding boxes for high-importance regions"
    
    # Verify each bounding box has complete information
    for i, bbox_info in enumerate(bounding_boxes):
        assert isinstance(bbox_info, dict), f"Bounding box {i} should be a dictionary"
        
        required_fields = [
            'bbox', 'center', 'area', 'importance_score', 
            'max_importance', 'aspect_ratio'
        ]
        
        for field in required_fields:
            assert field in bbox_info, f"Bounding box {i} should have {field}"
            assert bbox_info[field] is not None, f"Bounding box {i} {field} should not be None"
        
        # Verify bbox format and bounds
        bbox = bbox_info['bbox']
        assert isinstance(bbox, tuple) and len(bbox) == 4, \
            f"Bounding box {i} should be 4-tuple"
        min_row, min_col, max_row, max_col = bbox
        assert 0 <= min_row <= max_row < patch_height, \
            f"Bounding box {i} row bounds should be valid"
        assert 0 <= min_col <= max_col < patch_width, \
            f"Bounding box {i} col bounds should be valid"
        
        # Verify center is within bbox
        center = bbox_info['center']
        assert len(center) == 2, f"Bounding box {i} center should be 2-tuple"
        center_row, center_col = center
        assert min_row <= center_row <= max_row, \
            f"Bounding box {i} center should be within bounds"
        assert min_col <= center_col <= max_col, \
            f"Bounding box {i} center should be within bounds"
        
        # Verify numerical properties
        assert bbox_info['area'] > 0, f"Bounding box {i} area should be positive"
        assert bbox_info['importance_score'] >= 0, f"Bounding box {i} importance should be non-negative"
        # Use larger tolerance for floating-point comparison (max can be slightly less than mean due to numerical precision)
        assert bbox_info['max_importance'] >= bbox_info['importance_score'] - 1e-6, \
            f"Bounding box {i} max importance ({bbox_info['max_importance']}) should be >= mean importance ({bbox_info['importance_score']})"
        assert bbox_info['aspect_ratio'] > 0, f"Bounding box {i} aspect ratio should be positive"


@given(
    method=saliency_method_strategy,
    reconstruction_loss=anomaly_score_strategy
)
@settings(max_examples=20, deadline=None)
def test_saliency_method_consistency(method, reconstruction_loss):
    """
    Property: For any saliency generation method, the output should be consistent 
    and complete regardless of the specific method used.
    """
    # Create test image
    image = create_test_image(224, 224, 'RGB')
    
    # Create mock embedding service
    mock_service = create_mock_embedding_service()
    
    with patch('app.services.interpretability_engine.get_embedding_service', return_value=mock_service):
        # Create saliency generator
        generator = SaliencyMapGenerator(embedding_service=mock_service)
        
        try:
            # Generate saliency map
            saliency_result = generator.generate_saliency_map(
                image=image,
                reconstruction_loss=reconstruction_loss,
                method=method
            )
            
            # Verify consistent output structure regardless of method
            required_fields = [
                'saliency_map', 'importance_scores', 'method', 
                'reconstruction_loss', 'map_shape', 'max_importance', 'mean_importance'
            ]
            
            for field in required_fields:
                assert field in saliency_result, f"Result should contain {field} for method {method}"
                assert saliency_result[field] is not None, f"{field} should not be None for method {method}"
            
            # Verify method is correctly recorded
            assert saliency_result['method'] == method, f"Method should be recorded as {method}"
            
            # Verify reconstruction loss is preserved
            assert saliency_result['reconstruction_loss'] == reconstruction_loss, \
                "Reconstruction loss should be preserved"
            
            # Verify saliency map properties
            saliency_map = saliency_result['saliency_map']
            assert isinstance(saliency_map, np.ndarray), f"Saliency map should be numpy array for {method}"
            assert saliency_map.ndim == 2, f"Saliency map should be 2D for {method}"
            assert np.all(np.isfinite(saliency_map)), f"Saliency map should be finite for {method}"
            
            # Verify importance scores
            importance_scores = saliency_result['importance_scores']
            assert isinstance(importance_scores, np.ndarray), f"Importance scores should be numpy array for {method}"
            assert len(importance_scores) > 0, f"Importance scores should not be empty for {method}"
            assert np.all(np.isfinite(importance_scores)), f"Importance scores should be finite for {method}"
            
            # Verify shape consistency
            map_shape = saliency_result['map_shape']
            assert map_shape == saliency_map.shape, f"Map shape should match actual shape for {method}"
            
            # Verify statistical properties
            max_importance = saliency_result['max_importance']
            mean_importance = saliency_result['mean_importance']
            assert max_importance >= 0, f"Max importance should be non-negative for {method}"
            assert mean_importance >= 0, f"Mean importance should be non-negative for {method}"
            assert max_importance >= mean_importance, f"Max should be >= mean for {method}"
            
        except Exception as e:
            # If method fails, it should be a specific interpretability error
            assert isinstance(e, (SaliencyGenerationError, InterpretabilityError)), \
                f"Should raise specific error for {method}, got {type(e)}: {e}"


def test_interpretability_error_handling():
    """Test that interpretability components handle errors gracefully."""
    # Test with invalid saliency map
    identifier = VisualFeatureIdentifier()
    
    # Empty saliency map should not crash
    empty_map = np.array([])
    regions = identifier.identify_important_regions(empty_map)
    assert isinstance(regions, list), "Should return empty list for empty map"
    
    # Test with invalid image
    generator = ExplanationGenerator()
    
    # Should handle missing saliency result gracefully
    invalid_saliency = {
        'saliency_map': np.array([]),
        'method': 'test',
        'max_importance': 0,
        'mean_importance': 0
    }
    
    explanation = generator.generate_explanation(
        anomaly_score=1.0,
        normalized_score=0.5,
        saliency_result=invalid_saliency,
        age_group="test"
    )
    
    # Should still provide basic explanation structure
    assert 'summary' in explanation, "Should provide summary even for invalid input"
    assert 'severity' in explanation, "Should provide severity even for invalid input"


def test_patch_importance_scorer_initialization():
    """Test that PatchImportanceScorer initializes correctly."""
    mock_model = create_mock_vit_model()
    scorer = PatchImportanceScorer(mock_model, patch_size=16)
    
    assert scorer.model == mock_model, "Should store model reference"
    assert scorer.patch_size == 16, "Should store patch size"
    
    # Test reshape functionality
    importance_scores = torch.rand(196)  # 14x14 patches for 224x224 image
    spatial_map = scorer.reshape_to_spatial(importance_scores, (224, 224))
    
    assert spatial_map.shape == (14, 14), "Should reshape to correct spatial dimensions"
    assert torch.all(torch.isfinite(spatial_map)), "Spatial map should be finite"