"""
Pydantic schemas for analysis-related API endpoints.

This module contains request and response models for anomaly analysis,
interpretability results, and batch processing operations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from .drawings import DrawingResponse


class AnalysisMethod(str, Enum):
    """Enumeration for anomaly detection methods."""
    AUTOENCODER = "autoencoder"


class VisionModel(str, Enum):
    """Enumeration for vision models."""
    VIT = "vit"


class AnalysisRequest(BaseModel):
    """Request model for analyzing a drawing."""
    
    drawing_id: int = Field(..., gt=0, description="ID of the drawing to analyze")
    force_reanalysis: bool = Field(
        False, 
        description="Force re-analysis even if results exist"
    )


class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis of multiple drawings."""
    
    drawing_ids: List[int] = Field(
        ..., 
        min_length=1, 
        max_length=100,
        description="List of drawing IDs to analyze"
    )
    force_reanalysis: bool = Field(
        False, 
        description="Force re-analysis even if results exist"
    )
    
    @field_validator('drawing_ids')
    @classmethod
    def validate_drawing_ids(cls, v):
        """Validate that all drawing IDs are positive and unique."""
        if not all(id > 0 for id in v):
            raise ValueError('All drawing IDs must be positive')
        if len(v) != len(set(v)):
            raise ValueError('Drawing IDs must be unique')
        return v


class AnomalyAnalysisResponse(BaseModel):
    """Response model for anomaly analysis results."""
    
    id: int
    drawing_id: int
    anomaly_score: float = Field(..., description="Overall reconstruction loss on full 832-dimensional embedding")
    normalized_score: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Normalized anomaly score (0-100 scale: 0=no anomaly, 100=maximal anomaly)"
    )
    visual_anomaly_score: Optional[float] = Field(
        None, 
        description="Visual component reconstruction loss (dims 0-767)"
    )
    subject_anomaly_score: Optional[float] = Field(
        None, 
        description="Subject component reconstruction loss (dims 768-831)"
    )
    anomaly_attribution: Optional[str] = Field(
        None, 
        description="Primary anomaly source: 'visual', 'subject', 'both', or 'age'"
    )
    analysis_type: str = Field(
        "subject_aware", 
        description="Type of analysis performed"
    )
    subject_category: Optional[str] = Field(
        None, 
        description="Subject category used in analysis"
    )
    is_anomaly: bool = Field(..., description="Whether the drawing is flagged as anomalous")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in the anomaly decision"
    )
    age_group: str = Field(..., description="Age group used for analysis")
    method_used: AnalysisMethod = Field(..., description="Analysis method used")
    vision_model: VisionModel = Field(..., description="Vision model used")
    analysis_timestamp: datetime
    
    model_config = {"from_attributes": True}


class InterpretabilityResponse(BaseModel):
    """Response model for interpretability results."""
    
    saliency_map_url: str = Field(..., description="URL to saliency map image")
    overlay_image_url: str = Field(..., description="URL to overlay visualization")
    explanation_text: Optional[str] = Field(
        None, 
        description="Human-readable explanation"
    )
    importance_regions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of important regions with bounding boxes"
    )
    
    model_config = {"from_attributes": True}


class ComparisonExampleResponse(BaseModel):
    """Response model for comparison examples."""
    
    drawing_id: int
    filename: str
    age_years: float
    subject: Optional[str] = None
    similarity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Similarity score to the analyzed drawing"
    )
    anomaly_score: float = Field(..., description="Anomaly score of the comparison example")
    normalized_score: float = Field(
        ..., 
        ge=0.0, 
        le=100.0, 
        description="Normalized anomaly score of the comparison example (0-100 scale: 0=no anomaly, 100=maximal anomaly)"
    )


class AnalysisResultResponse(BaseModel):
    """Complete analysis result including drawing, analysis, and interpretability."""
    
    drawing: DrawingResponse
    analysis: AnomalyAnalysisResponse
    interpretability: Optional[InterpretabilityResponse] = None
    comparison_examples: List[ComparisonExampleResponse] = Field(
        default_factory=list,
        description="Similar normal examples from the same age group"
    )


class BatchAnalysisResponse(BaseModel):
    """Response model for batch analysis operations."""
    
    batch_id: str = Field(..., description="Unique identifier for the batch")
    total_drawings: int = Field(..., gt=0, description="Total number of drawings to analyze")
    completed: int = Field(..., ge=0, description="Number of completed analyses")
    failed: int = Field(..., ge=0, description="Number of failed analyses")
    status: str = Field(..., description="Batch processing status")
    results: List[AnalysisResultResponse] = Field(
        default_factory=list,
        description="Completed analysis results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details for failed analyses"
    )
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    @field_validator('completed', 'failed')
    @classmethod
    def validate_counts(cls, v, info):
        """Validate that completed + failed <= total_drawings."""
        if 'total_drawings' in info.data:
            total = info.data['total_drawings']
            if 'completed' in info.data and 'failed' in info.data:
                if info.data['completed'] + info.data['failed'] > total:
                    raise ValueError('completed + failed cannot exceed total_drawings')
        return v


class AnalysisHistoryResponse(BaseModel):
    """Response model for analysis history of a drawing."""
    
    drawing_id: int
    analyses: List[AnomalyAnalysisResponse]
    total_count: int


# Enhanced Interpretability Schemas

class InteractiveRegionResponse(BaseModel):
    """Response model for interactive saliency regions."""
    
    region_id: str = Field(..., description="Unique identifier for the region")
    bounding_box: List[int] = Field(..., description="Bounding box coordinates [x1, y1, x2, y2]")
    importance_score: float = Field(..., ge=0.0, le=1.0, description="Importance score for this region")
    spatial_location: str = Field(..., description="Spatial description (e.g., 'top-left', 'center')")
    hover_explanation: str = Field(..., description="Explanation shown on hover")
    click_explanation: str = Field(..., description="Detailed explanation shown on click")


class AttentionPatchResponse(BaseModel):
    """Response model for Vision Transformer attention patches."""
    
    patch_id: str = Field(..., description="Unique identifier for the patch")
    coordinates: List[int] = Field(..., description="Patch coordinates [x, y, width, height]")
    attention_weight: float = Field(..., ge=0.0, le=1.0, description="Attention weight for this patch")
    layer_index: int = Field(..., ge=0, description="Transformer layer index")
    head_index: int = Field(..., ge=0, description="Attention head index")


class InteractiveInterpretabilityResponse(BaseModel):
    """Response model for interactive interpretability data."""
    
    saliency_regions: List[InteractiveRegionResponse] = Field(..., description="Interactive regions with hover explanations")
    attention_patches: List[AttentionPatchResponse] = Field(..., description="Vision Transformer attention patch data")
    region_explanations: Dict[str, str] = Field(..., description="Explanations for each interactive region")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each explanation")
    interaction_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for interactive features")


class SimplifiedExplanationResponse(BaseModel):
    """Response model for simplified, non-technical explanations."""
    
    summary: str = Field(..., description="Simple, non-technical explanation")
    key_findings: List[str] = Field(..., description="Main points in accessible language")
    visual_indicators: List[Dict[str, Any]] = Field(..., description="Simple visual cues and their meanings")
    confidence_level: str = Field(..., description="High/Medium/Low confidence description")
    age_appropriate_context: str = Field(..., description="Context appropriate for the age group")
    recommendations: List[str] = Field(default_factory=list, description="Simple recommendations")


class ConfidenceMetricsResponse(BaseModel):
    """Response model for confidence metrics and reliability scores."""
    
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence in the analysis")
    explanation_reliability: float = Field(..., ge=0.0, le=1.0, description="Reliability of the explanation")
    model_certainty: float = Field(..., ge=0.0, le=1.0, description="Model's certainty in the prediction")
    data_sufficiency: str = Field(..., description="Sufficient/Limited/Insufficient data quality")
    warnings: List[str] = Field(default_factory=list, description="Confidence-related warnings")
    technical_details: Dict[str, Any] = Field(default_factory=dict, description="Technical confidence metrics")


class ExportRequest(BaseModel):
    """Request model for exporting interpretability results."""
    
    format: str = Field(..., description="Export format: pdf, png, csv, json, html")
    include_annotations: bool = Field(True, description="Include user annotations")
    include_comparisons: bool = Field(True, description="Include comparison examples")
    simplified_version: bool = Field(False, description="Use simplified explanations")
    export_options: Dict[str, Any] = Field(default_factory=dict, description="Additional export options")
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        """Validate export format."""
        allowed_formats = ['pdf', 'png', 'csv', 'json', 'html']
        if v.lower() not in allowed_formats:
            raise ValueError(f'Format must be one of: {", ".join(allowed_formats)}')
        return v.lower()


class ExportResponse(BaseModel):
    """Response model for export operations."""
    
    export_id: str = Field(..., description="Unique identifier for the export")
    file_path: str = Field(..., description="Path to the exported file")
    file_url: str = Field(..., description="URL to download the exported file")
    format: str = Field(..., description="Export format used")
    file_size: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="Export creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Export expiration timestamp")


class AnnotationRequest(BaseModel):
    """Request model for adding annotations to interpretability results."""
    
    region_id: str = Field(..., description="ID of the region being annotated")
    annotation_text: str = Field(..., min_length=1, max_length=500, description="User annotation text")
    annotation_type: str = Field(..., description="Type: note, question, concern, etc.")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    
    @field_validator('annotation_type')
    @classmethod
    def validate_annotation_type(cls, v):
        """Validate annotation type."""
        allowed_types = ['note', 'question', 'concern', 'observation', 'hypothesis']
        if v.lower() not in allowed_types:
            raise ValueError(f'Annotation type must be one of: {", ".join(allowed_types)}')
        return v.lower()


class ComparisonExamplesResponse(BaseModel):
    """Response model for comparison examples from the same age group."""
    
    normal_examples: List[Dict[str, Any]] = Field(..., description="Normal drawings from same age group")
    anomalous_examples: List[Dict[str, Any]] = Field(..., description="Other anomalous examples")
    explanation_context: str = Field(..., description="Context for the comparisons")
    age_group: str = Field(..., description="Age group for the examples")
    total_available: int = Field(..., description="Total examples available in this age group")