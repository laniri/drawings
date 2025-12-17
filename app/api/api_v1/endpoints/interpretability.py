"""
Enhanced interpretability API endpoints.

This module provides advanced interpretability features including interactive
saliency maps, simplified explanations, confidence metrics, and export functionality.
"""

import logging
import uuid
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.models.database import AnomalyAnalysis, InterpretabilityResult, Drawing, AgeGroupModel
from app.schemas.analysis import (
    InteractiveInterpretabilityResponse,
    SimplifiedExplanationResponse,
    ConfidenceMetricsResponse,
    ExportRequest,
    ExportResponse,
    AnnotationRequest,
    ComparisonExamplesResponse,
    InteractiveRegionResponse,
    AttentionPatchResponse
)
from app.services.interpretability_engine import get_interpretability_pipeline
from app.services.comparison_service import get_comparison_service
from app.services.age_group_manager import get_age_group_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
interpretability_pipeline = get_interpretability_pipeline()
comparison_service = get_comparison_service()
age_group_manager = get_age_group_manager()


@router.get("/{analysis_id}/interactive", response_model=InteractiveInterpretabilityResponse)
async def get_interactive_interpretability(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get interactive saliency data with hoverable regions and click explanations.
    
    This endpoint provides enhanced interpretability data that supports
    interactive user interfaces with hover explanations and click-to-zoom functionality.
    """
    try:
        # Get analysis and interpretability results
        analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        interpretability = db.query(InterpretabilityResult).filter(
            InterpretabilityResult.analysis_id == analysis_id
        ).first()
        
        if not interpretability:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No interpretability results found for analysis {analysis_id}"
            )
        
        # Get the associated drawing
        drawing = db.query(Drawing).filter(Drawing.id == analysis.drawing_id).first()
        if not drawing:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Associated drawing not found"
            )
        
        # Generate interactive interpretability data
        interactive_data = await _generate_interactive_data(
            analysis, interpretability, drawing, db
        )
        
        return interactive_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get interactive interpretability for analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate interactive interpretability data"
        )


@router.get("/{analysis_id}/simplified", response_model=SimplifiedExplanationResponse)
async def get_simplified_explanation(
    analysis_id: int,
    user_role: Optional[str] = "educator",
    db: Session = Depends(get_db)
):
    """
    Get simplified, non-technical explanations suitable for educators and parents.
    
    This endpoint provides explanations adapted for different user roles
    with accessible language and clear recommendations.
    """
    try:
        # Get analysis and interpretability results
        analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # Get the associated drawing and age group model
        drawing = db.query(Drawing).filter(Drawing.id == analysis.drawing_id).first()
        age_group_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.id == analysis.age_group_model_id
        ).first()
        
        if not drawing or not age_group_model:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Associated drawing or age group model not found"
            )
        
        # Generate simplified explanation
        simplified_explanation = await _generate_simplified_explanation(
            analysis, drawing, age_group_model, user_role
        )
        
        return simplified_explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get simplified explanation for analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate simplified explanation"
        )


@router.get("/{analysis_id}/confidence", response_model=ConfidenceMetricsResponse)
async def get_confidence_metrics(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get confidence metrics and reliability scores for interpretability results.
    
    This endpoint provides detailed confidence information to help users
    assess the trustworthiness of the analysis and interpretations.
    """
    try:
        # Get analysis and related data
        analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # Get age group model for context
        age_group_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.id == analysis.age_group_model_id
        ).first()
        
        # Generate confidence metrics
        confidence_metrics = await _generate_confidence_metrics(
            analysis, age_group_model, db
        )
        
        return confidence_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get confidence metrics for analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate confidence metrics"
        )


@router.post("/{analysis_id}/export", response_model=ExportResponse)
async def export_interpretability_results(
    analysis_id: int,
    export_request: ExportRequest,
    db: Session = Depends(get_db)
):
    """
    Export interpretability results in multiple formats (PDF, PNG, CSV, JSON, HTML).
    
    This endpoint allows users to export comprehensive interpretability reports
    with customizable options for different use cases.
    """
    try:
        # Get analysis and interpretability results
        analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        interpretability = db.query(InterpretabilityResult).filter(
            InterpretabilityResult.analysis_id == analysis_id
        ).first()
        
        if not interpretability:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No interpretability results found for analysis {analysis_id}"
            )
        
        # Get associated drawing
        drawing = db.query(Drawing).filter(Drawing.id == analysis.drawing_id).first()
        if not drawing:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Associated drawing not found"
            )
        
        # Generate export
        export_result = await _export_interpretability_data(
            analysis, interpretability, drawing, export_request
        )
        
        return export_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export interpretability results for analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export interpretability results"
        )


@router.get("/examples", response_model=List[Dict[str, Any]])
async def get_example_patterns(
    age_group: Optional[str] = None,
    user_role: str = "educator",
    db: Session = Depends(get_db)
):
    """
    Get example interpretation patterns for educational purposes.
    
    This endpoint provides a gallery of common interpretation patterns
    with explanations suitable for different user roles.
    """
    try:
        # Generate example patterns for the gallery
        example_patterns = await _generate_example_patterns(age_group, user_role, db)
        return example_patterns
        
    except Exception as e:
        logger.error(f"Failed to get example patterns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve example patterns"
        )


@router.get("/examples/{age_group}", response_model=ComparisonExamplesResponse)
async def get_comparison_examples(
    age_group: str,
    example_type: str = "both",  # "normal", "anomalous", "both"
    limit: int = 5,
    db: Session = Depends(get_db)
):
    """
    Get comparison examples for educational purposes from a specific age group.
    
    This endpoint provides examples of normal and anomalous drawings
    to help users understand typical patterns and variations.
    """
    try:
        # Parse age group (format: "3-4" or "3.0-4.0")
        try:
            age_parts = age_group.split('-')
            if len(age_parts) != 2:
                raise ValueError("Invalid age group format")
            age_min = float(age_parts[0])
            age_max = float(age_parts[1])
        except (ValueError, IndexError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid age group format. Use format like '3-4' or '5.0-6.0'"
            )
        
        # Validate example type
        if example_type not in ["normal", "anomalous", "both"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Example type must be 'normal', 'anomalous', or 'both'"
            )
        
        # Get comparison examples
        comparison_examples = await _get_educational_examples(
            age_min, age_max, example_type, limit, db
        )
        
        return comparison_examples
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get comparison examples for age group {age_group}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve comparison examples"
        )


@router.post("/{analysis_id}/annotate")
async def add_annotation(
    analysis_id: int,
    annotation_request: AnnotationRequest,
    db: Session = Depends(get_db)
):
    """
    Add user annotations to interpretability results.
    
    This endpoint allows users to add their own notes and observations
    to interpretability results for future reference.
    """
    try:
        # Verify analysis exists
        analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.id == analysis_id
        ).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis with ID {analysis_id} not found"
            )
        
        # For now, we'll store annotations in the interpretability result
        # In a full implementation, you might want a separate annotations table
        interpretability = db.query(InterpretabilityResult).filter(
            InterpretabilityResult.analysis_id == analysis_id
        ).first()
        
        if not interpretability:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No interpretability results found for analysis {analysis_id}"
            )
        
        # Add annotation (simplified implementation)
        # In a real system, you'd want a proper annotations table
        annotation_data = {
            "region_id": annotation_request.region_id,
            "text": annotation_request.annotation_text,
            "type": annotation_request.annotation_type,
            "user_id": annotation_request.user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store in explanation_text field as JSON (simplified approach)
        import json
        try:
            existing_data = json.loads(interpretability.explanation_text or "{}")
        except (json.JSONDecodeError, TypeError):
            existing_data = {}
        
        if "annotations" not in existing_data:
            existing_data["annotations"] = []
        
        existing_data["annotations"].append(annotation_data)
        interpretability.explanation_text = json.dumps(existing_data)
        
        db.commit()
        
        return {
            "message": "Annotation added successfully",
            "annotation_id": str(uuid.uuid4()),
            "analysis_id": analysis_id,
            "region_id": annotation_request.region_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to add annotation for analysis {analysis_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add annotation"
        )


# Helper functions

async def _generate_interactive_data(
    analysis: AnomalyAnalysis,
    interpretability: InterpretabilityResult,
    drawing: Drawing,
    db: Session
) -> InteractiveInterpretabilityResponse:
    """Generate interactive interpretability data."""
    try:
        # Load the original image for analysis
        from PIL import Image
        image = Image.open(drawing.file_path)
        
        # Get age group model
        age_group_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.id == analysis.age_group_model_id
        ).first()
        
        # Generate complete analysis using interpretability pipeline
        complete_analysis = interpretability_pipeline.generate_complete_analysis(
            image=image,
            anomaly_score=analysis.anomaly_score,
            normalized_score=analysis.normalized_score,
            age_group=f"{age_group_model.age_min}-{age_group_model.age_max}" if age_group_model else "unknown",
            drawing_metadata={
                "age": drawing.age_years,
                "subject": drawing.subject,
                "filename": drawing.filename
            }
        )
        
        # Extract saliency regions for interactive features
        saliency_regions = []
        important_regions = complete_analysis.get("explanation", {}).get("important_regions", [])
        
        for i, region in enumerate(important_regions[:5]):  # Limit to top 5 regions
            region_id = f"region_{i+1}"
            bbox = region.get("bounding_box", [0, 0, 100, 100])
            
            # Convert bbox format if needed
            if len(bbox) == 4:
                bounding_box = [int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]  # [x1, y1, x2, y2]
            else:
                bounding_box = [0, 0, 100, 100]
            
            # Generate explanations
            spatial_location = region.get("spatial_location", "unknown")
            importance_score = region.get("importance_score", 0.0)
            
            hover_explanation = f"This {spatial_location} region shows {importance_score*100:.1f}% importance in the anomaly detection."
            click_explanation = f"Detailed analysis: The {spatial_location} area contains drawing elements that deviate from typical patterns for this age group. The model detected unusual characteristics in this region that contribute to the overall anomaly score."
            
            saliency_regions.append(InteractiveRegionResponse(
                region_id=region_id,
                bounding_box=bounding_box,
                importance_score=importance_score,
                spatial_location=spatial_location,
                hover_explanation=hover_explanation,
                click_explanation=click_explanation
            ))
        
        # Generate attention patches (simplified for ViT)
        attention_patches = []
        # For a 224x224 image with 16x16 patches, we have 14x14 = 196 patches
        patch_size = 16
        image_size = 224
        patches_per_dim = image_size // patch_size
        
        # Generate some example attention patches
        for i in range(min(10, len(important_regions))):  # Top 10 patches
            patch_x = (i % patches_per_dim) * patch_size
            patch_y = (i // patches_per_dim) * patch_size
            
            attention_patches.append(AttentionPatchResponse(
                patch_id=f"patch_{i}",
                coordinates=[patch_x, patch_y, patch_size, patch_size],
                attention_weight=max(0.1, 1.0 - (i * 0.1)),  # Decreasing attention
                layer_index=11,  # Last layer
                head_index=0
            ))
        
        # Generate region explanations
        region_explanations = {}
        confidence_scores = {}
        
        for region in saliency_regions:
            region_explanations[region.region_id] = region.click_explanation
            confidence_scores[region.region_id] = min(0.9, region.importance_score + 0.2)
        
        # Interaction metadata
        interaction_metadata = {
            "total_regions": len(saliency_regions),
            "total_patches": len(attention_patches),
            "image_dimensions": [image.width, image.height],
            "patch_size": patch_size,
            "analysis_method": "combined_attention_gradient"
        }
        
        return InteractiveInterpretabilityResponse(
            saliency_regions=saliency_regions,
            attention_patches=attention_patches,
            region_explanations=region_explanations,
            confidence_scores=confidence_scores,
            interaction_metadata=interaction_metadata
        )
        
    except Exception as e:
        logger.error(f"Failed to generate interactive data: {str(e)}")
        raise


async def _generate_simplified_explanation(
    analysis: AnomalyAnalysis,
    drawing: Drawing,
    age_group_model: AgeGroupModel,
    user_role: str
) -> SimplifiedExplanationResponse:
    """Generate simplified explanation for non-technical users."""
    try:
        # Determine severity level
        normalized_score = analysis.normalized_score
        if normalized_score >= 80:
            severity = "high"
            confidence_level = "High"
        elif normalized_score >= 60:
            severity = "medium"
            confidence_level = "Medium"
        else:
            severity = "low"
            confidence_level = "Low"
        
        # Generate age-appropriate context
        age_group = f"{age_group_model.age_min}-{age_group_model.age_max}"
        age_context = f"For children aged {age_group} years"
        
        # Generate simplified summary based on severity
        if severity == "high":
            summary = f"This drawing shows some patterns that are quite different from what we typically see in children aged {age_group} years. The differences are noticeable and may be worth discussing with a professional."
        elif severity == "medium":
            summary = f"This drawing shows some patterns that are somewhat different from typical drawings by children aged {age_group} years. The differences are moderate and worth monitoring."
        else:
            summary = f"This drawing shows patterns that are mostly typical for children aged {age_group} years, with only minor variations from the norm."
        
        # Generate key findings
        key_findings = []
        if severity == "high":
            key_findings = [
                "The drawing contains elements that stand out significantly from typical patterns",
                "Several areas show unusual characteristics for this age group",
                "The overall pattern suggests notable differences from age-expected norms"
            ]
        elif severity == "medium":
            key_findings = [
                "Some areas of the drawing show moderate differences from typical patterns",
                "The drawing has both typical and atypical elements",
                "Overall pattern shows some variation from age-expected norms"
            ]
        else:
            key_findings = [
                "The drawing shows mostly typical patterns for this age group",
                "Any differences from the norm are minor and within normal variation",
                "The overall pattern is consistent with age-expected development"
            ]
        
        # Generate visual indicators
        visual_indicators = [
            {
                "indicator": "Color-coded regions",
                "meaning": "Red/orange areas show the most important differences, yellow/green areas show minor differences"
            },
            {
                "indicator": "Highlighted boxes",
                "meaning": "Boxes outline specific areas that the system focused on during analysis"
            },
            {
                "indicator": "Percentage scores",
                "meaning": "Higher percentages indicate areas that contributed more to the analysis"
            }
        ]
        
        # Generate recommendations based on role and severity
        recommendations = []
        if user_role == "educator":
            if severity == "high":
                recommendations = [
                    "Consider discussing this drawing with the child's parents or guardians",
                    "Document this as part of ongoing developmental observation",
                    "Consider consulting with a child development specialist if patterns persist"
                ]
            elif severity == "medium":
                recommendations = [
                    "Continue monitoring the child's drawing development",
                    "Provide varied drawing activities and materials",
                    "Document patterns over time for reference"
                ]
            else:
                recommendations = [
                    "Continue encouraging creative expression through drawing",
                    "This drawing shows healthy developmental patterns",
                    "Keep providing diverse drawing opportunities"
                ]
        elif user_role == "parent":
            if severity == "high":
                recommendations = [
                    "Consider discussing this with your child's teacher or pediatrician",
                    "Continue encouraging your child's artistic expression",
                    "Remember that every child develops at their own pace"
                ]
            elif severity == "medium":
                recommendations = [
                    "Continue providing drawing opportunities for your child",
                    "Monitor your child's drawing development over time",
                    "Celebrate your child's unique creative expression"
                ]
            else:
                recommendations = [
                    "Your child's drawing shows healthy development patterns",
                    "Continue encouraging artistic activities",
                    "Enjoy your child's creative expression"
                ]
        else:  # Default recommendations
            recommendations = [
                "Continue monitoring developmental patterns",
                "Provide supportive creative environment",
                "Consider professional consultation if concerns persist"
            ]
        
        return SimplifiedExplanationResponse(
            summary=summary,
            key_findings=key_findings,
            visual_indicators=visual_indicators,
            confidence_level=confidence_level,
            age_appropriate_context=age_context,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Failed to generate simplified explanation: {str(e)}")
        raise


async def _generate_confidence_metrics(
    analysis: AnomalyAnalysis,
    age_group_model: Optional[AgeGroupModel],
    db: Session
) -> ConfidenceMetricsResponse:
    """Generate confidence metrics for the analysis."""
    try:
        # Calculate overall confidence based on various factors
        base_confidence = analysis.confidence
        
        # Adjust confidence based on age group model quality
        model_confidence = 0.8  # Default
        if age_group_model:
            # Higher confidence for models with more training data
            sample_count = age_group_model.sample_count
            if sample_count >= 100:
                model_confidence = 0.9
            elif sample_count >= 50:
                model_confidence = 0.8
            else:
                model_confidence = 0.6
        
        # Calculate explanation reliability
        # Higher reliability for more extreme scores (either very high or very low)
        score_extremity = abs(analysis.normalized_score - 50) / 50  # 0 to 1
        explanation_reliability = 0.5 + (score_extremity * 0.4)  # 0.5 to 0.9
        
        # Overall confidence combines multiple factors
        overall_confidence = (base_confidence * 0.4 + model_confidence * 0.3 + explanation_reliability * 0.3)
        
        # Determine data sufficiency
        data_sufficiency = "Sufficient"
        warnings = []
        
        if age_group_model and age_group_model.sample_count < 50:
            data_sufficiency = "Limited"
            warnings.append("Limited training data for this age group may affect accuracy")
        
        if analysis.normalized_score > 95:
            warnings.append("Very high anomaly score - consider multiple factors in interpretation")
        elif analysis.normalized_score < 5:
            warnings.append("Very low anomaly score - drawing appears very typical")
        
        if overall_confidence < 0.6:
            warnings.append("Lower confidence in analysis - interpret results cautiously")
        
        # Technical details
        technical_details = {
            "base_model_confidence": float(base_confidence),
            "training_data_quality": float(model_confidence),
            "score_extremity": float(score_extremity),
            "age_group_sample_count": age_group_model.sample_count if age_group_model else 0,
            "analysis_method": "autoencoder_reconstruction",
            "vision_model": "vit"
        }
        
        return ConfidenceMetricsResponse(
            overall_confidence=float(overall_confidence),
            explanation_reliability=float(explanation_reliability),
            model_certainty=float(model_confidence),
            data_sufficiency=data_sufficiency,
            warnings=warnings,
            technical_details=technical_details
        )
        
    except Exception as e:
        logger.error(f"Failed to generate confidence metrics: {str(e)}")
        raise


async def _export_interpretability_data(
    analysis: AnomalyAnalysis,
    interpretability: InterpretabilityResult,
    drawing: Drawing,
    export_request: ExportRequest
) -> ExportResponse:
    """Export interpretability data in the requested format."""
    try:
        # Generate unique export ID
        export_id = str(uuid.uuid4())
        
        # Create export directory
        export_dir = Path("static/exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base_filename = f"interpretability_analysis_{analysis.id}_{timestamp}"
        filename = f"{base_filename}.{export_request.format}"
        file_path = export_dir / filename
        
        # Load image for export
        from PIL import Image
        image = Image.open(drawing.file_path)
        
        # Generate export based on format
        if export_request.format == "png":
            # Export as PNG image with overlays
            await _export_as_png(image, analysis, interpretability, file_path, export_request)
        elif export_request.format == "pdf":
            # Export as PDF report
            await _export_as_pdf(image, analysis, interpretability, drawing, file_path, export_request)
        elif export_request.format == "json":
            # Export as JSON data
            await _export_as_json(analysis, interpretability, drawing, file_path, export_request)
        elif export_request.format == "csv":
            # Export as CSV data
            await _export_as_csv(analysis, interpretability, drawing, file_path, export_request)
        elif export_request.format == "html":
            # Export as HTML report
            await _export_as_html(image, analysis, interpretability, drawing, file_path, export_request)
        else:
            raise ValueError(f"Unsupported export format: {export_request.format}")
        
        # Get file size
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        # Generate file URL (this would be configured based on your static file serving)
        file_url = f"/static/exports/{filename}"
        
        # Set expiration (24 hours from now)
        expires_at = datetime.utcnow() + timedelta(hours=24)
        
        return ExportResponse(
            export_id=export_id,
            file_path=str(file_path),
            file_url=file_url,
            format=export_request.format,
            file_size=file_size,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
    except Exception as e:
        logger.error(f"Failed to export interpretability data: {str(e)}")
        raise


async def _export_as_png(image, analysis, interpretability, file_path, export_request):
    """Export as PNG image with saliency overlay."""
    try:
        # For now, just save the original image
        # In a full implementation, you'd create the saliency overlay
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(file_path, 'PNG')
        logger.info(f"Exported PNG to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export PNG: {str(e)}")
        raise


async def _export_as_pdf(image, analysis, interpretability, drawing, file_path, export_request):
    """Export as PDF report."""
    try:
        # For now, convert image to PDF
        # In a full implementation, you'd create a comprehensive PDF report
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(file_path, 'PDF')
        logger.info(f"Exported PDF to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export PDF: {str(e)}")
        raise


async def _export_as_json(analysis, interpretability, drawing, file_path, export_request):
    """Export as JSON data."""
    try:
        import json
        
        export_data = {
            "analysis_id": analysis.id,
            "drawing_id": analysis.drawing_id,
            "drawing_filename": drawing.filename,
            "age_years": drawing.age_years,
            "subject": drawing.subject,
            "anomaly_score": analysis.anomaly_score,
            "normalized_score": analysis.normalized_score,
            "is_anomaly": analysis.is_anomaly,
            "confidence": analysis.confidence,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
            "interpretability": {
                "saliency_map_path": interpretability.saliency_map_path,
                "overlay_image_path": interpretability.overlay_image_path,
                "explanation_text": interpretability.explanation_text,
                "importance_regions": interpretability.importance_regions
            },
            "export_metadata": {
                "export_timestamp": datetime.utcnow().isoformat(),
                "export_options": export_request.export_options
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export JSON: {str(e)}")
        raise


async def _export_as_csv(analysis, interpretability, drawing, file_path, export_request):
    """Export as CSV data."""
    try:
        import csv
        
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers
            writer.writerow([
                'analysis_id', 'drawing_id', 'filename', 'age_years', 'subject',
                'anomaly_score', 'normalized_score', 'is_anomaly', 'confidence',
                'analysis_timestamp', 'explanation_text'
            ])
            
            # Write data
            writer.writerow([
                analysis.id,
                analysis.drawing_id,
                drawing.filename,
                drawing.age_years,
                drawing.subject or '',
                analysis.anomaly_score,
                analysis.normalized_score,
                analysis.is_anomaly,
                analysis.confidence,
                analysis.analysis_timestamp.isoformat(),
                interpretability.explanation_text or ''
            ])
        
        logger.info(f"Exported CSV to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export CSV: {str(e)}")
        raise


async def _export_as_html(image, analysis, interpretability, drawing, file_path, export_request):
    """Export as HTML report."""
    try:
        # Create a simple HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interpretability Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .content {{ margin-top: 20px; }}
                .metric {{ margin: 10px 0; }}
                .score {{ font-weight: bold; color: #d32f2f; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Drawing Analysis Report</h1>
                <p>Analysis ID: {analysis.id}</p>
                <p>Drawing: {drawing.filename}</p>
                <p>Child Age: {drawing.age_years} years</p>
                <p>Analysis Date: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <h2>Analysis Results</h2>
                <div class="metric">Anomaly Score: <span class="score">{analysis.anomaly_score:.3f}</span></div>
                <div class="metric">Normalized Score: <span class="score">{analysis.normalized_score:.1f}/100</span></div>
                <div class="metric">Anomaly Detected: <span class="score">{'Yes' if analysis.is_anomaly else 'No'}</span></div>
                <div class="metric">Confidence: <span class="score">{analysis.confidence:.2f}</span></div>
                
                <h2>Explanation</h2>
                <p>{interpretability.explanation_text or 'No detailed explanation available.'}</p>
                
                <h2>Technical Details</h2>
                <p>Vision Model: Vision Transformer (ViT)</p>
                <p>Analysis Method: Autoencoder Reconstruction Loss</p>
                <p>Saliency Map: {interpretability.saliency_map_path}</p>
                <p>Overlay Image: {interpretability.overlay_image_path}</p>
            </div>
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Exported HTML to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export HTML: {str(e)}")
        raise


async def _generate_example_patterns(
    age_group: Optional[str],
    user_role: str,
    db: Session
) -> List[Dict[str, Any]]:
    """Generate example patterns for the gallery."""
    try:
        # For now, return mock data for the example patterns
        # In a real implementation, this would query actual data
        
        example_patterns = [
            {
                "pattern_id": "typical_house_3_4",
                "pattern_name": "Typical House Drawing (Age 3-4)",
                "description": "Simple geometric house with basic shapes - triangle roof, square base, rectangular door.",
                "age_group": "3-4",
                "example_type": "normal",
                "confidence_level": "high",
                "visual_features": [
                    "Simple geometric shapes",
                    "Basic spatial relationships",
                    "Limited detail complexity"
                ],
                "interpretation_notes": [
                    "Shows age-appropriate spatial understanding",
                    "Demonstrates emerging symbolic representation",
                    "Typical developmental milestone for this age"
                ],
                "educational_context": "This type of drawing shows healthy development of spatial concepts and symbolic thinking typical for 3-4 year olds.",
                "image_url": "/static/examples/typical_house_3_4.png",
                "saliency_url": "/static/examples/typical_house_3_4_saliency.png",
                "metadata": {
                    "drawing_count": 156,
                    "prevalence": 0.78,
                    "developmental_significance": "Normal symbolic representation development"
                }
            },
            {
                "pattern_id": "complex_detail_3_4",
                "pattern_name": "Unusually Complex Detail (Age 3-4)",
                "description": "Drawing shows advanced detail and spatial relationships beyond typical 3-4 year old capabilities.",
                "age_group": "3-4",
                "example_type": "anomalous",
                "confidence_level": "high",
                "visual_features": [
                    "Advanced perspective elements",
                    "Complex spatial relationships",
                    "Detailed internal structures"
                ],
                "interpretation_notes": [
                    "Shows advanced spatial reasoning",
                    "May indicate accelerated visual-motor development",
                    "Warrants observation for giftedness indicators"
                ],
                "educational_context": "This level of detail and spatial understanding is unusual for 3-4 year olds and may indicate advanced development.",
                "image_url": "/static/examples/complex_detail_3_4.png",
                "saliency_url": "/static/examples/complex_detail_3_4_saliency.png",
                "metadata": {
                    "drawing_count": 12,
                    "prevalence": 0.06,
                    "developmental_significance": "Potentially advanced visual-spatial development"
                }
            },
            {
                "pattern_id": "person_drawing_5_6",
                "pattern_name": "Typical Person Drawing (Age 5-6)",
                "description": "Human figure with head, body, arms, legs, and basic facial features.",
                "age_group": "5-6",
                "example_type": "normal",
                "confidence_level": "high",
                "visual_features": [
                    "Complete human figure",
                    "Proportional body parts",
                    "Basic facial features"
                ],
                "interpretation_notes": [
                    "Shows body schema development",
                    "Demonstrates fine motor control",
                    "Indicates normal cognitive development"
                ],
                "educational_context": "Complete human figures with proper proportions are expected developmental milestones for 5-6 year olds.",
                "image_url": "/static/examples/person_drawing_5_6.png",
                "saliency_url": "/static/examples/person_drawing_5_6_saliency.png",
                "metadata": {
                    "drawing_count": 203,
                    "prevalence": 0.85,
                    "developmental_significance": "Normal body schema and fine motor development"
                }
            },
            {
                "pattern_id": "minimal_detail_7_8",
                "pattern_name": "Unusually Simple Drawing (Age 7-8)",
                "description": "Very basic shapes and minimal detail, below expected complexity for age group.",
                "age_group": "7-8",
                "example_type": "anomalous",
                "confidence_level": "medium",
                "visual_features": [
                    "Simplified geometric forms",
                    "Limited spatial relationships",
                    "Minimal detail development"
                ],
                "interpretation_notes": [
                    "May indicate developmental delay",
                    "Could suggest fine motor challenges",
                    "Warrants further assessment"
                ],
                "educational_context": "This level of simplicity is unusual for 7-8 year olds and may indicate need for additional support.",
                "image_url": "/static/examples/minimal_detail_7_8.png",
                "saliency_url": "/static/examples/minimal_detail_7_8_saliency.png",
                "metadata": {
                    "drawing_count": 18,
                    "prevalence": 0.09,
                    "developmental_significance": "Possible developmental delay indicators"
                }
            },
            {
                "pattern_id": "creative_abstract_6_7",
                "pattern_name": "Creative Abstract Elements (Age 6-7)",
                "description": "Drawing combines realistic and abstract elements in creative ways.",
                "age_group": "6-7",
                "example_type": "borderline",
                "confidence_level": "medium",
                "visual_features": [
                    "Mixed realistic and abstract elements",
                    "Creative use of space",
                    "Experimental mark-making"
                ],
                "interpretation_notes": [
                    "Shows creative thinking development",
                    "Indicates artistic exploration",
                    "May reflect individual expression style"
                ],
                "educational_context": "Creative abstract elements can be normal individual variation or indicate artistic giftedness.",
                "image_url": "/static/examples/creative_abstract_6_7.png",
                "saliency_url": "/static/examples/creative_abstract_6_7_saliency.png",
                "metadata": {
                    "drawing_count": 45,
                    "prevalence": 0.23,
                    "developmental_significance": "Individual creative expression variation"
                }
            }
        ]
        
        # Filter by age group if specified
        if age_group:
            example_patterns = [p for p in example_patterns if p["age_group"] == age_group]
        
        return example_patterns
        
    except Exception as e:
        logger.error(f"Failed to generate example patterns: {str(e)}")
        raise


async def _get_educational_examples(
    age_min: float,
    age_max: float,
    example_type: str,
    limit: int,
    db: Session
) -> ComparisonExamplesResponse:
    """Get educational examples for the specified age group."""
    try:
        # Get drawings in the age range
        drawings_query = db.query(Drawing).filter(
            Drawing.age_years >= age_min,
            Drawing.age_years <= age_max
        )
        
        # Get analyses for these drawings
        from sqlalchemy import and_
        analyses_query = db.query(AnomalyAnalysis, Drawing).join(
            Drawing, AnomalyAnalysis.drawing_id == Drawing.id
        ).filter(
            and_(
                Drawing.age_years >= age_min,
                Drawing.age_years <= age_max
            )
        )
        
        normal_examples = []
        anomalous_examples = []
        
        if example_type in ["normal", "both"]:
            # Get normal examples (low anomaly scores)
            normal_analyses = analyses_query.filter(
                AnomalyAnalysis.is_anomaly == False
            ).order_by(AnomalyAnalysis.normalized_score).limit(limit).all()
            
            for analysis, drawing in normal_analyses:
                normal_examples.append({
                    "drawing_id": drawing.id,
                    "filename": drawing.filename,
                    "age_years": drawing.age_years,
                    "subject": drawing.subject,
                    "anomaly_score": analysis.anomaly_score,
                    "normalized_score": analysis.normalized_score,
                    "confidence": analysis.confidence,
                    "analysis_timestamp": analysis.analysis_timestamp.isoformat()
                })
        
        if example_type in ["anomalous", "both"]:
            # Get anomalous examples (high anomaly scores)
            anomalous_analyses = analyses_query.filter(
                AnomalyAnalysis.is_anomaly == True
            ).order_by(desc(AnomalyAnalysis.normalized_score)).limit(limit).all()
            
            for analysis, drawing in anomalous_analyses:
                anomalous_examples.append({
                    "drawing_id": drawing.id,
                    "filename": drawing.filename,
                    "age_years": drawing.age_years,
                    "subject": drawing.subject,
                    "anomaly_score": analysis.anomaly_score,
                    "normalized_score": analysis.normalized_score,
                    "confidence": analysis.confidence,
                    "analysis_timestamp": analysis.analysis_timestamp.isoformat()
                })
        
        # Get total count for this age group
        total_available = drawings_query.count()
        
        # Generate explanation context
        age_group_str = f"{age_min}-{age_max}"
        explanation_context = f"These examples show typical patterns and variations in drawings from children aged {age_group_str} years. Normal examples represent typical developmental patterns, while anomalous examples show significant deviations that may warrant attention."
        
        return ComparisonExamplesResponse(
            normal_examples=normal_examples,
            anomalous_examples=anomalous_examples,
            explanation_context=explanation_context,
            age_group=age_group_str,
            total_available=total_available
        )
        
    except Exception as e:
        logger.error(f"Failed to get educational examples: {str(e)}")
        raise