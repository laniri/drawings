"""
Configuration API endpoints.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.config import settings
from app.schemas.models import (
    SystemConfigurationResponse,
    ConfigurationUpdateRequest,
    AgeGroupingStrategy,
    ThresholdUpdateRequest
)
from app.schemas.analysis import AnalysisMethod, VisionModel
from app.schemas.common import HealthCheckResponse, SuccessResponse
from app.services.threshold_manager import get_threshold_manager
from app.services.age_group_manager import get_age_group_manager
from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
threshold_manager = get_threshold_manager()
age_group_manager = get_age_group_manager()
model_manager = get_model_manager()


@router.get("/", response_model=SystemConfigurationResponse)
async def get_config(db: Session = Depends(get_db)):
    """
    Get current system configuration.
    
    This endpoint returns the current system configuration including
    model settings, threshold parameters, and age grouping strategy.
    """
    try:
        # Get current configuration from various sources
        # Get the current threshold percentile from threshold manager
        current_percentile = threshold_manager.get_current_percentile()
        
        config = SystemConfigurationResponse(
            vision_model=VisionModel.VIT,  # Currently only VIT is supported
            anomaly_detection_method=AnalysisMethod.AUTOENCODER,  # Currently only autoencoder
            threshold_percentile=current_percentile,  # Current or default percentile
            age_grouping_strategy=AgeGroupingStrategy.YEARLY,  # Default strategy
            min_samples_per_group=50,  # Default minimum samples
            max_age_group_span=4.0  # Default maximum span
        )
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )


@router.put("/", response_model=SuccessResponse)
async def update_config(
    request: ConfigurationUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update system configuration.
    
    This endpoint updates various system configuration settings
    including thresholds and age grouping parameters.
    """
    try:
        updates = {}
        
        # Update threshold if provided
        if request.threshold_percentile is not None:
            if not (50.0 <= request.threshold_percentile <= 99.9):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Threshold percentile must be between 50.0 and 99.9"
                )
            
            # Recalculate all thresholds
            threshold_results = threshold_manager.recalculate_all_thresholds(
                db, request.threshold_percentile
            )
            # Store the new percentile in threshold manager
            threshold_manager.set_current_percentile(request.threshold_percentile)
            updates["threshold_percentile"] = request.threshold_percentile
            updates["threshold_updates"] = threshold_results
        
        # Update age grouping settings
        if request.age_grouping_strategy is not None:
            updates["age_grouping_strategy"] = request.age_grouping_strategy.value
        
        if request.min_samples_per_group is not None:
            updates["min_samples_per_group"] = request.min_samples_per_group
        
        if request.max_age_group_span is not None:
            updates["max_age_group_span"] = request.max_age_group_span
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No configuration updates provided"
            )
        
        return SuccessResponse(
            message="Configuration updated successfully",
            data=updates
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update configuration: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update configuration"
        )


@router.put("/threshold", response_model=SuccessResponse)
async def update_threshold_settings(
    request: ThresholdUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update global threshold settings.
    
    This endpoint recalculates thresholds for all active models
    using the specified percentile value from the request body.
    """
    try:
        # Use percentile from request, or threshold if provided
        percentile = request.percentile if request.percentile is not None else 95.0
        
        # Validate percentile
        if not (50.0 <= percentile <= 99.9):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Percentile must be between 50.0 and 99.9"
            )
        
        # Recalculate all thresholds
        results = threshold_manager.recalculate_all_thresholds(db, percentile)
        
        # Store the new percentile in threshold manager
        if results["successful_updates"] > 0:
            threshold_manager.set_current_percentile(percentile)
        
        if results["failed_updates"] > 0:
            logger.warning(f"Some threshold updates failed: {results['errors']}")
        
        return SuccessResponse(
            message=f"Updated thresholds for {results['successful_updates']} models "
                   f"using {percentile}th percentile",
            data={
                "percentile": percentile,
                "successful_updates": results["successful_updates"],
                "failed_updates": results["failed_updates"],
                "errors": results["errors"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update threshold settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update threshold settings"
        )


@router.put("/age-grouping", response_model=SuccessResponse)
async def update_age_grouping(
    request: ConfigurationUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Modify age grouping strategy.
    
    This endpoint updates the age grouping configuration and can
    optionally trigger recreation of age group models.
    """
    try:
        # Process configuration updates
        
        updates = {}
        
        # Handle threshold updates first
        if request.threshold_percentile is not None:
            if not (50.0 <= request.threshold_percentile <= 99.9):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Threshold percentile must be between 50.0 and 99.9"
                )
            
            # Recalculate all thresholds
            threshold_results = threshold_manager.recalculate_all_thresholds(
                db, request.threshold_percentile
            )
            # Store the new percentile in threshold manager
            threshold_manager.set_current_percentile(request.threshold_percentile)
            updates["threshold_percentile"] = request.threshold_percentile
            updates["threshold_updates"] = threshold_results
        
        # Handle age grouping updates
        if request.age_grouping_strategy is not None:
            updates["age_grouping_strategy"] = request.age_grouping_strategy.value
        
        if request.min_samples_per_group is not None:
            updates["min_samples_per_group"] = request.min_samples_per_group
        
        if request.max_age_group_span is not None:
            updates["max_age_group_span"] = request.max_age_group_span
        
        if not updates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No configuration updates provided"
            )
        
        return SuccessResponse(
            message="Age grouping configuration updated",
            data=updates
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update age grouping: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update age grouping configuration"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: Session = Depends(get_db)):
    """
    System health check endpoint.
    
    This endpoint provides information about the health and status
    of various system components.
    """
    try:
        from datetime import datetime
        import os
        
        # Check database connectivity
        try:
            db.execute("SELECT 1")
            database_status = "healthy"
        except Exception as e:
            database_status = f"error: {str(e)}"
        
        # Check model availability
        try:
            from app.models.database import AgeGroupModel
            active_models = db.query(AgeGroupModel).filter(
                AgeGroupModel.is_active == True
            ).count()
            
            if active_models > 0:
                models_status = f"healthy ({active_models} active models)"
            else:
                models_status = "no active models"
        except Exception as e:
            models_status = f"error: {str(e)}"
        
        # Check storage
        try:
            uploads_dir = "uploads"
            models_dir = "static/models"
            
            uploads_exists = os.path.exists(uploads_dir)
            models_exists = os.path.exists(models_dir)
            
            if uploads_exists and models_exists:
                storage_status = "healthy"
            else:
                storage_status = f"directories missing: uploads={uploads_exists}, models={models_exists}"
        except Exception as e:
            storage_status = f"error: {str(e)}"
        
        # Determine overall status
        if all("error" not in status for status in [database_status, models_status, storage_status]):
            overall_status = "healthy"
        else:
            overall_status = "degraded"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",  # Would come from app metadata
            database=database_status,
            models=models_status,
            storage=storage_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )


@router.get("/stats")
async def get_system_stats(db: Session = Depends(get_db)):
    """
    Get comprehensive system statistics.
    
    This endpoint provides detailed statistics about the system
    including data distribution, model performance, and usage metrics.
    """
    try:
        # Get age distribution
        age_distribution = age_group_manager.analyze_age_distribution(db)
        
        # Get threshold statistics
        threshold_stats = threshold_manager.get_threshold_statistics(db)
        
        # Get age group coverage
        coverage_info = age_group_manager.get_age_group_coverage(db)
        
        # Get validation results
        validation_results = age_group_manager.validate_age_group_data(db)
        
        return {
            "age_distribution": age_distribution,
            "threshold_statistics": threshold_stats,
            "age_group_coverage": coverage_info,
            "validation_results": validation_results,
            "system_health": {
                "total_models": coverage_info["total_models"],
                "coverage_gaps": len(coverage_info["coverage_gaps"]),
                "validation_warnings": len(validation_results["warnings"]),
                "validation_errors": len(validation_results["errors"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@router.get("/subjects")
async def get_supported_subject_categories(db: Session = Depends(get_db)):
    """
    Get list of supported subject categories.
    
    This endpoint returns all supported subject categories that can be used
    when uploading drawings, along with usage statistics.
    """
    try:
        from app.schemas.drawings import SubjectCategory
        from app.models.database import Drawing
        from sqlalchemy import func
        
        # Get all supported subject categories from the enum
        supported_categories = [category.value for category in SubjectCategory]
        
        # Get usage statistics for each category
        usage_stats = db.query(
            Drawing.subject,
            func.count(Drawing.id).label('count')
        ).group_by(Drawing.subject).all()
        
        # Create usage dictionary
        usage_dict = {stat.subject: stat.count for stat in usage_stats if stat.subject}
        
        # Build response with categories and their usage
        categories_with_stats = []
        for category in supported_categories:
            categories_with_stats.append({
                "category": category,
                "display_name": category.replace("_", " ").title(),
                "usage_count": usage_dict.get(category, 0),
                "is_default": category == "unspecified"
            })
        
        # Sort by usage count (descending) then by name
        categories_with_stats.sort(key=lambda x: (-x["usage_count"], x["category"]))
        
        # Get total drawings with and without subjects
        total_drawings = db.query(Drawing).count()
        drawings_with_subject = db.query(Drawing).filter(Drawing.subject.isnot(None)).count()
        drawings_without_subject = total_drawings - drawings_with_subject
        
        return {
            "supported_categories": categories_with_stats,
            "total_categories": len(supported_categories),
            "statistics": {
                "total_drawings": total_drawings,
                "drawings_with_subject": drawings_with_subject,
                "drawings_without_subject": drawings_without_subject,
                "subject_coverage_percentage": round((drawings_with_subject / total_drawings * 100), 1) if total_drawings > 0 else 0
            },
            "subject_aware_analysis": {
                "enabled": True,
                "default_category": "unspecified",
                "hybrid_embedding_dimensions": 832,
                "visual_component_dimensions": 768,
                "subject_component_dimensions": 64
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported subject categories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve supported subject categories"
        )


@router.get("/subjects/statistics")
async def get_subject_specific_statistics(
    subject: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get subject-specific statistics and analysis data.
    
    This endpoint provides detailed statistics about drawings and analyses
    for specific subject categories or overall subject-related metrics.
    """
    try:
        from app.models.database import Drawing, AnomalyAnalysis
        from sqlalchemy import func, and_
        
        if subject:
            # Get statistics for a specific subject
            # Validate subject exists
            subject_drawings = db.query(Drawing).filter(Drawing.subject == subject).count()
            if subject_drawings == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No drawings found for subject '{subject}'"
                )
            
            # Get age distribution for this subject
            age_distribution = db.query(
                func.floor(Drawing.age_years).label('age_floor'),
                func.count(Drawing.id).label('count')
            ).filter(Drawing.subject == subject).group_by(func.floor(Drawing.age_years)).all()
            
            # Get analysis statistics for this subject
            analysis_stats = db.query(
                func.count(AnomalyAnalysis.id).label('total_analyses'),
                func.sum(func.cast(AnomalyAnalysis.is_anomaly, func.Integer)).label('anomaly_count'),
                func.avg(AnomalyAnalysis.anomaly_score).label('avg_anomaly_score'),
                func.avg(AnomalyAnalysis.normalized_score).label('avg_normalized_score'),
                func.avg(AnomalyAnalysis.confidence).label('avg_confidence')
            ).join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id).filter(
                Drawing.subject == subject
            ).first()
            
            # Get attribution breakdown for this subject
            attribution_stats = db.query(
                AnomalyAnalysis.anomaly_attribution,
                func.count(AnomalyAnalysis.id).label('count')
            ).join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id).filter(
                Drawing.subject == subject
            ).group_by(AnomalyAnalysis.anomaly_attribution).all()
            
            return {
                "subject": subject,
                "total_drawings": subject_drawings,
                "age_distribution": [
                    {
                        "age_group": f"{int(age_floor)}-{int(age_floor)+1}",
                        "count": count
                    }
                    for age_floor, count in age_distribution
                ],
                "analysis_statistics": {
                    "total_analyses": analysis_stats.total_analyses or 0,
                    "anomaly_count": analysis_stats.anomaly_count or 0,
                    "normal_count": (analysis_stats.total_analyses or 0) - (analysis_stats.anomaly_count or 0),
                    "anomaly_rate": round((analysis_stats.anomaly_count or 0) / max(1, analysis_stats.total_analyses or 1) * 100, 1),
                    "average_anomaly_score": round(analysis_stats.avg_anomaly_score or 0, 3),
                    "average_normalized_score": round(analysis_stats.avg_normalized_score or 0, 1),
                    "average_confidence": round(analysis_stats.avg_confidence or 0, 3)
                },
                "attribution_breakdown": [
                    {
                        "attribution_type": attribution.anomaly_attribution or "unknown",
                        "count": attribution.count,
                        "percentage": round(attribution.count / max(1, analysis_stats.total_analyses or 1) * 100, 1)
                    }
                    for attribution in attribution_stats
                ]
            }
        else:
            # Get overall subject statistics
            # Get subject distribution
            subject_distribution = db.query(
                Drawing.subject,
                func.count(Drawing.id).label('count')
            ).group_by(Drawing.subject).order_by(func.count(Drawing.id).desc()).all()
            
            # Get cross-subject analysis comparison
            cross_subject_stats = db.query(
                Drawing.subject,
                func.count(AnomalyAnalysis.id).label('total_analyses'),
                func.sum(func.cast(AnomalyAnalysis.is_anomaly, func.Integer)).label('anomaly_count'),
                func.avg(AnomalyAnalysis.anomaly_score).label('avg_anomaly_score'),
                func.avg(AnomalyAnalysis.normalized_score).label('avg_normalized_score')
            ).join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id).group_by(
                Drawing.subject
            ).order_by(func.count(AnomalyAnalysis.id).desc()).all()
            
            # Get attribution statistics across all subjects
            overall_attribution = db.query(
                AnomalyAnalysis.anomaly_attribution,
                func.count(AnomalyAnalysis.id).label('count')
            ).group_by(AnomalyAnalysis.anomaly_attribution).all()
            
            total_analyses = db.query(AnomalyAnalysis).count()
            
            return {
                "overview": {
                    "total_subjects_with_data": len([s for s in subject_distribution if s.subject]),
                    "total_drawings": sum(s.count for s in subject_distribution),
                    "total_analyses": total_analyses
                },
                "subject_distribution": [
                    {
                        "subject": subj.subject or "unspecified",
                        "drawing_count": subj.count,
                        "percentage": round(subj.count / sum(s.count for s in subject_distribution) * 100, 1)
                    }
                    for subj in subject_distribution
                ],
                "cross_subject_analysis": [
                    {
                        "subject": stat.subject or "unspecified",
                        "total_analyses": stat.total_analyses,
                        "anomaly_count": stat.anomaly_count or 0,
                        "anomaly_rate": round((stat.anomaly_count or 0) / max(1, stat.total_analyses) * 100, 1),
                        "average_anomaly_score": round(stat.avg_anomaly_score or 0, 3),
                        "average_normalized_score": round(stat.avg_normalized_score or 0, 1)
                    }
                    for stat in cross_subject_stats
                ],
                "overall_attribution_breakdown": [
                    {
                        "attribution_type": attr.anomaly_attribution or "unknown",
                        "count": attr.count,
                        "percentage": round(attr.count / max(1, total_analyses) * 100, 1)
                    }
                    for attr in overall_attribution
                ]
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get subject-specific statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subject-specific statistics"
        )


@router.get("/models/subject-aware")
async def get_subject_aware_model_status(db: Session = Depends(get_db)):
    """
    Get status of subject-aware model capabilities.
    
    This endpoint provides information about the current subject-aware
    modeling capabilities and model status.
    """
    try:
        from app.models.database import AgeGroupModel
        
        # Get all active models
        active_models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).all()
        
        # Check subject-aware capabilities
        subject_aware_models = []
        for model in active_models:
            supports_subjects = getattr(model, 'supports_subjects', True)  # Default to True for subject-aware system
            embedding_type = getattr(model, 'embedding_type', 'hybrid')
            
            subject_aware_models.append({
                "model_id": model.id,
                "age_range": f"{model.age_min}-{model.age_max}",
                "supports_subjects": supports_subjects,
                "embedding_type": embedding_type,
                "sample_count": model.sample_count,
                "created_timestamp": model.created_timestamp.isoformat(),
                "is_subject_aware": embedding_type == "hybrid" and supports_subjects
            })
        
        # Calculate overall subject-aware status
        total_models = len(active_models)
        subject_aware_count = sum(1 for model in subject_aware_models if model["is_subject_aware"])
        
        return {
            "subject_aware_system_status": {
                "enabled": True,
                "total_active_models": total_models,
                "subject_aware_models": subject_aware_count,
                "legacy_models": total_models - subject_aware_count,
                "coverage_percentage": round(subject_aware_count / max(1, total_models) * 100, 1)
            },
            "model_details": subject_aware_models,
            "architecture_info": {
                "hybrid_embedding_dimensions": 832,
                "visual_component_dimensions": 768,
                "subject_component_dimensions": 64,
                "supported_subject_categories": 64,
                "default_subject_category": "unspecified"
            },
            "capabilities": {
                "subject_aware_analysis": True,
                "anomaly_attribution": True,
                "component_level_scoring": True,
                "subject_specific_comparisons": True,
                "cross_subject_validation": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get subject-aware model status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve subject-aware model status"
        )


@router.post("/reset")
async def reset_system(
    confirm: bool = False,
    db: Session = Depends(get_db)
):
    """
    Reset system configuration and models.
    
    WARNING: This endpoint deactivates all models and clears caches.
    Use with caution in production environments.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must set confirm=true to reset system"
        )
    
    try:
        # Deactivate all models
        from app.models.database import AgeGroupModel
        models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).all()
        
        for model in models:
            model.is_active = False
        
        db.commit()
        
        # Clear caches
        model_manager.clear_model_cache()
        threshold_manager.clear_threshold_cache()
        
        logger.warning(f"System reset: deactivated {len(models)} models and cleared caches")
        
        return SuccessResponse(
            message=f"System reset completed: {len(models)} models deactivated",
            data={
                "deactivated_models": len(models),
                "caches_cleared": True
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to reset system: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset system"
        )