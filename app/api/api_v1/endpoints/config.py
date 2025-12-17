"""
Configuration API endpoints.
"""

import logging
from typing import Dict, Any
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