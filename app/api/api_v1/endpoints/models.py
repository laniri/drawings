"""
Model management API endpoints.
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.database import AgeGroupModel, Drawing, AnomalyAnalysis
from app.schemas.models import (
    AgeGroupModelResponse,
    ModelTrainingRequest,
    ThresholdUpdateRequest,
    ModelStatusResponse,
    ModelListResponse
)
from app.services.model_manager import get_model_manager, TrainingConfig
from app.services.age_group_manager import get_age_group_manager
from app.services.threshold_manager import get_threshold_manager
from app.services.data_sufficiency_service import (
    get_data_sufficiency_analyzer, get_data_augmentation_suggester,
    DataSufficiencyError
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
model_manager = get_model_manager()
age_group_manager = get_age_group_manager()
threshold_manager = get_threshold_manager()
data_sufficiency_analyzer = get_data_sufficiency_analyzer()
data_augmentation_suggester = get_data_augmentation_suggester()


class ModelTrainingTracker:
    """Simple tracker for model training progress"""
    def __init__(self):
        self.training_jobs = {}
    
    def start_training(self, job_id: str, age_min: float, age_max: float):
        self.training_jobs[job_id] = {
            "job_id": job_id,
            "age_range": (age_min, age_max),
            "status": "training",
            "progress": 0,
            "message": "Training started",
            "model_id": None,
            "error": None
        }
    
    def update_training(self, job_id: str, **kwargs):
        if job_id in self.training_jobs:
            self.training_jobs[job_id].update(kwargs)
    
    def get_training_status(self, job_id: str):
        return self.training_jobs.get(job_id)

training_tracker = ModelTrainingTracker()


@router.get("/age-groups", response_model=ModelListResponse)
async def list_age_group_models(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    List available age group models.
    
    This endpoint returns all age group models with their status,
    sample counts, and threshold information.
    """
    try:
        # Build query
        query = db.query(AgeGroupModel)
        if active_only:
            query = query.filter(AgeGroupModel.is_active == True)
        
        models = query.order_by(AgeGroupModel.age_min).all()
        
        # Count models by status
        total_count = len(models)
        active_count = sum(1 for m in models if m.is_active)
        training_count = 0  # Would need to track training jobs for this
        
        model_responses = []
        for model in models:
            # Determine model status based on data availability
            model_status = "ready" if model.is_active else "inactive"
            
            # Check if model has sufficient data
            if model.sample_count < 50:  # Configurable threshold
                model_status = "insufficient_data"
            
            model_response = AgeGroupModelResponse(
                id=model.id,
                age_min=model.age_min,
                age_max=model.age_max,
                model_type=model.model_type,
                vision_model=model.vision_model,
                sample_count=model.sample_count,
                threshold=model.threshold,
                status=model_status,
                created_timestamp=model.created_timestamp,
                is_active=model.is_active
            )
            model_responses.append(model_response)
        
        return ModelListResponse(
            models=model_responses,
            total_count=total_count,
            active_count=active_count,
            training_count=training_count
        )
        
    except Exception as e:
        logger.error(f"Failed to list age group models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model list"
        )


@router.post("/train")
async def train_age_group_model(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Train new age group model.
    
    This endpoint starts training a new autoencoder model for the specified
    age range. Training is performed in the background and progress can be
    tracked using the returned job ID.
    """
    try:
        # Validate age range
        if request.age_max <= request.age_min:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="age_max must be greater than age_min"
            )
        
        # Check if model already exists for this age range
        existing_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.age_min == request.age_min,
            AgeGroupModel.age_max == request.age_max,
            AgeGroupModel.is_active == True
        ).first()
        
        if existing_model:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Active model already exists for age range {request.age_min}-{request.age_max}"
            )
        
        # Check data availability
        drawing_count = db.query(Drawing).filter(
            Drawing.age_years >= request.age_min,
            Drawing.age_years < request.age_max
        ).count()
        
        if drawing_count < request.min_samples:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient data: {drawing_count} drawings available, "
                       f"need at least {request.min_samples}"
            )
        
        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())
        
        # Start training tracking
        training_tracker.start_training(job_id, request.age_min, request.age_max)
        
        # Schedule background training
        background_tasks.add_task(
            train_model_background,
            job_id,
            request,
            db
        )
        
        return {
            "job_id": job_id,
            "age_range": (request.age_min, request.age_max),
            "status": "training",
            "sample_count": drawing_count,
            "progress_url": f"/api/v1/models/training/{job_id}/status"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start model training"
        )


async def train_model_background(job_id: str, request: ModelTrainingRequest, db: Session):
    """Background task for model training"""
    try:
        logger.info(f"Starting background training for job {job_id}")
        
        # Update progress
        training_tracker.update_training(
            job_id,
            status="training",
            progress=10,
            message="Preparing training data"
        )
        
        # Create training configuration
        config = TrainingConfig(
            hidden_dims=[256, 128, 64, 32],  # Default architecture
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            early_stopping_patience=10
        )
        
        # Update progress
        training_tracker.update_training(
            job_id,
            progress=20,
            message="Starting model training"
        )
        
        # Train the model
        result = model_manager.train_age_group_model(
            age_min=request.age_min,
            age_max=request.age_max,
            config=config,
            db=db
        )
        
        # Update progress
        training_tracker.update_training(
            job_id,
            status="completed",
            progress=100,
            message="Training completed successfully",
            model_id=result["model_id"]
        )
        
        logger.info(f"Training job {job_id} completed successfully, model ID: {result['model_id']}")
        
    except Exception as e:
        training_tracker.update_training(
            job_id,
            status="failed",
            progress=0,
            message=f"Training failed: {str(e)}",
            error=str(e)
        )
        logger.error(f"Training job {job_id} failed: {str(e)}")


@router.get("/training/{job_id}/status")
async def get_training_status(job_id: str):
    """Get training job status."""
    status_info = training_tracker.get_training_status(job_id)
    
    if not status_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    return status_info


@router.put("/{model_id}/threshold", response_model=dict)
async def update_model_threshold(
    model_id: int,
    request: ThresholdUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update model threshold.
    
    This endpoint allows updating the anomaly detection threshold
    for a specific age group model. The threshold can be set directly
    or calculated from a percentile of validation data.
    """
    try:
        # Get the model
        model = db.query(AgeGroupModel).filter(AgeGroupModel.id == model_id).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {model_id} not found"
            )
        
        old_threshold = model.threshold
        
        if request.percentile is not None:
            # Calculate threshold from percentile
            threshold_info = threshold_manager.calculate_model_threshold(
                model_id, db, request.percentile
            )
            new_threshold = threshold_info["primary_threshold"]
        else:
            # Use provided threshold directly
            new_threshold = request.threshold
        
        # Update the threshold
        success = threshold_manager.update_model_threshold(model_id, new_threshold, db)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update threshold"
            )
        
        return {
            "model_id": model_id,
            "age_range": (model.age_min, model.age_max),
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "percentile_used": request.percentile,
            "updated_at": model.created_timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update threshold for model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update threshold"
        )


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(db: Session = Depends(get_db)):
    """
    Get model training and system status.
    
    This endpoint provides an overview of the model management system,
    including counts of models in different states and overall system health.
    """
    try:
        # Get model counts
        total_models = db.query(AgeGroupModel).count()
        active_models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).count()
        
        # Count models with insufficient data
        insufficient_data_models = db.query(AgeGroupModel).filter(
            AgeGroupModel.sample_count < 50,  # Configurable threshold
            AgeGroupModel.is_active == True
        ).count()
        
        # Get drawing and analysis counts
        total_drawings = db.query(Drawing).count()
        total_analyses = db.query(AnomalyAnalysis).count()
        
        # Get last training timestamp
        last_model = db.query(AgeGroupModel).order_by(
            AgeGroupModel.created_timestamp.desc()
        ).first()
        
        last_training = last_model.created_timestamp if last_model else None
        
        # Determine system status
        if active_models == 0:
            system_status = "no_models"
        elif insufficient_data_models > 0:
            system_status = "insufficient_data"
        elif total_drawings < 100:  # Minimum recommended drawings
            system_status = "limited_data"
        else:
            system_status = "healthy"
        
        # Count training jobs (would be 0 for now since we don't persist them)
        training_models = 0
        failed_models = total_models - active_models  # Simplified
        
        return ModelStatusResponse(
            total_models=total_models,
            active_models=active_models,
            training_models=training_models,
            failed_models=failed_models,
            total_drawings=total_drawings,
            total_analyses=total_analyses,
            system_status=system_status,
            last_training=last_training
        )
        
    except Exception as e:
        logger.error(f"Failed to get model status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model status"
        )


@router.post("/auto-create")
async def auto_create_age_groups(
    background_tasks: BackgroundTasks,
    force_recreate: bool = False,
    db: Session = Depends(get_db)
):
    """
    Automatically create age group models based on data distribution.
    
    This endpoint analyzes the available drawing data and creates
    appropriate age group models with sufficient sample sizes.
    """
    try:
        # Check if models already exist
        existing_models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).count()
        
        if existing_models > 0 and not force_recreate:
            return {
                "message": f"Found {existing_models} existing models. Use force_recreate=true to recreate.",
                "existing_models": existing_models
            }
        
        # Generate job ID for tracking
        import uuid
        job_id = str(uuid.uuid4())
        
        # Schedule background creation
        background_tasks.add_task(
            auto_create_models_background,
            job_id,
            force_recreate,
            db
        )
        
        return {
            "job_id": job_id,
            "status": "creating",
            "force_recreate": force_recreate,
            "progress_url": f"/api/v1/models/creation/{job_id}/status"
        }
        
    except Exception as e:
        logger.error(f"Failed to start auto-creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start model auto-creation"
        )


async def auto_create_models_background(job_id: str, force_recreate: bool, db: Session):
    """Background task for auto-creating models"""
    try:
        logger.info(f"Starting auto-creation job {job_id}")
        
        # Use age group manager to create models
        created_models = age_group_manager.create_age_groups(db, force_recreate)
        
        logger.info(f"Auto-creation job {job_id} completed: {len(created_models)} models created")
        
    except Exception as e:
        logger.error(f"Auto-creation job {job_id} failed: {str(e)}")


@router.get("/creation/{job_id}/status")
async def get_creation_status(job_id: str):
    """Get model creation job status."""
    # For now, return a simple response since we don't track creation jobs
    return {
        "job_id": job_id,
        "status": "completed",
        "message": "Model creation completed"
    }


@router.delete("/{model_id}")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """
    Delete (deactivate) an age group model.
    
    This endpoint deactivates a model rather than permanently deleting it
    to preserve analysis history.
    """
    try:
        model = db.query(AgeGroupModel).filter(AgeGroupModel.id == model_id).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model with ID {model_id} not found"
            )
        
        # Deactivate the model
        model.is_active = False
        db.commit()
        
        # Clear model from cache
        model_manager.clear_model_cache()
        
        logger.info(f"Model {model_id} deactivated")
        
        return {
            "model_id": model_id,
            "status": "deactivated",
            "message": f"Model {model_id} has been deactivated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model"
        )


# Data Sufficiency Endpoints

@router.get("/data-sufficiency/analyze")
async def analyze_data_sufficiency(
    age_groups: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Analyze data sufficiency for age groups.
    
    This endpoint analyzes the available data for specified age groups
    and provides warnings about insufficient data, unbalanced distributions,
    and other data quality issues.
    
    Args:
        age_groups: Comma-separated list of age ranges (e.g., "3-4,4-5,5-6")
                   If not provided, analyzes all existing age group models
    """
    try:
        # Parse age groups
        if age_groups:
            age_group_list = []
            for group_str in age_groups.split(','):
                parts = group_str.strip().split('-')
                if len(parts) != 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid age group format: {group_str}. Use format 'min-max'"
                    )
                age_min, age_max = float(parts[0]), float(parts[1])
                age_group_list.append((age_min, age_max))
        else:
            # Use existing age group models
            models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).all()
            age_group_list = [(m.age_min, m.age_max) for m in models]
            
            # If no models exist, create default age groups
            if not age_group_list:
                age_group_list = [(i, i+1) for i in range(3, 12)]  # 3-4, 4-5, ..., 11-12
        
        # Analyze each age group
        analysis_results = []
        for age_min, age_max in age_group_list:
            data_info = data_sufficiency_analyzer.analyze_age_group_data(age_min, age_max, db)
            analysis_results.append(data_info.to_dict())
        
        # Generate warnings
        warnings = data_sufficiency_analyzer.generate_data_warnings(age_group_list, db)
        warning_dicts = [w.to_dict() for w in warnings]
        
        # Generate merging suggestions
        merging_suggestions = data_sufficiency_analyzer.suggest_age_group_merging(age_group_list, db)
        suggestion_dicts = [s.to_dict() for s in merging_suggestions]
        
        return {
            "success": True,
            "age_groups_analyzed": len(age_group_list),
            "analysis_results": analysis_results,
            "warnings": warning_dicts,
            "merging_suggestions": suggestion_dicts,
            "summary": {
                "total_warnings": len(warnings),
                "critical_warnings": len([w for w in warnings if w.severity == "critical"]),
                "high_warnings": len([w for w in warnings if w.severity == "high"]),
                "merging_opportunities": len(merging_suggestions)
            }
        }
        
    except HTTPException:
        raise
    except DataSufficiencyError as e:
        logger.error(f"Data sufficiency analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in data sufficiency analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze data sufficiency"
        )


@router.get("/data-sufficiency/age-group/{age_min}/{age_max}")
async def analyze_specific_age_group(
    age_min: float,
    age_max: float,
    db: Session = Depends(get_db)
):
    """
    Analyze data sufficiency for a specific age group.
    
    This endpoint provides detailed analysis of data availability,
    quality, and distribution for a single age group.
    """
    try:
        # Validate age range
        if age_min >= age_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="age_min must be less than age_max"
            )
        
        if age_min < 2.0 or age_max > 18.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Age range must be within 2.0-18.0 years"
            )
        
        # Analyze the age group
        data_info = data_sufficiency_analyzer.analyze_age_group_data(age_min, age_max, db)
        
        # Generate augmentation suggestions
        augmentation_suggestions = data_augmentation_suggester.suggest_augmentation_strategies(data_info)
        
        return {
            "success": True,
            "age_group": {
                "min": age_min,
                "max": age_max
            },
            "data_analysis": data_info.to_dict(),
            "augmentation_suggestions": augmentation_suggestions,
            "recommendations": {
                "can_train": data_info.is_sufficient,
                "needs_augmentation": data_info.sample_count < 100,
                "quality_rating": "excellent" if data_info.data_quality_score > 0.8 else
                                "good" if data_info.data_quality_score > 0.6 else
                                "fair" if data_info.data_quality_score > 0.4 else "poor"
            }
        }
        
    except HTTPException:
        raise
    except DataSufficiencyError as e:
        logger.error(f"Age group analysis failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error in age group analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze age group"
        )


@router.post("/data-sufficiency/merge-age-groups")
async def merge_age_groups(
    original_groups: List[List[float]],  # List of [age_min, age_max] pairs
    merged_group: List[float],  # [age_min, age_max] for merged group
    db: Session = Depends(get_db)
):
    """
    Merge age groups to improve data sufficiency.
    
    This endpoint deactivates the original age group models and creates
    a new merged age group model with combined data.
    """
    try:
        # Validate input
        if len(merged_group) != 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="merged_group must contain exactly 2 values [age_min, age_max]"
            )
        
        merged_min, merged_max = merged_group
        if merged_min >= merged_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Merged group age_min must be less than age_max"
            )
        
        # Validate original groups
        for group in original_groups:
            if len(group) != 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each original group must contain exactly 2 values [age_min, age_max]"
                )
        
        # Check if merged group already exists
        existing_merged = db.query(AgeGroupModel).filter(
            AgeGroupModel.age_min == merged_min,
            AgeGroupModel.age_max == merged_max,
            AgeGroupModel.is_active == True
        ).first()
        
        if existing_merged:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Merged age group {merged_min}-{merged_max} already exists"
            )
        
        # Deactivate original models
        deactivated_models = []
        for age_min, age_max in original_groups:
            model = db.query(AgeGroupModel).filter(
                AgeGroupModel.age_min == age_min,
                AgeGroupModel.age_max == age_max,
                AgeGroupModel.is_active == True
            ).first()
            
            if model:
                model.is_active = False
                deactivated_models.append(model.id)
        
        # Calculate combined sample count
        combined_count = db.query(Drawing).filter(
            Drawing.age_years >= merged_min,
            Drawing.age_years < merged_max
        ).count()
        
        # Create new merged model (placeholder - would need actual training)
        import json
        merged_model = AgeGroupModel(
            age_min=merged_min,
            age_max=merged_max,
            model_type="autoencoder",
            vision_model="vit",
            parameters=json.dumps({
                "merged_from": original_groups,
                "created_by_merge": True
            }),
            sample_count=combined_count,
            threshold=0.95,  # Default threshold
            is_active=True
        )
        
        db.add(merged_model)
        db.commit()
        db.refresh(merged_model)
        
        logger.info(f"Merged age groups {original_groups} into {merged_min}-{merged_max}")
        
        return {
            "success": True,
            "merged_model_id": merged_model.id,
            "merged_age_group": [merged_min, merged_max],
            "deactivated_models": deactivated_models,
            "combined_sample_count": combined_count,
            "message": f"Successfully merged {len(original_groups)} age groups into {merged_min}-{merged_max}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to merge age groups: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to merge age groups"
        )


@router.get("/data-sufficiency/warnings")
async def get_data_warnings(
    severity: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get data sufficiency warnings for all age groups.
    
    This endpoint returns warnings about data quality issues,
    optionally filtered by severity level.
    """
    try:
        # Get all active age group models
        models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).all()
        age_groups = [(m.age_min, m.age_max) for m in models]
        
        # If no models exist, use default age groups
        if not age_groups:
            age_groups = [(i, i+1) for i in range(3, 12)]
        
        # Generate warnings
        warnings = data_sufficiency_analyzer.generate_data_warnings(age_groups, db)
        
        # Filter by severity if specified
        if severity:
            warnings = [w for w in warnings if w.severity == severity]
        
        warning_dicts = [w.to_dict() for w in warnings]
        
        # Group warnings by type
        warnings_by_type = {}
        for warning in warnings:
            warning_type = warning.warning_type
            if warning_type not in warnings_by_type:
                warnings_by_type[warning_type] = []
            warnings_by_type[warning_type].append(warning.to_dict())
        
        return {
            "success": True,
            "total_warnings": len(warnings),
            "severity_filter": severity,
            "warnings": warning_dicts,
            "warnings_by_type": warnings_by_type,
            "severity_counts": {
                "critical": len([w for w in warnings if w.severity == "critical"]),
                "high": len([w for w in warnings if w.severity == "high"]),
                "medium": len([w for w in warnings if w.severity == "medium"]),
                "low": len([w for w in warnings if w.severity == "low"])
            }
        }
        
    except DataSufficiencyError as e:
        logger.error(f"Failed to generate warnings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error generating warnings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate data warnings"
        )