"""
Training environment API endpoints for SageMaker and local training.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import ValidationError
from app.models.database import TrainingJob, TrainingReport
from app.schemas.models import (
    ModelDeploymentRequest,
    TrainingConfigRequest,
    TrainingEnvironment,
    TrainingJobResponse,
    TrainingReportResponse,
)
from app.services.local_training_environment import (
    LocalTrainingError,
    get_local_training_environment,
)
from app.services.model_deployment_service import (
    ModelDeploymentConfig,
    ModelDeploymentError,
    ModelExportError,
    ModelValidationError,
    get_model_deployment_service,
    get_model_exporter,
    get_model_validator,
)
from app.services.sagemaker_training_service import (
    SageMakerError,
    get_sagemaker_training_service,
)
from app.services.training_config import (
    TrainingConfig,
    TrainingConfigManager,
    get_training_config_manager,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
sagemaker_service = (
    get_sagemaker_training_service()
)  # May be None if AWS not configured
local_training_service = get_local_training_environment()
config_manager = get_training_config_manager()
model_exporter = get_model_exporter()
model_validator = get_model_validator()
model_deployment_service = get_model_deployment_service()


@router.post("/jobs", response_model=TrainingJobResponse)
async def submit_training_job(
    request: TrainingConfigRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Submit a new training job to either local or SageMaker environment.

    This endpoint creates and submits a training job based on the specified
    environment. For SageMaker jobs, it handles container building, data upload,
    and job submission. For local jobs, it starts training immediately.
    """
    try:
        # Convert request to training config
        training_config = TrainingConfig(
            job_name=request.job_name,
            environment=request.environment,
            dataset_folder=request.dataset_folder,
            metadata_file=request.metadata_file,
            epochs=request.epochs,
            sagemaker_instance_type=request.instance_type,
            sagemaker_instance_count=request.instance_count,
        )

        # Update data and optimizer configs
        training_config.data.batch_size = request.batch_size
        training_config.data.train_split = request.train_split
        training_config.data.validation_split = request.validation_split
        training_config.data.test_split = request.test_split
        training_config.optimizer.learning_rate = request.learning_rate

        # Validate configuration
        validation_result = config_manager.validate_config(training_config)
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid training configuration: {validation_result['errors']}",
            )

        if request.environment == TrainingEnvironment.LOCAL:
            # Submit local training job
            job_id = await submit_local_training_job(training_config, db)
        else:
            # Submit SageMaker training job
            job_id = await submit_sagemaker_training_job(
                training_config, background_tasks, db
            )

        # Get job details
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        return TrainingJobResponse(
            id=training_job.id,
            job_name=training_job.job_name,
            environment=training_job.environment,
            status=training_job.status,
            start_timestamp=training_job.start_timestamp,
            end_timestamp=training_job.end_timestamp,
            sagemaker_job_arn=training_job.sagemaker_job_arn,
        )

    except HTTPException:
        raise
    except (ValidationError, LocalTrainingError, SageMakerError) as e:
        logger.error(f"Training job submission failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error submitting training job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit training job",
        )


async def submit_local_training_job(config: TrainingConfig, db: Session) -> int:
    """Submit local training job."""
    try:
        # Prepare training data
        (
            train_embeddings,
            val_embeddings,
            test_embeddings,
        ) = local_training_service.prepare_training_data(
            config.dataset_folder, config.metadata_file, config
        )

        # Start training job
        job_id = local_training_service.start_training_job(
            config, train_embeddings, val_embeddings, db
        )

        logger.info(f"Local training job {job_id} submitted successfully")
        return job_id

    except Exception as e:
        logger.error(f"Failed to submit local training job: {str(e)}")
        raise LocalTrainingError(f"Local training submission failed: {str(e)}")


async def submit_sagemaker_training_job(
    config: TrainingConfig, background_tasks: BackgroundTasks, db: Session
) -> int:
    """Submit SageMaker training job."""
    try:
        # Check if SageMaker service is available
        if sagemaker_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SageMaker service is not available. Please configure AWS credentials.",
            )

        # Validate SageMaker configuration
        validation_result = sagemaker_service.validate_configuration()
        if not validation_result["is_valid"]:
            raise SageMakerError(
                f"SageMaker configuration invalid: {validation_result['errors']}"
            )

        # Schedule SageMaker job preparation in background
        background_tasks.add_task(prepare_and_submit_sagemaker_job, config, db)

        # Create initial job record
        training_job = TrainingJob(
            job_name=config.job_name,
            environment="sagemaker",
            config_parameters="{}",  # Will be updated by background task
            dataset_path=config.dataset_folder,
            status="preparing",
        )

        db.add(training_job)
        db.commit()
        db.refresh(training_job)

        logger.info(f"SageMaker training job {training_job.id} queued for preparation")
        return training_job.id

    except Exception as e:
        logger.error(f"Failed to submit SageMaker training job: {str(e)}")
        raise SageMakerError(f"SageMaker training submission failed: {str(e)}")


async def prepare_and_submit_sagemaker_job(config: TrainingConfig, db: Session):
    """Background task to prepare and submit SageMaker job."""
    try:
        logger.info(f"Preparing SageMaker job: {config.job_name}")

        # Build training container
        container_tag = sagemaker_service.container_builder.build_training_container()

        # Create execution role if needed
        role_arn = sagemaker_service.create_execution_role(
            "SageMakerDrawingAnomalyRole"
        )

        # Upload training data to S3
        s3_bucket = "drawing-anomaly-training"  # Should be configurable
        s3_prefix = f"training-data/{config.job_name}"

        s3_input_uri = sagemaker_service.upload_training_data(
            config.dataset_folder, config.metadata_file, s3_bucket, s3_prefix
        )

        # Set S3 output path
        s3_output_uri = f"s3://{s3_bucket}/training-output/{config.job_name}"

        # Push container to ECR (would need ECR repository setup)
        # ecr_uri = sagemaker_service.container_builder.push_to_ecr(container_tag, ecr_repo_uri)

        # For now, use the local container tag
        image_uri = container_tag

        # Submit training job
        job_id = sagemaker_service.submit_training_job(
            config, s3_input_uri, s3_output_uri, role_arn, image_uri, db
        )

        logger.info(f"SageMaker job prepared and submitted: {job_id}")

    except Exception as e:
        logger.error(f"Failed to prepare SageMaker job: {str(e)}")
        # Update job status to failed
        # This would require finding the job record and updating it


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(
    environment: Optional[TrainingEnvironment] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    List training jobs with optional filtering.

    This endpoint returns a list of training jobs, optionally filtered
    by environment (local/sagemaker) and status.
    """
    try:
        query = db.query(TrainingJob)

        if environment:
            query = query.filter(TrainingJob.environment == environment.value)

        if status:
            query = query.filter(TrainingJob.status == status)

        jobs = query.order_by(TrainingJob.start_timestamp.desc()).limit(limit).all()

        return [
            TrainingJobResponse(
                id=job.id,
                job_name=job.job_name,
                environment=job.environment,
                status=job.status,
                start_timestamp=job.start_timestamp,
                end_timestamp=job.end_timestamp,
                sagemaker_job_arn=job.sagemaker_job_arn,
            )
            for job in jobs
        ]

    except Exception as e:
        logger.error(f"Failed to list training jobs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training jobs",
        )


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_training_job_status(job_id: int, db: Session = Depends(get_db)):
    """
    Get detailed status of a specific training job.

    This endpoint returns comprehensive information about a training job,
    including progress, metrics, and environment-specific details.
    """
    try:
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not training_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found",
            )

        if training_job.environment == "local":
            status_info = local_training_service.get_job_status(job_id, db)
        else:
            if sagemaker_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="SageMaker service is not available",
                )
            status_info = sagemaker_service.get_job_status(job_id, db)

        return status_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve job status",
        )


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(job_id: int, db: Session = Depends(get_db)):
    """
    Cancel a running training job.

    This endpoint attempts to cancel a training job. For local jobs,
    it stops the training process. For SageMaker jobs, it stops the
    SageMaker training job.
    """
    try:
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not training_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found",
            )

        if training_job.status not in ["pending", "running", "preparing"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in status: {training_job.status}",
            )

        if training_job.environment == "local":
            success = local_training_service.cancel_training_job(job_id, db)
        else:
            if sagemaker_service is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="SageMaker service is not available",
                )
            success = sagemaker_service.cancel_training_job(job_id, db)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to cancel training job",
            )

        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Training job cancelled successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel training job",
        )


@router.get("/jobs/{job_id}/reports", response_model=List[TrainingReportResponse])
async def get_training_reports(job_id: int, db: Session = Depends(get_db)):
    """
    Get training reports for a specific job.

    This endpoint returns all training reports associated with a job,
    including metrics, model paths, and performance summaries.
    """
    try:
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not training_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found",
            )

        reports = (
            db.query(TrainingReport)
            .filter(TrainingReport.training_job_id == job_id)
            .order_by(TrainingReport.created_timestamp.desc())
            .all()
        )

        return [
            TrainingReportResponse(
                id=report.id,
                final_loss=report.final_loss,
                validation_accuracy=report.validation_accuracy,
                best_epoch=report.best_epoch,
                training_time_seconds=report.training_time_seconds,
                model_parameters_path=report.model_parameters_path,
                report_file_path=report.report_file_path,
                created_timestamp=report.created_timestamp,
            )
            for report in reports
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training reports: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve training reports",
        )


@router.post("/deploy")
async def deploy_trained_model(
    request: ModelDeploymentRequest, db: Session = Depends(get_db)
):
    """
    Deploy trained model parameters to production system.

    This endpoint loads trained model parameters and creates a new
    age group model for production use.
    """
    try:
        # Validate model parameters file exists
        from pathlib import Path

        model_path = Path(request.model_parameters_path)
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model parameters file not found: {request.model_parameters_path}",
            )

        # Check if model already exists for this age group
        if not request.replace_existing:
            existing_model = (
                db.query(AgeGroupModel)
                .filter(
                    AgeGroupModel.age_min == request.age_group_min,
                    AgeGroupModel.age_max == request.age_group_max,
                    AgeGroupModel.is_active == True,
                )
                .first()
            )

            if existing_model:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Model already exists for age group {request.age_group_min}-{request.age_group_max}",
                )

        # Load and validate model parameters
        import torch

        try:
            model_data = torch.load(model_path, map_location="cpu")
            model_architecture = model_data.get("model_architecture", {})
            training_config = model_data.get("training_config", {})
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model parameters file: {str(e)}",
            )

        # Create new age group model
        import json

        from app.models.database import AgeGroupModel

        # Deactivate existing model if replacing
        if request.replace_existing:
            existing_models = (
                db.query(AgeGroupModel)
                .filter(
                    AgeGroupModel.age_min == request.age_group_min,
                    AgeGroupModel.age_max == request.age_group_max,
                    AgeGroupModel.is_active == True,
                )
                .all()
            )

            for model in existing_models:
                model.is_active = False

        # Create new model record
        new_model = AgeGroupModel(
            age_min=request.age_group_min,
            age_max=request.age_group_max,
            model_type="autoencoder",
            vision_model="vit",
            parameters=json.dumps(
                {
                    "model_path": str(model_path),
                    "architecture": model_architecture,
                    "training_config": training_config,
                }
            ),
            sample_count=training_config.get("sample_count", 0),
            threshold=0.95,  # Default threshold, should be calculated
            is_active=True,
        )

        db.add(new_model)
        db.commit()
        db.refresh(new_model)

        logger.info(
            f"Deployed model {new_model.id} for age group {request.age_group_min}-{request.age_group_max}"
        )

        return {
            "model_id": new_model.id,
            "age_group_min": request.age_group_min,
            "age_group_max": request.age_group_max,
            "model_path": str(model_path),
            "replaced_existing": request.replace_existing,
            "status": "deployed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy trained model",
        )


@router.get("/environments/status")
async def get_training_environments_status():
    """
    Get status of available training environments.

    This endpoint returns information about local and SageMaker
    training environments, including availability and configuration.
    """
    try:
        # Get local environment info
        local_info = local_training_service.get_environment_info()

        # Get SageMaker environment info
        if sagemaker_service is not None:
            sagemaker_validation = sagemaker_service.validate_configuration()
        else:
            sagemaker_validation = {
                "is_valid": False,
                "errors": [
                    "SageMaker service not available - AWS credentials not configured"
                ],
            }

        return {
            "local": {
                "available": True,
                "device_info": local_info["device_info"],
                "memory_usage": local_info["memory_usage"],
                "active_jobs": len(local_info["active_jobs"]),
                "pytorch_version": local_info["pytorch_version"],
            },
            "sagemaker": {
                "available": sagemaker_validation["is_valid"],
                "configuration": sagemaker_validation["configuration"],
                "errors": sagemaker_validation["errors"],
                "warnings": sagemaker_validation["warnings"],
            },
        }

    except Exception as e:
        logger.error(f"Failed to get environment status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve environment status",
        )


@router.post("/sagemaker/setup")
async def setup_sagemaker_environment(
    s3_bucket: str, ecr_repository: Optional[str] = None
):
    """
    Setup SageMaker training environment.

    This endpoint helps set up the necessary AWS resources for
    SageMaker training, including IAM roles and container repositories.
    """
    try:
        # Check if SageMaker service is available
        if sagemaker_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SageMaker service is not available. Please configure AWS credentials.",
            )

        # Validate SageMaker access
        validation_result = sagemaker_service.validate_configuration()
        if not validation_result["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"SageMaker not properly configured: {validation_result['errors']}",
            )

        # Create execution role
        role_arn = sagemaker_service.create_execution_role(
            "SageMakerDrawingAnomalyRole"
        )

        # Build training container
        container_tag = sagemaker_service.container_builder.build_training_container()

        setup_info = {
            "execution_role_arn": role_arn,
            "training_container_tag": container_tag,
            "s3_bucket": s3_bucket,
            "ecr_repository": ecr_repository,
            "status": "ready",
        }

        if ecr_repository:
            # Push container to ECR
            ecr_uri = sagemaker_service.container_builder.push_to_ecr(
                container_tag, ecr_repository
            )
            setup_info["ecr_image_uri"] = ecr_uri

        return setup_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to setup SageMaker environment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup SageMaker environment",
        )


# Model Export and Deployment Endpoints


@router.post("/models/export")
async def export_model_from_training_job(
    training_job_id: int,
    age_group_min: float,
    age_group_max: float,
    export_format: str = "pytorch",
    db: Session = Depends(get_db),
):
    """
    Export trained model from training job in production-compatible format.

    This endpoint exports a trained model from a completed training job,
    creating a production-ready model file with metadata and validation.
    """
    try:
        # Validate age group range
        if age_group_min >= age_group_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid age group range: min must be less than max",
            )

        if age_group_min < 2.0 or age_group_max > 18.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Age group must be within range 2.0-18.0 years",
            )

        # Export model
        export_metadata = model_exporter.export_model_from_training_job(
            training_job_id=training_job_id,
            age_group_min=age_group_min,
            age_group_max=age_group_max,
            export_format=export_format,
            db=db,
        )

        return {
            "success": True,
            "model_id": export_metadata.model_id,
            "export_path": f"exports/models/{export_metadata.model_id}.{export_metadata.export_format}",
            "metadata": export_metadata.to_dict(),
            "message": "Model exported successfully",
        }

    except HTTPException:
        raise
    except ModelExportError as e:
        logger.error(f"Model export failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model export: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export model",
        )


@router.get("/models/exports")
async def list_exported_models():
    """
    List all exported models with their metadata.

    This endpoint returns a list of all models that have been exported,
    including their metadata, export timestamps, and file information.
    """
    try:
        exported_models = model_exporter.list_exported_models()

        return {
            "success": True,
            "count": len(exported_models),
            "models": exported_models,
        }

    except Exception as e:
        logger.error(f"Failed to list exported models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exported models",
        )


@router.post("/models/validate")
async def validate_exported_model(model_id: str):
    """
    Validate exported model for compatibility and integrity.

    This endpoint performs comprehensive validation of an exported model,
    checking file integrity, compatibility, and performance metrics.
    """
    try:
        # Find model metadata
        exported_models = model_exporter.list_exported_models()
        model_metadata = None

        for model in exported_models:
            if model.get("model_id") == model_id:
                model_metadata = model
                break

        if not model_metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exported model not found: {model_id}",
            )

        # Reconstruct metadata object
        from datetime import datetime

        from app.services.model_deployment_service import ModelExportMetadata

        metadata_dict = model_metadata.copy()
        metadata_dict["export_timestamp"] = datetime.fromisoformat(
            metadata_dict["export_timestamp"]
        )
        export_metadata = ModelExportMetadata(**metadata_dict)

        # Validate model
        validation_result = model_validator.validate_exported_model(export_metadata)

        return {
            "success": True,
            "model_id": model_id,
            "validation_result": validation_result,
            "is_valid": validation_result["is_valid"],
            "message": "Model validation completed",
        }

    except HTTPException:
        raise
    except ModelValidationError as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model validation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate model",
        )


@router.post("/models/deploy")
async def deploy_exported_model(
    model_export_path: str,
    age_group_min: float,
    age_group_max: float,
    replace_existing: bool = False,
    validate_before_deployment: bool = True,
    backup_existing: bool = True,
    db: Session = Depends(get_db),
):
    """
    Deploy exported model to production environment.

    This endpoint deploys an exported model to the production system,
    making it available for anomaly detection in the specified age group.
    """
    try:
        # Validate age group range
        if age_group_min >= age_group_max:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid age group range: min must be less than max",
            )

        # Create deployment configuration
        deployment_config = ModelDeploymentConfig(
            model_export_path=model_export_path,
            age_group_min=age_group_min,
            age_group_max=age_group_max,
            replace_existing=replace_existing,
            validate_before_deployment=validate_before_deployment,
            backup_existing=backup_existing,
            deployment_environment="production",
        )

        # Deploy model
        deployment_result = model_deployment_service.deploy_model(deployment_config, db)

        return {
            "success": deployment_result["success"],
            "model_id": deployment_result["model_id"],
            "deployment_path": deployment_result["deployment_path"],
            "backup_path": deployment_result.get("backup_path"),
            "validation_result": deployment_result.get("validation_result"),
            "database_updated": deployment_result["database_updated"],
            "warnings": deployment_result.get("warnings", []),
            "message": "Model deployed successfully",
        }

    except HTTPException:
        raise
    except ModelDeploymentError as e:
        logger.error(f"Model deployment failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during model deployment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy model",
        )


@router.get("/models/deployed")
async def list_deployed_models(db: Session = Depends(get_db)):
    """
    List all deployed models in production.

    This endpoint returns information about all models currently
    deployed and active in the production system.
    """
    try:
        deployed_models = model_deployment_service.list_deployed_models(db)

        return {
            "success": True,
            "count": len(deployed_models),
            "models": deployed_models,
        }

    except Exception as e:
        logger.error(f"Failed to list deployed models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve deployed models",
        )


@router.post("/models/{model_id}/undeploy")
async def undeploy_model(model_id: int, db: Session = Depends(get_db)):
    """
    Undeploy (deactivate) a deployed model.

    This endpoint deactivates a deployed model, removing it from
    active use in the production system.
    """
    try:
        success = model_deployment_service.undeploy_model(model_id, db)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Deployed model not found: {model_id}",
            )

        return {
            "success": True,
            "model_id": model_id,
            "status": "undeployed",
            "message": "Model undeployed successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to undeploy model {model_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to undeploy model",
        )
