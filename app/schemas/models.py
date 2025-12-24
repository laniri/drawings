"""
Pydantic schemas for model management and configuration.

This module contains request and response models for age group models,
threshold management, and system configuration.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .analysis import AnalysisMethod, VisionModel


class AgeGroupingStrategy(str, Enum):
    """Enumeration for age grouping strategies."""

    YEARLY = "yearly"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Enumeration for model training status."""

    TRAINING = "training"
    READY = "ready"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


class AgeGroupModelResponse(BaseModel):
    """Response model for age group model information."""

    id: int
    age_min: float = Field(..., ge=2.0, le=18.0)
    age_max: float = Field(..., ge=2.0, le=18.0)
    model_type: AnalysisMethod
    vision_model: VisionModel
    sample_count: int = Field(..., ge=0)
    threshold: float = Field(..., gt=0.0)
    status: ModelStatus
    created_timestamp: datetime
    is_active: bool

    model_config = {"from_attributes": True}

    @field_validator("age_max")
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_max is greater than age_min."""
        if "age_min" in info.data and v <= info.data["age_min"]:
            raise ValueError("age_max must be greater than age_min")
        return v


class ModelTrainingRequest(BaseModel):
    """Request model for training a new age group model."""

    age_min: float = Field(..., ge=2.0, le=18.0)
    age_max: float = Field(..., ge=2.0, le=18.0)
    model_type: AnalysisMethod = AnalysisMethod.AUTOENCODER
    vision_model: VisionModel = VisionModel.VIT
    min_samples: int = Field(
        50, ge=10, description="Minimum samples required for training"
    )

    @field_validator("age_max")
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_max is greater than age_min."""
        if "age_min" in info.data and v <= info.data["age_min"]:
            raise ValueError("age_max must be greater than age_min")
        return v


class ThresholdUpdateRequest(BaseModel):
    """Request model for updating model thresholds."""

    threshold: float = Field(..., gt=0.0, description="New threshold value")
    percentile: Optional[float] = Field(
        None,
        ge=50.0,
        le=99.9,
        description="Percentile for automatic threshold calculation",
    )


class SystemConfigurationResponse(BaseModel):
    """Response model for system configuration."""

    vision_model: VisionModel
    anomaly_detection_method: AnalysisMethod
    threshold_percentile: float = Field(..., ge=50.0, le=99.9)
    age_grouping_strategy: AgeGroupingStrategy
    min_samples_per_group: int = Field(..., ge=10)
    max_age_group_span: float = Field(..., gt=0.0, le=16.0)

    model_config = {"from_attributes": True}


class ConfigurationUpdateRequest(BaseModel):
    """Request model for updating system configuration."""

    threshold_percentile: Optional[float] = Field(
        None, ge=50.0, le=99.9, description="Percentile for threshold calculation"
    )
    age_grouping_strategy: Optional[AgeGroupingStrategy] = None
    min_samples_per_group: Optional[int] = Field(None, ge=10)
    max_age_group_span: Optional[float] = Field(None, gt=0.0, le=16.0)


class ModelStatusResponse(BaseModel):
    """Response model for model training and system status."""

    total_models: int = Field(..., ge=0)
    active_models: int = Field(..., ge=0)
    training_models: int = Field(..., ge=0)
    failed_models: int = Field(..., ge=0)
    total_drawings: int = Field(..., ge=0)
    total_analyses: int = Field(..., ge=0)
    system_status: str
    last_training: Optional[datetime] = None

    @field_validator("active_models", "training_models", "failed_models")
    @classmethod
    def validate_model_counts(cls, v, info):
        """Validate that model counts don't exceed total."""
        if "total_models" in info.data:
            total = info.data["total_models"]
            if v > total:
                raise ValueError(f"Model count cannot exceed total_models ({total})")
        return v


class ModelListResponse(BaseModel):
    """Response model for listing age group models."""

    models: List[AgeGroupModelResponse]
    total_count: int
    active_count: int
    training_count: int


class TrainingEnvironment(str, Enum):
    """Enumeration for training environments."""

    LOCAL = "local"
    SAGEMAKER = "sagemaker"


class TrainingConfigRequest(BaseModel):
    """Request model for training job configuration."""

    job_name: str = Field(..., description="Unique name for the training job")
    environment: TrainingEnvironment
    dataset_folder: str = Field(..., description="Path to folder containing drawings")
    metadata_file: str = Field(..., description="Path to metadata CSV/JSON file")

    # Training parameters
    learning_rate: float = Field(0.001, ge=1e-6, le=1.0)
    batch_size: int = Field(32, ge=1, le=512)
    epochs: int = Field(100, ge=1, le=1000)
    train_split: float = Field(0.7, ge=0.1, le=0.9)
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    test_split: float = Field(0.1, ge=0.05, le=0.3)

    # SageMaker specific parameters
    instance_type: Optional[str] = Field(
        "ml.m5.large", description="SageMaker instance type"
    )
    instance_count: int = Field(1, ge=1, le=10)

    @field_validator("train_split", "validation_split", "test_split")
    @classmethod
    def validate_splits_sum_to_one(cls, v, info):
        """Validate that train, validation, and test splits sum to 1.0."""
        if (
            "train_split" in info.data
            and "validation_split" in info.data
            and "test_split" in info.data
        ):
            total = (
                info.data["train_split"]
                + info.data["validation_split"]
                + info.data["test_split"]
            )
            if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                raise ValueError(
                    "train_split + validation_split + test_split must equal 1.0"
                )
        return v


class TrainingJobResponse(BaseModel):
    """Response model for training job information."""

    id: int
    job_name: str
    environment: str
    status: str
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    sagemaker_job_arn: Optional[str]

    model_config = {"from_attributes": True}


class TrainingReportResponse(BaseModel):
    """Response model for training report information."""

    id: int
    final_loss: float
    validation_accuracy: float
    best_epoch: int
    training_time_seconds: float
    model_parameters_path: str
    report_file_path: str
    created_timestamp: datetime

    model_config = {"from_attributes": True}


class ModelDeploymentRequest(BaseModel):
    """Request model for deploying trained model parameters."""

    model_parameters_path: str = Field(
        ..., description="Path to trained model parameters"
    )
    age_group_min: float = Field(..., ge=2.0, le=18.0)
    age_group_max: float = Field(..., ge=2.0, le=18.0)
    replace_existing: bool = Field(
        False, description="Whether to replace existing model for age group"
    )

    @field_validator("age_group_max")
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_group_max is greater than age_group_min."""
        if "age_group_min" in info.data and v <= info.data["age_group_min"]:
            raise ValueError("age_group_max must be greater than age_group_min")
        return v


class ModelExportFormat(str, Enum):
    """Enumeration for model export formats."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    PICKLE = "pickle"


class ModelExportRequest(BaseModel):
    """Request model for exporting trained models."""

    training_job_id: int = Field(..., description="ID of the training job to export")
    age_group_min: float = Field(..., ge=2.0, le=18.0)
    age_group_max: float = Field(..., ge=2.0, le=18.0)
    export_format: ModelExportFormat = Field(
        ModelExportFormat.PYTORCH, description="Export format"
    )

    @field_validator("age_group_max")
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_group_max is greater than age_group_min."""
        if "age_group_min" in info.data and v <= info.data["age_group_min"]:
            raise ValueError("age_group_max must be greater than age_group_min")
        return v


class ModelExportResponse(BaseModel):
    """Response model for model export results."""

    success: bool
    model_id: str
    export_path: str
    metadata: Dict[str, Any]
    message: str


class ModelValidationResponse(BaseModel):
    """Response model for model validation results."""

    success: bool
    model_id: str
    validation_result: Dict[str, Any]
    is_valid: bool
    message: str


class EnhancedModelDeploymentRequest(BaseModel):
    """Enhanced request model for deploying exported models."""

    model_export_path: str = Field(..., description="Path to exported model file")
    age_group_min: float = Field(..., ge=2.0, le=18.0)
    age_group_max: float = Field(..., ge=2.0, le=18.0)
    replace_existing: bool = Field(
        False, description="Whether to replace existing model"
    )
    validate_before_deployment: bool = Field(
        True, description="Validate model before deployment"
    )
    backup_existing: bool = Field(
        True, description="Backup existing model before replacement"
    )
    deployment_environment: str = Field(
        "production", description="Target deployment environment"
    )

    @field_validator("age_group_max")
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_group_max is greater than age_group_min."""
        if "age_group_min" in info.data and v <= info.data["age_group_min"]:
            raise ValueError("age_group_max must be greater than age_group_min")
        return v


class ModelDeploymentResponse(BaseModel):
    """Response model for model deployment results."""

    success: bool
    model_id: Optional[str]
    deployment_path: Optional[str]
    backup_path: Optional[str]
    validation_result: Optional[Dict[str, Any]]
    database_updated: bool
    warnings: List[str]
    message: str


class ExportedModelInfo(BaseModel):
    """Information about an exported model."""

    model_id: str
    export_timestamp: datetime
    training_job_id: int
    model_type: str
    model_version: str
    age_group_min: float
    age_group_max: float
    export_format: str
    file_size_bytes: int
    checksum: str
    training_metrics: Dict[str, Any]


class ExportedModelsListResponse(BaseModel):
    """Response model for listing exported models."""

    success: bool
    count: int
    models: List[Dict[str, Any]]


class DeployedModelInfo(BaseModel):
    """Information about a deployed model."""

    id: int
    age_min: float
    age_max: float
    model_type: str
    vision_model: str
    threshold: float
    created_timestamp: datetime
    deployment_info: Dict[str, Any]


class DeployedModelsListResponse(BaseModel):
    """Response model for listing deployed models."""

    success: bool
    count: int
    models: List[DeployedModelInfo]
