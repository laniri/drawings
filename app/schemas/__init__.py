# Pydantic schemas package

from .drawings import (
    ExpertLabel,
    DrawingUploadRequest,
    DrawingResponse,
    DrawingListResponse,
    DrawingFilterRequest,
)

from .analysis import (
    AnalysisMethod,
    VisionModel,
    AnalysisRequest,
    BatchAnalysisRequest,
    AnomalyAnalysisResponse,
    InterpretabilityResponse,
    AnalysisResultResponse,
    BatchAnalysisResponse,
    AnalysisHistoryResponse,
)

from .models import (
    AgeGroupingStrategy,
    ModelStatus,
    AgeGroupModelResponse,
    ModelTrainingRequest,
    ThresholdUpdateRequest,
    SystemConfigurationResponse,
    ConfigurationUpdateRequest,
    ModelStatusResponse,
    ModelListResponse,
)

from .common import (
    ErrorDetail,
    ErrorResponse,
    SuccessResponse,
    HealthCheckResponse,
    PaginationInfo,
    FileUploadResponse,
    ValidationErrorDetail,
    ImageFormat,
    ImageValidationRequest,
)

__all__ = [
    # Drawing schemas
    "ExpertLabel",
    "DrawingUploadRequest",
    "DrawingResponse",
    "DrawingListResponse",
    "DrawingFilterRequest",
    
    # Analysis schemas
    "AnalysisMethod",
    "VisionModel",
    "AnalysisRequest",
    "BatchAnalysisRequest",
    "AnomalyAnalysisResponse",
    "InterpretabilityResponse",
    "AnalysisResultResponse",
    "BatchAnalysisResponse",
    "AnalysisHistoryResponse",
    
    # Model management schemas
    "AgeGroupingStrategy",
    "ModelStatus",
    "AgeGroupModelResponse",
    "ModelTrainingRequest",
    "ThresholdUpdateRequest",
    "SystemConfigurationResponse",
    "ConfigurationUpdateRequest",
    "ModelStatusResponse",
    "ModelListResponse",
    
    # Common schemas
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",
    "HealthCheckResponse",
    "PaginationInfo",
    "FileUploadResponse",
    "ValidationErrorDetail",
    "ImageFormat",
    "ImageValidationRequest",
]