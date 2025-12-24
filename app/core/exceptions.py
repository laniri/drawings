"""
Custom exception classes and error handling utilities.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException, status


class DrawingAnalysisException(Exception):
    """Base exception for drawing analysis system."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(DrawingAnalysisException):
    """Raised when input validation fails."""

    pass


class ImageProcessingError(DrawingAnalysisException):
    """Raised when image processing fails."""

    pass


class ModelError(DrawingAnalysisException):
    """Raised when ML model operations fail."""

    pass


class StorageError(DrawingAnalysisException):
    """Raised when file storage operations fail."""

    pass


class ConfigurationError(DrawingAnalysisException):
    """Raised when configuration is invalid."""

    pass


class ResourceError(DrawingAnalysisException):
    """Raised when system resources are insufficient."""

    pass


class SecurityError(DrawingAnalysisException):
    """Raised when security validation fails."""

    pass


# HTTP Exception mappings
def create_http_exception(
    status_code: int, message: str, details: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create an HTTPException with consistent format."""
    return HTTPException(
        status_code=status_code,
        detail={"error": message, "details": details or {}, "type": "system_error"},
    )


def validation_error_to_http(error: ValidationError) -> HTTPException:
    """Convert ValidationError to HTTP 400."""
    return create_http_exception(
        status_code=status.HTTP_400_BAD_REQUEST,
        message=error.message,
        details=error.details,
    )


def image_processing_error_to_http(error: ImageProcessingError) -> HTTPException:
    """Convert ImageProcessingError to HTTP 422."""
    return create_http_exception(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        message=error.message,
        details=error.details,
    )


def model_error_to_http(error: ModelError) -> HTTPException:
    """Convert ModelError to HTTP 503."""
    return create_http_exception(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        message=error.message,
        details=error.details,
    )


def storage_error_to_http(error: StorageError) -> HTTPException:
    """Convert StorageError to HTTP 507."""
    return create_http_exception(
        status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
        message=error.message,
        details=error.details,
    )


def configuration_error_to_http(error: ConfigurationError) -> HTTPException:
    """Convert ConfigurationError to HTTP 500."""
    return create_http_exception(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message=error.message,
        details=error.details,
    )


def resource_error_to_http(error: ResourceError) -> HTTPException:
    """Convert ResourceError to HTTP 503."""
    return create_http_exception(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        message=error.message,
        details=error.details,
    )


def security_error_to_http(error: SecurityError) -> HTTPException:
    """Convert SecurityError to HTTP 403."""
    return create_http_exception(
        status_code=status.HTTP_403_FORBIDDEN,
        message=error.message,
        details=error.details,
    )
