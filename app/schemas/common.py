"""
Common Pydantic schemas and utilities.

This module contains shared schemas, error models, and utility functions
used across different API endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Model for error details in API responses."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(
        None, description="Detailed error information"
    )
    request_id: Optional[str] = Field(
        None, description="Request identifier for tracking"
    )


class SuccessResponse(BaseModel):
    """Standard success response model."""

    success: bool = True
    message: str = Field(..., description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    database: str = Field(..., description="Database status")
    models: str = Field(..., description="ML models status")
    storage: str = Field(..., description="File storage status")


class PaginationInfo(BaseModel):
    """Pagination information model."""

    page: int = Field(..., ge=1, description="Current page number")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_items: int = Field(..., ge=0, description="Total number of items")
    total_pages: int = Field(..., ge=0, description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class FileUploadResponse(BaseModel):
    """Response model for file upload operations."""

    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Stored file path")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    content_type: str = Field(..., description="MIME content type")
    upload_timestamp: str = Field(..., description="Upload timestamp")


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information."""

    field: str = Field(..., description="Field name")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(None, description="The invalid value")
    constraint: Optional[str] = Field(
        None, description="Validation constraint that failed"
    )


class ImageFormat(str, Enum):
    """Supported image formats."""

    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    BMP = "bmp"


class ImageValidationRequest(BaseModel):
    """Request model for image validation."""

    max_file_size: int = Field(
        10 * 1024 * 1024, description="Maximum file size in bytes"
    )
    allowed_formats: List[ImageFormat] = Field(
        default=[ImageFormat.PNG, ImageFormat.JPEG, ImageFormat.JPG, ImageFormat.BMP],
        description="Allowed image formats",
    )
    min_width: int = Field(32, ge=1, description="Minimum image width")
    min_height: int = Field(32, ge=1, description="Minimum image height")
    max_width: int = Field(4096, ge=1, description="Maximum image width")
    max_height: int = Field(4096, ge=1, description="Maximum image height")
