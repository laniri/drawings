"""
Pydantic schemas for drawing-related API endpoints.

This module contains request and response models for drawing upload,
retrieval, and management operations.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ExpertLabel(str, Enum):
    """Enumeration for expert labels on drawings."""
    NORMAL = "normal"
    CONCERN = "concern"
    SEVERE = "severe"


class DrawingUploadRequest(BaseModel):
    """Request model for uploading a drawing with metadata."""
    
    age_years: float = Field(
        ..., 
        ge=2.0, 
        le=18.0, 
        description="Child's age in years (2-18)"
    )
    subject: Optional[str] = Field(
        None, 
        max_length=50,
        description="Drawing subject (person, house, tree, etc.)"
    )
    expert_label: Optional[ExpertLabel] = Field(
        None,
        description="Expert assessment of the drawing"
    )
    drawing_tool: Optional[str] = Field(
        None,
        max_length=30,
        description="Tool used to create the drawing"
    )
    prompt: Optional[str] = Field(
        None,
        max_length=500,
        description="Prompt or instruction given to the child"
    )
    
    @field_validator('subject', 'drawing_tool', 'prompt')
    @classmethod
    def validate_optional_strings(cls, v):
        """Validate optional string fields - convert empty strings to None."""
        if v is not None and v.strip() == "":
            return None
        return v


class DrawingResponse(BaseModel):
    """Response model for drawing information."""
    
    id: int
    filename: str
    age_years: float
    subject: Optional[str]
    expert_label: Optional[str]
    drawing_tool: Optional[str]
    prompt: Optional[str]
    upload_timestamp: datetime
    
    model_config = {"from_attributes": True}


class DrawingListResponse(BaseModel):
    """Response model for listing multiple drawings."""
    
    drawings: List[DrawingResponse]
    total_count: int
    page: int = Field(ge=1, description="Current page number")
    page_size: int = Field(ge=1, le=100, description="Number of items per page")
    total_pages: int


class DrawingFilterRequest(BaseModel):
    """Request model for filtering drawings."""
    
    age_min: Optional[float] = Field(None, ge=2.0, le=18.0)
    age_max: Optional[float] = Field(None, ge=2.0, le=18.0)
    subject: Optional[str] = Field(None, max_length=50)
    expert_label: Optional[ExpertLabel] = None
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    
    @field_validator('age_max')
    @classmethod
    def validate_age_range(cls, v, info):
        """Validate that age_max is greater than age_min if both are provided."""
        if v is not None and 'age_min' in info.data and info.data['age_min'] is not None:
            if v <= info.data['age_min']:
                raise ValueError('age_max must be greater than age_min')
        return v