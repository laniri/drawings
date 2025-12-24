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


class SubjectCategory(str, Enum):
    """
    Enumeration for drawing subject categories.
    
    Supports up to 64 distinct categories using one-hot encoding.
    "unspecified" is the default category when subject information is unavailable.
    """
    # Default category (position 0 in one-hot encoding)
    UNSPECIFIED = "unspecified"
    
    # Objects
    TV = "TV"
    AIRPLANE = "airplane"
    APPLE = "apple"
    BED = "bed"
    BIKE = "bike"
    BOAT = "boat"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    CACTUS = "cactus"
    CAR = "car"
    CHAIR = "chair"
    CLOCK = "clock"
    COUCH = "couch"
    CUP = "cup"
    HAT = "hat"
    HOUSE = "house"
    ICE_CREAM = "ice cream"
    KEY = "key"
    LAMP = "lamp"
    MUSHROOM = "mushroom"
    PHONE = "phone"
    PIANO = "piano"
    SCISSORS = "scissors"
    TRAIN = "train"
    TREE = "tree"
    WATCH = "watch"
    
    # Animals
    BEAR = "bear"
    BEE = "bee"
    BIRD = "bird"
    CAMEL = "camel"
    CAT = "cat"
    COW = "cow"
    DOG = "dog"
    ELEPHANT = "elephant"
    FISH = "fish"
    FROG = "frog"
    HORSE = "horse"
    OCTOPUS = "octopus"
    RABBIT = "rabbit"
    SHEEP = "sheep"
    SNAIL = "snail"
    SPIDER = "spider"
    TIGER = "tiger"
    WHALE = "whale"
    
    # People and body parts
    FACE = "face"
    HAND = "hand"
    PERSON = "person"
    
    # Abstract/other categories
    FAMILY = "family"
    ABSTRACT = "abstract"
    OTHER = "other"


class DrawingUploadRequest(BaseModel):
    """Request model for uploading a drawing with metadata."""
    
    age_years: float = Field(
        ..., 
        ge=2.0, 
        le=18.0, 
        description="Child's age in years (2-18)"
    )
    subject: Optional[SubjectCategory] = Field(
        None,
        description="Drawing subject category"
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
    
    @field_validator('drawing_tool', 'prompt')
    @classmethod
    def validate_optional_strings(cls, v):
        """Validate optional string fields - convert empty/whitespace-only strings to None."""
        if v is not None and isinstance(v, str) and v.strip() == "":
            return None
        return v
    
    @field_validator('subject')
    @classmethod
    def validate_subject_category(cls, v):
        """Validate subject category - convert empty strings to None, validate enum values."""
        if v is not None and isinstance(v, str) and v.strip() == "":
            return None
        return v


class DrawingResponse(BaseModel):
    """Response model for drawing information."""
    
    id: int
    filename: str
    age_years: float
    subject: Optional[str]  # Returned as string from database
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
    subject: Optional[SubjectCategory] = Field(None)
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