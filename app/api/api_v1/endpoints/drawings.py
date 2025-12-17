"""
Drawing management API endpoints.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.core.database import get_db
from app.models.database import Drawing
from app.schemas.drawings import (
    DrawingUploadRequest, 
    DrawingResponse, 
    DrawingListResponse, 
    DrawingFilterRequest,
    ExpertLabel
)
from app.services import DataPipelineService, FileStorageService, ValidationResult, DrawingMetadata
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
data_pipeline = DataPipelineService()
file_storage = FileStorageService()


class UploadProgress:
    """Simple progress tracking for uploads"""
    def __init__(self):
        self.progress = {}
    
    def update(self, upload_id: str, progress: float, status: str):
        self.progress[upload_id] = {"progress": progress, "status": status}
    
    def get(self, upload_id: str):
        return self.progress.get(upload_id, {"progress": 0, "status": "not_found"})

upload_progress = UploadProgress()


@router.post("/upload", response_model=DrawingResponse, status_code=status.HTTP_201_CREATED)
async def upload_drawing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Drawing image file (PNG, JPEG, BMP)"),
    age_years: float = Form(..., ge=2.0, le=18.0, description="Child's age in years"),
    subject: Optional[str] = Form(None, description="Drawing subject"),
    expert_label: Optional[str] = Form(None, description="Expert assessment"),
    drawing_tool: Optional[str] = Form(None, description="Drawing tool used"),
    prompt: Optional[str] = Form(None, description="Drawing prompt"),
    db: Session = Depends(get_db)
):
    """
    Upload drawing with metadata.
    
    This endpoint accepts multipart form data with an image file and metadata.
    The image is validated, preprocessed, and stored along with the metadata.
    """
    try:
        # Validate file size
        if file.size and file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size {file.size} exceeds maximum {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Read file content
        file_content = await file.read()
        await file.seek(0)  # Reset for potential reuse
        
        # Validate image format and integrity
        validation_result = data_pipeline.validate_image(file_content, file.filename)
        if not validation_result.is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image: {validation_result.error_message}"
            )
        
        # Prepare metadata
        metadata_dict = {
            "age_years": age_years,
            "subject": subject,
            "expert_label": expert_label,
            "drawing_tool": drawing_tool,
            "prompt": prompt
        }
        
        # Validate metadata
        try:
            validated_metadata = data_pipeline.extract_metadata(metadata_dict)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata: {str(e)}"
            )
        
        # Save file to storage
        try:
            filename, file_path = await file_storage.save_uploaded_file(file, "drawings")
        except Exception as e:
            logger.error(f"File storage failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded file"
            )
        
        # Create database record
        try:
            drawing = Drawing(
                filename=filename,
                file_path=file_path,
                age_years=validated_metadata.age_years,
                subject=validated_metadata.subject,
                expert_label=validated_metadata.expert_label,
                drawing_tool=validated_metadata.drawing_tool,
                prompt=validated_metadata.prompt
            )
            
            db.add(drawing)
            db.commit()
            db.refresh(drawing)
            
            logger.info(f"Drawing uploaded successfully: ID={drawing.id}, filename={filename}")
            
            # Schedule background preprocessing (for future embedding generation)
            background_tasks.add_task(preprocess_drawing_background, drawing.id, file_path)
            
            return DrawingResponse.model_validate(drawing)
            
        except Exception as e:
            # Clean up file if database operation fails
            file_storage.delete_file(file_path)
            logger.error(f"Database operation failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save drawing metadata"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upload failed due to unexpected error"
        )


async def preprocess_drawing_background(drawing_id: int, file_path: str):
    """Background task for preprocessing uploaded drawings"""
    try:
        logger.info(f"Starting background preprocessing for drawing {drawing_id}")
        # This will be implemented in later tasks when embedding service is ready
        # For now, just log the preprocessing request
        logger.info(f"Background preprocessing queued for drawing {drawing_id} at {file_path}")
    except Exception as e:
        logger.error(f"Background preprocessing failed for drawing {drawing_id}: {str(e)}")


@router.get("/upload/progress/{upload_id}")
async def get_upload_progress(upload_id: str):
    """Get upload progress for large file uploads."""
    progress_info = upload_progress.get(upload_id)
    return progress_info


@router.get("/{drawing_id}", response_model=DrawingResponse)
async def get_drawing(drawing_id: int, db: Session = Depends(get_db)):
    """Retrieve drawing details by ID."""
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    return DrawingResponse.model_validate(drawing)


@router.get("/{drawing_id}/file")
async def get_drawing_file(drawing_id: int, db: Session = Depends(get_db)):
    """Retrieve the actual drawing file."""
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    # Check if file exists
    file_info = file_storage.get_file_info(drawing.file_path)
    if not file_info or not file_info["exists"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Drawing file not found on disk"
        )
    
    return FileResponse(
        path=drawing.file_path,
        filename=drawing.filename,
        media_type="image/*"
    )


@router.get("/", response_model=DrawingListResponse)
async def list_drawings(
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    subject: Optional[str] = None,
    expert_label: Optional[ExpertLabel] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    """List drawings with optional filtering and pagination."""
    
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page number must be >= 1"
        )
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100"
        )
    
    # Validate age range
    if age_min is not None and age_max is not None and age_max <= age_min:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="age_max must be greater than age_min"
        )
    
    # Build query with filters
    query = db.query(Drawing)
    
    if age_min is not None:
        query = query.filter(Drawing.age_years >= age_min)
    if age_max is not None:
        query = query.filter(Drawing.age_years <= age_max)
    if subject is not None:
        query = query.filter(Drawing.subject.ilike(f"%{subject}%"))
    if expert_label is not None:
        query = query.filter(Drawing.expert_label == expert_label.value)
    
    # Get total count for pagination
    total_count = query.count()
    
    # Apply pagination
    offset = (page - 1) * page_size
    drawings = query.order_by(Drawing.upload_timestamp.desc()).offset(offset).limit(page_size).all()
    
    # Calculate total pages
    total_pages = (total_count + page_size - 1) // page_size
    
    return DrawingListResponse(
        drawings=[DrawingResponse.model_validate(drawing) for drawing in drawings],
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.delete("/{drawing_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_drawing(drawing_id: int, db: Session = Depends(get_db)):
    """Delete drawing and associated data."""
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    # Delete associated file
    file_deleted = file_storage.delete_file(drawing.file_path)
    if not file_deleted:
        logger.warning(f"Failed to delete file for drawing {drawing_id}: {drawing.file_path}")
    
    # Delete database record (cascading will handle related records)
    db.delete(drawing)
    db.commit()
    
    logger.info(f"Drawing {drawing_id} deleted successfully")


@router.post("/batch/upload")
async def batch_upload_drawings(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple drawing files"),
    db: Session = Depends(get_db)
):
    """
    Upload multiple drawings in batch.
    
    This endpoint accepts multiple files and processes them in the background.
    Returns an upload ID for tracking progress.
    """
    if len(files) > 50:  # Reasonable limit for batch uploads
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 files allowed per batch upload"
        )
    
    # Generate upload ID for progress tracking
    import uuid
    upload_id = str(uuid.uuid4())
    
    # Initialize progress
    upload_progress.update(upload_id, 0.0, "started")
    
    # Schedule background processing
    background_tasks.add_task(process_batch_upload, upload_id, files, db)
    
    return {
        "upload_id": upload_id,
        "file_count": len(files),
        "status": "processing",
        "progress_url": f"/api/v1/drawings/upload/progress/{upload_id}"
    }


async def process_batch_upload(upload_id: str, files: List[UploadFile], db: Session):
    """Background task for processing batch uploads"""
    try:
        total_files = len(files)
        processed = 0
        successful = 0
        errors = []
        
        for i, file in enumerate(files):
            try:
                # Update progress
                progress = (i / total_files) * 100
                upload_progress.update(upload_id, progress, f"processing_file_{i+1}")
                
                # Process individual file (simplified version of single upload)
                file_content = await file.read()
                validation_result = data_pipeline.validate_image(file_content, file.filename)
                
                if validation_result.is_valid:
                    filename, file_path = await file_storage.save_uploaded_file(file, "drawings")
                    
                    # Create database record with minimal metadata
                    drawing = Drawing(
                        filename=filename,
                        file_path=file_path,
                        age_years=5.0,  # Default age for batch uploads
                        subject=None,
                        expert_label=None,
                        drawing_tool=None,
                        prompt="Batch upload"
                    )
                    
                    db.add(drawing)
                    db.commit()
                    successful += 1
                else:
                    errors.append(f"{file.filename}: {validation_result.error_message}")
                
                processed += 1
                
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
                processed += 1
        
        # Final progress update
        upload_progress.update(
            upload_id, 
            100.0, 
            f"completed_{successful}_successful_{len(errors)}_errors"
        )
        
        logger.info(f"Batch upload {upload_id} completed: {successful}/{total_files} successful")
        
    except Exception as e:
        upload_progress.update(upload_id, 0.0, f"failed_{str(e)}")
        logger.error(f"Batch upload {upload_id} failed: {str(e)}")


@router.get("/stats")
async def get_drawing_stats(db: Session = Depends(get_db)):
    """Get statistics about stored drawings."""
    try:
        total_drawings = db.query(Drawing).count()
        
        # Age distribution
        age_stats = db.query(Drawing.age_years).all()
        ages = [age[0] for age in age_stats]
        
        # Expert label distribution
        label_stats = {}
        for label in ExpertLabel:
            count = db.query(Drawing).filter(Drawing.expert_label == label.value).count()
            label_stats[label.value] = count
        
        # Storage stats
        storage_stats = file_storage.get_storage_stats()
        
        return {
            "total_drawings": total_drawings,
            "age_range": {
                "min": min(ages) if ages else None,
                "max": max(ages) if ages else None,
                "average": sum(ages) / len(ages) if ages else None
            },
            "expert_labels": label_stats,
            "storage": storage_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get drawing stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )