"""
Analysis operation API endpoints.
"""

import logging
import uuid
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.core.database import get_db
from app.models.database import Drawing, AnomalyAnalysis, InterpretabilityResult, DrawingEmbedding, AgeGroupModel
from app.schemas.analysis import (
    AnalysisRequest,
    BatchAnalysisRequest,
    AnalysisResultResponse,
    AnomalyAnalysisResponse,
    InterpretabilityResponse,
    BatchAnalysisResponse,
    AnalysisHistoryResponse,
    ComparisonExampleResponse
)
from app.schemas.drawings import DrawingResponse
from app.services.embedding_service import get_embedding_service
from app.services.model_manager import get_model_manager
from app.services.age_group_manager import get_age_group_manager
from app.services.threshold_manager import get_threshold_manager
from app.services.score_normalizer import get_score_normalizer
from app.services.interpretability_engine import get_interpretability_pipeline
from app.services.comparison_service import get_comparison_service
from app.utils.embedding_serialization import get_embedding_storage

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
embedding_service = get_embedding_service()
model_manager = get_model_manager()
age_group_manager = get_age_group_manager()
threshold_manager = get_threshold_manager()
score_normalizer = get_score_normalizer()
interpretability_engine = get_interpretability_pipeline()
comparison_service = get_comparison_service()


class BatchAnalysisTracker:
    """Simple tracker for batch analysis progress"""
    def __init__(self):
        self.batches = {}
    
    def create_batch(self, batch_id: str, drawing_ids: List[int]) -> None:
        self.batches[batch_id] = {
            "batch_id": batch_id,
            "total_drawings": len(drawing_ids),
            "completed": 0,
            "failed": 0,
            "status": "processing",
            "results": [],
            "errors": [],
            "started_at": datetime.utcnow(),
            "completed_at": None
        }
    
    def update_batch(self, batch_id: str, **kwargs) -> None:
        if batch_id in self.batches:
            self.batches[batch_id].update(kwargs)
    
    def get_batch(self, batch_id: str) -> Optional[dict]:
        return self.batches.get(batch_id)
    
    def add_result(self, batch_id: str, result: dict) -> None:
        if batch_id in self.batches:
            self.batches[batch_id]["results"].append(result)
            self.batches[batch_id]["completed"] += 1
    
    def add_error(self, batch_id: str, error: dict) -> None:
        if batch_id in self.batches:
            self.batches[batch_id]["errors"].append(error)
            self.batches[batch_id]["failed"] += 1

batch_tracker = BatchAnalysisTracker()


async def perform_single_analysis(drawing_id: int, db: Session, force_reanalysis: bool = False) -> AnalysisResultResponse:
    """
    Perform analysis on a single drawing.
    
    Args:
        drawing_id: ID of the drawing to analyze
        db: Database session
        force_reanalysis: Whether to force re-analysis
        
    Returns:
        Complete analysis result
    """
    # Get drawing
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    # Check if analysis already exists
    if not force_reanalysis:
        existing_analysis = db.query(AnomalyAnalysis).filter(
            AnomalyAnalysis.drawing_id == drawing_id
        ).order_by(desc(AnomalyAnalysis.analysis_timestamp)).first()
        
        if existing_analysis:
            # Return existing analysis
            interpretability = db.query(InterpretabilityResult).filter(
                InterpretabilityResult.analysis_id == existing_analysis.id
            ).first()
            
            # Get age group model for additional fields
            existing_age_group_model = db.query(AgeGroupModel).filter(
                AgeGroupModel.id == existing_analysis.age_group_model_id
            ).first()
            
            # Recalculate normalized score to ensure 0-100 scale compatibility
            try:
                recalculated_normalized_score = score_normalizer.normalize_score(
                    existing_analysis.anomaly_score, existing_analysis.age_group_model_id, db
                )
            except Exception as e:
                logger.warning(f"Failed to recalculate normalized score for existing analysis {existing_analysis.id}: {e}")
                # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                recalculated_normalized_score = max(0.0, min(100.0, existing_analysis.normalized_score))
            
            analysis_response = AnomalyAnalysisResponse(
                id=existing_analysis.id,
                drawing_id=existing_analysis.drawing_id,
                anomaly_score=existing_analysis.anomaly_score,
                normalized_score=recalculated_normalized_score,
                is_anomaly=existing_analysis.is_anomaly,
                confidence=existing_analysis.confidence,
                age_group=f"{existing_age_group_model.age_min}-{existing_age_group_model.age_max}" if existing_age_group_model else "unknown",
                method_used="autoencoder",
                vision_model="vit",
                analysis_timestamp=existing_analysis.analysis_timestamp
            )
            
            # Get comparison examples for existing analysis
            comparison_examples = []
            if existing_age_group_model:
                similar_examples = comparison_service.find_similar_normal_examples(
                    target_drawing_id=drawing_id,
                    age_group_min=existing_age_group_model.age_min,
                    age_group_max=existing_age_group_model.age_max,
                    db=db,
                    max_examples=3
                )
                
                comparison_examples = []
                for example in similar_examples:
                    # Recalculate normalized score for comparison examples
                    try:
                        recalculated_normalized_score = score_normalizer.normalize_score(
                            example["drawing_info"]["anomaly_score"], 
                            existing_age_group_model.id, 
                            db
                        )
                    except Exception as e:
                        logger.warning(f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}")
                        # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                        recalculated_normalized_score = max(0.0, min(100.0, example["drawing_info"]["normalized_score"]))
                    
                    comparison_examples.append(ComparisonExampleResponse(
                        drawing_id=example["drawing_id"],
                        filename=example["drawing_info"]["filename"],
                        age_years=example["drawing_info"]["age_years"],
                        subject=example["drawing_info"]["subject"],
                        similarity_score=example["similarity_score"],
                        anomaly_score=example["drawing_info"]["anomaly_score"],
                        normalized_score=recalculated_normalized_score
                    ))
            
            return AnalysisResultResponse(
                drawing=DrawingResponse.model_validate(drawing),
                analysis=analysis_response,
                interpretability=InterpretabilityResponse.model_validate(interpretability) if interpretability else None,
                comparison_examples=comparison_examples
            )
    
    # Get or generate embedding
    embedding_record = db.query(DrawingEmbedding).filter(
        DrawingEmbedding.drawing_id == drawing_id
    ).order_by(desc(DrawingEmbedding.created_timestamp)).first()
    
    if not embedding_record:
        # Generate embedding
        try:
            embedding_data = await embedding_service.generate_embedding_from_file(
                drawing.file_path, drawing.age_years
            )
            
            # Save embedding to database using serialization utilities
            embedding_storage = get_embedding_storage()
            serialized_data, dimension = embedding_storage.store_embedding(
                drawing_id=drawing_id,
                model_type="vit",
                embedding=embedding_data,
                age=drawing.age_years,
                use_cache=True
            )
            
            embedding_record = DrawingEmbedding(
                drawing_id=drawing_id,
                model_type="vit",
                embedding_vector=serialized_data,
                vector_dimension=dimension
            )
            db.add(embedding_record)
            db.commit()
            db.refresh(embedding_record)
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for drawing {drawing_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate embedding: {str(e)}"
            )
    
    # Deserialize embedding using serialization utilities
    embedding_storage = get_embedding_storage()
    embedding_data = embedding_storage.retrieve_embedding(
        drawing_id=drawing_id,
        model_type="vit",
        serialized_data=embedding_record.embedding_vector,
        age=drawing.age_years,
        use_cache=True
    )
    
    # Find appropriate age group model
    age_group_model = age_group_manager.find_appropriate_model(drawing.age_years, db)
    if not age_group_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No appropriate model found for age {drawing.age_years}"
        )
    
    # Compute anomaly score
    try:
        anomaly_score = model_manager.compute_reconstruction_loss(
            embedding_data, age_group_model.id, db
        )
        
        # Normalize score
        normalized_score = score_normalizer.normalize_score(
            anomaly_score, age_group_model.id, db
        )
        
        # Determine if anomaly
        is_anomaly, threshold_used, model_info = threshold_manager.is_anomaly(
            anomaly_score, drawing.age_years, db
        )
        
        # Calculate confidence (simple heuristic based on distance from threshold)
        if threshold_used > 0:
            confidence = min(1.0, abs(anomaly_score - threshold_used) / threshold_used)
        else:
            confidence = 0.5
        
    except Exception as e:
        logger.error(f"Failed to compute anomaly score for drawing {drawing_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute anomaly score: {str(e)}"
        )
    
    # Save analysis result
    analysis = AnomalyAnalysis(
        drawing_id=drawing_id,
        age_group_model_id=age_group_model.id,
        anomaly_score=anomaly_score,
        normalized_score=normalized_score,
        is_anomaly=is_anomaly,
        confidence=confidence
    )
    
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    
    # Generate interpretability if anomaly
    interpretability_result = None
    if is_anomaly:
        try:
            # Load the image for interpretability analysis
            from PIL import Image
            image = Image.open(drawing.file_path)
            
            # Generate complete interpretability analysis
            interp_result = interpretability_engine.generate_complete_analysis(
                image=image,
                anomaly_score=anomaly_score,
                normalized_score=normalized_score,
                age_group=f"{age_group_model.age_min}-{age_group_model.age_max}",
                drawing_metadata={
                    "age": drawing.age_years,
                    "subject": drawing.subject,
                    "filename": drawing.filename
                },
                save_visualizations=True,
                base_filename=f"drawing_{drawing_id}_analysis"
            )
            
            # Extract paths from the result
            saliency_map_path = interp_result.get("visualization_paths", {}).get("saliency_map", "")
            overlay_image_path = interp_result.get("visualization_paths", {}).get("overlay", "")
            explanation_text = interp_result.get("explanation", {}).get("summary", "")
            importance_regions = str(interp_result.get("important_regions", []))
            
            interpretability_record = InterpretabilityResult(
                analysis_id=analysis.id,
                saliency_map_path=saliency_map_path,
                overlay_image_path=overlay_image_path,
                explanation_text=explanation_text,
                importance_regions=importance_regions
            )
            
            db.add(interpretability_record)
            db.commit()
            db.refresh(interpretability_record)
            
            interpretability_result = InterpretabilityResponse.model_validate(interpretability_record)
            
        except Exception as e:
            logger.warning(f"Failed to generate interpretability for drawing {drawing_id}: {str(e)}")
            # Continue without interpretability
    
    logger.info(f"Analysis completed for drawing {drawing_id}: "
               f"score={anomaly_score:.6f}, anomaly={is_anomaly}")
    
    # Create analysis response with additional fields
    # Recalculate normalized score to ensure 0-100 scale compatibility
    try:
        recalculated_normalized_score = score_normalizer.normalize_score(
            analysis.anomaly_score, age_group_model.id, db
        )
    except Exception as e:
        logger.warning(f"Failed to recalculate normalized score for analysis {analysis.id}: {e}")
        # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
        recalculated_normalized_score = max(0.0, min(100.0, analysis.normalized_score))
    
    analysis_response = AnomalyAnalysisResponse(
        id=analysis.id,
        drawing_id=analysis.drawing_id,
        anomaly_score=analysis.anomaly_score,
        normalized_score=recalculated_normalized_score,
        is_anomaly=analysis.is_anomaly,
        confidence=analysis.confidence,
        age_group=f"{age_group_model.age_min}-{age_group_model.age_max}",
        method_used="autoencoder",
        vision_model="vit",
        analysis_timestamp=analysis.analysis_timestamp
    )
    
    # Get comparison examples for new analysis
    comparison_examples = []
    similar_examples = comparison_service.find_similar_normal_examples(
        target_drawing_id=drawing_id,
        age_group_min=age_group_model.age_min,
        age_group_max=age_group_model.age_max,
        db=db,
        max_examples=3
    )
    
    comparison_examples = []
    for example in similar_examples:
        # Recalculate normalized score for comparison examples
        try:
            recalculated_normalized_score = score_normalizer.normalize_score(
                example["drawing_info"]["anomaly_score"], 
                age_group_model.id, 
                db
            )
        except Exception as e:
            logger.warning(f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}")
            # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
            recalculated_normalized_score = max(0.0, min(100.0, example["drawing_info"]["normalized_score"]))
        
        comparison_examples.append(ComparisonExampleResponse(
            drawing_id=example["drawing_id"],
            filename=example["drawing_info"]["filename"],
            age_years=example["drawing_info"]["age_years"],
            subject=example["drawing_info"]["subject"],
            similarity_score=example["similarity_score"],
            anomaly_score=example["drawing_info"]["anomaly_score"],
            normalized_score=recalculated_normalized_score
        ))
    
    return AnalysisResultResponse(
        drawing=DrawingResponse.model_validate(drawing),
        analysis=analysis_response,
        interpretability=interpretability_result,
        comparison_examples=comparison_examples
    )


@router.get("/stats")
async def get_analysis_stats(db: Session = Depends(get_db)):
    """
    Get dashboard statistics for analyses and drawings.
    
    This endpoint provides comprehensive statistics for the dashboard
    including drawing counts, analysis results, and model status.
    """
    from sqlalchemy import func
    
    # Get total counts
    total_drawings = db.query(Drawing).count()
    total_analyses = db.query(AnomalyAnalysis).count()
    
    # Get anomaly counts - recalculate based on current thresholds
    # Get all analyses with their scores and associated drawings for age info
    analyses_with_drawings = db.query(
        AnomalyAnalysis.anomaly_score,
        Drawing.age_years
    ).join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id).all()
    
    # Recalculate anomaly classifications based on current thresholds
    anomaly_count = 0
    normal_count = 0
    
    for analysis in analyses_with_drawings:
        try:
            # Use threshold manager to determine if this score is anomalous
            is_anomaly, threshold_used, model_info = threshold_manager.is_anomaly(
                analysis.anomaly_score, analysis.age_years, db
            )
            if is_anomaly:
                anomaly_count += 1
            else:
                normal_count += 1
        except Exception as e:
            # If threshold calculation fails, fall back to normal classification
            logger.warning(f"Failed to recalculate anomaly status for age {analysis.age_years}: {e}")
            # We'll count this as normal to be conservative
            normal_count += 1
    
    # Get recent analyses with drawing info
    recent_analyses_query = db.query(
        AnomalyAnalysis.id,
        AnomalyAnalysis.drawing_id,
        Drawing.filename,
        Drawing.age_years,
        AnomalyAnalysis.anomaly_score,
        AnomalyAnalysis.is_anomaly,
        AnomalyAnalysis.analysis_timestamp
    ).join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id)\
     .order_by(desc(AnomalyAnalysis.analysis_timestamp))\
     .limit(10).all()
    
    recent_analyses = [
        {
            "id": analysis.id,
            "drawing_id": analysis.drawing_id,
            "filename": analysis.filename,
            "age_years": analysis.age_years,
            "anomaly_score": analysis.anomaly_score,
            "is_anomaly": analysis.is_anomaly,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat()
        }
        for analysis in recent_analyses_query
    ]
    
    # Get age distribution
    age_distribution_query = db.query(
        func.floor(Drawing.age_years).label('age_floor'),
        func.count(Drawing.id).label('count')
    ).group_by(func.floor(Drawing.age_years)).all()
    
    age_distribution = [
        {
            "age_group": f"{int(age_floor)}-{int(age_floor)+1}",
            "count": count
        }
        for age_floor, count in age_distribution_query
    ]
    
    # Get model status
    active_models = db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).count()
    latest_model = db.query(AgeGroupModel).order_by(desc(AgeGroupModel.created_timestamp)).first()
    
    model_status = {
        "vision_model": "vit",
        "is_loaded": embedding_service.is_ready() if embedding_service else False,
        "last_updated": latest_model.created_timestamp.isoformat() if latest_model else "",
        "active_age_groups": active_models
    }
    
    return {
        "total_drawings": total_drawings,
        "total_analyses": total_analyses,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "recent_analyses": recent_analyses,
        "age_distribution": age_distribution,
        "model_status": model_status
    }


@router.post("/analyze/{drawing_id}", response_model=AnalysisResultResponse)
async def analyze_drawing(
    drawing_id: int,
    request: AnalysisRequest = None,
    db: Session = Depends(get_db)
):
    """
    Analyze specific drawing for anomalies.
    
    This endpoint performs anomaly detection on a single drawing,
    generating embeddings, computing anomaly scores, and providing
    interpretability results if the drawing is flagged as anomalous.
    """
    force_reanalysis = request.force_reanalysis if request else False
    
    try:
        result = await perform_single_analysis(drawing_id, db, force_reanalysis)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis of drawing {drawing_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to unexpected error"
        )


@router.post("/batch", response_model=dict)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Batch analyze multiple drawings.
    
    This endpoint accepts a list of drawing IDs and processes them
    in the background, returning a batch ID for progress tracking.
    """
    # Validate that all drawings exist
    existing_drawings = db.query(Drawing.id).filter(
        Drawing.id.in_(request.drawing_ids)
    ).all()
    
    existing_ids = {d.id for d in existing_drawings}
    missing_ids = set(request.drawing_ids) - existing_ids
    
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drawings not found: {list(missing_ids)}"
        )
    
    # Create batch ID
    batch_id = str(uuid.uuid4())
    
    # Initialize batch tracking
    batch_tracker.create_batch(batch_id, request.drawing_ids)
    
    # Schedule background processing
    background_tasks.add_task(
        process_batch_analysis,
        batch_id,
        request.drawing_ids,
        request.force_reanalysis,
        db
    )
    
    return {
        "batch_id": batch_id,
        "total_drawings": len(request.drawing_ids),
        "status": "processing",
        "progress_url": f"/api/v1/analysis/batch/{batch_id}/progress"
    }


async def process_batch_analysis(
    batch_id: str,
    drawing_ids: List[int],
    force_reanalysis: bool,
    db: Session
):
    """Background task for processing batch analysis"""
    try:
        logger.info(f"Starting batch analysis {batch_id} for {len(drawing_ids)} drawings")
        
        for i, drawing_id in enumerate(drawing_ids):
            try:
                # Update progress
                progress = (i / len(drawing_ids)) * 100
                batch_tracker.update_batch(
                    batch_id,
                    status=f"processing_drawing_{i+1}_of_{len(drawing_ids)}"
                )
                
                # Perform analysis
                result = await perform_single_analysis(drawing_id, db, force_reanalysis)
                batch_tracker.add_result(batch_id, result.model_dump())
                
                logger.debug(f"Completed analysis for drawing {drawing_id} in batch {batch_id}")
                
            except Exception as e:
                error_info = {
                    "drawing_id": drawing_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                batch_tracker.add_error(batch_id, error_info)
                logger.error(f"Failed to analyze drawing {drawing_id} in batch {batch_id}: {str(e)}")
        
        # Mark batch as completed
        batch_tracker.update_batch(
            batch_id,
            status="completed",
            completed_at=datetime.utcnow()
        )
        
        batch_info = batch_tracker.get_batch(batch_id)
        logger.info(f"Batch analysis {batch_id} completed: "
                   f"{batch_info['completed']} successful, {batch_info['failed']} failed")
        
    except Exception as e:
        batch_tracker.update_batch(
            batch_id,
            status=f"failed: {str(e)}",
            completed_at=datetime.utcnow()
        )
        logger.error(f"Batch analysis {batch_id} failed: {str(e)}")


@router.get("/batch/{batch_id}/progress", response_model=BatchAnalysisResponse)
async def get_batch_progress(batch_id: str):
    """Get progress of batch analysis."""
    batch_info = batch_tracker.get_batch(batch_id)
    
    if not batch_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch {batch_id} not found"
        )
    
    return BatchAnalysisResponse(**batch_info)


@router.get("/{analysis_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(analysis_id: int, db: Session = Depends(get_db)):
    """
    Get analysis results by analysis ID.
    
    This endpoint retrieves a complete analysis result including
    the drawing information, anomaly analysis, and interpretability
    results if available.
    """
    analysis = db.query(AnomalyAnalysis).filter(
        AnomalyAnalysis.id == analysis_id
    ).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )
    
    # Get associated drawing
    drawing = db.query(Drawing).filter(Drawing.id == analysis.drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Associated drawing not found"
        )
    
    # Get interpretability results if available
    interpretability = db.query(InterpretabilityResult).filter(
        InterpretabilityResult.analysis_id == analysis_id
    ).first()
    
    # Get age group model for additional fields
    age_group_model = db.query(AgeGroupModel).filter(
        AgeGroupModel.id == analysis.age_group_model_id
    ).first()
    
    # Recalculate normalized score to ensure 0-100 scale compatibility
    try:
        recalculated_normalized_score = score_normalizer.normalize_score(
            analysis.anomaly_score, analysis.age_group_model_id, db
        )
    except Exception as e:
        logger.warning(f"Failed to recalculate normalized score for analysis {analysis.id}: {e}")
        # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
        recalculated_normalized_score = max(0.0, min(100.0, analysis.normalized_score))
    
    analysis_response = AnomalyAnalysisResponse(
        id=analysis.id,
        drawing_id=analysis.drawing_id,
        anomaly_score=analysis.anomaly_score,
        normalized_score=recalculated_normalized_score,
        is_anomaly=analysis.is_anomaly,
        confidence=analysis.confidence,
        age_group=f"{age_group_model.age_min}-{age_group_model.age_max}" if age_group_model else "unknown",
        method_used="autoencoder",
        vision_model="vit",
        analysis_timestamp=analysis.analysis_timestamp
    )
    
    # Get comparison examples for retrieved analysis
    comparison_examples = []
    if age_group_model:
        similar_examples = comparison_service.find_similar_normal_examples(
            target_drawing_id=drawing.id,
            age_group_min=age_group_model.age_min,
            age_group_max=age_group_model.age_max,
            db=db,
            max_examples=3
        )
        
        comparison_examples = []
        for example in similar_examples:
            # Recalculate normalized score for comparison examples
            try:
                recalculated_normalized_score = score_normalizer.normalize_score(
                    example["drawing_info"]["anomaly_score"], 
                    age_group_model.id, 
                    db
                )
            except Exception as e:
                logger.warning(f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}")
                # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                recalculated_normalized_score = max(0.0, min(100.0, example["drawing_info"]["normalized_score"]))
            
            comparison_examples.append(ComparisonExampleResponse(
                drawing_id=example["drawing_id"],
                filename=example["drawing_info"]["filename"],
                age_years=example["drawing_info"]["age_years"],
                subject=example["drawing_info"]["subject"],
                similarity_score=example["similarity_score"],
                anomaly_score=example["drawing_info"]["anomaly_score"],
                normalized_score=recalculated_normalized_score
            ))
    
    return AnalysisResultResponse(
        drawing=DrawingResponse.model_validate(drawing),
        analysis=analysis_response,
        interpretability=InterpretabilityResponse.model_validate(interpretability) if interpretability else None,
        comparison_examples=comparison_examples
    )


@router.post("/embeddings/{drawing_id}")
async def generate_embedding(
    drawing_id: int,
    db: Session = Depends(get_db)
):
    """
    Generate embedding for a drawing without requiring a trained model.
    
    This endpoint is used during the training phase to generate embeddings
    for all drawings before training the autoencoder models.
    """
    # Get drawing
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    # Check if embedding already exists
    existing_embedding = db.query(DrawingEmbedding).filter(
        DrawingEmbedding.drawing_id == drawing_id
    ).order_by(desc(DrawingEmbedding.created_timestamp)).first()
    
    if existing_embedding:
        return {
            "drawing_id": drawing_id,
            "status": "exists",
            "message": "Embedding already exists",
            "embedding_id": existing_embedding.id,
            "vector_dimension": existing_embedding.vector_dimension,
            "created_timestamp": existing_embedding.created_timestamp
        }
    
    # Generate embedding
    try:
        # Initialize embedding service if not ready
        if not embedding_service.is_ready():
            embedding_service.initialize()
        
        embedding_data = await embedding_service.generate_embedding_from_file(
            drawing.file_path, drawing.age_years
        )
        
        # Save embedding to database using serialization utilities
        embedding_storage = get_embedding_storage()
        serialized_data, dimension = embedding_storage.store_embedding(
            drawing_id=drawing_id,
            model_type="vit",
            embedding=embedding_data,
            age=drawing.age_years,
            use_cache=True
        )
        
        embedding_record = DrawingEmbedding(
            drawing_id=drawing_id,
            model_type="vit",
            embedding_vector=serialized_data,
            vector_dimension=dimension
        )
        db.add(embedding_record)
        db.commit()
        db.refresh(embedding_record)
        
        logger.info(f"Generated embedding for drawing {drawing_id}: dimension={len(embedding_data)}")
        
        return {
            "drawing_id": drawing_id,
            "status": "generated",
            "message": "Embedding generated successfully",
            "embedding_id": embedding_record.id,
            "vector_dimension": embedding_record.vector_dimension,
            "created_timestamp": embedding_record.created_timestamp
        }
        
    except Exception as e:
        logger.error(f"Failed to generate embedding for drawing {drawing_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {str(e)}"
        )


@router.get("/drawing/{drawing_id}", response_model=AnalysisHistoryResponse)
async def get_drawing_analyses(
    drawing_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get all analyses for a specific drawing.
    
    This endpoint returns the analysis history for a drawing,
    ordered by most recent first.
    """
    # Verify drawing exists
    drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Drawing with ID {drawing_id} not found"
        )
    
    # Get analyses for this drawing
    analyses = db.query(AnomalyAnalysis).filter(
        AnomalyAnalysis.drawing_id == drawing_id
    ).order_by(desc(AnomalyAnalysis.analysis_timestamp)).limit(limit).all()
    
    # Get total count
    total_count = db.query(AnomalyAnalysis).filter(
        AnomalyAnalysis.drawing_id == drawing_id
    ).count()
    
    # Convert analyses to response format with additional fields
    analysis_responses = []
    for analysis in analyses:
        age_group_model = db.query(AgeGroupModel).filter(
            AgeGroupModel.id == analysis.age_group_model_id
        ).first()
        
        # Recalculate normalized score to ensure 0-100 scale compatibility
        try:
            recalculated_normalized_score = score_normalizer.normalize_score(
                analysis.anomaly_score, analysis.age_group_model_id, db
            )
        except Exception as e:
            logger.warning(f"Failed to recalculate normalized score for analysis {analysis.id}: {e}")
            # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
            recalculated_normalized_score = max(0.0, min(100.0, analysis.normalized_score))
        
        analysis_response = AnomalyAnalysisResponse(
            id=analysis.id,
            drawing_id=analysis.drawing_id,
            anomaly_score=analysis.anomaly_score,
            normalized_score=recalculated_normalized_score,
            is_anomaly=analysis.is_anomaly,
            confidence=analysis.confidence,
            age_group=f"{age_group_model.age_min}-{age_group_model.age_max}" if age_group_model else "unknown",
            method_used="autoencoder",
            vision_model="vit",
            analysis_timestamp=analysis.analysis_timestamp
        )
        analysis_responses.append(analysis_response)
    
    return AnalysisHistoryResponse(
        drawing_id=drawing_id,
        analyses=analysis_responses,
        total_count=total_count
    )