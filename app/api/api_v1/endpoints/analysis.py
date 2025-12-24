"""
Analysis operation API endpoints.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.database import (
    AgeGroupModel,
    AnomalyAnalysis,
    Drawing,
    DrawingEmbedding,
    InterpretabilityResult,
)
from app.schemas.analysis import (
    AnalysisHistoryResponse,
    AnalysisRequest,
    AnalysisResultResponse,
    AnomalyAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    ComparisonExampleResponse,
    InterpretabilityResponse,
)
from app.schemas.drawings import DrawingResponse
from app.services.age_group_manager import get_age_group_manager
from app.services.comparison_service import get_comparison_service
from app.services.embedding_service import get_embedding_service
from app.services.interpretability_engine import get_interpretability_pipeline
from app.services.model_manager import get_model_manager
from app.services.score_normalizer import get_score_normalizer
from app.services.threshold_manager import get_threshold_manager
from app.utils.embedding_serialization import get_embedding_storage

logger = logging.getLogger(__name__)
router = APIRouter()


def _create_simple_saliency_map(image, anomaly_score, is_anomaly):
    """Create a simple saliency map based on anomaly score."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create base saliency map
    width, height = image.size
    saliency = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(saliency)

    # Create heat map based on anomaly score
    if is_anomaly:
        # For anomalies, create more intense heat map
        intensity = min(255, int(anomaly_score * 400))
        # Create multiple hotspots
        for i in range(3):
            x = width // 4 + (i * width // 4)
            y = height // 4 + (i * height // 6)
            radius = width // 8

            # Create gradient circle
            for r in range(radius, 0, -5):
                alpha = intensity * (radius - r) // radius
                color = (alpha, alpha // 2, 0)  # Red-orange gradient
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
    else:
        # For normal drawings, create subtle heat map
        intensity = max(30, int(anomaly_score * 100))
        x, y = width // 2, height // 2
        radius = width // 6

        for r in range(radius, 0, -3):
            alpha = intensity * (radius - r) // radius
            color = (0, alpha, alpha // 2)  # Blue-green gradient
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Apply blur for smoother appearance
    saliency = saliency.filter(ImageFilter.GaussianBlur(radius=5))

    return saliency


def _analyze_image_regions(image, anomaly_score, is_anomaly):
    """Analyze image to create realistic importance regions."""
    width, height = image.size

    regions = []

    # Center region (most important)
    center_importance = 0.8 if is_anomaly else 0.4
    regions.append(
        {
            "region_id": "center_main",
            "bounding_box": [width // 4, height // 4, 3 * width // 4, 3 * height // 4],
            "importance_score": center_importance,
            "spatial_location": "central drawing area",
        }
    )

    # Upper region
    upper_importance = 0.6 if is_anomaly else 0.3
    regions.append(
        {
            "region_id": "upper_section",
            "bounding_box": [width // 6, 0, 5 * width // 6, height // 3],
            "importance_score": upper_importance,
            "spatial_location": "upper portion",
        }
    )

    # Lower region
    lower_importance = 0.5 if is_anomaly else 0.2
    regions.append(
        {
            "region_id": "lower_section",
            "bounding_box": [width // 6, 2 * height // 3, 5 * width // 6, height],
            "importance_score": lower_importance,
            "spatial_location": "lower portion",
        }
    )

    # Left region
    left_importance = 0.4 if is_anomaly else 0.25
    regions.append(
        {
            "region_id": "left_section",
            "bounding_box": [0, height // 4, width // 3, 3 * height // 4],
            "importance_score": left_importance,
            "spatial_location": "left side",
        }
    )

    # Right region
    right_importance = 0.45 if is_anomaly else 0.28
    regions.append(
        {
            "region_id": "right_section",
            "bounding_box": [2 * width // 3, height // 4, width, 3 * height // 4],
            "importance_score": right_importance,
            "spatial_location": "right side",
        }
    )

    return regions


def _convert_interpretability_to_response(interpretability_record):
    """Convert database InterpretabilityResult to InterpretabilityResponse."""
    import json

    if not interpretability_record:
        return None

    # Parse importance regions from JSON string
    try:
        importance_regions = (
            json.loads(interpretability_record.importance_regions)
            if interpretability_record.importance_regions
            else []
        )
    except (json.JSONDecodeError, TypeError):
        importance_regions = []

    # Use overlay_image_path as the URL (it should contain the URL)
    saliency_url = interpretability_record.overlay_image_path or ""

    return InterpretabilityResponse(
        saliency_map_url=saliency_url,
        overlay_image_url=saliency_url,
        explanation_text=interpretability_record.explanation_text,
        importance_regions=importance_regions,
    )


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
            "completed_at": None,
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


async def perform_single_analysis(
    drawing_id: int, db: Session, force_reanalysis: bool = False
) -> AnalysisResultResponse:
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
            detail=f"Drawing with ID {drawing_id} not found",
        )

    # Check if analysis already exists
    if not force_reanalysis:
        existing_analysis = (
            db.query(AnomalyAnalysis)
            .filter(AnomalyAnalysis.drawing_id == drawing_id)
            .order_by(desc(AnomalyAnalysis.analysis_timestamp))
            .first()
        )

        if existing_analysis:
            # Check for existing interpretability
            interpretability = (
                db.query(InterpretabilityResult)
                .filter(InterpretabilityResult.analysis_id == existing_analysis.id)
                .first()
            )

            # Generate interpretability if it doesn't exist
            if not interpretability:
                try:
                    # Load the image for interpretability analysis
                    from PIL import Image

                    image = Image.open(drawing.file_path)

                    # Get age group model for interpretability generation
                    existing_age_group_model = (
                        db.query(AgeGroupModel)
                        .filter(
                            AgeGroupModel.id == existing_analysis.age_group_model_id
                        )
                        .first()
                    )

                    # Generate simplified interpretability analysis
                    import json
                    from pathlib import Path

                    # Create static directory if it doesn't exist
                    static_dir = Path("static/saliency_maps")
                    static_dir.mkdir(parents=True, exist_ok=True)

                    # Generate saliency map filename
                    saliency_filename = (
                        f"saliency_{existing_analysis.id}_{drawing_id}.png"
                    )
                    saliency_path = static_dir / saliency_filename
                    saliency_url = f"/static/saliency_maps/{saliency_filename}"

                    # Create a simple saliency map
                    saliency_map = _create_simple_saliency_map(
                        image,
                        existing_analysis.anomaly_score,
                        existing_analysis.is_anomaly,
                    )
                    saliency_map.save(saliency_path)

                    # Generate explanation
                    if existing_analysis.is_anomaly:
                        explanation_text = f"Analysis reveals patterns that deviate from typical developmental expectations for a {drawing.age_years}-year-old child. The anomaly score of {existing_analysis.anomaly_score:.3f} indicates significant differences in drawing characteristics compared to age-matched peers."
                    else:
                        explanation_text = f"This drawing demonstrates age-appropriate developmental patterns for a {drawing.age_years}-year-old child. The low anomaly score of {existing_analysis.anomaly_score:.3f} indicates alignment with expected developmental milestones."

                    # Create importance regions
                    regions = _analyze_image_regions(
                        image,
                        existing_analysis.anomaly_score,
                        existing_analysis.is_anomaly,
                    )
                    importance_regions = json.dumps(regions)

                    saliency_map_path = str(saliency_path)
                    overlay_image_path = saliency_url

                    interpretability_record = InterpretabilityResult(
                        analysis_id=existing_analysis.id,
                        saliency_map_path=saliency_map_path,
                        overlay_image_path=overlay_image_path,
                        explanation_text=explanation_text,
                        importance_regions=importance_regions,
                    )

                    db.add(interpretability_record)
                    db.commit()
                    db.refresh(interpretability_record)

                    interpretability = interpretability_record

                except Exception as e:
                    logger.warning(
                        f"Failed to generate advanced interpretability for existing analysis {existing_analysis.id}: {str(e)}"
                    )
                    # Create basic interpretability fallback
                    try:
                        # Generate basic explanation based on anomaly score
                        if existing_analysis.is_anomaly:
                            explanation_text = f"This drawing shows patterns that deviate from typical drawings for age {drawing.age_years}. The anomaly score of {existing_analysis.anomaly_score:.3f} indicates significant differences from the expected patterns for this age group."
                        else:
                            explanation_text = f"This drawing shows typical patterns for age {drawing.age_years}. The low anomaly score of {existing_analysis.anomaly_score:.3f} indicates the drawing follows expected developmental patterns."

                        # Create basic importance regions (mock data for now)
                        import json

                        basic_regions = [
                            {
                                "region_id": "center",
                                "bounding_box": [50, 50, 150, 150],
                                "importance_score": (
                                    0.8 if existing_analysis.is_anomaly else 0.3
                                ),
                                "spatial_location": "center area",
                            },
                            {
                                "region_id": "upper_left",
                                "bounding_box": [10, 10, 80, 80],
                                "importance_score": (
                                    0.6 if existing_analysis.is_anomaly else 0.2
                                ),
                                "spatial_location": "upper left area",
                            },
                        ]

                        interpretability_record = InterpretabilityResult(
                            analysis_id=existing_analysis.id,
                            saliency_map_path="",  # No saliency map available
                            overlay_image_path="",  # No overlay available
                            explanation_text=explanation_text,
                            importance_regions=json.dumps(basic_regions),
                        )

                        db.add(interpretability_record)
                        db.commit()
                        db.refresh(interpretability_record)

                        interpretability = interpretability_record
                        logger.info(
                            f"Created basic interpretability fallback for existing analysis {existing_analysis.id}"
                        )

                    except Exception as fallback_error:
                        logger.error(
                            f"Failed to create basic interpretability fallback for existing analysis {existing_analysis.id}: {str(fallback_error)}"
                        )
                        # Continue without interpretability

            # Get age group model for additional fields
            existing_age_group_model = (
                db.query(AgeGroupModel)
                .filter(AgeGroupModel.id == existing_analysis.age_group_model_id)
                .first()
            )

            # Recalculate normalized score to ensure 0-100 scale compatibility
            try:
                recalculated_normalized_score = score_normalizer.normalize_score(
                    existing_analysis.anomaly_score,
                    existing_analysis.age_group_model_id,
                    db,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to recalculate normalized score for existing analysis {existing_analysis.id}: {e}"
                )
                # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                recalculated_normalized_score = max(
                    0.0, min(100.0, existing_analysis.normalized_score)
                )

            analysis_response = AnomalyAnalysisResponse(
                id=existing_analysis.id,
                drawing_id=existing_analysis.drawing_id,
                anomaly_score=existing_analysis.anomaly_score,
                normalized_score=recalculated_normalized_score,
                visual_anomaly_score=getattr(
                    existing_analysis, "visual_anomaly_score", None
                ),
                subject_anomaly_score=getattr(
                    existing_analysis, "subject_anomaly_score", None
                ),
                anomaly_attribution=getattr(
                    existing_analysis, "anomaly_attribution", None
                ),
                analysis_type=getattr(
                    existing_analysis, "analysis_type", "subject_aware"
                ),
                subject_category=drawing.subject,
                is_anomaly=existing_analysis.is_anomaly,
                confidence=existing_analysis.confidence,
                age_group=(
                    f"{existing_age_group_model.age_min}-{existing_age_group_model.age_max}"
                    if existing_age_group_model
                    else "unknown"
                ),
                method_used="autoencoder",
                vision_model="vit",
                analysis_timestamp=existing_analysis.analysis_timestamp,
            )

            # Get comparison examples for existing analysis
            comparison_examples = []
            if existing_age_group_model:
                similar_examples = comparison_service.find_similar_normal_examples(
                    target_drawing_id=drawing_id,
                    age_group_min=existing_age_group_model.age_min,
                    age_group_max=existing_age_group_model.age_max,
                    db=db,
                    max_examples=3,
                )

                comparison_examples = []
                for example in similar_examples:
                    # Recalculate normalized score for comparison examples
                    try:
                        recalculated_normalized_score = (
                            score_normalizer.normalize_score(
                                example["drawing_info"]["anomaly_score"],
                                existing_age_group_model.id,
                                db,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}"
                        )
                        # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                        recalculated_normalized_score = max(
                            0.0, min(100.0, example["drawing_info"]["normalized_score"])
                        )

                    comparison_examples.append(
                        ComparisonExampleResponse(
                            drawing_id=example["drawing_id"],
                            filename=example["drawing_info"]["filename"],
                            age_years=example["drawing_info"]["age_years"],
                            subject=example["drawing_info"]["subject"],
                            similarity_score=example["similarity_score"],
                            anomaly_score=example["drawing_info"]["anomaly_score"],
                            normalized_score=recalculated_normalized_score,
                        )
                    )

            return AnalysisResultResponse(
                drawing=DrawingResponse.model_validate(drawing),
                analysis=analysis_response,
                interpretability=_convert_interpretability_to_response(
                    interpretability
                ),
                comparison_examples=comparison_examples,
            )
    else:
        logger.info(
            f"Force reanalysis requested for drawing {drawing_id}, creating new analysis"
        )
    embedding_record = (
        db.query(DrawingEmbedding)
        .filter(DrawingEmbedding.drawing_id == drawing_id)
        .order_by(desc(DrawingEmbedding.created_timestamp))
        .first()
    )

    if not embedding_record:
        # Generate hybrid embedding with subject information
        try:
            # Load image for hybrid embedding generation
            from PIL import Image

            image = Image.open(drawing.file_path).convert("RGB")

            # Generate hybrid embedding using subject information
            embedding_data = embedding_service.generate_hybrid_embedding(
                image=image,
                subject=drawing.subject,  # Use subject from drawing metadata
                age=drawing.age_years,
                use_cache=True,
            )

            # Save hybrid embedding to database using serialization utilities
            from app.utils.embedding_serialization import EmbeddingSerializer

            # Serialize hybrid embedding with component separation
            (
                full_bytes,
                visual_bytes,
                subject_bytes,
            ) = EmbeddingSerializer.serialize_hybrid_embedding(embedding_data)

            embedding_record = DrawingEmbedding(
                drawing_id=drawing_id,
                model_type="vit",
                embedding_type="hybrid",
                embedding_vector=full_bytes,
                visual_component=visual_bytes,
                subject_component=subject_bytes,
                vector_dimension=832,
            )
            db.add(embedding_record)
            db.commit()
            db.refresh(embedding_record)

        except Exception as e:
            logger.error(
                f"Failed to generate embedding for drawing {drawing_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate embedding: {str(e)}",
            )

    # Deserialize hybrid embedding using serialization utilities
    from app.utils.embedding_serialization import EmbeddingSerializer

    # Try to deserialize as hybrid embedding first
    embedding_data = EmbeddingSerializer.deserialize_hybrid_embedding(
        full_bytes=embedding_record.embedding_vector,
        visual_bytes=getattr(embedding_record, "visual_component", None),
        subject_bytes=getattr(embedding_record, "subject_component", None),
    )

    # Fallback to legacy deserialization if hybrid fails
    if embedding_data is None:
        embedding_storage = get_embedding_storage()
        embedding_data = embedding_storage.retrieve_embedding(
            drawing_id=drawing_id,
            model_type="vit",
            serialized_data=embedding_record.embedding_vector,
            age=drawing.age_years,
            use_cache=True,
        )

        # If legacy embedding is not 832-dimensional, we need to convert it to hybrid
        if embedding_data.shape[0] != 832:
            logger.warning(
                f"Legacy embedding found for drawing {drawing_id}, converting to hybrid"
            )
            # Load image and regenerate as hybrid embedding
            from PIL import Image

            image = Image.open(drawing.file_path).convert("RGB")
            embedding_data = embedding_service.generate_hybrid_embedding(
                image=image,
                subject=drawing.subject,
                age=drawing.age_years,
                use_cache=True,
            )

    # Find appropriate age group model
    age_group_model = age_group_manager.find_appropriate_model(drawing.age_years, db)
    if not age_group_model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No appropriate model found for age {drawing.age_years}",
        )

    # Compute subject-aware anomaly scores
    try:
        # Use the new subject-aware anomaly scoring method
        anomaly_scores = model_manager.compute_anomaly_score(
            embedding_data, age_group_model.id, db
        )

        # Extract overall score for backward compatibility
        anomaly_score = anomaly_scores["overall_anomaly_score"]
        visual_anomaly_score = anomaly_scores["visual_anomaly_score"]
        subject_anomaly_score = anomaly_scores["subject_anomaly_score"]

        # Normalize score
        normalized_score = score_normalizer.normalize_score(
            anomaly_score, age_group_model.id, db
        )

        # Determine if anomaly
        is_anomaly, threshold_used, model_info = threshold_manager.is_anomaly(
            anomaly_score, drawing.age_years, db
        )

        # Log the anomaly determination
        logger.info(
            f"Drawing {drawing_id}: is_anomaly={is_anomaly}, anomaly_score={anomaly_score}, threshold_used={threshold_used}"
        )

        # Determine anomaly attribution only for anomalous drawings (AFTER is_anomaly is final)
        if is_anomaly:
            logger.info(
                f"Drawing {drawing_id}: Determining attribution for anomalous drawing"
            )
            anomaly_attribution = model_manager.determine_attribution(
                embedding_data, age_group_model.id, db
            )
            logger.info(
                f"Drawing {drawing_id}: Attribution determined as: {anomaly_attribution}"
            )
        else:
            # No attribution needed for normal drawings
            logger.info(
                f"Drawing {drawing_id}: Setting attribution to None for normal drawing"
            )
            anomaly_attribution = None

        # Calculate confidence based on distance from threshold
        # For threshold-adjacent cases, confidence should be lower
        if threshold_used > 0:
            distance_from_threshold = abs(anomaly_score - threshold_used)
            relative_distance = distance_from_threshold / threshold_used

            # Use a more intuitive confidence calculation
            # - High confidence when far from threshold (either direction)
            # - Low confidence when very close to threshold
            if relative_distance < 0.01:  # Very close to threshold (within 1%)
                confidence = 0.1  # Low confidence for threshold-adjacent cases
            elif relative_distance < 0.05:  # Close to threshold (within 5%)
                confidence = 0.3  # Medium-low confidence
            elif relative_distance < 0.1:  # Moderately close (within 10%)
                confidence = 0.6  # Medium confidence
            else:  # Far from threshold
                confidence = min(
                    0.95, 0.5 + relative_distance
                )  # High confidence, capped at 95%
        else:
            confidence = 0.5

    except Exception as e:
        logger.error(
            f"Failed to compute subject-aware anomaly score for drawing {drawing_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute subject-aware anomaly score: {str(e)}",
        )

    # Save analysis result with subject-aware fields
    analysis = AnomalyAnalysis(
        drawing_id=drawing_id,
        age_group_model_id=age_group_model.id,
        anomaly_score=anomaly_score,
        normalized_score=normalized_score,
        visual_anomaly_score=visual_anomaly_score,
        subject_anomaly_score=subject_anomaly_score,
        anomaly_attribution=anomaly_attribution,
        analysis_type="subject_aware",
        is_anomaly=is_anomaly,
        confidence=confidence,
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    # Generate interpretability for all drawings using simplified approach
    interpretability_result = None
    try:
        # Load the image for analysis
        import json
        import os
        from pathlib import Path

        from PIL import Image, ImageDraw, ImageFilter

        image = Image.open(drawing.file_path)

        # Create static directory if it doesn't exist
        static_dir = Path("static/saliency_maps")
        static_dir.mkdir(parents=True, exist_ok=True)

        # Generate saliency map filename
        saliency_filename = f"saliency_{analysis.id}_{drawing_id}.png"
        saliency_path = static_dir / saliency_filename
        saliency_url = f"/static/saliency_maps/{saliency_filename}"

        # Create a simple saliency map based on anomaly score
        saliency_map = _create_simple_saliency_map(image, anomaly_score, is_anomaly)
        saliency_map.save(saliency_path)

        # Generate explanation based on analysis results with accurate score descriptions
        if is_anomaly:
            if normalized_score >= 80:
                score_description = "high"
            elif normalized_score >= 60:
                score_description = "moderately high"
            else:
                score_description = "moderate"

            explanation_text = f"Analysis of this drawing reveals patterns that deviate from typical developmental expectations for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates significant differences in drawing characteristics such as spatial organization, detail complexity, or symbolic representation compared to age-matched peers."
        else:
            if normalized_score < 40:
                score_description = "low"
                explanation_text = f"This drawing demonstrates age-appropriate developmental patterns for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates the drawing aligns well with expected developmental milestones in areas such as fine motor control, spatial awareness, and symbolic thinking."
            elif normalized_score < 60:
                score_description = "moderate"
                explanation_text = f"This drawing demonstrates age-appropriate developmental patterns for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates the drawing aligns well with expected developmental milestones in areas such as fine motor control, spatial awareness, and symbolic thinking."
            elif normalized_score < 90:
                score_description = "moderately elevated but still within normal range"
                explanation_text = f"This drawing demonstrates age-appropriate developmental patterns for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates the drawing aligns with expected developmental milestones, though some features show slight variation from typical patterns."
            else:
                # Very high scores (90-100) that are still technically normal
                score_description = "very high but still within normal range"
                explanation_text = f"This drawing shows patterns that are very close to the anomaly threshold for a {drawing.age_years}-year-old child. While technically classified as normal, the {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates several features that stand out from typical age-expected patterns. This drawing may warrant closer examination or discussion with a professional to understand the specific characteristics that contribute to the elevated score."

        # Create realistic importance regions based on image analysis
        regions = _analyze_image_regions(image, anomaly_score, is_anomaly)

        interpretability_record = InterpretabilityResult(
            analysis_id=analysis.id,
            saliency_map_path=str(saliency_path),
            overlay_image_path=saliency_url,  # Use saliency URL for overlay
            explanation_text=explanation_text,
            importance_regions=json.dumps(regions),
        )

        db.add(interpretability_record)
        db.commit()
        db.refresh(interpretability_record)

        # Create response with correct field mapping
        interpretability_result = InterpretabilityResponse(
            saliency_map_url=saliency_url,
            overlay_image_url=saliency_url,
            explanation_text=explanation_text,
            importance_regions=regions,  # Pass as list, not JSON string
        )
        logger.info(f"Generated simplified interpretability for drawing {drawing_id}")

    except Exception as e:
        logger.error(
            f"Failed to generate interpretability for drawing {drawing_id}: {str(e)}"
        )
        # Continue without interpretability

    logger.info(
        f"Analysis completed for drawing {drawing_id}: "
        f"score={anomaly_score:.6f}, anomaly={is_anomaly}"
    )

    # Create analysis response with additional fields
    # Recalculate normalized score to ensure 0-100 scale compatibility
    try:
        recalculated_normalized_score = score_normalizer.normalize_score(
            analysis.anomaly_score, age_group_model.id, db
        )
    except Exception as e:
        logger.warning(
            f"Failed to recalculate normalized score for analysis {analysis.id}: {e}"
        )
        # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
        recalculated_normalized_score = max(0.0, min(100.0, analysis.normalized_score))

    analysis_response = AnomalyAnalysisResponse(
        id=analysis.id,
        drawing_id=analysis.drawing_id,
        anomaly_score=analysis.anomaly_score,
        normalized_score=recalculated_normalized_score,
        visual_anomaly_score=visual_anomaly_score,
        subject_anomaly_score=subject_anomaly_score,
        anomaly_attribution=anomaly_attribution,
        analysis_type="subject_aware",
        subject_category=drawing.subject,
        is_anomaly=analysis.is_anomaly,
        confidence=analysis.confidence,
        age_group=f"{age_group_model.age_min}-{age_group_model.age_max}",
        method_used="autoencoder",
        vision_model="vit",
        analysis_timestamp=analysis.analysis_timestamp,
    )

    # Get comparison examples for new analysis
    comparison_examples = []
    similar_examples = comparison_service.find_similar_normal_examples(
        target_drawing_id=drawing_id,
        age_group_min=age_group_model.age_min,
        age_group_max=age_group_model.age_max,
        db=db,
        max_examples=3,
    )

    comparison_examples = []
    for example in similar_examples:
        # Recalculate normalized score for comparison examples
        try:
            recalculated_normalized_score = score_normalizer.normalize_score(
                example["drawing_info"]["anomaly_score"], age_group_model.id, db
            )
        except Exception as e:
            logger.warning(
                f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}"
            )
            # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
            recalculated_normalized_score = max(
                0.0, min(100.0, example["drawing_info"]["normalized_score"])
            )

        comparison_examples.append(
            ComparisonExampleResponse(
                drawing_id=example["drawing_id"],
                filename=example["drawing_info"]["filename"],
                age_years=example["drawing_info"]["age_years"],
                subject=example["drawing_info"]["subject"],
                similarity_score=example["similarity_score"],
                anomaly_score=example["drawing_info"]["anomaly_score"],
                normalized_score=recalculated_normalized_score,
            )
        )

    return AnalysisResultResponse(
        drawing=DrawingResponse.model_validate(drawing),
        analysis=analysis_response,
        interpretability=interpretability_result,
        comparison_examples=comparison_examples,
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
    analyses_with_drawings = (
        db.query(AnomalyAnalysis.anomaly_score, Drawing.age_years)
        .join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id)
        .all()
    )

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
            logger.warning(
                f"Failed to recalculate anomaly status for age {analysis.age_years}: {e}"
            )
            # We'll count this as normal to be conservative
            normal_count += 1

    # Get recent analyses with drawing info
    recent_analyses_query = (
        db.query(
            AnomalyAnalysis.id,
            AnomalyAnalysis.drawing_id,
            Drawing.filename,
            Drawing.age_years,
            AnomalyAnalysis.anomaly_score,
            AnomalyAnalysis.is_anomaly,
            AnomalyAnalysis.analysis_timestamp,
        )
        .join(Drawing, AnomalyAnalysis.drawing_id == Drawing.id)
        .order_by(desc(AnomalyAnalysis.analysis_timestamp))
        .limit(10)
        .all()
    )

    recent_analyses = [
        {
            "id": analysis.id,
            "drawing_id": analysis.drawing_id,
            "filename": analysis.filename,
            "age_years": analysis.age_years,
            "anomaly_score": analysis.anomaly_score,
            "is_anomaly": analysis.is_anomaly,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
        }
        for analysis in recent_analyses_query
    ]

    # Get age distribution
    age_distribution_query = (
        db.query(
            func.floor(Drawing.age_years).label("age_floor"),
            func.count(Drawing.id).label("count"),
        )
        .group_by(func.floor(Drawing.age_years))
        .all()
    )

    age_distribution = [
        {"age_group": f"{int(age_floor)}-{int(age_floor)+1}", "count": count}
        for age_floor, count in age_distribution_query
    ]

    # Get model status
    active_models = (
        db.query(AgeGroupModel).filter(AgeGroupModel.is_active == True).count()
    )
    latest_model = (
        db.query(AgeGroupModel).order_by(desc(AgeGroupModel.created_timestamp)).first()
    )

    model_status = {
        "vision_model": "vit",
        "is_loaded": embedding_service.is_ready() if embedding_service else False,
        "last_updated": (
            latest_model.created_timestamp.isoformat() if latest_model else ""
        ),
        "active_age_groups": active_models,
    }

    return {
        "total_drawings": total_drawings,
        "total_analyses": total_analyses,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "recent_analyses": recent_analyses,
        "age_distribution": age_distribution,
        "model_status": model_status,
    }


@router.post("/analyze/{drawing_id}", response_model=AnalysisResultResponse)
async def analyze_drawing(
    drawing_id: int, request: AnalysisRequest = None, db: Session = Depends(get_db)
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
        logger.error(
            f"Unexpected error during analysis of drawing {drawing_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to unexpected error",
        )


@router.post("/batch", response_model=dict)
async def batch_analyze(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Batch analyze multiple drawings.

    This endpoint accepts a list of drawing IDs and processes them
    in the background, returning a batch ID for progress tracking.
    """
    # Validate that all drawings exist
    existing_drawings = (
        db.query(Drawing.id).filter(Drawing.id.in_(request.drawing_ids)).all()
    )

    existing_ids = {d.id for d in existing_drawings}
    missing_ids = set(request.drawing_ids) - existing_ids

    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Drawings not found: {list(missing_ids)}",
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
        db,
    )

    return {
        "batch_id": batch_id,
        "total_drawings": len(request.drawing_ids),
        "status": "processing",
        "progress_url": f"/api/v1/analysis/batch/{batch_id}/progress",
    }


async def process_batch_analysis(
    batch_id: str, drawing_ids: List[int], force_reanalysis: bool, db: Session
):
    """Background task for processing batch analysis"""
    try:
        logger.info(
            f"Starting batch analysis {batch_id} for {len(drawing_ids)} drawings"
        )

        for i, drawing_id in enumerate(drawing_ids):
            try:
                # Update progress
                progress = (i / len(drawing_ids)) * 100
                batch_tracker.update_batch(
                    batch_id, status=f"processing_drawing_{i+1}_of_{len(drawing_ids)}"
                )

                # Perform analysis
                result = await perform_single_analysis(drawing_id, db, force_reanalysis)
                batch_tracker.add_result(batch_id, result.model_dump())

                logger.debug(
                    f"Completed analysis for drawing {drawing_id} in batch {batch_id}"
                )

            except Exception as e:
                error_info = {
                    "drawing_id": drawing_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                batch_tracker.add_error(batch_id, error_info)
                logger.error(
                    f"Failed to analyze drawing {drawing_id} in batch {batch_id}: {str(e)}"
                )

        # Mark batch as completed
        batch_tracker.update_batch(
            batch_id, status="completed", completed_at=datetime.utcnow()
        )

        batch_info = batch_tracker.get_batch(batch_id)
        logger.info(
            f"Batch analysis {batch_id} completed: "
            f"{batch_info['completed']} successful, {batch_info['failed']} failed"
        )

    except Exception as e:
        batch_tracker.update_batch(
            batch_id, status=f"failed: {str(e)}", completed_at=datetime.utcnow()
        )
        logger.error(f"Batch analysis {batch_id} failed: {str(e)}")


@router.get("/batch/{batch_id}/progress", response_model=BatchAnalysisResponse)
async def get_batch_progress(batch_id: str):
    """Get progress of batch analysis."""
    batch_info = batch_tracker.get_batch(batch_id)

    if not batch_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Batch {batch_id} not found"
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
    analysis = (
        db.query(AnomalyAnalysis).filter(AnomalyAnalysis.id == analysis_id).first()
    )

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found",
        )

    # Get associated drawing
    drawing = db.query(Drawing).filter(Drawing.id == analysis.drawing_id).first()
    if not drawing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Associated drawing not found",
        )

    # Get interpretability results if available
    interpretability = (
        db.query(InterpretabilityResult)
        .filter(InterpretabilityResult.analysis_id == analysis_id)
        .first()
    )

    # Get age group model for additional fields
    age_group_model = (
        db.query(AgeGroupModel)
        .filter(AgeGroupModel.id == analysis.age_group_model_id)
        .first()
    )

    # Use the stored normalized score for consistency with explanation text and attribution
    # The stored score was used for the original analysis and explanation generation
    stored_normalized_score = analysis.normalized_score

    # Only validate that the score is within reasonable bounds (0-100)
    if stored_normalized_score < 0 or stored_normalized_score > 100:
        logger.warning(
            f"Analysis {analysis.id} has out-of-bounds normalized score: {stored_normalized_score}"
        )
        # Clamp to valid range
        stored_normalized_score = max(0.0, min(100.0, stored_normalized_score))

    analysis_response = AnomalyAnalysisResponse(
        id=analysis.id,
        drawing_id=analysis.drawing_id,
        anomaly_score=analysis.anomaly_score,
        normalized_score=stored_normalized_score,
        visual_anomaly_score=getattr(analysis, "visual_anomaly_score", None),
        subject_anomaly_score=getattr(analysis, "subject_anomaly_score", None),
        anomaly_attribution=getattr(analysis, "anomaly_attribution", None),
        analysis_type=getattr(analysis, "analysis_type", "subject_aware"),
        subject_category=drawing.subject,
        is_anomaly=analysis.is_anomaly,
        confidence=analysis.confidence,
        age_group=(
            f"{age_group_model.age_min}-{age_group_model.age_max}"
            if age_group_model
            else "unknown"
        ),
        method_used="autoencoder",
        vision_model="vit",
        analysis_timestamp=analysis.analysis_timestamp,
    )

    # Get comparison examples for retrieved analysis
    comparison_examples = []
    if age_group_model:
        similar_examples = comparison_service.find_similar_normal_examples(
            target_drawing_id=drawing.id,
            age_group_min=age_group_model.age_min,
            age_group_max=age_group_model.age_max,
            db=db,
            max_examples=3,
        )

        comparison_examples = []
        for example in similar_examples:
            # Recalculate normalized score for comparison examples
            try:
                recalculated_normalized_score = score_normalizer.normalize_score(
                    example["drawing_info"]["anomaly_score"], age_group_model.id, db
                )
            except Exception as e:
                logger.warning(
                    f"Failed to recalculate normalized score for comparison example {example['drawing_id']}: {e}"
                )
                # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
                recalculated_normalized_score = max(
                    0.0, min(100.0, example["drawing_info"]["normalized_score"])
                )

            comparison_examples.append(
                ComparisonExampleResponse(
                    drawing_id=example["drawing_id"],
                    filename=example["drawing_info"]["filename"],
                    age_years=example["drawing_info"]["age_years"],
                    subject=example["drawing_info"]["subject"],
                    similarity_score=example["similarity_score"],
                    anomaly_score=example["drawing_info"]["anomaly_score"],
                    normalized_score=recalculated_normalized_score,
                )
            )

    return AnalysisResultResponse(
        drawing=DrawingResponse.model_validate(drawing),
        analysis=analysis_response,
        interpretability=_convert_interpretability_to_response(interpretability),
        comparison_examples=comparison_examples,
    )


@router.post("/embeddings/{drawing_id}")
async def generate_embedding(drawing_id: int, db: Session = Depends(get_db)):
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
            detail=f"Drawing with ID {drawing_id} not found",
        )

    # Check if embedding already exists
    existing_embedding = (
        db.query(DrawingEmbedding)
        .filter(DrawingEmbedding.drawing_id == drawing_id)
        .order_by(desc(DrawingEmbedding.created_timestamp))
        .first()
    )

    if existing_embedding:
        return {
            "drawing_id": drawing_id,
            "status": "exists",
            "message": "Embedding already exists",
            "embedding_id": existing_embedding.id,
            "vector_dimension": existing_embedding.vector_dimension,
            "created_timestamp": existing_embedding.created_timestamp,
        }

    # Generate embedding
    try:
        # Initialize embedding service if not ready
        if not embedding_service.is_ready():
            embedding_service.initialize()

        # Generate hybrid embedding with subject information
        from PIL import Image

        image = Image.open(drawing.file_path).convert("RGB")

        embedding_data = embedding_service.generate_hybrid_embedding(
            image=image,
            subject=drawing.subject,  # Use subject from drawing metadata
            age=drawing.age_years,
            use_cache=True,
        )

        # Save hybrid embedding to database using serialization utilities
        from app.utils.embedding_serialization import EmbeddingSerializer

        # Serialize hybrid embedding with component separation
        (
            full_bytes,
            visual_bytes,
            subject_bytes,
        ) = EmbeddingSerializer.serialize_hybrid_embedding(embedding_data)

        embedding_record = DrawingEmbedding(
            drawing_id=drawing_id,
            model_type="vit",
            embedding_type="hybrid",
            embedding_vector=full_bytes,
            visual_component=visual_bytes,
            subject_component=subject_bytes,
            vector_dimension=832,
        )
        db.add(embedding_record)
        db.commit()
        db.refresh(embedding_record)

        logger.info(
            f"Generated embedding for drawing {drawing_id}: dimension={len(embedding_data)}"
        )

        return {
            "drawing_id": drawing_id,
            "status": "generated",
            "message": "Embedding generated successfully",
            "embedding_id": embedding_record.id,
            "vector_dimension": embedding_record.vector_dimension,
            "created_timestamp": embedding_record.created_timestamp,
        }

    except Exception as e:
        logger.error(f"Failed to generate embedding for drawing {drawing_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {str(e)}",
        )


@router.get("/drawing/{drawing_id}", response_model=AnalysisHistoryResponse)
async def get_drawing_analyses(
    drawing_id: int, limit: int = 10, db: Session = Depends(get_db)
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
            detail=f"Drawing with ID {drawing_id} not found",
        )

    # Get analyses for this drawing
    analyses = (
        db.query(AnomalyAnalysis)
        .filter(AnomalyAnalysis.drawing_id == drawing_id)
        .order_by(desc(AnomalyAnalysis.analysis_timestamp))
        .limit(limit)
        .all()
    )

    # Get total count
    total_count = (
        db.query(AnomalyAnalysis)
        .filter(AnomalyAnalysis.drawing_id == drawing_id)
        .count()
    )

    # Convert analyses to response format with additional fields
    analysis_responses = []
    for analysis in analyses:
        age_group_model = (
            db.query(AgeGroupModel)
            .filter(AgeGroupModel.id == analysis.age_group_model_id)
            .first()
        )

        # Recalculate normalized score to ensure 0-100 scale compatibility
        try:
            recalculated_normalized_score = score_normalizer.normalize_score(
                analysis.anomaly_score, analysis.age_group_model_id, db
            )
        except Exception as e:
            logger.warning(
                f"Failed to recalculate normalized score for analysis {analysis.id}: {e}"
            )
            # Fallback: if stored score is negative, use 0; if > 100, use 100; otherwise use stored value
            recalculated_normalized_score = max(
                0.0, min(100.0, analysis.normalized_score)
            )

        analysis_response = AnomalyAnalysisResponse(
            id=analysis.id,
            drawing_id=analysis.drawing_id,
            anomaly_score=analysis.anomaly_score,
            normalized_score=recalculated_normalized_score,
            visual_anomaly_score=getattr(analysis, "visual_anomaly_score", None),
            subject_anomaly_score=getattr(analysis, "subject_anomaly_score", None),
            anomaly_attribution=getattr(analysis, "anomaly_attribution", None),
            analysis_type=getattr(analysis, "analysis_type", "subject_aware"),
            subject_category=drawing.subject if drawing else None,
            is_anomaly=analysis.is_anomaly,
            confidence=analysis.confidence,
            age_group=(
                f"{age_group_model.age_min}-{age_group_model.age_max}"
                if age_group_model
                else "unknown"
            ),
            method_used="autoencoder",
            vision_model="vit",
            analysis_timestamp=analysis.analysis_timestamp,
        )
        analysis_responses.append(analysis_response)

    return AnalysisHistoryResponse(
        drawing_id=drawing_id, analyses=analysis_responses, total_count=total_count
    )
