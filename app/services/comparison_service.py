"""
Comparison service for finding similar normal examples.

This service provides functionality to find similar normal drawings
from the same age group for comparison purposes when displaying
analysis results.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models.database import AnomalyAnalysis, Drawing, DrawingEmbedding
from app.utils.embedding_serialization import get_embedding_storage

logger = logging.getLogger(__name__)


class ComparisonService:
    """Service for finding similar normal examples for comparison."""

    def __init__(self):
        self.embedding_storage = get_embedding_storage()

    def find_similar_normal_examples(
        self,
        target_drawing_id: int,
        age_group_min: float,
        age_group_max: float,
        db: Session,
        max_examples: int = 3,
        similarity_threshold: float = 0.8,
        subject_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find similar normal examples from the same age group and optionally same subject.

        Args:
            target_drawing_id: ID of the drawing to find comparisons for
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            db: Database session
            max_examples: Maximum number of examples to return
            similarity_threshold: Minimum similarity score (0-1)
            subject_category: Optional subject category to filter by

        Returns:
            List of similar normal examples with metadata
        """
        try:
            # Get target drawing embedding
            target_embedding = self._get_drawing_embedding(target_drawing_id, db)
            if target_embedding is None:
                logger.warning(f"No embedding found for drawing {target_drawing_id}")
                return []

            # Find normal drawings in the same age group and subject
            normal_drawings = self._get_normal_drawings_in_age_group(
                age_group_min, age_group_max, target_drawing_id, db, subject_category
            )

            if not normal_drawings:
                # If no subject-specific examples found, try fallback without subject filter
                if subject_category and subject_category != "unspecified":
                    logger.info(
                        f"No normal drawings found for subject '{subject_category}' in age group {age_group_min}-{age_group_max}, trying fallback"
                    )
                    normal_drawings = self._get_normal_drawings_in_age_group(
                        age_group_min, age_group_max, target_drawing_id, db, None
                    )

                if not normal_drawings:
                    logger.info(
                        f"No normal drawings found in age group {age_group_min}-{age_group_max}"
                    )
                    return []

            # Calculate similarities and find best matches
            similar_examples = []

            for drawing_id, drawing_info in normal_drawings.items():
                # Get embedding for this drawing
                candidate_embedding = self._get_drawing_embedding(drawing_id, db)
                if candidate_embedding is None:
                    continue

                # Calculate similarity
                similarity = self._calculate_cosine_similarity(
                    target_embedding, candidate_embedding
                )

                if similarity >= similarity_threshold:
                    similar_examples.append(
                        {
                            "drawing_id": drawing_id,
                            "similarity_score": float(similarity),
                            "drawing_info": drawing_info,
                        }
                    )

            # Sort by similarity and return top examples
            similar_examples.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_examples[:max_examples]

        except Exception as e:
            logger.error(f"Failed to find similar normal examples: {str(e)}")
            return []

    def _get_drawing_embedding(
        self, drawing_id: int, db: Session
    ) -> Optional[np.ndarray]:
        """Get embedding for a drawing."""
        try:
            # Get embedding record
            embedding_record = (
                db.query(DrawingEmbedding)
                .filter(DrawingEmbedding.drawing_id == drawing_id)
                .order_by(DrawingEmbedding.created_timestamp.desc())
                .first()
            )

            if not embedding_record:
                return None

            # Get drawing for age information
            drawing = db.query(Drawing).filter(Drawing.id == drawing_id).first()
            if not drawing:
                return None

            # Deserialize embedding
            embedding = self.embedding_storage.retrieve_embedding(
                drawing_id=drawing_id,
                model_type=embedding_record.model_type,
                serialized_data=embedding_record.embedding_vector,
                age=drawing.age_years,
                use_cache=True,
            )

            return embedding

        except Exception as e:
            logger.error(f"Failed to get embedding for drawing {drawing_id}: {str(e)}")
            return None

    def _get_normal_drawings_in_age_group(
        self,
        age_group_min: float,
        age_group_max: float,
        exclude_drawing_id: int,
        db: Session,
        subject_category: Optional[str] = None,
        max_candidates: int = 50,
    ) -> Dict[int, Dict[str, Any]]:
        """Get normal drawings in the specified age group and optionally subject category."""
        try:
            # Build base query for drawings in age group that have been analyzed as normal
            query = (
                db.query(
                    Drawing.id,
                    Drawing.filename,
                    Drawing.age_years,
                    Drawing.subject,
                    Drawing.file_path,
                    AnomalyAnalysis.anomaly_score,
                    AnomalyAnalysis.normalized_score,
                )
                .join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id)
                .join(DrawingEmbedding, Drawing.id == DrawingEmbedding.drawing_id)
                .filter(
                    and_(
                        Drawing.age_years >= age_group_min,
                        Drawing.age_years <= age_group_max,
                        Drawing.id != exclude_drawing_id,
                        AnomalyAnalysis.is_anomaly == False,
                    )
                )
            )

            # Add subject filter if specified
            if subject_category and subject_category != "unspecified":
                query = query.filter(Drawing.subject == subject_category)

            query = query.order_by(
                AnomalyAnalysis.normalized_score.asc()  # Prefer most normal examples
            ).limit(max_candidates)

            results = query.all()

            # Convert to dictionary format
            normal_drawings = {}
            for result in results:
                normal_drawings[result.id] = {
                    "filename": result.filename,
                    "age_years": result.age_years,
                    "subject": result.subject,
                    "file_path": result.file_path,
                    "anomaly_score": result.anomaly_score,
                    "normalized_score": result.normalized_score,
                }

            return normal_drawings

        except Exception as e:
            logger.error(f"Failed to get normal drawings in age group: {str(e)}")
            return {}

    def _calculate_cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

            # Ensure result is in valid range
            return max(0.0, min(1.0, float(similarity)))

        except Exception as e:
            logger.error(f"Failed to calculate cosine similarity: {str(e)}")
            return 0.0

    def get_comparison_statistics(
        self, age_group_min: float, age_group_max: float, db: Session
    ) -> Dict[str, Any]:
        """Get statistics about available comparison examples in an age group."""
        try:
            # Count total drawings in age group
            total_drawings = (
                db.query(Drawing)
                .filter(
                    and_(
                        Drawing.age_years >= age_group_min,
                        Drawing.age_years <= age_group_max,
                    )
                )
                .count()
            )

            # Count distinct normal drawings with embeddings
            normal_with_embeddings = (
                db.query(Drawing.id)
                .join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id)
                .join(DrawingEmbedding, Drawing.id == DrawingEmbedding.drawing_id)
                .filter(
                    and_(
                        Drawing.age_years >= age_group_min,
                        Drawing.age_years <= age_group_max,
                        AnomalyAnalysis.is_anomaly == False,
                    )
                )
                .distinct()
                .count()
            )

            # Count distinct anomalous drawings
            anomalous_count = (
                db.query(Drawing.id)
                .join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id)
                .filter(
                    and_(
                        Drawing.age_years >= age_group_min,
                        Drawing.age_years <= age_group_max,
                        AnomalyAnalysis.is_anomaly == True,
                    )
                )
                .distinct()
                .count()
            )

            return {
                "age_group": f"{age_group_min}-{age_group_max}",
                "total_drawings": total_drawings,
                "normal_with_embeddings": normal_with_embeddings,
                "anomalous_count": anomalous_count,
                "comparison_availability": normal_with_embeddings >= 3,
            }

        except Exception as e:
            logger.error(f"Failed to get comparison statistics: {str(e)}")
            return {
                "age_group": f"{age_group_min}-{age_group_max}",
                "total_drawings": 0,
                "normal_with_embeddings": 0,
                "anomalous_count": 0,
                "comparison_availability": False,
            }

    def get_subject_specific_examples(
        self,
        age_group_min: float,
        age_group_max: float,
        subject_category: str,
        db: Session,
        max_examples: int = 5,
        include_anomalous: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get subject-specific examples for comparison and educational purposes.

        Args:
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            subject_category: Subject category to filter by
            db: Database session
            max_examples: Maximum number of examples per category
            include_anomalous: Whether to include anomalous examples

        Returns:
            Dictionary with 'normal' and optionally 'anomalous' example lists
        """
        try:
            examples = {"normal": []}

            # Get normal examples
            normal_query = (
                db.query(
                    Drawing.id,
                    Drawing.filename,
                    Drawing.age_years,
                    Drawing.subject,
                    Drawing.file_path,
                    AnomalyAnalysis.anomaly_score,
                    AnomalyAnalysis.normalized_score,
                )
                .join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id)
                .filter(
                    and_(
                        Drawing.age_years >= age_group_min,
                        Drawing.age_years <= age_group_max,
                        Drawing.subject == subject_category,
                        AnomalyAnalysis.is_anomaly == False,
                    )
                )
                .order_by(AnomalyAnalysis.normalized_score.asc())
                .limit(max_examples)
            )

            normal_results = normal_query.all()
            for result in normal_results:
                examples["normal"].append(
                    {
                        "drawing_id": result.id,
                        "filename": result.filename,
                        "age_years": result.age_years,
                        "subject": result.subject,
                        "file_path": result.file_path,
                        "anomaly_score": result.anomaly_score,
                        "normalized_score": result.normalized_score,
                        "category": "normal",
                    }
                )

            # Get anomalous examples if requested
            if include_anomalous:
                examples["anomalous"] = []

                anomalous_query = (
                    db.query(
                        Drawing.id,
                        Drawing.filename,
                        Drawing.age_years,
                        Drawing.subject,
                        Drawing.file_path,
                        AnomalyAnalysis.anomaly_score,
                        AnomalyAnalysis.normalized_score,
                    )
                    .join(AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id)
                    .filter(
                        and_(
                            Drawing.age_years >= age_group_min,
                            Drawing.age_years <= age_group_max,
                            Drawing.subject == subject_category,
                            AnomalyAnalysis.is_anomaly == True,
                        )
                    )
                    .order_by(AnomalyAnalysis.normalized_score.desc())
                    .limit(max_examples)
                )

                anomalous_results = anomalous_query.all()
                for result in anomalous_results:
                    examples["anomalous"].append(
                        {
                            "drawing_id": result.id,
                            "filename": result.filename,
                            "age_years": result.age_years,
                            "subject": result.subject,
                            "file_path": result.file_path,
                            "anomaly_score": result.anomaly_score,
                            "normalized_score": result.normalized_score,
                            "category": "anomalous",
                        }
                    )

            return examples

        except Exception as e:
            logger.error(f"Failed to get subject-specific examples: {str(e)}")
            return {"normal": [], "anomalous": [] if include_anomalous else None}

    def get_comparison_examples_with_fallback(
        self,
        target_drawing_id: int,
        age_group_min: float,
        age_group_max: float,
        subject_category: Optional[str],
        db: Session,
        max_examples: int = 3,
    ) -> Dict[str, Any]:
        """
        Get comparison examples with fallback strategy when subject-specific examples are unavailable.

        Args:
            target_drawing_id: ID of the drawing to find comparisons for
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            subject_category: Subject category to filter by
            db: Session
            max_examples: Maximum number of examples to return

        Returns:
            Dictionary containing examples and metadata about fallback strategy used
        """
        try:
            result = {
                "examples": [],
                "fallback_used": False,
                "fallback_reason": None,
                "subject_requested": subject_category,
                "subject_matched": subject_category,
            }

            # Try subject-specific examples first
            if subject_category and subject_category != "unspecified":
                examples = self.find_similar_normal_examples(
                    target_drawing_id=target_drawing_id,
                    age_group_min=age_group_min,
                    age_group_max=age_group_max,
                    db=db,
                    max_examples=max_examples,
                    subject_category=subject_category,
                )

                if examples:
                    result["examples"] = examples
                    return result
                else:
                    # No subject-specific examples found, try fallback
                    result["fallback_used"] = True
                    result["fallback_reason"] = (
                        f"No normal examples found for subject '{subject_category}'"
                    )

            # Fallback: get examples without subject filter
            examples = self.find_similar_normal_examples(
                target_drawing_id=target_drawing_id,
                age_group_min=age_group_min,
                age_group_max=age_group_max,
                db=db,
                max_examples=max_examples,
                subject_category=None,
            )

            result["examples"] = examples
            result["subject_matched"] = (
                "any" if result["fallback_used"] else subject_category
            )

            if not examples and not result["fallback_used"]:
                result["fallback_used"] = True
                result["fallback_reason"] = (
                    f"No normal examples found in age group {age_group_min}-{age_group_max}"
                )

            return result

        except Exception as e:
            logger.error(f"Failed to get comparison examples with fallback: {str(e)}")
            return {
                "examples": [],
                "fallback_used": True,
                "fallback_reason": f"Error: {str(e)}",
                "subject_requested": subject_category,
                "subject_matched": None,
            }


# Global service instance
_comparison_service = None


def get_comparison_service() -> ComparisonService:
    """Get the global comparison service instance."""
    global _comparison_service
    if _comparison_service is None:
        _comparison_service = ComparisonService()
    return _comparison_service
