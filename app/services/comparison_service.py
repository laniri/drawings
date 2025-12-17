"""
Comparison service for finding similar normal examples.

This service provides functionality to find similar normal drawings
from the same age group for comparison purposes when displaying
analysis results.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.models.database import Drawing, DrawingEmbedding, AnomalyAnalysis
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
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find similar normal examples from the same age group.
        
        Args:
            target_drawing_id: ID of the drawing to find comparisons for
            age_group_min: Minimum age for the age group
            age_group_max: Maximum age for the age group
            db: Database session
            max_examples: Maximum number of examples to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of similar normal examples with metadata
        """
        try:
            # Get target drawing embedding
            target_embedding = self._get_drawing_embedding(target_drawing_id, db)
            if target_embedding is None:
                logger.warning(f"No embedding found for drawing {target_drawing_id}")
                return []
            
            # Find normal drawings in the same age group
            normal_drawings = self._get_normal_drawings_in_age_group(
                age_group_min, age_group_max, target_drawing_id, db
            )
            
            if not normal_drawings:
                logger.info(f"No normal drawings found in age group {age_group_min}-{age_group_max}")
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
                    similar_examples.append({
                        "drawing_id": drawing_id,
                        "similarity_score": float(similarity),
                        "drawing_info": drawing_info
                    })
            
            # Sort by similarity and return top examples
            similar_examples.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_examples[:max_examples]
            
        except Exception as e:
            logger.error(f"Failed to find similar normal examples: {str(e)}")
            return []
    
    def _get_drawing_embedding(self, drawing_id: int, db: Session) -> Optional[np.ndarray]:
        """Get embedding for a drawing."""
        try:
            # Get embedding record
            embedding_record = db.query(DrawingEmbedding).filter(
                DrawingEmbedding.drawing_id == drawing_id
            ).order_by(DrawingEmbedding.created_timestamp.desc()).first()
            
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
                use_cache=True
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
        max_candidates: int = 50
    ) -> Dict[int, Dict[str, Any]]:
        """Get normal drawings in the specified age group."""
        try:
            # Query for drawings in age group that have been analyzed as normal
            query = db.query(
                Drawing.id,
                Drawing.filename,
                Drawing.age_years,
                Drawing.subject,
                Drawing.file_path,
                AnomalyAnalysis.anomaly_score,
                AnomalyAnalysis.normalized_score
            ).join(
                AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id
            ).join(
                DrawingEmbedding, Drawing.id == DrawingEmbedding.drawing_id
            ).filter(
                and_(
                    Drawing.age_years >= age_group_min,
                    Drawing.age_years <= age_group_max,
                    Drawing.id != exclude_drawing_id,
                    AnomalyAnalysis.is_anomaly == False
                )
            ).order_by(
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
                    "normalized_score": result.normalized_score
                }
            
            return normal_drawings
            
        except Exception as e:
            logger.error(f"Failed to get normal drawings in age group: {str(e)}")
            return {}
    
    def _calculate_cosine_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
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
        self,
        age_group_min: float,
        age_group_max: float,
        db: Session
    ) -> Dict[str, Any]:
        """Get statistics about available comparison examples in an age group."""
        try:
            # Count total drawings in age group
            total_drawings = db.query(Drawing).filter(
                and_(
                    Drawing.age_years >= age_group_min,
                    Drawing.age_years <= age_group_max
                )
            ).count()
            
            # Count distinct normal drawings with embeddings
            normal_with_embeddings = db.query(Drawing.id).join(
                AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id
            ).join(
                DrawingEmbedding, Drawing.id == DrawingEmbedding.drawing_id
            ).filter(
                and_(
                    Drawing.age_years >= age_group_min,
                    Drawing.age_years <= age_group_max,
                    AnomalyAnalysis.is_anomaly == False
                )
            ).distinct().count()
            
            # Count distinct anomalous drawings
            anomalous_count = db.query(Drawing.id).join(
                AnomalyAnalysis, Drawing.id == AnomalyAnalysis.drawing_id
            ).filter(
                and_(
                    Drawing.age_years >= age_group_min,
                    Drawing.age_years <= age_group_max,
                    AnomalyAnalysis.is_anomaly == True
                )
            ).distinct().count()
            
            return {
                "age_group": f"{age_group_min}-{age_group_max}",
                "total_drawings": total_drawings,
                "normal_with_embeddings": normal_with_embeddings,
                "anomalous_count": anomalous_count,
                "comparison_availability": normal_with_embeddings >= 3
            }
            
        except Exception as e:
            logger.error(f"Failed to get comparison statistics: {str(e)}")
            return {
                "age_group": f"{age_group_min}-{age_group_max}",
                "total_drawings": 0,
                "normal_with_embeddings": 0,
                "anomalous_count": 0,
                "comparison_availability": False
            }


# Global service instance
_comparison_service = None


def get_comparison_service() -> ComparisonService:
    """Get the global comparison service instance."""
    global _comparison_service
    if _comparison_service is None:
        _comparison_service = ComparisonService()
    return _comparison_service