"""
Threshold Management Service for configurable anomaly detection thresholds.

This service handles threshold calculation, management, and dynamic updates
for anomaly detection in children's drawings analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.database import AgeGroupModel, AnomalyAnalysis, Drawing, DrawingEmbedding
from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)


class ThresholdManagerError(Exception):
    """Base exception for threshold manager errors."""
    pass


class ThresholdCalculationError(ThresholdManagerError):
    """Raised when threshold calculation fails."""
    pass


@dataclass
class ThresholdConfig:
    """Configuration for threshold management."""
    default_percentile: float = 95.0  # Default 95th percentile
    min_samples_for_calculation: int = 30  # Minimum samples needed for reliable threshold
    confidence_levels: List[float] = None  # Multiple confidence levels
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [90.0, 95.0, 99.0]


class ThresholdManager:
    """Manager for threshold calculation and dynamic updates."""
    
    def __init__(self, config: ThresholdConfig = None):
        self.config = config or ThresholdConfig()
        self.model_manager = get_model_manager()
        self._cached_thresholds = {}  # Cache for computed thresholds
        self._current_percentile = self.config.default_percentile  # Store current percentile
    
    def calculate_percentile_threshold(self, 
                                     scores: np.ndarray, 
                                     percentile: float) -> float:
        """
        Calculate threshold based on percentile of scores.
        
        Args:
            scores: Array of anomaly scores
            percentile: Percentile value (0-100)
            
        Returns:
            Threshold value at the specified percentile
        """
        if len(scores) == 0:
            raise ThresholdCalculationError("Cannot calculate threshold from empty scores array")
        
        if not (0 <= percentile <= 100):
            raise ThresholdCalculationError(f"Percentile must be between 0 and 100, got {percentile}")
        
        # Remove any non-finite values
        valid_scores = scores[np.isfinite(scores)]
        
        if len(valid_scores) == 0:
            raise ThresholdCalculationError("No valid (finite) scores found for threshold calculation")
        
        if len(valid_scores) < self.config.min_samples_for_calculation:
            logger.warning(f"Only {len(valid_scores)} valid scores available for threshold calculation "
                         f"(minimum recommended: {self.config.min_samples_for_calculation})")
        
        # Calculate percentile threshold
        threshold = np.percentile(valid_scores, percentile)
        
        if not np.isfinite(threshold):
            raise ThresholdCalculationError(f"Calculated threshold is not finite: {threshold}")
        
        logger.info(f"Calculated {percentile}th percentile threshold: {threshold:.6f} "
                   f"from {len(valid_scores)} scores")
        
        return float(threshold)
    
    def calculate_model_threshold(self, 
                                age_group_model_id: int, 
                                db: Session,
                                percentile: Optional[float] = None) -> Dict:
        """
        Calculate threshold for a specific age group model using existing analysis results.
        
        This optimized version uses existing anomaly scores from the database instead
        of recalculating reconstruction losses, making it much faster.
        
        Args:
            age_group_model_id: ID of the age group model
            db: Database session
            percentile: Percentile to use (defaults to config default)
            
        Returns:
            Dictionary containing threshold information
        """
        if percentile is None:
            percentile = self.config.default_percentile
        
        try:
            # Get the age group model
            age_group_model = db.query(AgeGroupModel).filter(
                AgeGroupModel.id == age_group_model_id
            ).first()
            
            if not age_group_model:
                raise ThresholdCalculationError(f"Age group model {age_group_model_id} not found")
            
            # Get existing anomaly scores for this age group model from analysis results
            # This is much faster than recalculating reconstruction losses
            analyses = db.query(AnomalyAnalysis).filter(
                AnomalyAnalysis.age_group_model_id == age_group_model_id
            ).all()
            
            if not analyses:
                raise ThresholdCalculationError(
                    f"No analysis results found for age group model {age_group_model_id}"
                )
            
            # Extract anomaly scores
            reconstruction_losses = [analysis.anomaly_score for analysis in analyses]
            
            if not reconstruction_losses:
                raise ThresholdCalculationError(
                    f"No valid anomaly scores found for model {age_group_model_id}"
                )
            
            scores_array = np.array(reconstruction_losses)
            
            # Calculate thresholds for multiple confidence levels
            thresholds = {}
            for conf_level in self.config.confidence_levels:
                thresholds[f"percentile_{conf_level}"] = self.calculate_percentile_threshold(
                    scores_array, conf_level
                )
            
            # Always calculate the requested percentile if it's not already in confidence_levels
            primary_threshold_key = f"percentile_{percentile}"
            if primary_threshold_key not in thresholds:
                thresholds[primary_threshold_key] = self.calculate_percentile_threshold(
                    scores_array, percentile
                )
            
            # Calculate additional statistics
            statistics = {
                "mean": float(np.mean(scores_array)),
                "std": float(np.std(scores_array)),
                "min": float(np.min(scores_array)),
                "max": float(np.max(scores_array)),
                "median": float(np.median(scores_array)),
                "sample_count": len(scores_array)
            }
            
            result = {
                "age_group_model_id": age_group_model_id,
                "age_range": (age_group_model.age_min, age_group_model.age_max),
                "thresholds": thresholds,
                "statistics": statistics,
                "primary_threshold": thresholds[primary_threshold_key],
                "percentile_used": percentile
            }
            
            # Cache the result
            cache_key = f"{age_group_model_id}_{percentile}"
            self._cached_thresholds[cache_key] = result
            
            logger.info(f"Calculated thresholds for model {age_group_model_id}: "
                       f"primary={result['primary_threshold']:.6f} from {len(analyses)} analyses")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate threshold for model {age_group_model_id}: {str(e)}")
            raise ThresholdCalculationError(f"Threshold calculation failed: {str(e)}")
    
    def update_model_threshold(self, 
                             age_group_model_id: int, 
                             new_threshold: float,
                             db: Session) -> bool:
        """
        Update the threshold for a specific age group model.
        
        Args:
            age_group_model_id: ID of the age group model
            new_threshold: New threshold value
            db: Database session
            
        Returns:
            True if update was successful
        """
        try:
            # Validate threshold value
            if not np.isfinite(new_threshold):
                raise ThresholdCalculationError(f"Threshold must be finite, got {new_threshold}")
            
            if new_threshold < 0:
                raise ThresholdCalculationError(f"Threshold must be non-negative, got {new_threshold}")
            
            # Get the age group model
            age_group_model = db.query(AgeGroupModel).filter(
                AgeGroupModel.id == age_group_model_id
            ).first()
            
            if not age_group_model:
                raise ThresholdCalculationError(f"Age group model {age_group_model_id} not found")
            
            # Update threshold
            old_threshold = age_group_model.threshold
            age_group_model.threshold = new_threshold
            db.commit()
            
            # Clear cached thresholds for this model
            keys_to_remove = [key for key in self._cached_thresholds.keys() 
                            if key.startswith(f"{age_group_model_id}_")]
            for key in keys_to_remove:
                del self._cached_thresholds[key]
            
            logger.info(f"Updated threshold for model {age_group_model_id}: "
                       f"{old_threshold:.6f} -> {new_threshold:.6f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update threshold for model {age_group_model_id}: {str(e)}")
            db.rollback()
            raise ThresholdCalculationError(f"Threshold update failed: {str(e)}")
    
    def recalculate_all_thresholds(self, 
                                 db: Session, 
                                 percentile: Optional[float] = None) -> Dict:
        """
        Recalculate thresholds for all active age group models.
        
        Args:
            db: Database session
            percentile: Percentile to use (defaults to config default)
            
        Returns:
            Dictionary containing results for all models
        """
        if percentile is None:
            percentile = self.config.default_percentile
        
        # Get all active models
        models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).all()
        
        results = {
            "total_models": len(models),
            "successful_updates": 0,
            "failed_updates": 0,
            "model_results": {},
            "errors": []
        }
        
        for model in models:
            try:
                # Calculate new threshold
                threshold_info = self.calculate_model_threshold(
                    model.id, db, percentile
                )
                
                # Update the model
                success = self.update_model_threshold(
                    model.id, 
                    threshold_info["primary_threshold"], 
                    db
                )
                
                if success:
                    results["successful_updates"] += 1
                    results["model_results"][model.id] = {
                        "success": True,
                        "age_range": (model.age_min, model.age_max),
                        "old_threshold": model.threshold,
                        "new_threshold": threshold_info["primary_threshold"],
                        "statistics": threshold_info["statistics"]
                    }
                else:
                    results["failed_updates"] += 1
                    results["errors"].append(f"Failed to update model {model.id}")
                
            except Exception as e:
                results["failed_updates"] += 1
                error_msg = f"Model {model.id}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(f"Failed to recalculate threshold for model {model.id}: {str(e)}")
        
        # Store the current percentile if recalculation was successful
        if results["successful_updates"] > 0:
            self.set_current_percentile(percentile)
        
        logger.info(f"Threshold recalculation complete: "
                   f"{results['successful_updates']} successful, "
                   f"{results['failed_updates']} failed")
        
        return results
    
    def get_threshold_for_age(self, 
                            age: float, 
                            db: Session) -> Optional[float]:
        """
        Get the appropriate threshold for a given age.
        
        Args:
            age: Age to get threshold for
            db: Database session
            
        Returns:
            Threshold value or None if no appropriate model found
        """
        # Find the appropriate model for this age
        from app.services.age_group_manager import get_age_group_manager
        age_group_manager = get_age_group_manager()
        
        model = age_group_manager.find_appropriate_model(age, db)
        
        if model:
            return model.threshold
        
        return None
    
    def is_anomaly(self, 
                   anomaly_score: float, 
                   age: float, 
                   db: Session) -> Tuple[bool, float, Optional[str]]:
        """
        Determine if a score represents an anomaly for a given age.
        
        Args:
            anomaly_score: The computed anomaly score
            age: Age of the subject
            db: Database session
            
        Returns:
            Tuple of (is_anomaly, threshold_used, model_info)
        """
        threshold = self.get_threshold_for_age(age, db)
        
        if threshold is None:
            logger.warning(f"No threshold available for age {age}")
            return False, 0.0, "No appropriate model found"
        
        is_anomaly = anomaly_score > threshold
        
        # Get model info for context
        from app.services.age_group_manager import get_age_group_manager
        age_group_manager = get_age_group_manager()
        model = age_group_manager.find_appropriate_model(age, db)
        
        model_info = None
        if model:
            model_info = f"Age group {model.age_min}-{model.age_max} (model {model.id})"
        
        return is_anomaly, threshold, model_info
    
    def get_current_percentile(self) -> float:
        """Get the current threshold percentile."""
        return self._current_percentile
    
    def set_current_percentile(self, percentile: float) -> None:
        """Set the current threshold percentile."""
        if 50.0 <= percentile <= 99.9:
            self._current_percentile = percentile
        else:
            raise ValueError("Percentile must be between 50.0 and 99.9")
    
    def get_threshold_statistics(self, db: Session) -> Dict:
        """
        Get statistics about all thresholds in the system.
        
        Args:
            db: Database session
            
        Returns:
            Dictionary containing threshold statistics
        """
        models = db.query(AgeGroupModel).filter(
            AgeGroupModel.is_active == True
        ).all()
        
        if not models:
            return {
                "total_models": 0,
                "threshold_range": (0, 0),
                "mean_threshold": 0,
                "model_thresholds": []
            }
        
        thresholds = [model.threshold for model in models]
        
        return {
            "total_models": len(models),
            "threshold_range": (min(thresholds), max(thresholds)),
            "mean_threshold": np.mean(thresholds),
            "std_threshold": np.std(thresholds),
            "model_thresholds": [
                {
                    "model_id": model.id,
                    "age_range": (model.age_min, model.age_max),
                    "threshold": model.threshold,
                    "sample_count": model.sample_count
                }
                for model in models
            ]
        }
    
    def clear_threshold_cache(self) -> None:
        """Clear the threshold calculation cache."""
        self._cached_thresholds.clear()
        logger.info("Threshold cache cleared")


# Global threshold manager instance
_threshold_manager = None


def get_threshold_manager() -> ThresholdManager:
    """Get the global threshold manager instance."""
    global _threshold_manager
    if _threshold_manager is None:
        _threshold_manager = ThresholdManager()
    return _threshold_manager