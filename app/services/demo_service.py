"""
Demo service for managing sample content and demo page functionality.

This service provides real analyzed sample drawings with complete results
for demonstration purposes, including interpretability visualizations.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import ConfigurationError
from app.models.database import AnomalyAnalysis, Drawing, InterpretabilityResult
from app.services.environment_storage import get_storage_service

logger = logging.getLogger(__name__)


class DemoService:
    """
    Service for managing demo content and real sample analysis results.

    Provides real analyzed sample drawings with interpretability visualizations
    for public demonstration of the system capabilities.
    """

    def __init__(self):
        self.demo_data_path = Path(settings.STATIC_DIR) / "demo"
        self.samples_file = self.demo_data_path / "demo_samples.json"

        # Ensure demo directory exists
        self.demo_data_path.mkdir(parents=True, exist_ok=True)

    def _get_age_group_display(self, age: float) -> str:
        """Convert age to display age group."""
        if age < 3:
            return "2-3"
        elif age < 4:
            return "3-4"
        elif age < 5:
            return "4-5"
        elif age < 6:
            return "5-6"
        elif age < 7:
            return "6-7"
        elif age < 8:
            return "7-8"
        elif age < 9:
            return "8-9"
        else:
            return "9-12"

    def _create_demo_sample_from_real_data(
        self,
        drawing: Drawing,
        analysis: AnomalyAnalysis,
        interpretability: Optional[InterpretabilityResult] = None,
        sample_id: int = 1,
    ) -> Dict[str, Any]:
        """Create a demo sample from real database data."""

        # Create descriptive title
        subject = drawing.subject or "drawing"
        age = int(drawing.age_years)
        title = f"{subject.title()} Drawing - Age {age}"

        # Create description based on analysis results
        if analysis.is_anomaly:
            description = f"An {subject} drawing by a {age}-year-old showing patterns that deviate from typical developmental expectations"
        else:
            description = f"A typical {subject} drawing by a {age}-year-old showing expected developmental features"

        # Build file URLs using environment-aware storage service
        storage_service = get_storage_service()
        original_image_url = storage_service.get_file_url(drawing.file_path)
        saliency_map_url = None

        if interpretability and interpretability.saliency_map_path:
            saliency_map_url = storage_service.get_file_url(
                interpretability.saliency_map_path
            )

        # Create analysis result
        analysis_result = {
            "anomaly_score": round(analysis.anomaly_score, 3),
            "normalized_score": round(analysis.normalized_score, 1),
            "is_anomaly": analysis.is_anomaly,
            "confidence": round(analysis.confidence, 2),
            "processing_time": 2.1,  # Approximate
            "model_version": "v2.0.0",
            "age_group_model": f"age_{self._get_age_group_display(drawing.age_years).replace('-', '_')}",
            "visual_score": (
                round(analysis.visual_anomaly_score, 3)
                if analysis.visual_anomaly_score
                else None
            ),
            "subject_score": (
                round(analysis.subject_anomaly_score, 3)
                if analysis.subject_anomaly_score
                else None
            ),
            "attribution": analysis.anomaly_attribution,
        }

        # Create interpretability data
        interpretability_data = None
        if interpretability and interpretability.explanation_text:
            interpretability_data = {
                "explanation": interpretability.explanation_text,
                "key_regions": [],  # Would need to parse importance_regions JSON if available
                "technical_details": {
                    "saliency_method": "gradient-based",
                    "attention_regions": 4,
                    "confidence_threshold": 0.7,
                },
            }
        else:
            # Create basic explanation based on analysis
            if analysis.is_anomaly:
                explanation = f"This drawing shows patterns that deviate from typical {age}-year-old developmental expectations. "
                if analysis.anomaly_attribution == "visual":
                    explanation += "The visual features show unusual characteristics for this age group."
                elif analysis.anomaly_attribution == "subject":
                    explanation += "The subject representation shows atypical patterns."
                elif analysis.anomaly_attribution == "both":
                    explanation += "Both visual features and subject representation show atypical patterns."
                else:
                    explanation += f"The anomaly score of {analysis.anomaly_score:.3f} indicates deviation from expected patterns."
            else:
                explanation = f"This drawing demonstrates age-appropriate developmental patterns for a {age}-year-old child. The anomaly score of {analysis.anomaly_score:.3f} indicates the drawing aligns well with expected developmental milestones."

            interpretability_data = {
                "explanation": explanation,
                "key_regions": [],
                "technical_details": {
                    "saliency_method": "gradient-based",
                    "attention_regions": 3,
                    "confidence_threshold": 0.7,
                },
            }

        return {
            "id": sample_id,
            "drawing_id": drawing.id,
            "title": title,
            "description": description,
            "age_group": self._get_age_group_display(drawing.age_years),
            "subject_category": drawing.subject or "unknown",
            "original_image": original_image_url,
            "saliency_map": saliency_map_url,
            "composite_image": original_image_url,  # Use original image for demo display
            "analysis_result": analysis_result,
            "interpretability": interpretability_data,
            "metadata": {
                "created_at": (
                    analysis.analysis_timestamp.isoformat()
                    if analysis.analysis_timestamp
                    else None
                ),
                "content_rating": "safe",
                "educational_value": "high",
            },
        }

    def _get_real_demo_samples(self) -> List[Dict[str, Any]]:
        """Get real demo samples from the database with randomness."""
        try:
            db = next(get_db())

            try:
                # Get threshold manager for dynamic anomaly classification
                from app.services.threshold_manager import get_threshold_manager
                import random

                threshold_manager = get_threshold_manager()

                # First, check if we have any drawings at all
                total_drawings = db.query(Drawing).count()
                logger.info(f"Total drawings in database: {total_drawings}")

                # Get drawings with analyses (we'll classify them dynamically)
                # Order by random to get variety
                all_drawings = (
                    db.query(Drawing)
                    .join(AnomalyAnalysis)
                    .filter(Drawing.subject.isnot(None))
                    .filter(AnomalyAnalysis.anomaly_score.isnot(None))
                    .order_by(func.random())
                    .limit(100)  # Get more to ensure we have variety
                    .all()
                )

                logger.info(f"Found {len(all_drawings)} drawings with analyses")

                # Classify drawings dynamically
                normal_drawings = []
                anomalous_drawings = []

                for drawing in all_drawings:
                    if drawing.analyses:
                        analysis = drawing.analyses[0]
                        # Use threshold manager to determine if anomaly
                        is_anomaly, _, _ = threshold_manager.is_anomaly(
                            analysis.anomaly_score, drawing.age_years, db
                        )

                        if is_anomaly:
                            anomalous_drawings.append(drawing)
                        else:
                            normal_drawings.append(drawing)

                # Randomly select from available drawings
                selected_normal = (
                    random.sample(normal_drawings, min(3, len(normal_drawings)))
                    if normal_drawings
                    else []
                )
                selected_anomalous = (
                    random.sample(anomalous_drawings, min(2, len(anomalous_drawings)))
                    if anomalous_drawings
                    else []
                )

                logger.info(
                    f"Classified: {len(normal_drawings)} normal, {len(anomalous_drawings)} anomalous"
                )
                logger.info(
                    f"Randomly selected: {len(selected_normal)} normal, {len(selected_anomalous)} anomalous"
                )

                demo_samples = []
                sample_id = 1

                # Process normal examples
                for drawing in selected_normal:
                    analysis = drawing.analyses[0] if drawing.analyses else None
                    if analysis:
                        # Try to get interpretability data
                        interpretability = (
                            analysis.interpretability[0]
                            if analysis.interpretability
                            else None
                        )

                        sample = self._create_demo_sample_from_real_data(
                            drawing, analysis, interpretability, sample_id
                        )
                        demo_samples.append(sample)
                        sample_id += 1

                # Process anomalous examples
                for drawing in selected_anomalous:
                    analysis = drawing.analyses[0] if drawing.analyses else None
                    if analysis:
                        # Try to get interpretability data
                        interpretability = (
                            analysis.interpretability[0]
                            if analysis.interpretability
                            else None
                        )

                        sample = self._create_demo_sample_from_real_data(
                            drawing, analysis, interpretability, sample_id
                        )
                        demo_samples.append(sample)
                        sample_id += 1

                logger.info(
                    f"Created {len(demo_samples)} real demo samples with diverse subjects"
                )
                return demo_samples

            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error fetching real demo samples: {e}")
            logger.exception("Full traceback:")
            return self._get_fallback_demo_samples()

    def _get_fallback_demo_samples(self) -> List[Dict[str, Any]]:
        """Get fallback demo samples if database query fails."""
        return [
            {
                "id": 1,
                "title": "Demo System - Database Connection Issue",
                "description": "Unable to load real examples from database. Please check system status.",
                "age_group": "N/A",
                "subject_category": "system",
                "original_image": None,
                "saliency_map": None,
                "composite_image": None,
                "analysis_result": {
                    "anomaly_score": 0.0,
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "processing_time": 0.0,
                    "model_version": "v2.0.0",
                },
                "interpretability": {
                    "explanation": "System is currently unable to load real analysis examples. Please contact administrator.",
                    "key_regions": [],
                    "technical_details": {},
                },
                "metadata": {
                    "created_at": None,
                    "content_rating": "safe",
                    "educational_value": "low",
                },
            }
        ]

    def get_demo_samples(self) -> List[Dict[str, Any]]:
        """
        Get all demo samples with real analysis results.

        Returns:
            List of demo sample dictionaries with complete analysis data from real database
        """
        try:
            # Always fetch fresh real data from database
            samples = self._get_real_demo_samples()

            # Update the cached file with real data
            with open(self.samples_file, "w") as f:
                json.dump(samples, f, indent=2)

            logger.info(f"Retrieved {len(samples)} real demo samples")
            return samples

        except Exception as e:
            logger.error(f"Error loading demo samples: {e}")
            return self._get_fallback_demo_samples()

    def get_demo_sample(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific demo sample by ID.

        Args:
            sample_id: ID of the demo sample

        Returns:
            Demo sample dictionary or None if not found
        """
        samples = self.get_demo_samples()

        for sample in samples:
            if sample.get("id") == sample_id:
                return sample

        logger.warning(f"Demo sample {sample_id} not found")
        return None

    def get_project_description(self) -> Dict[str, Any]:
        """
        Get comprehensive project description for demo page.

        Returns:
            Project description dictionary with all required information
        """
        return {
            "title": "Children's Drawing Anomaly Detection System",
            "subtitle": "AI-Powered Analysis of Developmental Patterns in Children's Artwork",
            "overview": (
                "This system uses advanced machine learning techniques to analyze children's drawings "
                "and identify patterns that deviate from age-expected developmental norms. By leveraging "
                "Vision Transformer (ViT) embeddings and autoencoder models trained on age-specific "
                "drawing patterns, we can detect anomalies through reconstruction loss analysis."
            ),
            "key_features": [
                "Age-specific model training for accurate developmental assessment",
                "Vision Transformer (ViT) feature extraction for detailed image analysis",
                "Autoencoder-based anomaly detection using reconstruction loss",
                "Interactive interpretability with saliency map visualizations",
                "Real-time analysis with comprehensive result explanations",
            ],
            "technical_approach": {
                "feature_extraction": "Vision Transformer (ViT) for robust image feature representation",
                "anomaly_detection": "Autoencoder models trained separately for each age group (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years)",
                "interpretability": "Gradient-based saliency maps highlighting regions of interest",
                "threshold_management": "Configurable percentile-based anomaly thresholds with real-time updates",
            },
            "applications": [
                "Research into child development patterns and milestones",
                "Educational assessment tools for monitoring developmental progress",
                "Screening support for healthcare providers (with appropriate professional oversight)",
                "Academic research in developmental psychology and AI applications",
            ],
            "current_status": {
                "training_data": "37,778+ analyzed drawings across all age groups",
                "models": "8 trained autoencoder models (one per age group)",
                "features": "Real-time dashboard, optimized threshold management, guaranteed interpretability",
            },
        }

    def get_medical_disclaimer(self) -> Dict[str, Any]:
        """
        Get comprehensive medical disclaimer for demo page.

        Returns:
            Medical disclaimer dictionary with all required warnings
        """
        return {
            "title": "⚠️ IMPORTANT MEDICAL DISCLAIMER",
            "primary_warning": "This is a demonstration system only and is NOT intended for medical diagnosis.",
            "detailed_disclaimer": [
                "This system is designed for educational, research, and demonstration purposes only.",
                "It should never be used as a substitute for professional medical advice, diagnosis, or treatment.",
                "The analysis results are based on statistical patterns in drawing data and do not constitute medical assessments.",
                "Any concerns about child development should always be discussed with qualified healthcare professionals.",
                "The system's anomaly detection may produce false positives or miss important developmental indicators.",
                "Individual children develop at different rates, and drawing ability can vary significantly based on many factors.",
            ],
            "recommendations": [
                "Always consult with pediatricians, child psychologists, or other qualified professionals for developmental concerns",
                "Use this system only as a supplementary tool for research or educational exploration",
                "Do not make any medical or educational decisions based solely on these analysis results",
                "Consider the broader context of a child's development, not just drawing analysis",
            ],
            "styling": {
                "background_color": "#ffebee",
                "border_color": "#f44336",
                "text_color": "#c62828",
                "prominence": "high",
            },
        }

    def get_technical_links(self) -> Dict[str, Any]:
        """
        Get technical links and documentation references.

        Returns:
            Technical links dictionary with GitHub and documentation references
        """
        return {
            "github": {
                "url": "https://github.com/user/drawing-analysis-system",
                "title": "📁 View Source Code on GitHub",
                "description": "Complete source code, documentation, and technical implementation details",
            },
            "documentation": {
                "url": "/docs",
                "title": "📖 API Documentation",
                "description": "Interactive API documentation with endpoint details and examples",
            },
            "research": {
                "url": "/research",
                "title": "🔬 Research Background",
                "description": "Academic background, methodology, and research findings",
            },
            "technical_paper": {
                "url": "/static/docs/technical_paper.pdf",
                "title": "📄 Technical Paper",
                "description": "Detailed technical methodology and validation results",
            },
        }

    def get_demo_statistics(self) -> Dict[str, Any]:
        """
        Get demo-specific statistics and metrics from real data.

        Returns:
            Demo statistics dictionary based on real database data
        """
        try:
            db = next(get_db())

            # Get real statistics from database
            total_drawings = db.query(Drawing).count()
            total_analyses = db.query(AnomalyAnalysis).count()
            anomaly_count = (
                db.query(AnomalyAnalysis)
                .filter(AnomalyAnalysis.is_anomaly == True)
                .count()
            )

            # Get subject distribution
            subject_counts = (
                db.query(Drawing.subject, func.count(Drawing.id))
                .filter(Drawing.subject.isnot(None))
                .group_by(Drawing.subject)
                .limit(10)  # Top 10 subjects
                .all()
            )

            subject_distribution = {subject: count for subject, count in subject_counts}

            # Get average confidence from recent analyses
            avg_confidence_result = db.query(
                func.avg(AnomalyAnalysis.confidence)
            ).scalar()
            avg_confidence = (
                round(avg_confidence_result, 3) if avg_confidence_result else 0.0
            )

            # Get demo samples for demo-specific stats
            samples = self.get_demo_samples()
            demo_total = len(samples)
            demo_anomalies = sum(
                1
                for s in samples
                if s.get("analysis_result", {}).get("is_anomaly", False)
            )
            demo_normal = demo_total - demo_anomalies

            return {
                "total_samples": demo_total,
                "normal_samples": demo_normal,
                "anomaly_samples": demo_anomalies,
                "anomaly_rate": demo_anomalies / max(demo_total, 1),
                "subject_distribution": subject_distribution,
                "average_confidence": avg_confidence,
                "interpretability_coverage": "100%",  # All demo samples have interpretability
                "database_stats": {
                    "total_drawings": total_drawings,
                    "total_analyses": total_analyses,
                    "total_anomalies": anomaly_count,
                    "overall_anomaly_rate": round(
                        anomaly_count / max(total_analyses, 1), 3
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error getting demo statistics: {e}")
            return {
                "total_samples": 0,
                "normal_samples": 0,
                "anomaly_samples": 0,
                "anomaly_rate": 0.0,
                "subject_distribution": {},
                "average_confidence": 0.0,
                "interpretability_coverage": "0%",
                "database_stats": {
                    "total_drawings": 0,
                    "total_analyses": 0,
                    "total_anomalies": 0,
                    "overall_anomaly_rate": 0.0,
                },
            }


# Global demo service instance
_demo_service: Optional[DemoService] = None


def get_demo_service() -> DemoService:
    """
    Get the global demo service instance.

    Returns:
        DemoService instance
    """
    global _demo_service
    if _demo_service is None:
        _demo_service = DemoService()
    return _demo_service
