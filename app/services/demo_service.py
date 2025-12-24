"""
Demo service for managing sample content and demo page functionality.

This service provides pre-analyzed sample drawings with complete results
for demonstration purposes, including interpretability visualizations.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.core.config import settings
from app.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class DemoService:
    """
    Service for managing demo content and sample analysis results.
    
    Provides pre-analyzed sample drawings with interpretability visualizations
    for public demonstration of the system capabilities.
    """
    
    def __init__(self):
        self.demo_data_path = Path(settings.STATIC_DIR) / "demo"
        self.samples_file = self.demo_data_path / "demo_samples.json"
        
        # Ensure demo directory exists
        self.demo_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize demo samples if not exists
        self._initialize_demo_samples()
    
    def _initialize_demo_samples(self):
        """Initialize demo samples with pre-analyzed results."""
        if not self.samples_file.exists():
            logger.info("Initializing demo samples")
            self._create_default_demo_samples()
    
    def _create_default_demo_samples(self):
        """Create default demo samples with analysis results."""
        default_samples = [
            {
                "id": 1,
                "title": "House Drawing - Age 5",
                "description": "A typical house drawing by a 5-year-old showing expected developmental features",
                "age_group": "5-6",
                "original_image": "/static/demo/sample_1_original.png",
                "saliency_map": "/static/demo/sample_1_saliency.png",
                "composite_image": "/static/demo/sample_1_composite.png",
                "analysis_result": {
                    "anomaly_score": 0.15,
                    "is_anomaly": False,
                    "confidence": 0.92,
                    "processing_time": 2.3,
                    "model_version": "v1.0",
                    "age_group_model": "age_5_6"
                },
                "interpretability": {
                    "explanation": "The model focused on typical house elements: roof, walls, door, and windows. These features align well with expected developmental patterns for this age group.",
                    "key_regions": [
                        {"region": "roof", "importance": 0.85, "description": "Triangular roof shape is age-appropriate"},
                        {"region": "walls", "importance": 0.78, "description": "Rectangular wall structure shows spatial understanding"},
                        {"region": "door", "importance": 0.65, "description": "Central door placement is typical"},
                        {"region": "windows", "importance": 0.72, "description": "Window symmetry indicates developing fine motor skills"}
                    ],
                    "technical_details": {
                        "saliency_method": "gradient-based",
                        "attention_regions": 4,
                        "confidence_threshold": 0.7
                    }
                },
                "metadata": {
                    "created_at": "2024-01-15T10:30:00Z",
                    "content_rating": "safe",
                    "educational_value": "high"
                }
            },
            {
                "id": 2,
                "title": "Family Portrait - Age 7",
                "description": "A family drawing showing advanced detail and emotional expression typical for age 7",
                "age_group": "7-8", 
                "original_image": "/static/demo/sample_2_original.png",
                "saliency_map": "/static/demo/sample_2_saliency.png",
                "composite_image": "/static/demo/sample_2_composite.png",
                "analysis_result": {
                    "anomaly_score": 0.08,
                    "is_anomaly": False,
                    "confidence": 0.96,
                    "processing_time": 2.1,
                    "model_version": "v1.0",
                    "age_group_model": "age_7_8"
                },
                "interpretability": {
                    "explanation": "The model identified well-developed human figure representation with appropriate proportions and facial features for this age group. The emotional expression and detail level are consistent with typical 7-year-old development.",
                    "key_regions": [
                        {"region": "faces", "importance": 0.91, "description": "Detailed facial features show emotional awareness"},
                        {"region": "body_proportions", "importance": 0.83, "description": "Improved body proportions indicate spatial development"},
                        {"region": "clothing_details", "importance": 0.67, "description": "Clothing details show attention to visual elements"},
                        {"region": "background", "importance": 0.45, "description": "Simple background maintains focus on figures"}
                    ],
                    "technical_details": {
                        "saliency_method": "gradient-based",
                        "attention_regions": 5,
                        "confidence_threshold": 0.7
                    }
                },
                "metadata": {
                    "created_at": "2024-01-15T11:15:00Z",
                    "content_rating": "safe",
                    "educational_value": "high"
                }
            },
            {
                "id": 3,
                "title": "Abstract Pattern - Age 4",
                "description": "An abstract drawing with unusual patterns that triggered anomaly detection",
                "age_group": "4-5",
                "original_image": "/static/demo/sample_3_original.png", 
                "saliency_map": "/static/demo/sample_3_saliency.png",
                "composite_image": "/static/demo/sample_3_composite.png",
                "analysis_result": {
                    "anomaly_score": 0.78,
                    "is_anomaly": True,
                    "confidence": 0.89,
                    "processing_time": 2.7,
                    "model_version": "v1.0",
                    "age_group_model": "age_4_5"
                },
                "interpretability": {
                    "explanation": "The model detected unusual pattern complexity and spatial organization that differs from typical 4-year-old drawings. The high level of abstract thinking and geometric precision is uncommon for this age group.",
                    "key_regions": [
                        {"region": "geometric_patterns", "importance": 0.94, "description": "Complex geometric patterns unusual for age 4"},
                        {"region": "symmetry", "importance": 0.87, "description": "Advanced symmetrical organization"},
                        {"region": "line_precision", "importance": 0.81, "description": "Precise line control beyond typical motor skills"},
                        {"region": "spatial_planning", "importance": 0.76, "description": "Sophisticated spatial planning evident"}
                    ],
                    "technical_details": {
                        "saliency_method": "gradient-based",
                        "attention_regions": 6,
                        "confidence_threshold": 0.7
                    }
                },
                "metadata": {
                    "created_at": "2024-01-15T14:20:00Z",
                    "content_rating": "safe",
                    "educational_value": "very_high"
                }
            }
        ]
        
        # Save demo samples to file
        with open(self.samples_file, 'w') as f:
            json.dump(default_samples, f, indent=2)
        
        logger.info(f"Created {len(default_samples)} default demo samples")
    
    def get_demo_samples(self) -> List[Dict[str, Any]]:
        """
        Get all demo samples with analysis results.
        
        Returns:
            List of demo sample dictionaries with complete analysis data
        """
        try:
            if self.samples_file.exists():
                with open(self.samples_file, 'r') as f:
                    samples = json.load(f)
                
                logger.info(f"Retrieved {len(samples)} demo samples")
                return samples
            else:
                logger.warning("Demo samples file not found, creating default samples")
                self._create_default_demo_samples()
                return self.get_demo_samples()
                
        except Exception as e:
            logger.error(f"Error loading demo samples: {e}")
            return []
    
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
                "Real-time analysis with comprehensive result explanations"
            ],
            "technical_approach": {
                "feature_extraction": "Vision Transformer (ViT) for robust image feature representation",
                "anomaly_detection": "Autoencoder models trained separately for each age group (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years)",
                "interpretability": "Gradient-based saliency maps highlighting regions of interest",
                "threshold_management": "Configurable percentile-based anomaly thresholds with real-time updates"
            },
            "applications": [
                "Research into child development patterns and milestones",
                "Educational assessment tools for monitoring developmental progress", 
                "Screening support for healthcare providers (with appropriate professional oversight)",
                "Academic research in developmental psychology and AI applications"
            ],
            "current_status": {
                "training_data": "37,778+ analyzed drawings across all age groups",
                "models": "8 trained autoencoder models (one per age group)",
                "features": "Real-time dashboard, optimized threshold management, guaranteed interpretability"
            }
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
                "Individual children develop at different rates, and drawing ability can vary significantly based on many factors."
            ],
            "recommendations": [
                "Always consult with pediatricians, child psychologists, or other qualified professionals for developmental concerns",
                "Use this system only as a supplementary tool for research or educational exploration",
                "Do not make any medical or educational decisions based solely on these analysis results",
                "Consider the broader context of a child's development, not just drawing analysis"
            ],
            "styling": {
                "background_color": "#ffebee",
                "border_color": "#f44336", 
                "text_color": "#c62828",
                "prominence": "high"
            }
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
                "description": "Complete source code, documentation, and technical implementation details"
            },
            "documentation": {
                "url": "/docs",
                "title": "📖 API Documentation", 
                "description": "Interactive API documentation with endpoint details and examples"
            },
            "research": {
                "url": "/research",
                "title": "🔬 Research Background",
                "description": "Academic background, methodology, and research findings"
            },
            "technical_paper": {
                "url": "/static/docs/technical_paper.pdf",
                "title": "📄 Technical Paper",
                "description": "Detailed technical methodology and validation results"
            }
        }
    
    def get_demo_statistics(self) -> Dict[str, Any]:
        """
        Get demo-specific statistics and metrics.
        
        Returns:
            Demo statistics dictionary
        """
        samples = self.get_demo_samples()
        
        total_samples = len(samples)
        anomaly_samples = sum(1 for s in samples if s.get("analysis_result", {}).get("is_anomaly", False))
        normal_samples = total_samples - anomaly_samples
        
        age_groups = {}
        for sample in samples:
            age_group = sample.get("age_group", "unknown")
            age_groups[age_group] = age_groups.get(age_group, 0) + 1
        
        avg_confidence = sum(
            s.get("analysis_result", {}).get("confidence", 0) for s in samples
        ) / max(total_samples, 1)
        
        return {
            "total_samples": total_samples,
            "normal_samples": normal_samples,
            "anomaly_samples": anomaly_samples,
            "anomaly_rate": anomaly_samples / max(total_samples, 1),
            "age_group_distribution": age_groups,
            "average_confidence": round(avg_confidence, 3),
            "interpretability_coverage": "100%"  # All demo samples have interpretability
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