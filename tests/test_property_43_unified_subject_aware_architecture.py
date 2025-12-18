"""
Property-Based Test for Unified Subject-Aware Architecture

**Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
**Validates: Requirements 12.2, 12.4**

This test validates that all models use the unified subject-aware architecture
with consistent embedding dimensions and metadata across the system.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
import json

from app.services.model_manager import ModelManager, ModelManagerError
from app.models.database import AgeGroupModel


class TestUnifiedSubjectAwareArchitecture:
    """Test unified subject-aware architecture properties."""
    
    def create_mock_age_group_model(self, 
                                  model_id: int,
                                  age_min: float, 
                                  age_max: float,
                                  supports_subjects: bool = True,
                                  embedding_type: str = "hybrid",
                                  subject_categories: List[str] = None) -> Mock:
        """Create a mock age group model."""
        model = Mock()
        model.id = model_id
        model.age_min = age_min
        model.age_max = age_max
        model.model_type = "autoencoder"
        model.vision_model = "vit"
        model.supports_subjects = supports_subjects
        model.embedding_type = embedding_type
        model.subject_categories = json.dumps(subject_categories or ["house", "person", "tree"])
        model.sample_count = 50
        model.threshold = 0.95
        model.is_active = True
        model.created_timestamp = Mock()
        
        # Mock parameters
        parameters = {
            "training_config": {"epochs": 100, "learning_rate": 0.001},
            "architecture": {"input_dim": 832, "hidden_dims": [256, 128, 64]},
            "subject_distribution": {"house": 20, "person": 15, "tree": 15},
            "supported_subjects": subject_categories or ["house", "person", "tree"],
            "embedding_type": embedding_type,
            "embedding_dimensions": {
                "total": 832,
                "visual": 768,
                "subject": 64
            }
        }
        model.parameters = json.dumps(parameters)
        
        return model
    
    def create_mock_database_session(self, models: List[Mock]) -> Mock:
        """Create a mock database session with age group models."""
        mock_db = Mock()
        
        def mock_query_side_effect(model_class):
            query_mock = Mock()
            if model_class == AgeGroupModel:
                query_mock.all.return_value = models
                # For individual model queries
                def filter_side_effect(*args, **kwargs):
                    filtered_mock = Mock()
                    filtered_mock.first.return_value = models[0] if models else None
                    return filtered_mock
                query_mock.filter.side_effect = filter_side_effect
            return query_mock
        
        mock_db.query.side_effect = mock_query_side_effect
        return mock_db
    
    @given(
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=100),  # model_id
                st.floats(min_value=3.0, max_value=8.0),  # age_min
                st.floats(min_value=8.1, max_value=12.0), # age_max
                st.booleans(),  # supports_subjects
                st.sampled_from(["hybrid", "visual", "legacy"])  # embedding_type
            ),
            min_size=2,
            max_size=8
        )
    )
    @settings(max_examples=20)
    def test_unified_architecture_validation(self, model_specs):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        For any collection of age group models, the system should validate that all models
        use the unified subject-aware architecture and identify any legacy models.
        """
        # Create mock models based on specifications
        models = []
        for model_id, age_min, age_max, supports_subjects, embedding_type in model_specs:
            assume(age_max > age_min)  # Ensure valid age ranges
            
            model = self.create_mock_age_group_model(
                model_id, age_min, age_max, supports_subjects, embedding_type
            )
            models.append(model)
        
        # Create mock database session
        mock_db = self.create_mock_database_session(models)
        
        # Create model manager
        manager = ModelManager()
        
        # Validate unified architecture
        validation_result = manager.validate_unified_subject_aware_architecture(mock_db)
        
        # Property: Validation should complete successfully
        assert validation_result is not None
        assert "total_models" in validation_result
        assert "subject_aware_models" in validation_result
        assert "legacy_models" in validation_result
        assert "is_unified" in validation_result
        
        # Property: Total models should match input
        assert validation_result["total_models"] == len(models)
        
        # Property: Subject-aware and legacy counts should sum to total
        assert (validation_result["subject_aware_models"] + 
                validation_result["legacy_models"]) == validation_result["total_models"]
        
        # Property: Subject-aware models should have correct characteristics
        expected_subject_aware = sum(
            1 for _, _, _, supports_subjects, embedding_type in model_specs
            if supports_subjects and embedding_type == "hybrid"
        )
        assert validation_result["subject_aware_models"] == expected_subject_aware
        
        # Property: Legacy models should be identified correctly
        expected_legacy = len(models) - expected_subject_aware
        assert validation_result["legacy_models"] == expected_legacy
        
        # Property: System is unified only if all models are subject-aware
        expected_unified = (expected_legacy == 0)
        assert validation_result["is_unified"] == expected_unified
        
        # Property: Recommendations should be provided for legacy models
        if expected_legacy > 0:
            assert len(validation_result["recommendations"]) > 0
            assert any("retrain" in rec.lower() for rec in validation_result["recommendations"])
        else:
            assert any("unified" in rec.lower() for rec in validation_result["recommendations"])
    
    def test_all_subject_aware_models_unified(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        When all models use the subject-aware architecture, the system should
        report unified architecture status.
        """
        # Create all subject-aware models
        models = [
            self.create_mock_age_group_model(1, 3.0, 4.0, True, "hybrid"),
            self.create_mock_age_group_model(2, 4.0, 5.0, True, "hybrid"),
            self.create_mock_age_group_model(3, 5.0, 6.0, True, "hybrid"),
            self.create_mock_age_group_model(4, 6.0, 7.0, True, "hybrid")
        ]
        
        # Create mock database session
        mock_db = self.create_mock_database_session(models)
        
        # Create model manager
        manager = ModelManager()
        
        # Validate unified architecture
        validation_result = manager.validate_unified_subject_aware_architecture(mock_db)
        
        # Property: All models should be subject-aware
        assert validation_result["subject_aware_models"] == 4
        assert validation_result["legacy_models"] == 0
        assert validation_result["is_unified"] == True
        
        # Property: No invalid models should be reported
        assert len(validation_result["invalid_models"]) == 0
        
        # Property: Recommendations should confirm unified architecture
        assert any("unified" in rec.lower() for rec in validation_result["recommendations"])
    
    def test_mixed_architecture_models(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        When models have mixed architectures, the system should identify
        legacy models and provide retraining recommendations.
        """
        # Create mixed models
        models = [
            self.create_mock_age_group_model(1, 3.0, 4.0, True, "hybrid"),    # Subject-aware
            self.create_mock_age_group_model(2, 4.0, 5.0, False, "visual"),  # Legacy
            self.create_mock_age_group_model(3, 5.0, 6.0, True, "hybrid"),   # Subject-aware
            self.create_mock_age_group_model(4, 6.0, 7.0, False, "legacy")   # Legacy
        ]
        
        # Create mock database session
        mock_db = self.create_mock_database_session(models)
        
        # Create model manager
        manager = ModelManager()
        
        # Validate unified architecture
        validation_result = manager.validate_unified_subject_aware_architecture(mock_db)
        
        # Property: Should identify correct counts
        assert validation_result["subject_aware_models"] == 2
        assert validation_result["legacy_models"] == 2
        assert validation_result["is_unified"] == False
        
        # Property: Invalid models should be identified
        assert len(validation_result["invalid_models"]) == 2
        
        # Property: Each invalid model should have correct information
        for invalid_model in validation_result["invalid_models"]:
            assert "id" in invalid_model
            assert "age_range" in invalid_model
            assert "supports_subjects" in invalid_model
            assert "embedding_type" in invalid_model
            
            # Should be one of the legacy models
            assert (invalid_model["supports_subjects"] == False or 
                   invalid_model["embedding_type"] != "hybrid")
        
        # Property: Recommendations should suggest retraining
        assert any("retrain" in rec.lower() for rec in validation_result["recommendations"])
        assert any("hybrid" in rec.lower() for rec in validation_result["recommendations"])
    
    def test_model_info_consistency(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        Model information should consistently report subject-aware architecture details
        including embedding dimensions and supported subjects.
        """
        # Create a subject-aware model
        model = self.create_mock_age_group_model(
            1, 4.0, 5.0, True, "hybrid", ["house", "person", "tree", "car"]
        )
        
        # Create mock database session
        mock_db = self.create_mock_database_session([model])
        
        # Create model manager
        manager = ModelManager()
        
        # Get model info
        model_info = manager.get_model_info(1, mock_db)
        
        # Property: Model info should include subject-aware details
        assert model_info["supports_subjects"] == True
        assert model_info["embedding_type"] == "hybrid"
        assert len(model_info["subject_categories"]) == 4
        assert "house" in model_info["subject_categories"]
        assert "person" in model_info["subject_categories"]
        
        # Property: Embedding dimensions should be consistent
        embedding_dims = model_info["embedding_dimensions"]
        assert embedding_dims["total"] == 832
        assert embedding_dims["visual"] == 768
        assert embedding_dims["subject"] == 64
        
        # Property: Subject distribution should be available
        assert "subject_distribution" in model_info
        assert isinstance(model_info["subject_distribution"], dict)
    
    def test_empty_model_collection(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        The validation should handle empty model collections gracefully.
        """
        # Create empty model collection
        models = []
        
        # Create mock database session
        mock_db = self.create_mock_database_session(models)
        
        # Create model manager
        manager = ModelManager()
        
        # Validate unified architecture
        validation_result = manager.validate_unified_subject_aware_architecture(mock_db)
        
        # Property: Should handle empty collection
        assert validation_result["total_models"] == 0
        assert validation_result["subject_aware_models"] == 0
        assert validation_result["legacy_models"] == 0
        assert validation_result["is_unified"] == True  # Vacuously true
        assert len(validation_result["invalid_models"]) == 0
    
    def test_architecture_consistency_across_age_groups(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        All age groups should use consistent subject-aware architecture
        with the same embedding dimensions and structure.
        """
        # Create models for different age groups with consistent architecture
        age_ranges = [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0), (6.0, 7.0), (7.0, 8.0)]
        models = []
        
        for i, (age_min, age_max) in enumerate(age_ranges):
            model = self.create_mock_age_group_model(
                i + 1, age_min, age_max, True, "hybrid", 
                ["house", "person", "tree"]  # Consistent subjects
            )
            models.append(model)
        
        # Create mock database session
        mock_db = self.create_mock_database_session(models)
        
        # Create model manager
        manager = ModelManager()
        
        # Validate unified architecture
        validation_result = manager.validate_unified_subject_aware_architecture(mock_db)
        
        # Property: All models should be subject-aware
        assert validation_result["subject_aware_models"] == len(age_ranges)
        assert validation_result["legacy_models"] == 0
        assert validation_result["is_unified"] == True
        
        # Property: Check consistency across all models
        for model in models:
            model_info = manager.get_model_info(model.id, mock_db)
            
            # Each model should have consistent architecture
            assert model_info["supports_subjects"] == True
            assert model_info["embedding_type"] == "hybrid"
            assert model_info["embedding_dimensions"]["total"] == 832
            assert model_info["embedding_dimensions"]["visual"] == 768
            assert model_info["embedding_dimensions"]["subject"] == 64
            
            # Subject categories should be consistent
            assert len(model_info["subject_categories"]) == 3
            assert set(model_info["subject_categories"]) == {"house", "person", "tree"}
    
    def test_validation_error_handling(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 43: Unified Subject-Aware Architecture**
        **Validates: Requirements 12.2, 12.4**
        
        The validation should handle database errors gracefully.
        """
        # Create mock database that raises an exception
        mock_db = Mock()
        mock_db.query.side_effect = Exception("Database connection error")
        
        # Create model manager
        manager = ModelManager()
        
        # Property: Should raise ModelManagerError for database issues
        with pytest.raises(ModelManagerError) as exc_info:
            manager.validate_unified_subject_aware_architecture(mock_db)
        
        assert "Architecture validation failed" in str(exc_info.value)