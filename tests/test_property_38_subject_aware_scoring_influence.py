"""
Property-Based Test for Subject-Aware Scoring Influence

**Feature: children-drawing-anomaly-detection, Property 38: Subject-Aware Scoring Influence**
**Validates: Requirements 4.2**

This test validates that when subject information is provided, the Drawing_Analysis_System
SHALL use subject-aware scoring that considers subject-specific patterns within the age group.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.model_manager import ModelManager, ModelManagerError
from app.models.database import AgeGroupModel, Drawing
from app.schemas.drawings import SubjectCategory


class TestSubjectAwareScoringInfluence:
    """Test subject-aware scoring influence on anomaly detection."""

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject1=st.sampled_from([category.value for category in SubjectCategory]),
        subject2=st.sampled_from([category.value for category in SubjectCategory]),
        visual_variation=st.floats(min_value=0.1, max_value=2.0),
        subject_variation=st.floats(min_value=0.1, max_value=2.0)
    )
    @settings(max_examples=30, deadline=10000)
    def test_subject_aware_scoring_influence_property(self, age, subject1, subject2, visual_variation, subject_variation):
        """
        **Feature: children-drawing-anomaly-detection, Property 38: Subject-Aware Scoring Influence**
        **Validates: Requirements 4.2**
        
        Property: For any two drawings with the same visual features but different subjects,
        the subject-aware scoring should produce different component-specific scores that
        reflect the subject-specific patterns within the age group.
        """
        # Skip if subjects are the same (no difference expected)
        assume(subject1 != subject2)
        
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.supports_subjects = True
        mock_age_group_model.embedding_type = "hybrid"
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create identical visual embeddings (768 dimensions)
        base_visual_embedding = np.random.rand(768).astype(np.float32)
        
        # Create different subject encodings (64-dimensional one-hot vectors)
        subject1_encoding = np.zeros(64, dtype=np.float32)
        subject1_encoding[hash(subject1) % 64] = 1.0  # Different positions for different subjects
        
        subject2_encoding = np.zeros(64, dtype=np.float32)
        subject2_encoding[hash(subject2) % 64] = 1.0
        
        # Create hybrid embeddings with same visual but different subjects
        hybrid_embedding1 = np.concatenate([base_visual_embedding, subject1_encoding])
        hybrid_embedding2 = np.concatenate([base_visual_embedding, subject2_encoding])
        
        # Ensure exactly 832 dimensions
        assert hybrid_embedding1.shape[0] == 832
        assert hybrid_embedding2.shape[0] == 832
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the subject-aware reconstruction loss method to simulate different subject responses
        def mock_subject_aware_loss(embedding, model_id, db):
            # Extract subject component to determine which subject this is
            subject_component = embedding[768:]
            
            # Simulate different reconstruction losses based on subject
            if np.array_equal(subject_component, subject1_encoding):
                return {
                    "overall_loss": 0.5 * visual_variation,
                    "visual_loss": 0.3 * visual_variation,
                    "subject_loss": 0.2 * subject_variation  # Subject1 specific loss
                }
            else:
                return {
                    "overall_loss": 0.6 * visual_variation,
                    "visual_loss": 0.3 * visual_variation,  # Same visual, so same visual loss
                    "subject_loss": 0.3 * subject_variation  # Subject2 specific loss (different)
                }
        
        with patch.object(model_manager, 'compute_subject_aware_reconstruction_loss', side_effect=mock_subject_aware_loss):
            # Test scoring for both embeddings
            result1 = model_manager.compute_anomaly_score(hybrid_embedding1, 1, mock_db)
            result2 = model_manager.compute_anomaly_score(hybrid_embedding2, 1, mock_db)
            
            # Verify that both results have the expected structure
            for result in [result1, result2]:
                assert isinstance(result, dict)
                assert "overall_anomaly_score" in result
                assert "visual_anomaly_score" in result
                assert "subject_anomaly_score" in result
            
            # Since visual components are identical, visual scores should be the same
            assert abs(result1["visual_anomaly_score"] - result2["visual_anomaly_score"]) < 1e-6
            
            # Since subject components are different, subject scores should be different
            # (unless by coincidence they map to the same reconstruction loss or same hash position)
            if subject_variation > 0.1 and not np.array_equal(subject1_encoding, subject2_encoding):
                # Only check if subjects actually map to different encodings
                subject_score_diff = abs(result1["subject_anomaly_score"] - result2["subject_anomaly_score"])
                # Allow for some tolerance but expect meaningful difference
                assert subject_score_diff > 1e-6 or result1["subject_anomaly_score"] != result2["subject_anomaly_score"]

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory]),
        visual_noise_level=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=20, deadline=8000)
    def test_subject_component_isolation_property(self, age, subject, visual_noise_level):
        """
        **Feature: children-drawing-anomaly-detection, Property 38: Subject-Aware Scoring Influence**
        **Validates: Requirements 4.2**
        
        Property: The subject-aware scoring should be able to isolate subject-specific
        contributions to the anomaly score, independent of visual variations.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.supports_subjects = True
        mock_age_group_model.embedding_type = "hybrid"
        
        # Create visual embedding with some noise
        visual_embedding = np.random.rand(768).astype(np.float32) * visual_noise_level
        
        # Create subject encoding
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        
        # Create hybrid embedding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the subject-aware reconstruction loss to return component-specific losses
        mock_losses = {
            "overall_loss": 0.5,
            "visual_loss": 0.3 + visual_noise_level * 0.2,  # Visual loss affected by noise
            "subject_loss": 0.1  # Subject loss independent of visual noise
        }
        
        with patch.object(model_manager, 'compute_subject_aware_reconstruction_loss', return_value=mock_losses):
            result = model_manager.compute_anomaly_score(hybrid_embedding, 1, mock_db)
            
            # Verify that component-specific scores are returned
            assert "visual_anomaly_score" in result
            assert "subject_anomaly_score" in result
            
            # Verify that the scores match the expected component isolation
            assert result["visual_anomaly_score"] == mock_losses["visual_loss"]
            assert result["subject_anomaly_score"] == mock_losses["subject_loss"]
            
            # Verify that overall score incorporates both components
            assert result["overall_anomaly_score"] == mock_losses["overall_loss"]

    def test_subject_aware_scoring_validates_hybrid_architecture(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 38: Subject-Aware Scoring Influence**
        **Validates: Requirements 4.2**
        
        Test that subject-aware scoring validates the hybrid embedding architecture
        before performing component-specific analysis.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create model manager
        model_manager = ModelManager()
        
        # Test with non-hybrid embedding (wrong dimension)
        invalid_embedding = np.random.rand(768).astype(np.float32)  # Only visual, no subject
        
        with pytest.raises(ModelManagerError) as exc_info:
            model_manager.compute_anomaly_score(invalid_embedding, 1, mock_db)
        
        # Verify error mentions hybrid embedding requirement
        error_message = str(exc_info.value)
        assert "832" in error_message or "hybrid" in error_message.lower()

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory])
    )
    @settings(max_examples=15, deadline=6000)
    def test_subject_aware_scoring_consistency_property(self, age, subject):
        """
        **Feature: children-drawing-anomaly-detection, Property 38: Subject-Aware Scoring Influence**
        **Validates: Requirements 4.2**
        
        Property: Subject-aware scoring should be consistent - the same hybrid embedding
        should always produce the same component-specific scores when analyzed multiple times.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        
        # Create consistent hybrid embedding
        np.random.seed(42)  # Fixed seed for consistency
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock consistent reconstruction losses
        consistent_losses = {
            "overall_loss": 0.42,
            "visual_loss": 0.25,
            "subject_loss": 0.17
        }
        
        with patch.object(model_manager, 'compute_subject_aware_reconstruction_loss', return_value=consistent_losses):
            # Compute scores multiple times
            result1 = model_manager.compute_anomaly_score(hybrid_embedding, 1, mock_db)
            result2 = model_manager.compute_anomaly_score(hybrid_embedding, 1, mock_db)
            result3 = model_manager.compute_anomaly_score(hybrid_embedding, 1, mock_db)
            
            # Verify consistency across multiple computations
            assert result1["overall_anomaly_score"] == result2["overall_anomaly_score"] == result3["overall_anomaly_score"]
            assert result1["visual_anomaly_score"] == result2["visual_anomaly_score"] == result3["visual_anomaly_score"]
            assert result1["subject_anomaly_score"] == result2["subject_anomaly_score"] == result3["subject_anomaly_score"]