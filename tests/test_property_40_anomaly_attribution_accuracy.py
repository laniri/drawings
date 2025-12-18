"""
Property-Based Test for Anomaly Attribution Accuracy

**Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
**Validates: Requirements 4.6**

This test validates that the system SHALL indicate whether the anomaly is age-related,
subject-related, or both when generating anomaly scores.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.model_manager import ModelManager, ModelManagerError
from app.models.database import AgeGroupModel, Drawing
from app.schemas.drawings import SubjectCategory


class TestAnomalyAttributionAccuracy:
    """Test anomaly attribution accuracy in subject-aware anomaly detection."""

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory]),
        visual_anomaly_level=st.floats(min_value=0.0, max_value=2.0),
        subject_anomaly_level=st.floats(min_value=0.0, max_value=2.0),
        threshold=st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=40, deadline=12000)
    def test_anomaly_attribution_accuracy_property(self, age, subject, visual_anomaly_level, subject_anomaly_level, threshold):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Property: For any hybrid embedding with known component-specific anomaly levels,
        the attribution system should correctly identify whether the anomaly is primarily
        visual, subject-related, both, or age-related based on component-specific thresholds.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = threshold
        mock_age_group_model.supports_subjects = True
        mock_age_group_model.embedding_type = "hybrid"
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Calculate component-specific thresholds (same logic as in determine_attribution)
        visual_threshold = threshold * (768 / 832)
        subject_threshold = threshold * (64 / 832)
        
        # Create component-specific scores based on anomaly levels
        visual_score = visual_anomaly_level * visual_threshold
        subject_score = subject_anomaly_level * subject_threshold
        overall_score = max(visual_score, subject_score)  # Simplified overall score
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the compute_anomaly_score method to return controlled scores
        mock_scores = {
            "overall_anomaly_score": overall_score,
            "visual_anomaly_score": visual_score,
            "subject_anomaly_score": subject_score
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=mock_scores):
            # Test attribution determination
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            
            # Verify attribution accuracy based on component anomaly levels
            visual_anomalous = visual_score > visual_threshold
            subject_anomalous = subject_score > subject_threshold
            overall_anomalous = overall_score > threshold
            
            if visual_anomalous and subject_anomalous:
                # Both components are anomalous
                assert attribution == "both"
            elif visual_anomalous and not subject_anomalous:
                # Only visual component is anomalous
                assert attribution == "visual"
            elif subject_anomalous and not visual_anomalous:
                # Only subject component is anomalous
                assert attribution == "subject"
            elif overall_anomalous and not visual_anomalous and not subject_anomalous:
                # Overall anomalous but components are not - likely age-related
                assert attribution == "age"
            else:
                # No clear anomaly - should default to age
                assert attribution == "age"

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory])
    )
    @settings(max_examples=20, deadline=8000)
    def test_visual_only_attribution_property(self, age, subject):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Property: When only the visual component has high reconstruction loss,
        the attribution should be "visual".
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock scores with high visual, low subject anomaly
        visual_threshold = 0.5 * (768 / 832)
        subject_threshold = 0.5 * (64 / 832)
        
        mock_scores = {
            "overall_anomaly_score": 0.6,
            "visual_anomaly_score": visual_threshold * 1.5,  # Above threshold
            "subject_anomaly_score": subject_threshold * 0.5  # Below threshold
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=mock_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution == "visual"

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory])
    )
    @settings(max_examples=20, deadline=8000)
    def test_subject_only_attribution_property(self, age, subject):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Property: When only the subject component has high reconstruction loss,
        the attribution should be "subject".
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock scores with low visual, high subject anomaly
        visual_threshold = 0.5 * (768 / 832)
        subject_threshold = 0.5 * (64 / 832)
        
        mock_scores = {
            "overall_anomaly_score": 0.4,
            "visual_anomaly_score": visual_threshold * 0.5,  # Below threshold
            "subject_anomaly_score": subject_threshold * 1.5  # Above threshold
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=mock_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution == "subject"

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory])
    )
    @settings(max_examples=20, deadline=8000)
    def test_both_components_attribution_property(self, age, subject):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Property: When both visual and subject components have high reconstruction loss,
        the attribution should be "both".
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock scores with both components above threshold
        visual_threshold = 0.5 * (768 / 832)
        subject_threshold = 0.5 * (64 / 832)
        
        mock_scores = {
            "overall_anomaly_score": 0.8,
            "visual_anomaly_score": visual_threshold * 1.5,  # Above threshold
            "subject_anomaly_score": subject_threshold * 1.5  # Above threshold
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=mock_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution == "both"

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory])
    )
    @settings(max_examples=15, deadline=6000)
    def test_age_related_attribution_property(self, age, subject):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Property: When overall score is high but individual components are not anomalous,
        the attribution should be "age" (indicating potential age-related anomaly).
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[hash(subject) % 64] = 1.0
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock scores with high overall but low components (age-related pattern)
        visual_threshold = 0.5 * (768 / 832)
        subject_threshold = 0.5 * (64 / 832)
        
        mock_scores = {
            "overall_anomaly_score": 0.7,  # Above overall threshold
            "visual_anomaly_score": visual_threshold * 0.8,  # Below threshold
            "subject_anomaly_score": subject_threshold * 0.8  # Below threshold
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=mock_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution == "age"

    def test_attribution_consistency_across_calls(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Test that attribution determination is consistent - the same embedding and scores
        should always produce the same attribution.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create consistent hybrid embedding
        np.random.seed(42)  # Fixed seed for consistency
        visual_embedding = np.random.rand(768).astype(np.float32)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[10] = 1.0  # Fixed subject encoding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock consistent scores
        consistent_scores = {
            "overall_anomaly_score": 0.6,
            "visual_anomaly_score": 0.4,
            "subject_anomaly_score": 0.2
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=consistent_scores):
            # Test multiple calls
            attribution1 = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            attribution2 = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            attribution3 = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            
            # Verify consistency
            assert attribution1 == attribution2 == attribution3

    def test_attribution_handles_edge_cases(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Test that attribution determination handles edge cases gracefully.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding
        hybrid_embedding = np.random.rand(832).astype(np.float32)
        
        # Create model manager
        model_manager = ModelManager()
        
        # Test case 1: All scores are zero
        zero_scores = {
            "overall_anomaly_score": 0.0,
            "visual_anomaly_score": 0.0,
            "subject_anomaly_score": 0.0
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=zero_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution in ["visual", "subject", "both", "age"]  # Should handle gracefully
        
        # Test case 2: Very high scores
        high_scores = {
            "overall_anomaly_score": 10.0,
            "visual_anomaly_score": 8.0,
            "subject_anomaly_score": 9.0
        }
        
        with patch.object(model_manager, 'compute_anomaly_score', return_value=high_scores):
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            assert attribution == "both"  # Both components clearly anomalous

    def test_cross_age_group_comparison_integration(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 40: Anomaly Attribution Accuracy**
        **Validates: Requirements 4.6**
        
        Test that the cross-age-group comparison method works correctly for age-related detection.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create multiple mock age group models
        mock_models = []
        for i, (age_min, age_max) in enumerate([(2, 3), (3, 4), (4, 5), (5, 6)]):
            model = Mock(spec=AgeGroupModel)
            model.id = i + 1
            model.age_min = age_min
            model.age_max = age_max
            model.is_active = True
            model.supports_subjects = True
            model.embedding_type = "hybrid"
            mock_models.append(model)
        
        # Mock database query to return all models
        mock_db.query.return_value.filter.return_value.all.return_value = mock_models
        
        # Create hybrid embedding
        hybrid_embedding = np.random.rand(832).astype(np.float32)
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock compute_anomaly_score to return different scores for different age groups
        def mock_compute_score(embedding, model_id, db):
            # Simulate lower scores for different age groups (age-related pattern)
            base_score = 0.8 - (model_id * 0.1)  # Decreasing scores for different ages
            return {
                "overall_anomaly_score": base_score,
                "visual_anomaly_score": base_score * 0.7,
                "subject_anomaly_score": base_score * 0.3
            }
        
        with patch.object(model_manager, 'compute_anomaly_score', side_effect=mock_compute_score):
            # Test cross-age-group comparison
            cross_age_scores = model_manager.compare_across_age_groups(
                hybrid_embedding, 4.5, mock_db
            )
            
            # Verify that we get scores for different age groups
            assert isinstance(cross_age_scores, dict)
            assert len(cross_age_scores) > 0
            
            # Verify that scores are different across age groups (indicating age-related pattern)
            scores_list = list(cross_age_scores.values())
            if len(scores_list) > 1:
                # Check that not all scores are identical (some variation expected)
                assert not all(score == scores_list[0] for score in scores_list)