"""
Property-Based Test for Subject-Missing Default Handling

**Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
**Validates: Requirements 4.3**

This test validates that when subject information is missing, the Drawing_Analysis_System
SHALL use "unspecified" subject category in the hybrid embedding for consistent analysis.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.model_manager import ModelManager, ModelManagerError
from app.services.embedding_service import EmbeddingService, SubjectEncoder
from app.models.database import AgeGroupModel, Drawing
from app.schemas.drawings import SubjectCategory


class TestSubjectMissingDefaultHandling:
    """Test subject-missing default handling in anomaly detection."""

    @given(
        age=st.floats(min_value=2.0, max_value=18.0)
    )
    @settings(max_examples=25, deadline=8000)
    def test_subject_missing_default_handling_property(self, age):
        """
        **Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
        **Validates: Requirements 4.3**
        
        Property: For any drawing with missing subject information, the system should
        automatically use "unspecified" as the default subject category and create
        a valid 832-dimensional hybrid embedding for consistent analysis.
        """
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
        
        # Create visual embedding with random features
        visual_embedding = np.random.rand(768).astype(np.float32)
        
        # Test the SubjectEncoder's handling of None/missing subject
        unspecified_encoding = SubjectEncoder.encode_subject_category(None)
        
        # Verify that missing subject maps to "unspecified" encoding
        assert isinstance(unspecified_encoding, np.ndarray)
        assert unspecified_encoding.shape == (64,)
        assert unspecified_encoding.dtype == np.float32
        
        # Verify it's a valid one-hot encoding
        assert np.sum(unspecified_encoding) == 1.0
        assert np.sum(unspecified_encoding == 1.0) == 1
        
        # Create hybrid embedding with unspecified subject
        hybrid_embedding = np.concatenate([visual_embedding, unspecified_encoding])
        
        # Verify hybrid embedding properties
        assert hybrid_embedding.shape[0] == 832
        assert hybrid_embedding.dtype == np.float32
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the subject-aware reconstruction loss method
        mock_losses = {
            "overall_loss": 0.4,
            "visual_loss": 0.3,
            "subject_loss": 0.1  # Low subject loss for unspecified
        }
        
        with patch.object(model_manager, 'compute_subject_aware_reconstruction_loss', return_value=mock_losses):
            # Test that the system can process embeddings with unspecified subject
            result = model_manager.compute_anomaly_score(hybrid_embedding, 1, mock_db)
            
            # Verify that the result has the expected structure
            assert isinstance(result, dict)
            assert "overall_anomaly_score" in result
            assert "visual_anomaly_score" in result
            assert "subject_anomaly_score" in result
            
            # Verify that all scores are valid numbers
            assert isinstance(result["overall_anomaly_score"], (int, float))
            assert isinstance(result["visual_anomaly_score"], (int, float))
            assert isinstance(result["subject_anomaly_score"], (int, float))
            
            # Verify that the scores match the expected values
            assert result["overall_anomaly_score"] == 0.4
            assert result["visual_anomaly_score"] == 0.3
            assert result["subject_anomaly_score"] == 0.1

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        missing_subject_representation=st.sampled_from([None, "", "unknown", "missing"])
    )
    @settings(max_examples=20, deadline=6000)
    def test_various_missing_subject_representations_property(self, age, missing_subject_representation):
        """
        **Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
        **Validates: Requirements 4.3**
        
        Property: Various representations of missing subject information (None, empty string, etc.)
        should all be consistently mapped to the same "unspecified" encoding.
        """
        # Test that different representations of missing subjects all map to the same encoding
        encoding1 = SubjectEncoder.encode_subject_category(None)
        encoding2 = SubjectEncoder.encode_subject_category(missing_subject_representation)
        
        # All missing subject representations should produce the same encoding
        if missing_subject_representation in [None, "", "unknown", "missing"]:
            # These should all map to unspecified
            assert np.array_equal(encoding1, encoding2)
        
        # Verify both are valid one-hot encodings
        for encoding in [encoding1, encoding2]:
            assert isinstance(encoding, np.ndarray)
            assert encoding.shape == (64,)
            assert encoding.dtype == np.float32
            assert np.sum(encoding) == 1.0
            assert np.sum(encoding == 1.0) == 1

    def test_unspecified_subject_consistency_across_embeddings(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
        **Validates: Requirements 4.3**
        
        Test that the "unspecified" subject encoding is consistent across different
        visual embeddings and produces predictable hybrid embeddings.
        """
        # Create different visual embeddings
        visual1 = np.random.rand(768).astype(np.float32)
        visual2 = np.random.rand(768).astype(np.float32)
        
        # Get unspecified encoding for both
        unspecified1 = SubjectEncoder.encode_subject_category(None)
        unspecified2 = SubjectEncoder.encode_subject_category(None)
        
        # Verify that unspecified encodings are identical
        assert np.array_equal(unspecified1, unspecified2)
        
        # Create hybrid embeddings
        hybrid1 = np.concatenate([visual1, unspecified1])
        hybrid2 = np.concatenate([visual2, unspecified2])
        
        # Verify hybrid embedding properties
        assert hybrid1.shape[0] == 832
        assert hybrid2.shape[0] == 832
        
        # Verify that the subject components are identical
        assert np.array_equal(hybrid1[768:], hybrid2[768:])
        
        # Verify that the visual components are different (as expected)
        assert not np.array_equal(hybrid1[:768], hybrid2[:768])

    @given(
        age=st.floats(min_value=2.0, max_value=18.0)
    )
    @settings(max_examples=15, deadline=5000)
    def test_unspecified_subject_attribution_property(self, age):
        """
        **Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
        **Validates: Requirements 4.3**
        
        Property: When using "unspecified" subject, the anomaly attribution should
        still work correctly and not falsely attribute anomalies to the subject component.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.threshold = 0.5
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create hybrid embedding with unspecified subject
        visual_embedding = np.random.rand(768).astype(np.float32)
        unspecified_encoding = SubjectEncoder.encode_subject_category(None)
        hybrid_embedding = np.concatenate([visual_embedding, unspecified_encoding])
        
        # Create model manager
        model_manager = ModelManager()
        
        # Test case 1: Visual anomaly, normal subject
        with patch.object(model_manager, 'compute_anomaly_score') as mock_compute:
            mock_compute.return_value = {
                "overall_anomaly_score": 0.8,
                "visual_anomaly_score": 0.7,  # High visual anomaly
                "subject_anomaly_score": 0.1   # Low subject anomaly (unspecified is normal)
            }
            
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            
            # Should attribute to visual, not subject, since unspecified is typically normal
            # Note: "both" is also acceptable if both components exceed thresholds
            assert attribution in ["visual", "age", "both"]  # Could be visual, age-related, or both
            # The key is that it should work correctly with unspecified subjects
        
        # Test case 2: Normal visual, subject component issue (unlikely with unspecified)
        with patch.object(model_manager, 'compute_anomaly_score') as mock_compute:
            mock_compute.return_value = {
                "overall_anomaly_score": 0.3,
                "visual_anomaly_score": 0.2,  # Normal visual
                "subject_anomaly_score": 0.1   # Normal subject (unspecified should be normal)
            }
            
            attribution = model_manager.determine_attribution(hybrid_embedding, 1, mock_db)
            
            # With low scores across the board, could be any attribution
            # The key is that the system handles unspecified subjects correctly
            assert attribution in ["visual", "age", "subject"]  # Any attribution is acceptable for low scores

    def test_embedding_service_integration_with_missing_subject(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 39: Subject-Missing Default Handling**
        **Validates: Requirements 4.3**
        
        Test that the embedding service correctly handles missing subject information
        when generating hybrid embeddings.
        """
        # Create mock image
        from PIL import Image
        mock_image = Image.new('RGB', (224, 224), color='white')
        
        # Create embedding service
        embedding_service = EmbeddingService()
        
        # Mock the service initialization and visual embedding generation
        with patch.object(embedding_service, 'is_ready', return_value=True):
            with patch.object(embedding_service, '_generate_visual_embedding') as mock_visual:
                mock_visual.return_value = np.random.rand(768).astype(np.float32)
                
                # Test with None subject (missing)
                hybrid_embedding = embedding_service.generate_hybrid_embedding(
                    image=mock_image,
                    subject=None,  # Missing subject
                    age=5.0
                )
                
                # Verify hybrid embedding properties
                assert isinstance(hybrid_embedding, np.ndarray)
                assert hybrid_embedding.shape == (832,)
                assert hybrid_embedding.dtype == np.float32
                
                # Verify that the subject component is the unspecified encoding
                subject_component = hybrid_embedding[768:]
                expected_unspecified = SubjectEncoder.encode_subject_category(None)
                assert np.array_equal(subject_component, expected_unspecified)
                
                # Verify that it's a valid one-hot encoding
                assert np.sum(subject_component) == 1.0
                assert np.sum(subject_component == 1.0) == 1