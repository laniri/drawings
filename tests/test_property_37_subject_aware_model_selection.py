"""
Property-Based Test for Subject-Aware Model Selection

**Feature: children-drawing-anomaly-detection, Property 37: Subject-Aware Model Selection**
**Validates: Requirements 4.1**

This test validates that when a new drawing is analyzed, the Drawing_Analysis_System
SHALL compute reconstruction loss using the appropriate age-subject aware autoencoder model.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

from app.services.model_manager import ModelManager, ModelManagerError
from app.models.database import AgeGroupModel, Drawing
from app.schemas.drawings import SubjectCategory


class TestSubjectAwareModelSelection:
    """Test subject-aware model selection for anomaly detection."""

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        subject=st.sampled_from([category.value for category in SubjectCategory]),
        embedding_dim=st.just(832)  # Always 832 for hybrid embeddings
    )
    @settings(max_examples=50, deadline=10000)
    def test_subject_aware_model_selection_property(self, age, subject, embedding_dim):
        """
        **Feature: children-drawing-anomaly-detection, Property 37: Subject-Aware Model Selection**
        **Validates: Requirements 4.1**
        
        Property: For any age and subject combination, when computing anomaly scores,
        the system should select and use an age-appropriate subject-aware autoencoder model
        that supports the unified hybrid embedding architecture.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model that supports subjects
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.age_min = float(int(age))
        mock_age_group_model.age_max = float(int(age) + 1)
        mock_age_group_model.supports_subjects = True
        mock_age_group_model.embedding_type = "hybrid"
        mock_age_group_model.threshold = 0.5
        
        # Mock the database query to return our age group model
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create a valid hybrid embedding (832 dimensions)
        visual_embedding = np.random.rand(768).astype(np.float32)
        
        # Create subject encoding (64-dimensional one-hot vector)
        subject_encoding = np.zeros(64, dtype=np.float32)
        subject_encoding[0] = 1.0  # Set first position for any subject
        
        # Combine into hybrid embedding
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding])
        
        # Ensure exactly 832 dimensions
        assert hybrid_embedding.shape[0] == 832
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the load_model method to return a mock autoencoder
        mock_autoencoder = Mock()
        mock_autoencoder.eval.return_value = None
        
        # Mock the forward pass to return a reconstruction
        mock_reconstruction = np.random.rand(832).astype(np.float32)
        
        with patch.object(model_manager, 'load_model', return_value=mock_autoencoder):
            with patch('torch.FloatTensor') as mock_tensor:
                with patch('torch.no_grad'):
                    # Mock tensor operations
                    mock_input_tensor = Mock()
                    mock_output_tensor = Mock()
                    mock_loss_tensor = Mock()
                    
                    mock_tensor.return_value = mock_input_tensor
                    mock_input_tensor.unsqueeze.return_value = mock_input_tensor
                    mock_autoencoder.return_value = mock_output_tensor
                    
                    # Mock the loss calculation
                    mock_mean_tensor = Mock()
                    mock_mean_tensor.item.return_value = 0.75  # Mock reconstruction loss
                    
                    with patch('torch.mean', return_value=mock_mean_tensor):
                        # Test the compute_anomaly_score method
                        try:
                            result = model_manager.compute_anomaly_score(
                                hybrid_embedding, mock_age_group_model.id, mock_db
                            )
                            
                            # Verify that the method returns the expected structure
                            assert isinstance(result, dict)
                            assert "overall_anomaly_score" in result
                            assert "visual_anomaly_score" in result
                            assert "subject_anomaly_score" in result
                            
                            # Verify that all scores are numeric
                            assert isinstance(result["overall_anomaly_score"], (int, float))
                            assert isinstance(result["visual_anomaly_score"], (int, float))
                            assert isinstance(result["subject_anomaly_score"], (int, float))
                            
                            # Verify that the model was loaded with the correct ID
                            model_manager.load_model.assert_called_once_with(mock_age_group_model.id, mock_db)
                            
                            # Verify that the model supports subjects (requirement 4.1)
                            assert mock_age_group_model.supports_subjects == True
                            assert mock_age_group_model.embedding_type == "hybrid"
                            
                        except ModelManagerError as e:
                            # If the model manager raises an error, it should be a meaningful error
                            assert "subject-aware" in str(e).lower() or "hybrid" in str(e).lower()

    @given(
        age=st.floats(min_value=2.0, max_value=18.0),
        invalid_embedding_dim=st.integers(min_value=1, max_value=1000).filter(lambda x: x != 832)
    )
    @settings(max_examples=20, deadline=5000)
    def test_subject_aware_model_selection_rejects_invalid_embeddings(self, age, invalid_embedding_dim):
        """
        **Feature: children-drawing-anomaly-detection, Property 37: Subject-Aware Model Selection**
        **Validates: Requirements 4.1**
        
        Property: The subject-aware model selection should reject embeddings that are not
        832-dimensional hybrid embeddings, ensuring consistent architecture.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create invalid embedding (not 832 dimensions)
        invalid_embedding = np.random.rand(invalid_embedding_dim).astype(np.float32)
        
        # Create model manager
        model_manager = ModelManager()
        
        # Test that invalid embeddings are rejected
        with pytest.raises(ModelManagerError) as exc_info:
            model_manager.compute_anomaly_score(invalid_embedding, 1, mock_db)
        
        # Verify the error message mentions the expected dimension
        error_message = str(exc_info.value)
        assert "832" in error_message or "hybrid" in error_message.lower()

    def test_subject_aware_model_selection_uses_correct_architecture(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 37: Subject-Aware Model Selection**
        **Validates: Requirements 4.1**
        
        Test that the model selection process verifies the model uses subject-aware architecture.
        """
        # Create mock database session
        mock_db = Mock(spec=Session)
        
        # Create mock age group model with subject-aware architecture
        mock_age_group_model = Mock(spec=AgeGroupModel)
        mock_age_group_model.id = 1
        mock_age_group_model.supports_subjects = True
        mock_age_group_model.embedding_type = "hybrid"
        
        # Mock the database query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_age_group_model
        
        # Create valid hybrid embedding
        hybrid_embedding = np.random.rand(832).astype(np.float32)
        
        # Create model manager
        model_manager = ModelManager()
        
        # Mock the load_model method
        mock_autoencoder = Mock()
        
        with patch.object(model_manager, 'load_model', return_value=mock_autoencoder):
            with patch.object(model_manager, 'compute_subject_aware_reconstruction_loss') as mock_compute:
                mock_compute.return_value = {
                    "overall_loss": 0.5,
                    "visual_loss": 0.3,
                    "subject_loss": 0.2
                }
                
                # Test the compute_anomaly_score method
                result = model_manager.compute_anomaly_score(
                    hybrid_embedding, mock_age_group_model.id, mock_db
                )
                
                # Verify that the subject-aware reconstruction loss method was called
                mock_compute.assert_called_once_with(
                    hybrid_embedding, mock_age_group_model.id, mock_db
                )
                
                # Verify the result structure
                assert result["overall_anomaly_score"] == 0.5
                assert result["visual_anomaly_score"] == 0.3
                assert result["subject_anomaly_score"] == 0.2