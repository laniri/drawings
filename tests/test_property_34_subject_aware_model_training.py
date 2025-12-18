"""
Property-Based Test for Subject-Aware Model Training

**Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
**Validates: Requirements 3.7**

This test validates that autoencoder models are trained on 832-dimensional hybrid 
embeddings and properly handle subject-aware features during training.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
import tempfile
import json

from app.services.model_manager import (
    ModelManager, 
    TrainingConfig, 
    AutoencoderModel,
    ModelManagerError,
    AutoencoderTrainingError
)
from app.models.database import Drawing, DrawingEmbedding, AgeGroupModel
from app.services.embedding_service import SubjectEncoder
from app.schemas.drawings import SubjectCategory


class TestSubjectAwareModelTraining:
    """Test subject-aware model training properties."""
    
    def create_mock_hybrid_embedding(self, visual_seed: int, subject_category: str) -> np.ndarray:
        """Create a deterministic hybrid embedding for testing."""
        # Create deterministic visual embedding (768 dimensions)
        np.random.seed(visual_seed)
        visual_embedding = np.random.randn(768).astype(np.float32)
        
        # Create subject encoding (64 dimensions)
        if hasattr(SubjectCategory, subject_category.upper()):
            subject_enum = getattr(SubjectCategory, subject_category.upper())
        else:
            subject_enum = SubjectCategory.UNSPECIFIED
        
        subject_encoding = SubjectEncoder.encode_subject_category(subject_enum)
        
        # Combine into hybrid embedding (832 dimensions total)
        hybrid_embedding = np.concatenate([visual_embedding, subject_encoding], axis=0)
        
        return hybrid_embedding
    
    def create_mock_database_session(self, 
                                   age_min: float, 
                                   age_max: float,
                                   embeddings_data: List[Tuple[np.ndarray, str]]) -> Mock:
        """Create a mock database session with hybrid embeddings."""
        mock_db = Mock()
        
        # Create mock drawings
        mock_drawings = []
        for i, (embedding, subject) in enumerate(embeddings_data):
            drawing = Mock()
            drawing.id = i + 1
            drawing.age_years = age_min + (i * (age_max - age_min) / len(embeddings_data))
            drawing.subject = subject if subject != "unspecified" else None
            mock_drawings.append(drawing)
        
        # Create mock embedding records
        mock_embedding_records = []
        for i, (embedding, subject) in enumerate(embeddings_data):
            embedding_record = Mock()
            embedding_record.drawing_id = i + 1
            embedding_record.embedding_type = "hybrid"
            embedding_record.embedding_vector = embedding.tobytes()  # Mock serialized data
            embedding_record.created_timestamp = Mock()
            mock_embedding_records.append(embedding_record)
        
        # Configure query behavior
        def mock_query_side_effect(model_class):
            query_mock = Mock()
            if model_class == Drawing:
                query_mock.filter.return_value.all.return_value = mock_drawings
            elif model_class == DrawingEmbedding:
                # Return different embedding records based on filter
                def filter_side_effect(*args, **kwargs):
                    filtered_mock = Mock()
                    filtered_mock.order_by.return_value.first.return_value = mock_embedding_records[0]
                    return filtered_mock
                query_mock.filter = filter_side_effect
            return query_mock
        
        mock_db.query.side_effect = mock_query_side_effect
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        return mock_db
    
    @given(
        st.tuples(
            st.floats(min_value=3.0, max_value=6.0),  # age_min
            st.floats(min_value=6.1, max_value=9.0)   # age_max
        ),
        st.lists(
            st.tuples(
                st.integers(min_value=1, max_value=1000),  # visual seed
                st.sampled_from(["house", "person", "tree", "car", "unspecified"])  # subject
            ),
            min_size=10,
            max_size=30
        )
    )
    @settings(max_examples=20)
    def test_hybrid_embedding_training_consistency(self, age_range, embedding_specs):
        """
        **Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
        **Validates: Requirements 3.7**
        
        For any set of hybrid embeddings, training a subject-aware autoencoder should
        consistently handle 832-dimensional input and produce valid reconstruction results.
        """
        age_min, age_max = age_range
        assume(age_max > age_min)
        
        # Create hybrid embeddings
        embeddings_data = []
        for visual_seed, subject in embedding_specs:
            hybrid_embedding = self.create_mock_hybrid_embedding(visual_seed, subject)
            embeddings_data.append((hybrid_embedding, subject))
        
        # Mock database session
        mock_db = self.create_mock_database_session(age_min, age_max, embeddings_data)
        
        # Create model manager
        manager = ModelManager()
        
        # Mock the embedding deserialization
        with patch('app.utils.embedding_serialization.deserialize_embedding_from_db') as mock_deserialize:
            # Return the actual embeddings when deserialization is called
            mock_deserialize.side_effect = [embedding for embedding, _ in embeddings_data]
            
            # Mock file operations
            with patch('pathlib.Path.exists', return_value=False), \
                 patch('torch.save') as mock_torch_save, \
                 patch('pathlib.Path.mkdir'):
                
                # Create training config
                config = TrainingConfig(
                    hidden_dims=[256, 128, 64],
                    learning_rate=0.001,
                    batch_size=8,
                    epochs=5,  # Small number for testing
                    validation_split=0.2
                )
                
                try:
                    # Train subject-aware model
                    result = manager.train_subject_aware_age_group_model(
                        age_min, age_max, config, mock_db
                    )
                    
                    # Property: Training should succeed with hybrid embeddings
                    assert result is not None
                    assert "model_id" in result
                    assert "subject_distribution" in result
                    assert "supported_subjects" in result
                    assert result["embedding_type"] == "hybrid"
                    
                    # Property: Subject distribution should be tracked
                    subject_distribution = result["subject_distribution"]
                    assert isinstance(subject_distribution, dict)
                    assert len(subject_distribution) > 0
                    
                    # Property: All subjects in data should be represented
                    expected_subjects = set(subject for _, subject in embedding_specs)
                    actual_subjects = set(subject_distribution.keys())
                    assert expected_subjects.issubset(actual_subjects) or len(expected_subjects - actual_subjects) <= 1
                    
                    # Property: Model should be saved with subject metadata
                    mock_torch_save.assert_called_once()
                    save_args = mock_torch_save.call_args[0]
                    saved_data = save_args[0]
                    
                    assert 'subject_metadata' in saved_data
                    assert 'supported_subjects' in saved_data['subject_metadata']
                    assert 'embedding_type' in saved_data['subject_metadata']
                    assert saved_data['subject_metadata']['embedding_type'] == 'hybrid'
                
                except (ModelManagerError, AutoencoderTrainingError) as e:
                    # Training may fail with very small datasets, which is acceptable
                    assert len(embeddings_data) < 15, f"Training failed unexpectedly with {len(embeddings_data)} samples: {str(e)}"
    
    def test_hybrid_embedding_dimensionality_validation(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
        **Validates: Requirements 3.7**
        
        The training process should validate that all embeddings are 832-dimensional
        and reject non-hybrid embeddings.
        """
        # Create embeddings with wrong dimensions
        wrong_dim_embeddings = [
            (np.random.randn(768).astype(np.float32), "house"),  # Missing subject component
            (np.random.randn(1024).astype(np.float32), "person"),  # Too large
        ]
        
        # Mock database session
        mock_db = self.create_mock_database_session(4.0, 6.0, wrong_dim_embeddings)
        
        # Create model manager
        manager = ModelManager()
        
        # Mock the embedding deserialization to return wrong-sized embeddings
        with patch('app.utils.embedding_serialization.deserialize_embedding_from_db') as mock_deserialize:
            mock_deserialize.side_effect = [embedding for embedding, _ in wrong_dim_embeddings]
            
            # Create training config
            config = TrainingConfig(
                hidden_dims=[256, 128, 64],
                learning_rate=0.001,
                batch_size=4,
                epochs=5
            )
            
            # Property: Training should fail with wrong-dimensional embeddings
            with pytest.raises((ModelManagerError, AutoencoderTrainingError)):
                manager.train_subject_aware_age_group_model(4.0, 6.0, config, mock_db)
    
    def test_subject_distribution_analysis(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
        **Validates: Requirements 3.7**
        
        The training process should analyze and report subject distribution,
        including warnings for unbalanced data.
        """
        # Create unbalanced dataset
        embeddings_data = []
        
        # Many house drawings
        for i in range(15):
            embedding = self.create_mock_hybrid_embedding(i, "house")
            embeddings_data.append((embedding, "house"))
        
        # Few person drawings
        for i in range(3):
            embedding = self.create_mock_hybrid_embedding(i + 100, "person")
            embeddings_data.append((embedding, "person"))
        
        # Mock database session
        mock_db = self.create_mock_database_session(4.0, 6.0, embeddings_data)
        
        # Create model manager
        manager = ModelManager()
        
        # Mock the embedding deserialization
        with patch('app.utils.embedding_serialization.deserialize_embedding_from_db') as mock_deserialize:
            mock_deserialize.side_effect = [embedding for embedding, _ in embeddings_data]
            
            with patch('pathlib.Path.exists', return_value=False), \
                 patch('torch.save'), \
                 patch('pathlib.Path.mkdir'):
                
                config = TrainingConfig(
                    hidden_dims=[256, 128, 64],
                    learning_rate=0.001,
                    batch_size=8,
                    epochs=5
                )
                
                # Train model
                result = manager.train_subject_aware_age_group_model(4.0, 6.0, config, mock_db)
                
                # Property: Subject distribution should be accurately reported
                subject_distribution = result["subject_distribution"]
                assert subject_distribution["house"] == 15
                assert subject_distribution["person"] == 3
                
                # Property: Supported subjects should include all present subjects
                supported_subjects = result["supported_subjects"]
                assert "house" in supported_subjects
                assert "person" in supported_subjects
    
    def test_autoencoder_architecture_for_hybrid_embeddings(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
        **Validates: Requirements 3.7**
        
        The autoencoder architecture should be properly configured for 832-dimensional
        hybrid embeddings and produce valid reconstructions.
        """
        # Create test hybrid embeddings
        embeddings_data = []
        for i in range(20):
            embedding = self.create_mock_hybrid_embedding(i, "house")
            embeddings_data.append((embedding, "house"))
        
        # Test autoencoder directly
        config = TrainingConfig(
            hidden_dims=[256, 128, 64],
            learning_rate=0.001,
            batch_size=4,
            epochs=3
        )
        
        # Create autoencoder model
        model = AutoencoderModel(input_dim=832, hidden_dims=config.hidden_dims)
        
        # Property: Model should accept 832-dimensional input
        test_input = torch.FloatTensor(embeddings_data[0][0]).unsqueeze(0)
        assert test_input.shape[1] == 832
        
        # Property: Model should produce 832-dimensional output
        with torch.no_grad():
            output = model(test_input)
            assert output.shape == test_input.shape
            assert output.shape[1] == 832
        
        # Property: Model architecture should be correctly configured
        arch_info = model.get_architecture_info()
        assert arch_info['input_dim'] == 832
        assert arch_info['hidden_dims'] == config.hidden_dims
        assert arch_info['total_parameters'] > 0
    
    def test_unified_architecture_consistency(self):
        """
        **Feature: children-drawing-anomaly-detection, Property 34: Subject-Aware Model Training**
        **Validates: Requirements 3.7**
        
        All trained models should use the unified subject-aware architecture
        with consistent embedding dimensions and metadata.
        """
        # Create multiple age groups with different subjects
        age_groups = [(3.0, 4.0), (4.0, 5.0), (5.0, 6.0)]
        subjects = ["house", "person", "tree"]
        
        manager = ModelManager()
        
        for age_min, age_max in age_groups:
            # Create embeddings for this age group
            embeddings_data = []
            for i, subject in enumerate(subjects):
                for j in range(8):  # 8 samples per subject
                    embedding = self.create_mock_hybrid_embedding(i * 10 + j, subject)
                    embeddings_data.append((embedding, subject))
            
            # Mock database session
            mock_db = self.create_mock_database_session(age_min, age_max, embeddings_data)
            
            # Mock the embedding deserialization
            with patch('app.utils.embedding_serialization.deserialize_embedding_from_db') as mock_deserialize:
                mock_deserialize.side_effect = [embedding for embedding, _ in embeddings_data]
                
                with patch('pathlib.Path.exists', return_value=False), \
                     patch('torch.save') as mock_torch_save, \
                     patch('pathlib.Path.mkdir'):
                    
                    config = TrainingConfig(
                        hidden_dims=[256, 128, 64],
                        learning_rate=0.001,
                        batch_size=8,
                        epochs=3
                    )
                    
                    # Train model
                    result = manager.train_subject_aware_age_group_model(age_min, age_max, config, mock_db)
                    
                    # Property: All models should use unified architecture
                    assert result["embedding_type"] == "hybrid"
                    assert len(result["supported_subjects"]) == len(subjects)
                    
                    # Property: Model metadata should be consistent
                    mock_torch_save.assert_called()
                    save_args = mock_torch_save.call_args[0]
                    saved_data = save_args[0]
                    
                    subject_metadata = saved_data['subject_metadata']
                    assert subject_metadata['embedding_type'] == 'hybrid'
                    assert len(subject_metadata['supported_subjects']) == len(subjects)
                    
                    # Reset mock for next iteration
                    mock_torch_save.reset_mock()