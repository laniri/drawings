"""
Property-based tests for model export compatibility.

**Feature: children-drawing-anomaly-detection, Property 10: Model Export Compatibility**
**Validates: Requirements 3.5**
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

from app.services.model_deployment_service import (
    ModelExporter,
    ModelValidator,
    ModelExportMetadata,
    ModelExportError,
    ModelValidationError
)
from app.services.model_manager import AutoencoderModel
from app.models.database import TrainingJob, TrainingReport
from app.core.database import get_db
from sqlalchemy.orm import Session


# Hypothesis strategies for generating test data
valid_dimension_strategy = st.integers(min_value=32, max_value=1024)
valid_age_strategy = st.floats(min_value=2.0, max_value=18.0, allow_nan=False, allow_infinity=False)
valid_parameter_count_strategy = st.integers(min_value=1000, max_value=1000000)
valid_loss_strategy = st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False)

@st.composite
def model_architecture_strategy(draw):
    """Generate valid model architecture."""
    input_dim = draw(valid_dimension_strategy)
    hidden_dims = draw(st.lists(
        st.integers(min_value=16, max_value=512),
        min_size=1,
        max_size=4
    ))
    
    return {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'latent_dim': min(hidden_dims[-1] if hidden_dims else 64, 128),
        'model_type': 'autoencoder',
        'total_parameters': sum([input_dim * hidden_dims[0]] + 
                               [hidden_dims[i] * hidden_dims[i+1] for i in range(len(hidden_dims)-1)] +
                               [hidden_dims[-1] * (hidden_dims[-1] // 2)]) * 2  # Approximate
    }

@st.composite
def training_metrics_strategy(draw):
    """Generate valid training metrics."""
    final_loss = draw(valid_loss_strategy)
    best_loss = draw(st.floats(min_value=0.001, max_value=final_loss))
    
    return {
        'final_loss': final_loss,
        'best_loss': best_loss,
        'validation_accuracy': draw(st.floats(min_value=0.1, max_value=1.0)),
        'best_epoch': draw(st.integers(min_value=1, max_value=1000)),
        'training_time_seconds': draw(st.floats(min_value=10.0, max_value=3600.0))
    }

@st.composite
def age_group_strategy(draw):
    """Generate valid age group ranges."""
    age_min = draw(st.floats(min_value=2.0, max_value=16.0))
    age_max = draw(st.floats(min_value=age_min + 0.5, max_value=18.0))
    return (age_min, age_max)

def create_test_model(architecture: Dict[str, Any]) -> nn.Module:
    """Create a test autoencoder model."""
    input_dim = architecture['input_dim']
    hidden_dims = architecture['hidden_dims']
    
    model = AutoencoderModel(input_dim, hidden_dims)
    
    # Initialize with small random weights for consistency
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=0.1)
        else:
            nn.init.constant_(param, 0.01)
    
    return model


@given(
    architecture=model_architecture_strategy(),
    age_group=age_group_strategy(),
    training_metrics=training_metrics_strategy(),
    export_format=st.sampled_from(['pytorch', 'pickle'])  # Skip ONNX for simplicity
)
@settings(max_examples=50, deadline=None)
def test_model_export_compatibility(architecture, age_group, training_metrics, export_format):
    """
    **Feature: children-drawing-anomaly-detection, Property 10: Model Export Compatibility**
    **Validates: Requirements 3.5**
    
    Property: For any trained model that meets performance criteria, the exported 
    parameters should be loadable by the production system.
    """
    age_min, age_max = age_group
    
    # Create test model
    model = create_test_model(architecture)
    model.eval()
    
    # Create model info
    model_info = {
        **architecture,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    # Initialize exporter
    exporter = ModelExporter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Temporarily override export directory
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export model directly
            export_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=age_min,
                age_group_max=age_max,
                export_format=export_format
            )
            
            # Verify export metadata is complete
            assert export_metadata.model_id is not None
            assert export_metadata.export_format == export_format
            assert export_metadata.age_group_min == age_min
            assert export_metadata.age_group_max == age_max
            assert export_metadata.input_dimension == architecture['input_dim']
            assert export_metadata.parameter_count > 0
            assert export_metadata.file_size_bytes > 0
            assert export_metadata.checksum != ""
            
            # Verify export file exists
            export_file = exporter.export_dir / f"{export_metadata.model_id}.{export_format}"
            assert export_file.exists()
            assert export_file.stat().st_size > 0
            
            # Verify metadata file exists
            metadata_file = exporter.export_dir / f"{export_metadata.model_id}_metadata.json"
            assert metadata_file.exists()
            
            # Test loading exported model (compatibility check)
            if export_format == "pytorch":
                # Load PyTorch model
                checkpoint = torch.load(export_file, map_location='cpu')
                assert isinstance(checkpoint, dict)
                assert 'model_state_dict' in checkpoint
                assert 'model_architecture' in checkpoint
                assert 'metadata' in checkpoint
                
                # Verify model can be reconstructed
                loaded_architecture = checkpoint['model_architecture']
                reconstructed_model = AutoencoderModel(
                    loaded_architecture['input_dim'],
                    loaded_architecture.get('hidden_dims', architecture['hidden_dims'])
                )
                reconstructed_model.load_state_dict(checkpoint['model_state_dict'])
                reconstructed_model.eval()
                
                # Test that reconstructed model produces same output as original
                test_input = torch.randn(1, architecture['input_dim'])
                with torch.no_grad():
                    original_output = model(test_input)
                    reconstructed_output = reconstructed_model(test_input)
                    
                    # Outputs should be very close (allowing for small numerical differences)
                    assert torch.allclose(original_output, reconstructed_output, atol=1e-6)
                
            elif export_format == "pickle":
                # Load pickle model
                import pickle
                with open(export_file, 'rb') as f:
                    data = pickle.load(f)
                
                assert isinstance(data, dict)
                assert 'model' in data
                assert 'metadata' in data
                
                loaded_model = data['model']
                loaded_model.eval()
                
                # Test that loaded model produces same output as original
                test_input = torch.randn(1, architecture['input_dim'])
                with torch.no_grad():
                    original_output = model(test_input)
                    loaded_output = loaded_model(test_input)
                    
                    # Outputs should be identical for pickle format
                    assert torch.allclose(original_output, loaded_output, atol=1e-8)
            
            # Verify metadata can be loaded and parsed
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            assert metadata_dict['model_id'] == export_metadata.model_id
            assert metadata_dict['export_format'] == export_format
            assert metadata_dict['age_group_min'] == age_min
            assert metadata_dict['age_group_max'] == age_max
            assert metadata_dict['input_dimension'] == architecture['input_dim']
            
        finally:
            # Restore original export directory
            exporter.export_dir = original_export_dir


@given(
    architecture=model_architecture_strategy(),
    age_group=age_group_strategy(),
    export_format=st.sampled_from(['pytorch', 'pickle'])
)
@settings(max_examples=30, deadline=None)
def test_model_validation_after_export(architecture, age_group, export_format):
    """
    Property: Exported models should pass validation checks for compatibility and integrity.
    """
    age_min, age_max = age_group
    
    # Create test model
    model = create_test_model(architecture)
    model.eval()
    
    # Create model info
    model_info = {
        **architecture,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    # Initialize services
    exporter = ModelExporter()
    validator = ModelValidator(export_dir=exporter.export_dir)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override export directory
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        validator.export_dir = exporter.export_dir
        
        try:
            # Export model
            export_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=age_min,
                age_group_max=age_max,
                export_format=export_format
            )
            
            # Validate exported model
            validation_result = validator.validate_exported_model(export_metadata)
            
            # Validation should pass for properly exported models
            assert validation_result['is_valid'] == True
            assert len(validation_result['errors']) == 0
            
            # Check specific validation components
            assert validation_result['integrity_checks']['checksum'] == 'passed'
            assert validation_result['compatibility_checks']['is_compatible'] == True
            assert validation_result['compatibility_checks']['version_check'] == 'passed'
            assert validation_result['compatibility_checks']['architecture_check'] == 'passed'
            assert validation_result['compatibility_checks']['format_check'] == 'passed'
            
        finally:
            exporter.export_dir = original_export_dir


@given(
    architecture=model_architecture_strategy(),
    age_group=age_group_strategy()
)
@settings(max_examples=20, deadline=None)
def test_export_format_consistency(architecture, age_group):
    """
    Property: The same model exported in different formats should maintain compatibility.
    """
    age_min, age_max = age_group
    
    # Create test model
    model = create_test_model(architecture)
    model.eval()
    
    model_info = {
        **architecture,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    exporter = ModelExporter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export in multiple formats
            pytorch_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=age_min,
                age_group_max=age_max,
                export_format="pytorch"
            )
            
            pickle_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=age_min,
                age_group_max=age_max,
                export_format="pickle"
            )
            
            # Core metadata should be consistent across formats
            assert pytorch_metadata.age_group_min == pickle_metadata.age_group_min
            assert pytorch_metadata.age_group_max == pickle_metadata.age_group_max
            assert pytorch_metadata.input_dimension == pickle_metadata.input_dimension
            assert pytorch_metadata.parameter_count == pickle_metadata.parameter_count
            assert pytorch_metadata.model_type == pickle_metadata.model_type
            
            # Both exports should be valid
            validator = ModelValidator(export_dir=exporter.export_dir)
            
            pytorch_validation = validator.validate_exported_model(pytorch_metadata)
            pickle_validation = validator.validate_exported_model(pickle_metadata)
            
            assert pytorch_validation['is_valid'] == True
            assert pickle_validation['is_valid'] == True
            
        finally:
            exporter.export_dir = original_export_dir


@given(
    architecture=model_architecture_strategy(),
    invalid_age_group=st.one_of(
        st.tuples(
            st.floats(min_value=18.1, max_value=25.0),  # Age too high
            st.floats(min_value=19.0, max_value=26.0)
        ),
        st.tuples(
            st.floats(min_value=0.0, max_value=1.9),    # Age too low
            st.floats(min_value=1.0, max_value=2.0)
        ),
        st.tuples(
            st.floats(min_value=10.0, max_value=15.0),  # Invalid range (min >= max)
            st.floats(min_value=5.0, max_value=10.0)
        )
    )
)
@settings(max_examples=20, deadline=None)
def test_export_validation_rejects_invalid_age_groups(architecture, invalid_age_group):
    """
    Property: Model export should reject invalid age group ranges.
    """
    age_min, age_max = invalid_age_group
    
    # Skip cases where the range is actually valid due to floating point precision
    assume(age_min >= age_max or age_min < 2.0 or age_max > 18.0)
    
    model = create_test_model(architecture)
    model_info = {
        **architecture,
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    exporter = ModelExporter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export should either fail or produce a model that fails validation
            try:
                export_metadata = exporter.export_model_direct(
                    model=model,
                    model_info=model_info,
                    age_group_min=age_min,
                    age_group_max=age_max,
                    export_format="pytorch"
                )
                
                # If export succeeds, validation should catch the invalid age group
                validator = ModelValidator()
                validation_result = validator.validate_exported_model(export_metadata)
                
                # Should have compatibility errors for invalid age groups
                assert not validation_result['compatibility_checks']['is_compatible'] or \
                       len(validation_result['compatibility_checks']['errors']) > 0 or \
                       len(validation_result['compatibility_checks']['warnings']) > 0
                
            except (ModelExportError, ValueError):
                # Export rejection is also acceptable
                pass
                
        finally:
            exporter.export_dir = original_export_dir


def test_export_with_corrupted_model():
    """
    Test that export handles corrupted or invalid models gracefully.
    """
    exporter = ModelExporter()
    
    # Create a "model" that's not actually a valid PyTorch model
    class InvalidModel:
        def parameters(self):
            return []
        
        def state_dict(self):
            # This will cause an error during export
            raise RuntimeError("Invalid model state")
    
    invalid_model = InvalidModel()
    
    model_info = {
        'input_dim': 512,
        'hidden_dims': [256, 128],
        'model_type': 'autoencoder',
        'parameter_count': 0
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export should fail gracefully
            with pytest.raises(ModelExportError):
                exporter.export_model_direct(
                    model=invalid_model,
                    model_info=model_info,
                    age_group_min=5.0,
                    age_group_max=10.0,
                    export_format="pytorch"
                )
                
        finally:
            exporter.export_dir = original_export_dir


def test_export_metadata_completeness():
    """
    Test that export metadata contains all required fields.
    """
    # Create simple test model
    model = AutoencoderModel(128, [64, 32])
    model_info = {
        'input_dim': 128,
        'hidden_dims': [64, 32],
        'model_type': 'autoencoder',
        'parameter_count': sum(p.numel() for p in model.parameters())
    }
    
    exporter = ModelExporter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            export_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=5.0,
                age_group_max=10.0,
                export_format="pytorch"
            )
            
            # Check all required metadata fields
            required_fields = [
                'model_id', 'export_timestamp', 'training_job_id', 'model_type',
                'model_version', 'architecture_hash', 'parameter_count',
                'input_dimension', 'output_dimension', 'age_group_min', 'age_group_max',
                'training_metrics', 'compatibility_version', 'export_format',
                'file_size_bytes', 'checksum'
            ]
            
            metadata_dict = export_metadata.to_dict()
            for field in required_fields:
                assert field in metadata_dict, f"Missing required field: {field}"
                assert metadata_dict[field] is not None, f"Field {field} is None"
            
            # Check specific field types and values
            assert isinstance(export_metadata.model_id, str)
            assert len(export_metadata.model_id) > 0
            assert export_metadata.parameter_count > 0
            assert export_metadata.input_dimension == 128
            assert export_metadata.output_dimension == 128  # Autoencoder
            assert export_metadata.age_group_min == 5.0
            assert export_metadata.age_group_max == 10.0
            assert export_metadata.export_format == "pytorch"
            assert export_metadata.file_size_bytes > 0
            assert len(export_metadata.checksum) == 64  # SHA256 hex length
            
        finally:
            exporter.export_dir = original_export_dir


def test_list_exported_models():
    """
    Test that exported models can be listed and retrieved.
    """
    exporter = ModelExporter()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        original_export_dir = exporter.export_dir
        exporter.export_dir = Path(temp_dir) / "exports"
        exporter.export_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initially no models
            models = exporter.list_exported_models()
            assert len(models) == 0
            
            # Export a model
            model = AutoencoderModel(64, [32, 16])
            model_info = {
                'input_dim': 64,
                'hidden_dims': [32, 16],
                'model_type': 'autoencoder',
                'parameter_count': sum(p.numel() for p in model.parameters())
            }
            
            export_metadata = exporter.export_model_direct(
                model=model,
                model_info=model_info,
                age_group_min=3.0,
                age_group_max=6.0,
                export_format="pytorch"
            )
            
            # Should now have one model
            models = exporter.list_exported_models()
            assert len(models) == 1
            
            # Check model information
            model_info = models[0]
            assert model_info['model_id'] == export_metadata.model_id
            assert model_info['export_format'] == 'pytorch'
            assert model_info['age_group_min'] == 3.0
            assert model_info['age_group_max'] == 6.0
            
        finally:
            exporter.export_dir = original_export_dir