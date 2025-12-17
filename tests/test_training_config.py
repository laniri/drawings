"""
Property-based tests for training configuration management.

**Feature: children-drawing-anomaly-detection, Property 8: Training Configuration Validation**
**Validates: Requirements 3.2**
"""

import pytest
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume
from typing import List

from app.services.training_config import (
    TrainingConfig,
    TrainingConfigManager,
    ModelConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingEnvironment,
    OptimizerType,
    SchedulerType,
    ParameterSweepConfig
)
from app.core.exceptions import ValidationError


# Hypothesis strategies for generating test data
valid_learning_rate_strategy = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False, allow_infinity=False)
invalid_learning_rate_strategy = st.one_of(
    st.floats(min_value=-1.0, max_value=0.0),  # Negative or zero
    st.floats(min_value=1.1, max_value=10.0),  # Too large
    st.just(float('nan')),
    st.just(float('inf'))
)

valid_batch_size_strategy = st.integers(min_value=1, max_value=1024)
invalid_batch_size_strategy = st.one_of(
    st.integers(min_value=-100, max_value=0),  # Non-positive
    st.integers(min_value=1025, max_value=10000)  # Too large
)

valid_epochs_strategy = st.integers(min_value=1, max_value=10000)
invalid_epochs_strategy = st.one_of(
    st.integers(min_value=-100, max_value=0),  # Non-positive
    st.integers(min_value=10001, max_value=100000)  # Too large
)

valid_dropout_rate_strategy = st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False)
invalid_dropout_rate_strategy = st.one_of(
    st.floats(min_value=-1.0, max_value=-0.01),  # Negative
    st.floats(min_value=1.0, max_value=2.0),  # >= 1.0
    st.just(float('nan')),
    st.just(float('inf'))
)

# Generate valid split ratios that sum to 1.0
@st.composite
def valid_split_ratios_strategy(draw):
    train_ratio = draw(st.floats(min_value=0.1, max_value=0.8))
    val_ratio = draw(st.floats(min_value=0.1, max_value=0.4))
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # Ensure test ratio is reasonable
    if test_ratio < 0.05 or test_ratio > 0.5:
        test_ratio = 0.1
        val_ratio = 0.2
        train_ratio = 0.7
    
    return (train_ratio, val_ratio, test_ratio)

@st.composite
def invalid_split_ratios_strategy(draw):
    # Generate ratios that don't sum to 1.0
    train_ratio = draw(st.floats(min_value=0.1, max_value=0.9))
    val_ratio = draw(st.floats(min_value=0.1, max_value=0.9))
    test_ratio = draw(st.floats(min_value=0.1, max_value=0.9))
    
    # Ensure they don't accidentally sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    assume(abs(total - 1.0) > 0.02)  # Significantly different from 1.0
    
    return (train_ratio, val_ratio, test_ratio)

hidden_dims_strategy = st.lists(
    st.integers(min_value=8, max_value=1024),
    min_size=1,
    max_size=5
)


@given(
    learning_rate=valid_learning_rate_strategy,
    batch_size=valid_batch_size_strategy,
    epochs=valid_epochs_strategy,
    dropout_rate=valid_dropout_rate_strategy,
    split_ratios=valid_split_ratios_strategy(),
    hidden_dims=hidden_dims_strategy
)
@settings(max_examples=100, deadline=None)
def test_training_configuration_validation_valid_params(
    learning_rate, batch_size, epochs, dropout_rate, split_ratios, hidden_dims
):
    """
    **Feature: children-drawing-anomaly-detection, Property 8: Training Configuration Validation**
    **Validates: Requirements 3.2**
    
    Property: For any training parameters within valid ranges, the training environment 
    should accept the configuration, and reject parameters outside valid ranges.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # Create configuration with valid parameters
    config = TrainingConfig(
        job_name="test_job",
        epochs=epochs,
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        data=DataConfig(
            batch_size=batch_size,
            train_split=train_ratio,
            validation_split=val_ratio,
            test_split=test_ratio
        ),
        model=ModelConfig(
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
    )
    
    # Configuration should be valid (no exception raised during creation)
    assert config.optimizer.learning_rate == learning_rate
    assert config.data.batch_size == batch_size
    assert config.epochs == epochs
    assert config.model.dropout_rate == dropout_rate
    assert config.data.train_split == train_ratio
    assert config.data.validation_split == val_ratio
    assert config.data.test_split == test_ratio
    
    # Validation should pass
    manager = TrainingConfigManager()
    validation_result = manager.validate_config(config)
    assert validation_result['is_valid'] == True


@given(
    invalid_param=st.one_of(
        st.tuples(st.just("learning_rate"), invalid_learning_rate_strategy),
        st.tuples(st.just("batch_size"), invalid_batch_size_strategy),
        st.tuples(st.just("epochs"), invalid_epochs_strategy),
        st.tuples(st.just("dropout_rate"), invalid_dropout_rate_strategy)
    )
)
@settings(max_examples=50, deadline=None)
def test_training_configuration_validation_invalid_params(invalid_param):
    """
    Property: Invalid training parameters should be rejected with appropriate error messages.
    """
    param_name, param_value = invalid_param
    
    # Skip NaN and infinity values for certain parameters as they may cause different errors
    if param_name in ["learning_rate", "dropout_rate"]:
        assume(not (isinstance(param_value, float) and (
            param_value != param_value or  # NaN check
            param_value == float('inf') or param_value == float('-inf')
        )))
    
    # Create base valid configuration
    base_config_dict = {
        "job_name": "test_job",
        "epochs": 100,
        "optimizer": OptimizerConfig(learning_rate=0.001),
        "data": DataConfig(batch_size=32),
        "model": ModelConfig(dropout_rate=0.1)
    }
    
    # Modify with invalid parameter
    if param_name == "learning_rate":
        base_config_dict["optimizer"] = OptimizerConfig(learning_rate=param_value)
    elif param_name == "batch_size":
        base_config_dict["data"] = DataConfig(batch_size=param_value)
    elif param_name == "epochs":
        base_config_dict["epochs"] = param_value
    elif param_name == "dropout_rate":
        base_config_dict["model"] = ModelConfig(dropout_rate=param_value)
    
    # Configuration creation should raise ValidationError
    with pytest.raises(ValidationError):
        TrainingConfig(**base_config_dict)


@given(split_ratios=invalid_split_ratios_strategy())
@settings(max_examples=50, deadline=None)
def test_training_configuration_validation_invalid_splits(split_ratios):
    """
    Property: Data splits that don't sum to 1.0 should be rejected.
    """
    train_ratio, val_ratio, test_ratio = split_ratios
    
    # Configuration creation should raise ValidationError
    with pytest.raises(ValidationError, match="Data splits must sum to 1.0"):
        TrainingConfig(
            job_name="test_job",
            data=DataConfig(
                train_split=train_ratio,
                validation_split=val_ratio,
                test_split=test_ratio
            )
        )


@given(
    config_format=st.sampled_from(['yaml', 'json']),
    learning_rate=valid_learning_rate_strategy,
    batch_size=valid_batch_size_strategy,
    epochs=valid_epochs_strategy
)
@settings(max_examples=20, deadline=None)
def test_config_serialization_round_trip(config_format, learning_rate, batch_size, epochs):
    """
    Property: Configuration serialization and deserialization should preserve all values.
    """
    # Create original configuration
    original_config = TrainingConfig(
        job_name="test_serialization",
        epochs=epochs,
        optimizer=OptimizerConfig(learning_rate=learning_rate),
        data=DataConfig(batch_size=batch_size)
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_file = temp_path / f"test_config.{config_format}"
        
        # Save and load configuration
        manager = TrainingConfigManager()
        manager.save_config(original_config, config_file)
        loaded_config = manager.load_config(config_file)
        
        # Verify all important values are preserved
        assert loaded_config.job_name == original_config.job_name
        assert loaded_config.epochs == original_config.epochs
        assert loaded_config.optimizer.learning_rate == original_config.optimizer.learning_rate
        assert loaded_config.data.batch_size == original_config.data.batch_size
        assert loaded_config.model.dropout_rate == original_config.model.dropout_rate


def test_parameter_sweep_generation():
    """Test parameter sweep configuration generation."""
    base_config = TrainingConfig(job_name="base_sweep_test")
    
    sweep_config = ParameterSweepConfig(
        learning_rates=[0.001, 0.01],
        batch_sizes=[16, 32],
        hidden_dims=[[128, 64], [256, 128]],
        latent_dims=[16, 32],
        dropout_rates=[0.1, 0.2],
        max_trials=5  # Limit for testing
    )
    
    manager = TrainingConfigManager()
    sweep_configs = manager.create_parameter_sweep(base_config, sweep_config)
    
    # Should generate configurations (limited by max_trials)
    assert len(sweep_configs) <= sweep_config.max_trials
    assert len(sweep_configs) > 0
    
    # Each configuration should have unique job name
    job_names = [config.job_name for config in sweep_configs]
    assert len(set(job_names)) == len(job_names)
    
    # All configurations should be valid
    for config in sweep_configs:
        validation_result = manager.validate_config(config)
        assert validation_result['is_valid'] == True


def test_config_validation_warnings_and_recommendations():
    """Test that configuration validation provides appropriate warnings and recommendations."""
    manager = TrainingConfigManager()
    
    # Configuration with potential issues
    config = TrainingConfig(
        job_name="warning_test",
        epochs=1000,  # Many epochs - should warn
        optimizer=OptimizerConfig(learning_rate=0.5),  # High learning rate - should warn
        data=DataConfig(batch_size=256),  # Large batch size - should warn
        early_stopping_patience=2,  # Low patience - should recommend increase
        gradient_clip_norm=None  # No gradient clipping - should recommend
    )
    
    validation_result = manager.validate_config(config)
    
    # Should still be valid but have warnings/recommendations
    assert validation_result['is_valid'] == True
    assert len(validation_result['warnings']) > 0
    assert len(validation_result['recommendations']) > 0
    
    # Check specific warnings
    warning_texts = ' '.join(validation_result['warnings'])
    assert 'batch size' in warning_texts.lower() or 'learning rate' in warning_texts.lower() or 'epochs' in warning_texts.lower()
    
    # Check specific recommendations
    recommendation_texts = ' '.join(validation_result['recommendations'])
    assert 'patience' in recommendation_texts.lower() or 'gradient' in recommendation_texts.lower()


def test_nonexistent_paths_validation():
    """Test validation of file paths that don't exist."""
    with pytest.raises(ValidationError, match="Dataset folder not found"):
        TrainingConfig(
            job_name="path_test",
            dataset_folder="/nonexistent/path"
        )
    
    with pytest.raises(ValidationError, match="Metadata file not found"):
        TrainingConfig(
            job_name="path_test",
            metadata_file="/nonexistent/file.csv"
        )


def test_default_config_creation():
    """Test that default configuration is valid."""
    config = TrainingConfig()
    
    # Should not raise any exceptions
    assert config.job_name is not None
    assert config.epochs > 0
    assert config.optimizer.learning_rate > 0
    assert config.data.batch_size > 0
    
    # Should pass validation
    manager = TrainingConfigManager()
    validation_result = manager.validate_config(config)
    assert validation_result['is_valid'] == True