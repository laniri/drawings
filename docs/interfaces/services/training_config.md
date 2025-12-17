# Training Config Service

Training Configuration Management Service for Children's Drawing Anomaly Detection System

This module provides Hydra-based configuration management for training parameters,
validation, and parameter sweep capabilities.

## Class: TrainingEnvironment

Training environment options.

## Class: OptimizerType

Optimizer types for training.

## Class: SchedulerType

Learning rate scheduler types.

## Class: ModelConfig

Configuration for model architecture.

## Class: OptimizerConfig

Configuration for optimizer.

## Class: SchedulerConfig

Configuration for learning rate scheduler.

## Class: DataConfig

Configuration for data handling.

## Class: TrainingConfig

Complete training configuration.

## Class: ParameterSweepConfig

Configuration for parameter sweeps.

## Class: TrainingConfigManager

Manager for training configurations with Hydra integration.

### create_default_configs

Create default configuration files.

**Signature**: `create_default_configs()`

### load_config

Load training configuration from file or create default.

Args:
    config_path: Path to configuration file
    
Returns:
    TrainingConfig object

**Signature**: `load_config(config_path)`

### save_config

Save training configuration to file.

Args:
    config: TrainingConfig object to save
    config_path: Path where to save the configuration

**Signature**: `save_config(config, config_path)`

### validate_config

Validate training configuration and return validation results.

Args:
    config: TrainingConfig to validate
    
Returns:
    Dictionary with validation results

**Signature**: `validate_config(config)`

### create_parameter_sweep

Create parameter sweep configurations.

Args:
    base_config: Base configuration to modify
    sweep_config: Parameter sweep configuration
    
Returns:
    List of TrainingConfig objects for the sweep

**Signature**: `create_parameter_sweep(base_config, sweep_config)`

### get_hydra_config

Convert TrainingConfig to Hydra DictConfig.

Args:
    config: TrainingConfig to convert
    
Returns:
    Hydra DictConfig object

**Signature**: `get_hydra_config(config)`

