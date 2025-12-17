# Training Config Contract

## Overview
Service contract for Training Config (service)

**Source File**: `app/services/training_config.py`

## Interface Specification

### Classes

#### TrainingEnvironment

Training environment options.

**Inherits from**: str, Enum

#### OptimizerType

Optimizer types for training.

**Inherits from**: str, Enum

#### SchedulerType

Learning rate scheduler types.

**Inherits from**: str, Enum

#### ModelConfig

Configuration for model architecture.

**Attributes**:

- `encoder_dim: int`
- `hidden_dims: List[int]`
- `latent_dim: int`
- `dropout_rate: float`
- `activation: str`
- `batch_norm: bool`

#### OptimizerConfig

Configuration for optimizer.

**Attributes**:

- `type: OptimizerType`
- `learning_rate: float`
- `weight_decay: float`
- `momentum: float`
- `betas: tuple`

#### SchedulerConfig

Configuration for learning rate scheduler.

**Attributes**:

- `type: SchedulerType`
- `step_size: int`
- `gamma: float`
- `T_max: int`

#### DataConfig

Configuration for data handling.

**Attributes**:

- `batch_size: int`
- `num_workers: int`
- `pin_memory: bool`
- `train_split: float`
- `validation_split: float`
- `test_split: float`
- `stratify_by_age: bool`
- `age_group_size: float`

#### TrainingConfig

Complete training configuration.

**Attributes**:

- `job_name: str`
- `experiment_name: str`
- `environment: TrainingEnvironment`
- `device: str`
- `mixed_precision: bool`
- `dataset_folder: str`
- `metadata_file: str`
- `data: DataConfig`
- `model: ModelConfig`
- `epochs: int`
- `early_stopping_patience: int`
- `early_stopping_min_delta: float`
- `gradient_clip_norm: Optional[float]`
- `optimizer: OptimizerConfig`
- `scheduler: SchedulerConfig`
- `validation_frequency: int`
- `checkpoint_frequency: int`
- `save_best_only: bool`
- `log_frequency: int`
- `save_plots: bool`
- `plot_frequency: int`
- `sagemaker_instance_type: str`
- `sagemaker_instance_count: int`
- `sagemaker_volume_size: int`
- `sagemaker_max_runtime: int`
- `output_dir: str`
- `model_save_path: str`
- `log_save_path: str`

#### ParameterSweepConfig

Configuration for parameter sweeps.

**Attributes**:

- `learning_rates: List[float]`
- `batch_sizes: List[int]`
- `hidden_dims: <ast.Subscript object at 0x1104e42d0>`
- `latent_dims: List[int]`
- `dropout_rates: List[float]`
- `max_trials: int`
- `optimization_metric: str`
- `optimization_direction: str`

#### TrainingConfigManager

Manager for training configurations with Hydra integration.

## Methods

### create_default_configs

Create default configuration files.

**Signature**: `create_default_configs()`

### load_config

Load training configuration from file or create default.

Args:
    config_path: Path to configuration file
    
Returns:
    TrainingConfig object

**Signature**: `load_config(config_path: <ast.Subscript object at 0x1104d6810>) -> TrainingConfig`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config_path` | `<ast.Subscript object at 0x1104d6810>` | Parameter description |

**Returns**: `TrainingConfig`

### save_config

Save training configuration to file.

Args:
    config: TrainingConfig object to save
    config_path: Path where to save the configuration

**Signature**: `save_config(config: TrainingConfig, config_path: Union[str, Path])`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config` | `TrainingConfig` | Parameter description |
| `config_path` | `Union[str, Path]` | Parameter description |

### validate_config

Validate training configuration and return validation results.

Args:
    config: TrainingConfig to validate
    
Returns:
    Dictionary with validation results

**Signature**: `validate_config(config: TrainingConfig) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config` | `TrainingConfig` | Parameter description |

**Returns**: `Dict[str, Any]`

### create_parameter_sweep

Create parameter sweep configurations.

Args:
    base_config: Base configuration to modify
    sweep_config: Parameter sweep configuration
    
Returns:
    List of TrainingConfig objects for the sweep

**Signature**: `create_parameter_sweep(base_config: TrainingConfig, sweep_config: ParameterSweepConfig) -> List[TrainingConfig]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `base_config` | `TrainingConfig` | Parameter description |
| `sweep_config` | `ParameterSweepConfig` | Parameter description |

**Returns**: `List[TrainingConfig]`

### get_hydra_config

Convert TrainingConfig to Hydra DictConfig.

Args:
    config: TrainingConfig to convert
    
Returns:
    Hydra DictConfig object

**Signature**: `get_hydra_config(config: TrainingConfig) -> DictConfig`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config` | `TrainingConfig` | Parameter description |

**Returns**: `DictConfig`

## Defined Interfaces

### TrainingConfigManagerInterface

**Type**: Protocol
**Implemented by**: TrainingConfigManager

**Methods**:

- `create_default_configs()`
- `load_config(config_path: <ast.Subscript object at 0x1104d6810>) -> TrainingConfig`
- `save_config(config: TrainingConfig, config_path: Union[str, Path])`
- `validate_config(config: TrainingConfig) -> Dict[str, Any]`
- `create_parameter_sweep(base_config: TrainingConfig, sweep_config: ParameterSweepConfig) -> List[TrainingConfig]`
- `get_hydra_config(config: TrainingConfig) -> DictConfig`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/training_config.py`
- Last validated: 2025-12-16 15:47:04

