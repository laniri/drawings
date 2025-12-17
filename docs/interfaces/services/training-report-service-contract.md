# Training Report Service Contract

## Overview
Service contract for Training Report Service (service)

**Source File**: `app/services/training_report_service.py`

## Interface Specification

### Classes

#### TrainingReportError

Base exception for training report generation errors.

**Inherits from**: Exception

#### MetricsCalculationError

Raised when metrics calculation fails.

**Inherits from**: TrainingReportError

#### VisualizationError

Raised when visualization generation fails.

**Inherits from**: TrainingReportError

#### TrainingMetrics

Container for comprehensive training metrics.

**Attributes**:

- `final_train_loss: float`
- `final_val_loss: float`
- `best_val_loss: float`
- `best_epoch: int`
- `total_epochs: int`
- `training_time_seconds: float`
- `train_loss_mean: float`
- `train_loss_std: float`
- `val_loss_mean: float`
- `val_loss_std: float`
- `convergence_epoch: Optional[int]`
- `early_stopping_triggered: bool`
- `overfitting_detected: bool`
- `generalization_gap: float`
- `reconstruction_error_stats: Dict[str, float]`
- `anomaly_detection_threshold: float`
- `validation_accuracy_estimate: float`
- `loss_variance: float`
- `gradient_norm_stats: <ast.Subscript object at 0x110830150>`
- `learning_rate_schedule: List[float]`

#### ModelArchitectureInfo

Container for model architecture information.

**Attributes**:

- `model_type: str`
- `input_dimension: int`
- `hidden_dimensions: List[int]`
- `latent_dimension: int`
- `total_parameters: int`
- `trainable_parameters: int`
- `model_size_mb: float`
- `activation_functions: List[str]`
- `dropout_rate: float`

#### TrainingConfiguration

Container for training configuration details.

**Attributes**:

- `learning_rate: float`
- `batch_size: int`
- `epochs: int`
- `optimizer: str`
- `loss_function: str`
- `early_stopping_patience: int`
- `min_delta: float`
- `data_split_ratios: Dict[str, float]`
- `device_used: str`
- `random_seed: Optional[int]`

#### TrainingReportGenerator

Generator for comprehensive training reports with visualizations.

## Methods

### to_dict

Convert metrics to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### to_dict

Convert architecture info to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### to_dict

Convert configuration to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### generate_comprehensive_report

Generate comprehensive training report with all metrics and visualizations.

Args:
    training_job_id: ID of the training job
    training_result: Training results dictionary
    model_info: Optional model information
    db: Database session
    
Returns:
    Dictionary containing complete report information
    
Raises:
    TrainingReportError: If report generation fails

**Signature**: `generate_comprehensive_report(training_job_id: int, training_result: Dict[str, Any], model_info: Optional[Dict], db: Session) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `training_job_id` | `int` | Parameter description |
| `training_result` | `Dict[str, Any]` | Parameter description |
| `model_info` | `Optional[Dict]` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict[str, Any]`

## Dependencies

- `app.models.database.TrainingJob`
- `app.models.database.TrainingReport`

## Defined Interfaces

### TrainingMetricsInterface

**Type**: Protocol
**Implemented by**: TrainingMetrics

**Methods**:

- `to_dict() -> Dict[str, Any]`

### ModelArchitectureInfoInterface

**Type**: Protocol
**Implemented by**: ModelArchitectureInfo

**Methods**:

- `to_dict() -> Dict[str, Any]`

### TrainingConfigurationInterface

**Type**: Protocol
**Implemented by**: TrainingConfiguration

**Methods**:

- `to_dict() -> Dict[str, Any]`

### TrainingReportGeneratorInterface

**Type**: Protocol
**Implemented by**: TrainingReportGenerator

**Methods**:

- `generate_comprehensive_report(training_job_id: int, training_result: Dict[str, Any], model_info: Optional[Dict], db: Session) -> Dict[str, Any]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/training_report_service.py`
- Last validated: 2025-12-16 15:47:04

