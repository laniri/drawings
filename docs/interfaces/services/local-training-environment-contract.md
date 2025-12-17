# Local Training Environment Contract

## Overview
Service contract for Local Training Environment (service)

**Source File**: `app/services/local_training_environment.py`

## Interface Specification

### Classes

#### LocalTrainingError

Base exception for local training errors.

**Inherits from**: Exception

#### DeviceDetectionError

Raised when device detection fails.

**Inherits from**: LocalTrainingError

#### TrainingProgressError

Raised when training progress monitoring fails.

**Inherits from**: LocalTrainingError

#### TrainingProgress

Container for training progress information.

**Attributes**:

- `job_id: int`
- `epoch: int`
- `total_epochs: int`
- `batch: int`
- `total_batches: int`
- `train_loss: float`
- `val_loss: Optional[float]`
- `learning_rate: float`
- `elapsed_time: float`
- `estimated_remaining: Optional[float]`
- `memory_usage: Optional[Dict]`

#### DeviceManager

Enhanced device manager with detailed GPU/CPU detection and monitoring.

#### TrainingProgressMonitor

Monitor and log training progress with real-time updates.

#### LocalTrainingEnvironment

Local training environment with comprehensive monitoring and logging.

#### EnhancedAutoencoderTrainer

Enhanced autoencoder trainer with progress monitoring and device management.

**Inherits from**: AutoencoderTrainer

## Methods

### epoch_progress

Get epoch progress as percentage.

**Signature**: `epoch_progress() -> float`

**Type**: Property

**Returns**: `float`

### batch_progress

Get batch progress within current epoch as percentage.

**Signature**: `batch_progress() -> float`

**Type**: Property

**Returns**: `float`

### device

Get the current device.

**Signature**: `device() -> torch.device`

**Type**: Property

**Returns**: `torch.device`

### device_info

Get device information.

**Signature**: `device_info() -> Dict`

**Type**: Property

**Returns**: `Dict`

### get_memory_usage

Get current memory usage if available.

**Signature**: `get_memory_usage() -> Optional[Dict]`

**Returns**: `Optional[Dict]`

### clear_cache

Clear device cache if applicable.

**Signature**: `clear_cache() -> None`

**Returns**: `None`

### optimize_for_training

Apply device-specific optimizations for training.

**Signature**: `optimize_for_training() -> None`

**Returns**: `None`

### add_callback

Add a callback function to be called on progress updates.

**Signature**: `add_callback(callback: Callable[<ast.List object at 0x110434a90>, <ast.Constant object at 0x110435150>]) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `callback` | `Callable[<ast.List object at 0x110434a90>, <ast.Constant object at 0x110435150>]` | Parameter description |

**Returns**: `None`

### start_epoch

Mark the start of a new epoch.

**Signature**: `start_epoch(epoch: int, total_batches: int) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `epoch` | `int` | Parameter description |
| `total_batches` | `int` | Parameter description |

**Returns**: `None`

### update_batch

Update progress for current batch.

**Signature**: `update_batch(batch: int, train_loss: float, learning_rate: float) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `batch` | `int` | Parameter description |
| `train_loss` | `float` | Parameter description |
| `learning_rate` | `float` | Parameter description |

**Returns**: `None`

### update_epoch

Update progress at end of epoch.

**Signature**: `update_epoch(epoch: int, train_loss: float, val_loss: float) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `epoch` | `int` | Parameter description |
| `train_loss` | `float` | Parameter description |
| `val_loss` | `float` | Parameter description |

**Returns**: `None`

### get_latest_progress

Get the latest progress update.

**Signature**: `get_latest_progress() -> Optional[TrainingProgress]`

**Returns**: `Optional[TrainingProgress]`

### get_history

Get complete training history.

**Signature**: `get_history() -> List[TrainingProgress]`

**Returns**: `List[TrainingProgress]`

### stop

Stop monitoring.

**Signature**: `stop() -> None`

**Returns**: `None`

### get_environment_info

Get comprehensive environment information.

**Signature**: `get_environment_info() -> Dict`

**Returns**: `Dict`

### prepare_training_data

Prepare training data from dataset folder.

Args:
    dataset_folder: Path to folder containing images
    metadata_file: Path to metadata file
    config: Training configuration
    
Returns:
    Tuple of (train_embeddings, val_embeddings, test_embeddings)

**Signature**: `prepare_training_data(dataset_folder: str, metadata_file: str, config: TrainingConfig) -> Tuple[<ast.Attribute object at 0x110479990>, <ast.Attribute object at 0x110479890>, <ast.Attribute object at 0x110479790>]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `dataset_folder` | `str` | Parameter description |
| `metadata_file` | `str` | Parameter description |
| `config` | `TrainingConfig` | Parameter description |

**Returns**: `Tuple[<ast.Attribute object at 0x110479990>, <ast.Attribute object at 0x110479890>, <ast.Attribute object at 0x110479790>]`

### start_training_job

Start a local training job.

Args:
    config: Training configuration
    train_embeddings: Training embeddings
    val_embeddings: Validation embeddings
    db: Database session
    
Returns:
    Training job ID

**Signature**: `start_training_job(config: TrainingConfig, train_embeddings: np.ndarray, val_embeddings: np.ndarray, db: Session) -> int`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config` | `TrainingConfig` | Parameter description |
| `train_embeddings` | `np.ndarray` | Parameter description |
| `val_embeddings` | `np.ndarray` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `int`

### get_job_status

Get status of a training job.

**Signature**: `get_job_status(job_id: int, db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `job_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### list_training_jobs

List all training jobs.

**Signature**: `list_training_jobs(db: Session) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `List[Dict]`

### cancel_training_job

Cancel an active training job.

**Signature**: `cancel_training_job(job_id: int, db: Session) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `job_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `bool`

### train

Enhanced training with progress monitoring and validation split.

Args:
    train_embeddings: Training embeddings
    val_embeddings: Validation embeddings
    
Returns:
    Dictionary containing training results and metrics

**Signature**: `train(train_embeddings: np.ndarray, val_embeddings: np.ndarray) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `train_embeddings` | `np.ndarray` | Parameter description |
| `val_embeddings` | `np.ndarray` | Parameter description |

**Returns**: `Dict`

## Dependencies

- `app.models.database.TrainingJob`
- `app.models.database.TrainingReport`
- `app.services.training_config.TrainingConfig`
- `app.services.training_config.TrainingConfigManager`
- `app.services.dataset_preparation.DatasetPreparationService`
- `app.services.dataset_preparation.SplitConfig`
- `app.services.embedding_service.get_embedding_service`
- `app.services.model_manager.AutoencoderModel`
- `app.services.model_manager.AutoencoderTrainer`
- `app.services.model_manager.EarlyStopping`

## Defined Interfaces

### TrainingProgressInterface

**Type**: Protocol
**Implemented by**: TrainingProgress

**Methods**:

- `epoch_progress() -> float`
- `batch_progress() -> float`

### DeviceManagerInterface

**Type**: Protocol
**Implemented by**: DeviceManager

**Methods**:

- `device() -> torch.device`
- `device_info() -> Dict`
- `get_memory_usage() -> Optional[Dict]`
- `clear_cache() -> None`
- `optimize_for_training() -> None`

### TrainingProgressMonitorInterface

**Type**: Protocol
**Implemented by**: TrainingProgressMonitor

**Methods**:

- `add_callback(callback: Callable[<ast.List object at 0x110434a90>, <ast.Constant object at 0x110435150>]) -> None`
- `start_epoch(epoch: int, total_batches: int) -> None`
- `update_batch(batch: int, train_loss: float, learning_rate: float) -> None`
- `update_epoch(epoch: int, train_loss: float, val_loss: float) -> None`
- `get_latest_progress() -> Optional[TrainingProgress]`
- `get_history() -> List[TrainingProgress]`
- `stop() -> None`

### LocalTrainingEnvironmentInterface

**Type**: Protocol
**Implemented by**: LocalTrainingEnvironment

**Methods**:

- `get_environment_info() -> Dict`
- `prepare_training_data(dataset_folder: str, metadata_file: str, config: TrainingConfig) -> Tuple[<ast.Attribute object at 0x110479990>, <ast.Attribute object at 0x110479890>, <ast.Attribute object at 0x110479790>]`
- `start_training_job(config: TrainingConfig, train_embeddings: np.ndarray, val_embeddings: np.ndarray, db: Session) -> int`
- `get_job_status(job_id: int, db: Session) -> Dict`
- `list_training_jobs(db: Session) -> List[Dict]`
- `cancel_training_job(job_id: int, db: Session) -> bool`

### EnhancedAutoencoderTrainerInterface

**Type**: Protocol
**Implemented by**: EnhancedAutoencoderTrainer

**Methods**:

- `train(train_embeddings: np.ndarray, val_embeddings: np.ndarray) -> Dict`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/local_training_environment.py`
- Last validated: 2025-12-16 15:47:04

