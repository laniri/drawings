# Local Training Environment Service

Local Training Environment Service for Children's Drawing Anomaly Detection System

This module provides local autoencoder training with PyTorch, GPU/CPU device detection,
and comprehensive training progress monitoring and logging.

## Class: LocalTrainingError

Base exception for local training errors.

## Class: DeviceDetectionError

Raised when device detection fails.

## Class: TrainingProgressError

Raised when training progress monitoring fails.

## Class: TrainingProgress

Container for training progress information.

### epoch_progress

Get epoch progress as percentage.

**Signature**: `epoch_progress()`

### batch_progress

Get batch progress within current epoch as percentage.

**Signature**: `batch_progress()`

## Class: DeviceManager

Enhanced device manager with detailed GPU/CPU detection and monitoring.

### device

Get the current device.

**Signature**: `device()`

### device_info

Get device information.

**Signature**: `device_info()`

### get_memory_usage

Get current memory usage if available.

**Signature**: `get_memory_usage()`

### clear_cache

Clear device cache if applicable.

**Signature**: `clear_cache()`

### optimize_for_training

Apply device-specific optimizations for training.

**Signature**: `optimize_for_training()`

## Class: TrainingProgressMonitor

Monitor and log training progress with real-time updates.

### add_callback

Add a callback function to be called on progress updates.

**Signature**: `add_callback(callback)`

### start_epoch

Mark the start of a new epoch.

**Signature**: `start_epoch(epoch, total_batches)`

### update_batch

Update progress for current batch.

**Signature**: `update_batch(batch, train_loss, learning_rate)`

### update_epoch

Update progress at end of epoch.

**Signature**: `update_epoch(epoch, train_loss, val_loss)`

### get_latest_progress

Get the latest progress update.

**Signature**: `get_latest_progress()`

### get_history

Get complete training history.

**Signature**: `get_history()`

### stop

Stop monitoring.

**Signature**: `stop()`

## Class: LocalTrainingEnvironment

Local training environment with comprehensive monitoring and logging.

### get_environment_info

Get comprehensive environment information.

**Signature**: `get_environment_info()`

### prepare_training_data

Prepare training data from dataset folder.

Args:
    dataset_folder: Path to folder containing images
    metadata_file: Path to metadata file
    config: Training configuration
    
Returns:
    Tuple of (train_embeddings, val_embeddings, test_embeddings)

**Signature**: `prepare_training_data(dataset_folder, metadata_file, config)`

### start_training_job

Start a local training job.

Args:
    config: Training configuration
    train_embeddings: Training embeddings
    val_embeddings: Validation embeddings
    db: Database session
    
Returns:
    Training job ID

**Signature**: `start_training_job(config, train_embeddings, val_embeddings, db)`

### get_job_status

Get status of a training job.

**Signature**: `get_job_status(job_id, db)`

### list_training_jobs

List all training jobs.

**Signature**: `list_training_jobs(db)`

### cancel_training_job

Cancel an active training job.

**Signature**: `cancel_training_job(job_id, db)`

## Class: EnhancedAutoencoderTrainer

Enhanced autoencoder trainer with progress monitoring and device management.

### train

Enhanced training with progress monitoring and validation split.

Args:
    train_embeddings: Training embeddings
    val_embeddings: Validation embeddings
    
Returns:
    Dictionary containing training results and metrics

**Signature**: `train(train_embeddings, val_embeddings)`

