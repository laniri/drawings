# Offline Training Tutorial for Children's Drawing Anomaly Detection System

This comprehensive tutorial covers how to perform offline training of autoencoder models for anomaly detection in children's drawings. The system supports both automated and manual training workflows with comprehensive monitoring and configuration options.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Fast Offline Training (Recommended)](#fast-offline-training-recommended)
4. [Quick Start (Automated Training)](#quick-start-automated-training)
5. [Manual Training Workflow](#manual-training-workflow)
6. [Advanced Configuration](#advanced-configuration)
7. [Local Training Environment](#local-training-environment)
8. [Monitoring and Progress Tracking](#monitoring-and-progress-tracking)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

The offline training system uses Vision Transformer (ViT) embeddings and autoencoder models to detect anomalies in children's drawings. The system:

- **Generates ViT embeddings** from drawing images (769-dimensional vectors)
- **Trains age-specific autoencoder models** for reconstruction-based anomaly detection
- **Supports local training** with GPU/CPU detection and optimization
- **Provides comprehensive monitoring** with real-time progress tracking
- **Handles data sufficiency analysis** and automatic age group creation

### Architecture Overview

```
Drawing Images → ViT Embeddings → Age-Grouped Datasets → Autoencoder Training → Anomaly Detection Models
```

## Prerequisites

### System Requirements

- **Python 3.11+** with virtual environment
- **PyTorch** with CUDA support (optional but recommended)
- **Minimum 8GB RAM** (16GB+ recommended for large datasets)
- **GPU with 4GB+ VRAM** (optional, will use CPU if unavailable)

### Software Dependencies

```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# All dependencies should already be installed via requirements.txt
pip install -r requirements-dev.txt
```

### Data Requirements

- **Minimum 10 drawings per age group** for training
- **Recommended 50+ drawings per age group** for robust models
- **Age range 3-12 years** (system supports 2-18 years)
- **Supported formats**: PNG, JPEG, BMP (max 10MB per file)

## Fast Offline Training (Recommended)

**⚡ This is the fastest and most reliable way to train models when you have existing embeddings.**

The offline training script bypasses the web server entirely and trains models directly against the database. This approach:

- **Runs 3-5x faster** than server-based training
- **Never times out** or blocks the web server
- **Uses optimized training parameters** (50 epochs vs 100)
- **Provides real-time progress updates**
- **Works with existing embeddings**

### Prerequisites for Offline Training

1. **Database with drawings and embeddings** (run `train_models.py --skip-training` first if needed)
2. **Server must be stopped** (offline training accesses database directly)

### Step 1: Stop the Server (if running)

```bash
# Find and kill any running uvicorn processes
ps aux | grep uvicorn
kill <process_id>

# Or use Ctrl+C if running in terminal
```

### Step 2: Run Offline Training

```bash
# Activate virtual environment (CRITICAL - must use project's venv)
source venv/bin/activate

# Verify you're using the correct Python
which python  # Should show: /Users/itay/Desktop/drawings/venv/bin/python

# Train all default age groups (3-6, 6-9, 9-12 years)
python train_models_offline.py

# Or specify custom age groups
python train_models_offline.py --age-groups "3-6,6-9,9-12"

# Set minimum samples per group (default: 10)
python train_models_offline.py --min-samples 20

# Force retrain existing models
python train_models_offline.py --force
```

**Important**: You must use the project's virtual environment (`venv/bin/activate`) as it contains all required dependencies.

### Step 3: Verify Results

```bash
# Start the server to test trained models
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Check model status via API
curl "http://localhost:8000/api/v1/models/status"

# Test anomaly detection
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/1"
```

### Expected Output

```
=== Offline Model Training ===

✓ Database connection established
Training 3 age groups: [(3.0, 6.0), (6.0, 9.0), (9.0, 12.0)]

Training model for age group 3.0-6.0 years...
  Fetching embeddings for age group 3.0-6.0...
  Found 12543 drawings in age range
  Retrieved 12543 embeddings (0 missing)
  ✓ Using 12543 embeddings for training
  ⏳ Starting training with 50 epochs...
  ✓ Training completed in 45.2 seconds
  ✓ Model ID: 1
  ✓ Threshold: 0.0234
  ✓ Final validation loss: 0.018456

Training model for age group 6.0-9.0 years...
  [... similar output ...]

=== Training Summary ===
Total time: 156.3 seconds
Successful models: 3
Failed models: 0

✓ Successfully trained models:
  - Age 3.0-6.0: Model 1 (threshold: 0.0234)
  - Age 6.0-9.0: Model 2 (threshold: 0.0198)
  - Age 9.0-12.0: Model 3 (threshold: 0.0267)

=== Training Complete ===
You can now start the server and test the trained models:
  source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Advantages of Offline Training

- **Speed**: 3-5x faster than server-based training
- **Reliability**: No timeouts or server blocking
- **Resource efficiency**: Direct database access
- **Better error handling**: Clear progress and error messages
- **Optimized parameters**: Tuned for faster convergence

## Quick Start (Automated Training)

### Step 1: Start the Backend Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the FastAPI backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Generate Sample Data (Optional)

If you don't have real drawing data, generate sample drawings:

```bash
# Generate 95 sample drawings across age groups
python create_sample_drawings.py

# Upload sample drawings to the system
python upload_sample_drawings.py
```

### Step 3: Run Automated Training

```bash
# Complete automated training workflow
python train_models.py
```

This script will:
1. Fetch all drawings from the database
2. Generate ViT embeddings for each drawing
3. Train autoencoder models for age groups (3-6, 6-9, 9-12 years)
4. Set up anomaly detection thresholds
5. Test the trained system

### Step 4: Verify Training Results

```bash
# Check system status
curl "http://localhost:8000/api/v1/models/status"

# List trained models
curl "http://localhost:8000/api/v1/models/age-groups"

# Test anomaly detection
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/1"
```

## Manual Training Workflow

### Step 1: Data Preparation and Analysis

#### Check Data Availability

```bash
# Analyze data sufficiency for all age groups
curl "http://localhost:8000/api/v1/models/data-sufficiency/analyze"

# Analyze specific age group
curl "http://localhost:8000/api/v1/models/data-sufficiency/age-group/3.0/6.0"
```

#### Upload Drawings (if needed)

```python
import requests

# Upload a single drawing
url = "http://localhost:8000/api/v1/drawings/upload"
files = {'file': open('drawing.png', 'rb')}
data = {
    'age_years': 5.5,
    'subject': 'house',
    'expert_label': 'normal'
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### Step 2: Generate Embeddings

Generate ViT embeddings for all drawings:

```bash
# Generate embeddings for a specific drawing
curl -X POST "http://localhost:8000/api/v1/analysis/embeddings/1"

# Or use the batch processing script
python -c "
import requests
import json

# Get all drawings
response = requests.get('http://localhost:8000/api/v1/drawings/')
drawings = response.json()['drawings']

# Generate embeddings for each
for drawing in drawings:
    drawing_id = drawing['id']
    response = requests.post(f'http://localhost:8000/api/v1/analysis/embeddings/{drawing_id}')
    result = response.json()
    print(f'Drawing {drawing_id}: {result[\"status\"]}')
"
```

### Step 3: Configure Training Parameters

Create a custom training configuration:

```python
# training_config.py
from app.services.training_config import TrainingConfig, ModelConfig, OptimizerConfig, DataConfig

# Create custom configuration
config = TrainingConfig(
    job_name="custom_autoencoder_training",
    epochs=150,
    model=ModelConfig(
        hidden_dims=[512, 256, 128, 64],
        latent_dim=32,
        dropout_rate=0.1
    ),
    optimizer=OptimizerConfig(
        learning_rate=0.001,
        weight_decay=1e-5
    ),
    data=DataConfig(
        batch_size=32,
        validation_split=0.2
    ),
    early_stopping_patience=15,
    save_plots=True
)

# Save configuration
from app.services.training_config import get_training_config_manager
config_manager = get_training_config_manager()
config_manager.save_config(config, "custom_training_config.yaml")
```

### Step 4: Train Models for Specific Age Groups

```bash
# Train model for specific age range
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{
       "age_min": 3.0,
       "age_max": 6.0,
       "min_samples": 10
     }'

# Response includes job_id for tracking
# {
#   "job_id": "uuid-string",
#   "age_range": [3.0, 6.0],
#   "status": "training",
#   "sample_count": 25,
#   "progress_url": "/api/v1/models/training/{job_id}/status"
# }
```

### Step 5: Monitor Training Progress

```bash
# Check training status
curl "http://localhost:8000/api/v1/models/training/{job_id}/status"

# Monitor in real-time with watch
watch -n 5 'curl -s "http://localhost:8000/api/v1/models/training/{job_id}/status" | jq'
```

### Step 6: Auto-Create Age Groups (Alternative)

Let the system automatically determine optimal age groups:

```bash
# Auto-create age groups based on data distribution
curl -X POST "http://localhost:8000/api/v1/models/auto-create"

# Force recreation of existing models
curl -X POST "http://localhost:8000/api/v1/models/auto-create?force_recreate=true"
```

## Advanced Configuration

### Custom Training Configuration

Create detailed training configurations using the configuration system:

```python
from app.services.training_config import (
    TrainingConfig, ModelConfig, OptimizerConfig, 
    SchedulerConfig, DataConfig, TrainingEnvironment,
    OptimizerType, SchedulerType
)

# Advanced configuration
advanced_config = TrainingConfig(
    job_name="advanced_autoencoder_training",
    experiment_name="drawing_anomaly_detection_v2",
    environment=TrainingEnvironment.LOCAL,
    device="auto",  # "auto", "cpu", "cuda", "mps"
    mixed_precision=True,  # Enable for faster training on compatible GPUs
    
    # Model architecture
    model=ModelConfig(
        encoder_dim=512,
        hidden_dims=[512, 256, 128, 64, 32],
        latent_dim=16,
        dropout_rate=0.15,
        activation="relu",
        batch_norm=True
    ),
    
    # Training parameters
    epochs=200,
    early_stopping_patience=20,
    early_stopping_min_delta=1e-5,
    gradient_clip_norm=1.0,
    
    # Optimization
    optimizer=OptimizerConfig(
        type=OptimizerType.ADAMW,
        learning_rate=0.0005,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    ),
    
    # Learning rate scheduling
    scheduler=SchedulerConfig(
        type=SchedulerType.COSINE,
        T_max=200
    ),
    
    # Data handling
    data=DataConfig(
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        train_split=0.7,
        validation_split=0.2,
        test_split=0.1,
        stratify_by_age=True,
        age_group_size=1.0
    ),
    
    # Monitoring and output
    validation_frequency=1,
    checkpoint_frequency=10,
    save_best_only=True,
    log_frequency=5,
    save_plots=True,
    plot_frequency=10,
    
    # Output paths
    output_dir="outputs/advanced_training",
    model_save_path="models/advanced",
    log_save_path="logs/advanced"
)
```

### Parameter Sweeps

Perform hyperparameter optimization:

```python
from app.services.training_config import ParameterSweepConfig

# Define parameter sweep
sweep_config = ParameterSweepConfig(
    learning_rates=[0.0001, 0.0005, 0.001, 0.005],
    batch_sizes=[16, 32, 64],
    hidden_dims=[
        [256, 128, 64],
        [512, 256, 128, 64],
        [1024, 512, 256, 128]
    ],
    latent_dims=[16, 32, 64],
    dropout_rates=[0.0, 0.1, 0.2],
    max_trials=20,
    optimization_metric="validation_loss",
    optimization_direction="minimize"
)

# Generate sweep configurations
config_manager = get_training_config_manager()
sweep_configs = config_manager.create_parameter_sweep(base_config, sweep_config)

print(f"Generated {len(sweep_configs)} configurations for parameter sweep")
```

## Local Training Environment

### Device Detection and Optimization

The system automatically detects and optimizes for available hardware:

```python
from app.services.local_training_environment import get_local_training_environment

# Get training environment
training_env = get_local_training_environment()

# Check environment info
env_info = training_env.get_environment_info()
print(f"Device: {env_info['device_info']['type']}")
print(f"Memory: {env_info['device_info'].get('total_memory_gb', 'N/A')} GB")
```

### Manual Local Training

For complete control over the training process:

```python
from app.services.local_training_environment import LocalTrainingEnvironment
from app.services.training_config import TrainingConfig
from app.core.database import get_db

# Initialize training environment
training_env = LocalTrainingEnvironment()

# Prepare training data from dataset folder
train_embeddings, val_embeddings, test_embeddings = training_env.prepare_training_data(
    dataset_folder="sample_drawings",
    metadata_file="metadata.json",
    config=training_config
)

# Start training job
db = next(get_db())
job_id = training_env.start_training_job(
    config=training_config,
    train_embeddings=train_embeddings,
    val_embeddings=val_embeddings,
    db=db
)

print(f"Training job started with ID: {job_id}")

# Monitor progress
while True:
    status = training_env.get_job_status(job_id, db)
    print(f"Status: {status['status']}")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(10)
```

### Custom Dataset Preparation

Prepare datasets with custom splitting strategies:

```python
from app.services.dataset_preparation import DatasetPreparationService, SplitConfig

# Create custom split configuration
split_config = SplitConfig(
    train_ratio=0.8,
    validation_ratio=0.15,
    test_ratio=0.05,
    stratify_by_age=True,
    age_group_size=0.5,  # 6-month age groups
    random_seed=42
)

# Prepare dataset
dataset_service = DatasetPreparationService()
dataset_split = dataset_service.prepare_dataset(
    dataset_folder="drawings",
    metadata_file="metadata.csv",
    split_config=split_config
)

print(f"Train: {dataset_split.train_count}")
print(f"Validation: {dataset_split.validation_count}")
print(f"Test: {dataset_split.test_count}")
```

## Monitoring and Progress Tracking

### Real-time Training Monitoring

Monitor training progress in real-time:

```python
from app.services.local_training_environment import TrainingProgressMonitor

# Create progress monitor
monitor = TrainingProgressMonitor(job_id=1, total_epochs=100)

# Add custom callback
def log_progress(progress):
    print(f"Epoch {progress.epoch}/{progress.total_epochs}: "
          f"Loss {progress.train_loss:.6f}, "
          f"Progress {progress.epoch_progress:.1f}%")

monitor.add_callback(log_progress)

# Monitor will be called automatically during training
```

### Training History and Metrics

Access detailed training history:

```bash
# Get training history for a job
curl "http://localhost:8000/api/v1/models/training/{job_id}/history"

# Get model performance metrics
curl "http://localhost:8000/api/v1/models/{model_id}/metrics"
```

### System Health Monitoring

Monitor system health and resource usage:

```python
from app.services.health_monitor import get_health_monitor

health_monitor = get_health_monitor()

# Get system health status
health_status = health_monitor.get_system_health()
print(f"CPU Usage: {health_status['cpu_percent']}%")
print(f"Memory Usage: {health_status['memory_percent']}%")
print(f"GPU Usage: {health_status.get('gpu_percent', 'N/A')}%")

# Get training resource usage
resource_usage = health_monitor.get_training_resource_usage()
print(f"Active training jobs: {resource_usage['active_jobs']}")
print(f"Memory per job: {resource_usage['memory_per_job_mb']} MB")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Server Timeout During Training

**Problem**: Server becomes unresponsive during training, `train_models.py` times out with "Read timed out" error.

**Cause**: Training process blocks the FastAPI server, making it unresponsive to new requests.

**Solutions**:

```bash
# Solution 1: Use offline training (RECOMMENDED)
# Stop the server first
ps aux | grep uvicorn
kill <process_id>

# Run offline training
source venv/bin/activate
python train_models_offline.py

# Solution 2: Restart server and use smaller batches
# Kill blocked server
pkill -f uvicorn

# Restart server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Use train_models.py with existing embeddings
python train_models.py --skip-embeddings

# Solution 3: Train one age group at a time via API
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 3.0, "age_max": 6.0, "min_samples": 10}'
```

#### 2. Insufficient Data Error

```bash
# Error: "Insufficient data: X drawings available, need at least Y"

# Solution 1: Lower minimum sample requirement
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{
       "age_min": 3.0,
       "age_max": 6.0,
       "min_samples": 5
     }'

# Solution 2: Merge age groups
curl -X POST "http://localhost:8000/api/v1/models/data-sufficiency/merge-age-groups" \
     -H "Content-Type: application/json" \
     -d '{
       "original_groups": [[3.0, 4.0], [4.0, 5.0], [5.0, 6.0]],
       "merged_group": [3.0, 6.0]
     }'

# Solution 3: Generate more sample data
python create_sample_drawings.py
python upload_sample_drawings.py
```

#### 2. CUDA Out of Memory

```python
# Reduce batch size in training configuration
config.data.batch_size = 16  # Reduce from 32

# Enable gradient checkpointing (if available)
config.gradient_checkpointing = True

# Use mixed precision training
config.mixed_precision = True
```

#### 3. Training Convergence Issues

```python
# Adjust learning rate
config.optimizer.learning_rate = 0.0001  # Reduce learning rate

# Increase patience for early stopping
config.early_stopping_patience = 20

# Add learning rate scheduling
config.scheduler.type = SchedulerType.STEP
config.scheduler.step_size = 30
config.scheduler.gamma = 0.5
```

#### 4. Embedding Generation Failures

```bash
# Check embedding service status
curl "http://localhost:8000/api/v1/analysis/embedding-service/status"

# Reinitialize embedding service
curl -X POST "http://localhost:8000/api/v1/analysis/embedding-service/initialize"

# Check individual drawing processing
curl -X POST "http://localhost:8000/api/v1/analysis/embeddings/1" -v
```

### Debug Mode Training

Enable detailed logging for debugging:

```python
import logging

# Enable debug logging
logging.getLogger("app.services.local_training_environment").setLevel(logging.DEBUG)
logging.getLogger("app.services.model_manager").setLevel(logging.DEBUG)

# Run training with detailed logs
python train_models.py
```

### Performance Optimization

#### GPU Optimization

```python
# Optimize for GPU training
config.data.pin_memory = True
config.data.num_workers = 4
config.mixed_precision = True

# Enable cuDNN optimizations
import torch
torch.backends.cudnn.benchmark = True
```

#### CPU Optimization

```python
# Optimize for CPU training
import torch
torch.set_num_threads(8)  # Set based on CPU cores

config.data.num_workers = 0  # Disable multiprocessing on CPU
config.data.batch_size = 16  # Smaller batch size for CPU
```

## Best Practices

### 1. Data Quality

- **Ensure consistent image quality** (resolution, format, lighting)
- **Validate age labels** for accuracy
- **Balance age distribution** across groups
- **Remove corrupted or invalid images**

### 2. Training Configuration

- **Start with default configurations** and adjust based on results
- **Use validation split** of 20-30% for reliable evaluation
- **Enable early stopping** to prevent overfitting
- **Save training plots** for analysis

### 3. Model Management

- **Train separate models** for different age groups
- **Use appropriate thresholds** (95th percentile recommended)
- **Regularly retrain models** with new data
- **Monitor model performance** over time

### 4. Resource Management

- **Monitor system resources** during training
- **Use GPU when available** for faster training
- **Adjust batch size** based on available memory
- **Clean up model cache** periodically

### 5. Validation and Testing

- **Test models** on held-out data
- **Validate anomaly detection** with known cases
- **Compare model performance** across age groups
- **Document training parameters** and results

### Example Complete Workflow

```bash
#!/bin/bash
# complete_training_workflow.sh

echo "Starting complete offline training workflow..."

# 1. Start backend
echo "Starting backend server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 10

# 2. Generate sample data (if needed)
echo "Generating sample data..."
python create_sample_drawings.py
python upload_sample_drawings.py

# 3. Analyze data sufficiency
echo "Analyzing data sufficiency..."
curl -s "http://localhost:8000/api/v1/models/data-sufficiency/analyze" | jq

# 4. Run automated training
echo "Starting automated training..."
python train_models.py

# 5. Verify results
echo "Verifying training results..."
curl -s "http://localhost:8000/api/v1/models/status" | jq
curl -s "http://localhost:8000/api/v1/models/age-groups" | jq

# 6. Test anomaly detection
echo "Testing anomaly detection..."
curl -s -X POST "http://localhost:8000/api/v1/analysis/analyze/1" | jq

echo "Training workflow completed successfully!"

# Cleanup
kill $BACKEND_PID
```

This tutorial provides comprehensive coverage of offline training capabilities in the children's drawing anomaly detection system. The system is designed to be flexible and robust, supporting both automated workflows for quick setup and manual configuration for advanced use cases.