# Model Training Guide

This guide explains how to train the anomaly detection models for the Children's Drawing Anomaly Detection System.

## Overview

The system uses a two-stage approach:
1. **Feature Extraction**: Vision Transformer (ViT) generates 769-dimensional embeddings
2. **Anomaly Detection**: Age-specific autoencoder models detect anomalies via reconstruction loss

## Quick Start

### Automated Training (Recommended)

```bash
# 1. Ensure backend is running
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &

# 2. Run complete training pipeline
python train_models.py
```

This script will:
- Fetch all drawings from the database
- Generate ViT embeddings for each drawing
- Train autoencoder models for age groups
- Set up anomaly detection thresholds

### Manual Training Steps

If you need more control over the process:

#### Step 1: Generate Sample Data (Optional)

```bash
# Create 95 synthetic drawings across age groups
python create_sample_drawings.py

# Upload them to the system
python upload_sample_drawings.py
```

#### Step 2: Generate Embeddings

```bash
# For each drawing, generate ViT embeddings
for i in {1..95}; do
  curl -X POST "http://localhost:8000/api/v1/analysis/embeddings/$i"
done
```

#### Step 3: Train Models

```bash
# Train early childhood model (3-6 years)
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 3.0, "age_max": 6.0, "min_samples": 10}'

# Train middle childhood model (6-9 years)
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 6.0, "age_max": 9.0, "min_samples": 10}'

# Train late childhood model (9-12 years)
curl -X POST "http://localhost:8000/api/v1/models/train" \
     -H "Content-Type: application/json" \
     -d '{"age_min": 9.0, "age_max": 12.0, "min_samples": 10}'
```

#### Step 4: Monitor Training

```bash
# Check overall system status
curl "http://localhost:8000/api/v1/models/status"

# View trained models
curl "http://localhost:8000/api/v1/models/age-groups"
```

## Training Requirements

### Data Requirements

- **Minimum samples per age group**: 10 drawings
- **Recommended samples**: 50+ drawings per age group
- **Age range**: 3-12 years
- **Image formats**: PNG, JPEG, BMP
- **Image size**: Automatically resized to 224x224 for ViT

### System Requirements

- **Memory**: 4GB+ RAM (8GB+ recommended)
- **Storage**: 1GB+ for models and embeddings
- **GPU**: Optional (Apple Silicon MPS or CUDA supported)
- **Python**: 3.11+ with virtual environment

## Model Architecture

### Vision Transformer (ViT)
- **Model**: `google/vit-base-patch16-224`
- **Input**: 224x224 RGB images
- **Output**: 768-dimensional embeddings
- **Augmentation**: Age information concatenated (769 dimensions total)

### Autoencoder Models
- **Architecture**: Encoder-decoder with bottleneck
- **Input**: 769-dimensional ViT embeddings
- **Loss**: Mean Squared Error (reconstruction loss)
- **Training**: Age-specific models for better performance

### Anomaly Detection
- **Method**: Reconstruction loss thresholding
- **Threshold**: 95th percentile of training reconstruction losses
- **Scoring**: Normalized anomaly scores (0-1 range)

## Training Output

After successful training, you should see:

```json
{
  "total_models": 3,
  "active_models": 3,
  "training_models": 0,
  "failed_models": 0,
  "total_drawings": 95,
  "total_analyses": 0,
  "system_status": "ready"
}
```

### Model Information

Each trained model includes:
- **Age range**: Min/max ages for the model
- **Sample count**: Number of training samples
- **Threshold**: Anomaly detection threshold
- **Status**: Training status (active/insufficient_data/failed)
- **Created timestamp**: When the model was trained

## Validation

### Test Anomaly Detection

```bash
# Analyze a drawing to test the system
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/1"
```

Expected response:
```json
{
  "drawing": {...},
  "analysis": {
    "anomaly_score": 0.052,
    "normalized_score": 0.052,
    "is_anomaly": false,
    "confidence": 0.60,
    "age_group": "6.0-9.0"
  }
}
```

### Web Interface Validation

1. Open http://localhost:5173
2. Check dashboard shows:
   - Total drawings count
   - Active models count
   - Age distribution chart
   - Recent analyses

## Troubleshooting

### Common Training Issues

1. **"No embeddings found for age range"**
   - Generate embeddings first using the embeddings endpoint
   - Ensure drawings exist in the specified age range

2. **"Insufficient samples for training"**
   - Upload more drawings for the age group
   - Reduce `min_samples` parameter (minimum 10)

3. **"Vision Transformer preprocessing failed"**
   - Check NumPy version (should be <2.0.0)
   - Verify PyTorch and transformers installation

4. **Training jobs fail silently**
   - Check backend logs for detailed error messages
   - Verify database connectivity
   - Ensure sufficient memory available

### Performance Optimization

- **GPU Acceleration**: System automatically detects and uses available GPU/MPS
- **Batch Processing**: Embeddings generated in batches for efficiency
- **Model Caching**: ViT model cached after first load
- **Embedding Caching**: Generated embeddings stored in database

## Advanced Configuration

### Custom Age Groups

Modify the age groups in `train_models.py`:

```python
age_groups = [
    (3.0, 5.0, "Preschool"),
    (5.0, 8.0, "Early school"),
    (8.0, 12.0, "Late school")
]
```

### Threshold Adjustment

```bash
# Update threshold percentile for all models
curl -X PUT "http://localhost:8000/api/v1/config/threshold?percentile=90.0"
```

### Model Retraining

```bash
# Reset system and retrain
curl -X POST "http://localhost:8000/api/v1/config/reset?confirm=true"
python train_models.py
```

## Production Deployment

For production use:
1. Use real drawing data instead of synthetic samples
2. Increase minimum samples per age group (50+)
3. Implement proper data validation and quality checks
4. Set up monitoring and alerting for model performance
5. Consider distributed training for large datasets