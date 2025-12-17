# Model Manager Service

Model Manager Service for age-based modeling and anomaly detection.

This service handles autoencoder architecture, training pipelines, and model management
for age-group specific anomaly detection in children's drawings.

## Class: ModelManagerError

Base exception for model manager errors.

## Class: AutoencoderTrainingError

Raised when autoencoder training fails.

## Class: ModelLoadingError

Raised when model loading fails.

## Class: TrainingConfig

Configuration for autoencoder training.

## Class: AutoencoderModel

Autoencoder architecture for embedding reconstruction.

### forward

Forward pass through autoencoder.

**Signature**: `forward(x)`

### encode

Encode input to latent representation.

**Signature**: `encode(x)`

### decode

Decode latent representation to reconstruction.

**Signature**: `decode(z)`

### get_architecture_info

Get information about the model architecture.

**Signature**: `get_architecture_info()`

## Class: EarlyStopping

Smart early stopping utility that handles gradient explosion recovery.

## Class: AutoencoderTrainer

Trainer for autoencoder models.

### train

Train autoencoder on embeddings.

Args:
    embeddings: Array of embeddings for training
    
Returns:
    Dictionary containing training results and metrics

**Signature**: `train(embeddings)`

## Class: ModelManager

Manager for age-based autoencoder models.

### train_age_group_model

Train an autoencoder model for a specific age group.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    config: Training configuration
    db: Database session
    
Returns:
    Dictionary containing training results and model info

**Signature**: `train_age_group_model(age_min, age_max, config, db)`

### load_model

Load a trained autoencoder model.

Args:
    age_group_model_id: ID of the age group model
    db: Database session
    
Returns:
    Loaded autoencoder model

**Signature**: `load_model(age_group_model_id, db)`

### compute_reconstruction_loss

Compute reconstruction loss for an embedding using a specific model.

Args:
    embedding: Input embedding vector
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Reconstruction loss value

**Signature**: `compute_reconstruction_loss(embedding, age_group_model_id, db)`

### get_model_info

Get information about a specific model.

**Signature**: `get_model_info(age_group_model_id, db)`

### list_models

List all age group models.

**Signature**: `list_models(db)`

### clear_model_cache

Clear the loaded models cache.

**Signature**: `clear_model_cache()`

