# Model Manager Contract

## Overview
Service contract for Model Manager (service)

**Source File**: `app/services/model_manager.py`

## Interface Specification

### Classes

#### ModelManagerError

Base exception for model manager errors.

**Inherits from**: Exception

#### AutoencoderTrainingError

Raised when autoencoder training fails.

**Inherits from**: ModelManagerError

#### ModelLoadingError

Raised when model loading fails.

**Inherits from**: ModelManagerError

#### TrainingConfig

Configuration for autoencoder training.

**Attributes**:

- `hidden_dims: List[int]`
- `learning_rate: float`
- `batch_size: int`
- `epochs: int`
- `validation_split: float`
- `early_stopping_patience: int`
- `min_delta: float`
- `device: str`

#### AutoencoderModel

Autoencoder architecture for embedding reconstruction.

#### EarlyStopping

Smart early stopping utility that handles gradient explosion recovery.

#### AutoencoderTrainer

Trainer for autoencoder models.

#### ModelManager

Manager for age-based autoencoder models.

## Methods

### forward

Forward pass through autoencoder.

**Signature**: `forward(x: torch.Tensor) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `x` | `torch.Tensor` | Parameter description |

**Returns**: `torch.Tensor`

### encode

Encode input to latent representation.

**Signature**: `encode(x: torch.Tensor) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `x` | `torch.Tensor` | Parameter description |

**Returns**: `torch.Tensor`

### decode

Decode latent representation to reconstruction.

**Signature**: `decode(z: torch.Tensor) -> torch.Tensor`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `z` | `torch.Tensor` | Parameter description |

**Returns**: `torch.Tensor`

### get_architecture_info

Get information about the model architecture.

**Signature**: `get_architecture_info() -> Dict`

**Returns**: `Dict`

### train

Train autoencoder on embeddings.

Args:
    embeddings: Array of embeddings for training
    
Returns:
    Dictionary containing training results and metrics

**Signature**: `train(embeddings: np.ndarray) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `embeddings` | `np.ndarray` | Parameter description |

**Returns**: `Dict`

### train_age_group_model

Train an autoencoder model for a specific age group.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    config: Training configuration
    db: Database session
    
Returns:
    Dictionary containing training results and model info

**Signature**: `train_age_group_model(age_min: float, age_max: float, config: TrainingConfig, db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_min` | `float` | Parameter description |
| `age_max` | `float` | Parameter description |
| `config` | `TrainingConfig` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### load_model

Load a trained autoencoder model.

Args:
    age_group_model_id: ID of the age group model
    db: Database session
    
Returns:
    Loaded autoencoder model

**Signature**: `load_model(age_group_model_id: int, db: Session) -> AutoencoderModel`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `AutoencoderModel`

### compute_reconstruction_loss

Compute reconstruction loss for an embedding using a specific model.

Args:
    embedding: Input embedding vector
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Reconstruction loss value

**Signature**: `compute_reconstruction_loss(embedding: np.ndarray, age_group_model_id: int, db: Session) -> float`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `embedding` | `np.ndarray` | Parameter description |
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `float`

### get_model_info

Get information about a specific model.

**Signature**: `get_model_info(age_group_model_id: int, db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### list_models

List all age group models.

**Signature**: `list_models(db: Session) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `List[Dict]`

### clear_model_cache

Clear the loaded models cache.

**Signature**: `clear_model_cache() -> None`

**Returns**: `None`

## Dependencies

- `app.models.database.AgeGroupModel`
- `app.models.database.Drawing`
- `app.models.database.DrawingEmbedding`
- `app.services.embedding_service.get_embedding_service`

## Defined Interfaces

### AutoencoderModelInterface

**Type**: Protocol
**Implemented by**: AutoencoderModel

**Methods**:

- `forward(x: torch.Tensor) -> torch.Tensor`
- `encode(x: torch.Tensor) -> torch.Tensor`
- `decode(z: torch.Tensor) -> torch.Tensor`
- `get_architecture_info() -> Dict`

### AutoencoderTrainerInterface

**Type**: Protocol
**Implemented by**: AutoencoderTrainer

**Methods**:

- `train(embeddings: np.ndarray) -> Dict`

### ModelManagerInterface

**Type**: Protocol
**Implemented by**: ModelManager

**Methods**:

- `train_age_group_model(age_min: float, age_max: float, config: TrainingConfig, db: Session) -> Dict`
- `load_model(age_group_model_id: int, db: Session) -> AutoencoderModel`
- `compute_reconstruction_loss(embedding: np.ndarray, age_group_model_id: int, db: Session) -> float`
- `get_model_info(age_group_model_id: int, db: Session) -> Dict`
- `list_models(db: Session) -> List[Dict]`
- `clear_model_cache() -> None`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/model_manager.py`
- Last validated: 2025-12-16 15:47:04

