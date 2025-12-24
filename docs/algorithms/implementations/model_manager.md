# ModelManager Algorithm Implementation

**Source File**: `app/services/model_manager.py`
**Last Updated**: 2025-12-18 23:17:04

## Overview

Manager for age-based autoencoder models.

## Computational Complexity Analysis

*This section analyzes the time and space complexity characteristics of the algorithm.*

Complexity analysis not available.

## Performance Analysis

*This section provides performance benchmarks and scalability characteristics.*

### Scalability Analysis

Consider profiling with representative datasets to determine scalability characteristics.

### Optimization Recommendations

- Implement caching for expensive computations

## Validation Methodology

*This section describes the testing and validation approach for the algorithm.*

### Testing Methods

- Calculate training metrics for embeddings
- Returns:
    Dictionary with validation results Clear the loaded models cache

### Validation Criteria

- Convergence of training process
- Generalization to unseen data
- Stability across different initializations

### Accuracy Metrics

- AUC-ROC
- False Positive Rate
- True Positive Rate

### Edge Cases

The following edge cases should be tested:

- Very large values for age_min
- Very large original_embedding
- Single-element embedding
- Empty embedding
- Zero value for age_max
- Negative values for age_max
- Very large embeddings
- Zero value for age_min
- Negative values for age_group_model_id
- Negative values for age_min
- Very large reconstructed_embedding
- Negative values for current_age
- Single-element reconstructed_embedding
- Very large embedding
- Empty original_embedding
- Single-element embeddings
- Empty reconstructed_embedding
- Zero value for age_group_model_id
- Very large values for current_age
- Very large values for age_group_model_id
- Zero value for current_age
- Single-element original_embedding
- Empty embeddings
- Very large values for age_max

## Implementation Details

### Methods

#### `train_subject_aware_age_group_model`

Train a subject-aware autoencoder model for a specific age group.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    config: Training configuration
    db: Database session
    
Returns:
    Dictionary containing training results and model info

**Parameters:**
- `self` (Any)
- `age_min` (float)
- `age_max` (float)
- `config` (TrainingConfig)
- `db` (Session)

**Returns:** Dict

#### `train_age_group_model`

Train an autoencoder model for a specific age group.

This method now delegates to the subject-aware training method to ensure
all models use the unified subject-aware architecture.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    config: Training configuration
    db: Database session
    
Returns:
    Dictionary containing training results and model info

**Parameters:**
- `self` (Any)
- `age_min` (float)
- `age_max` (float)
- `config` (TrainingConfig)
- `db` (Session)

**Returns:** Dict

#### `load_model`

Load a trained subject-aware autoencoder model.

Args:
    age_group_model_id: ID of the age group model
    db: Database session
    
Returns:
    Loaded autoencoder model

**Parameters:**
- `self` (Any)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** AutoencoderModel

#### `compute_reconstruction_loss`

Compute reconstruction loss for a hybrid embedding using a specific model.

Args:
    embedding: Input hybrid embedding vector (832-dimensional)
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Reconstruction loss value

**Parameters:**
- `self` (Any)
- `embedding` (np.ndarray)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** float

#### `compute_subject_aware_reconstruction_loss`

Compute component-specific reconstruction losses for a hybrid embedding.

Args:
    embedding: Input hybrid embedding vector (832-dimensional)
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Dictionary with overall, visual, and subject component losses

**Parameters:**
- `self` (Any)
- `embedding` (np.ndarray)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** Dict[str, float]

#### `get_model_info`

Get information about a specific subject-aware model.

**Parameters:**
- `self` (Any)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** Dict

#### `list_models`

List all age group models.

**Parameters:**
- `self` (Any)
- `db` (Session)

**Returns:** List[Dict]

#### `compute_anomaly_score`

Compute subject-aware anomaly scores for a hybrid embedding.

This method computes overall reconstruction loss on the full 832-dimensional
hybrid embedding and provides component-specific loss calculations for
visual (dims 0-767) and subject (dims 768-831) components.

Args:
    embedding: Input hybrid embedding vector (832-dimensional)
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Dictionary containing:
    - overall_anomaly_score: Overall reconstruction loss on full embedding
    - visual_anomaly_score: Visual component reconstruction loss (dims 0-767)
    - subject_anomaly_score: Subject component reconstruction loss (dims 768-831)

**Parameters:**
- `self` (Any)
- `embedding` (np.ndarray)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** Dict[str, float]

#### `determine_attribution`

Determine anomaly attribution based on component-specific reconstruction losses.

This method implements attribution decision rules by:
1. Calculating component-specific thresholds (visual, subject)
2. Comparing component losses to their respective thresholds
3. Determining primary anomaly source based on threshold exceedance
4. Supporting cross-age-group comparison for age-related detection

Args:
    embedding: Input hybrid embedding vector (832-dimensional)
    age_group_model_id: ID of the age group model to use
    db: Database session
    
Returns:
    Attribution string: "visual", "subject", "both", or "age"

**Parameters:**
- `self` (Any)
- `embedding` (np.ndarray)
- `age_group_model_id` (int)
- `db` (Session)

**Returns:** str

#### `compare_across_age_groups`

Compare reconstruction loss across different age group models for age-related detection.

This method supports cross-age-group comparison by computing the drawing's
anomaly score using models from different age groups to determine if the
anomaly is age-related.

Args:
    embedding: Input hybrid embedding vector (832-dimensional)
    current_age: Current age of the drawing
    db: Database session
    
Returns:
    Dictionary mapping age group ranges to anomaly scores

**Parameters:**
- `self` (Any)
- `embedding` (np.ndarray)
- `current_age` (float)
- `db` (Session)

**Returns:** Dict[str, float]

#### `validate_unified_subject_aware_architecture`

Validate that all models use the unified subject-aware architecture.

Returns:
    Dictionary with validation results

**Parameters:**
- `self` (Any)
- `db` (Session)

**Returns:** Dict[str, Any]

#### `clear_model_cache`

Clear the loaded models cache.

**Parameters:**
- `self` (Any)

**Returns:** None

## LaTeX Mathematical Notation

*For formal mathematical documentation and publication:*

```latex
\section{ModelManager Algorithm}
```

## References and Standards

- **IEEE Standard 830-1998**: Software Requirements Specifications
- **Algorithm Documentation**: Following IEEE guidelines for algorithm specification
- **Mathematical Notation**: Standard mathematical notation and LaTeX formatting

---

*This documentation was automatically generated from source code analysis.*
*Generated on: 2025-12-18 23:17:04*