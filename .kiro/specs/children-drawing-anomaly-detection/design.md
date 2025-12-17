# Design Document: Children's Drawing Anomaly Detection System

## Overview

The Children's Drawing Anomaly Detection System is a machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected norms. The system employs deep learning vision models to extract meaningful features from drawings, statistical methods to model age-appropriate patterns, and interpretability techniques to explain anomaly decisions.

The system follows a modular architecture with clear separation between data processing, machine learning inference, and user interface components. This design enables researchers and clinicians to efficiently process individual drawings or batches while providing interpretable insights into potential developmental, emotional, or perceptual anomalies.

## Architecture

The system follows a layered architecture with the following main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface Layer                      │
├─────────────────────────────────────────────────────────────┤
│                      API Gateway                            │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline  │  Embedding    │  Model        │  Interp.  │
│  Service        │  Service      │  Manager      │  Engine   │
├─────────────────────────────────────────────────────────────┤
│                   Configuration Manager                     │
├─────────────────────────────────────────────────────────────┤
│              Storage Layer (Models, Data, Cache)            │
└─────────────────────────────────────────────────────────────┘
```

**Key Architectural Principles:**
- **Modularity**: Each component has a single responsibility and clear interfaces
- **Extensibility**: New vision models and anomaly detection methods can be easily added
- **Scalability**: Components can be scaled independently based on load
- **Configurability**: Model selection, thresholds, and parameters are externally configurable

## Components and Interfaces

### Data Pipeline Service
**Responsibility**: Image preprocessing, validation, and metadata management

**Key Methods:**
- `validate_image(image_data: bytes) -> ValidationResult`
- `preprocess_image(image: Image, target_size: Tuple[int, int]) -> Tensor`
- `extract_metadata(upload_data: Dict) -> DrawingMetadata`

**Interfaces:**
- Input: Raw image files (PNG, JPEG, BMP) with metadata
- Output: Preprocessed tensors and validated metadata objects

### Embedding Service
**Responsibility**: Converting drawings into numerical feature vectors

**Key Methods:**
- `load_vision_model(model_type: str, config: Dict) -> VisionModel`
- `generate_embedding(image_tensor: Tensor, age: float) -> Embedding`
- `batch_embed(images: List[Tensor], ages: List[float]) -> List[Embedding]`

**Supported Models:**
- CLIP (primary recommendation for sketch-like content)
- Vision Transformer (ViT) for global spatial relationships
- ResNet (fallback for standard CNN approach)

### Model Manager
**Responsibility**: Age-based modeling and anomaly detection

**Key Methods:**
- `train_age_group_model(embeddings: List[Embedding], method: str) -> AgeGroupModel`
- `compute_anomaly_score(embedding: Embedding, age_group: str) -> AnomalyScore`
- `update_thresholds(validation_scores: List[float], percentile: float) -> Dict[str, float]`

**Anomaly Detection Methods:**
- Mahalanobis Distance (default)
- VAE Reconstruction Error
- One-Class SVM / Isolation Forest

### Interpretability Engine
**Responsibility**: Generating explanations for anomaly decisions

**Key Methods:**
- `generate_saliency_map(image: Tensor, model: VisionModel, score: float) -> SaliencyMap`
- `create_attention_visualization(image: Tensor, attention_weights: Tensor) -> AttentionMap`
- `explain_anomaly(drawing_data: DrawingData, result: AnomalyResult) -> Explanation`

## Data Models

### Core Data Structures

```python
@dataclass
class DrawingMetadata:
    age_years: float
    subject: Optional[str]  # "person", "house", "tree", etc.
    expert_label: Optional[ExpertLabel]
    drawing_tool: Optional[str]
    prompt: Optional[str]
    timestamp: datetime

@dataclass
class Embedding:
    vector: np.ndarray  # Fixed-length feature vector
    model_type: str
    age_years: float
    metadata: DrawingMetadata

@dataclass
class AnomalyResult:
    score: float
    normalized_score: float
    is_anomaly: bool
    confidence: float
    age_group: str
    method_used: str

@dataclass
class SaliencyMap:
    heatmap: np.ndarray
    overlay_image: np.ndarray
    importance_regions: List[BoundingBox]
    explanation_text: str
```

### Age Group Modeling

**Age Grouping Strategy:**
- Default: 1-year intervals (3-4, 4-5, 5-6, etc.)
- Automatic merging when insufficient data (< 50 samples)
- Configurable grouping intervals

**Model Storage:**
```python
@dataclass
class AgeGroupModel:
    age_range: Tuple[float, float]
    method: str  # "mahalanobis", "vae", "one_class_svm"
    parameters: Dict  # Method-specific parameters
    training_stats: Dict  # Sample count, validation metrics
    threshold: float
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Image Format Validation Consistency
*For any* uploaded file, the system should accept it if and only if it is a valid PNG, JPEG, or BMP image with valid age metadata in the range 2-18 years
**Validates: Requirements 1.1, 1.3**

### Property 2: Image Preprocessing Uniformity
*For any* valid input image, the preprocessing pipeline should produce output tensors with identical dimensions and normalized pixel values in the expected range
**Validates: Requirements 1.2**

### Property 3: Metadata Persistence
*For any* drawing upload with optional metadata (subject, expert labels), the provided metadata should be stored and retrievable during analysis
**Validates: Requirements 1.4, 1.5**

### Property 4: Embedding Dimensionality Consistency
*For any* set of drawings processed with the same model configuration, all generated embeddings should have identical vector dimensions
**Validates: Requirements 2.1, 2.4**

### Property 5: Age-Augmented Embedding Consistency
*For any* drawing with age information, when age augmentation is enabled, the embedding should include age data and have consistent dimensionality across all age-augmented embeddings
**Validates: Requirements 2.3**

### Property 6: Statistical Distribution Computation
*For any* set of training embeddings for an age group, the Age_Group_Model should compute mathematically correct mean vectors and covariance matrices
**Validates: Requirements 3.1, 3.2**

### Property 7: Anomaly Score Generation
*For any* processed drawing, the system should generate a valid anomaly score (finite number) using the appropriate age group model
**Validates: Requirements 4.1**

### Property 8: Score Normalization Consistency
*For any* set of anomaly scores across different age groups, normalized scores should be comparable and fall within expected ranges
**Validates: Requirements 4.2**

### Property 9: Threshold-Based Anomaly Detection
*For any* drawing with computed anomaly score, the drawing should be flagged as anomalous if and only if its score exceeds the configured threshold
**Validates: Requirements 5.3**

### Property 10: Threshold Recalculation Accuracy
*For any* set of validation scores, the 95th percentile threshold should be mathematically correct and properly applied to subsequent analyses
**Validates: Requirements 5.1, 5.5**

### Property 11: Dynamic Threshold Updates
*For any* threshold configuration change, subsequent analyses should use the new threshold without requiring system restart
**Validates: Requirements 5.2, 5.4**

### Property 12: Interpretability Generation Completeness
*For any* drawing flagged as anomalous, the system should generate corresponding saliency maps and explanations
**Validates: Requirements 6.1, 6.4, 6.5**

### Property 13: Error Handling Robustness
*For any* invalid input or system failure condition, the system should provide appropriate error messages and continue processing valid inputs
**Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**

### Property 14: Batch Processing Consistency
*For any* batch of drawings, each drawing should receive the same analysis result whether processed individually or as part of the batch
**Validates: Requirements 7.5**

## Error Handling

The system implements comprehensive error handling across all components:

### Input Validation Errors
- **Invalid Image Formats**: Reject with specific format requirements
- **Corrupted Images**: Detect and isolate corrupted files
- **Missing Age Data**: Prompt for required information or apply defaults
- **Out-of-Range Ages**: Validate against acceptable age bounds (2-18 years)

### Processing Errors
- **Model Loading Failures**: Fallback to alternative models or cached versions
- **Embedding Generation Errors**: Retry with different preprocessing or model parameters
- **Insufficient Training Data**: Merge adjacent age groups or provide warnings
- **Memory Constraints**: Implement batch processing with appropriate queue management

### Analysis Errors
- **Anomaly Scoring Failures**: Implement fallback scoring methods
- **Threshold Calculation Errors**: Use default thresholds with appropriate warnings
- **Interpretability Generation Failures**: Provide basic explanations or skip visualization

### System-Level Error Handling
- **Resource Exhaustion**: Queue requests with estimated processing times
- **Network/Storage Issues**: Implement retry mechanisms with exponential backoff
- **Configuration Errors**: Validate configurations on startup and provide clear error messages

## Testing Strategy

The system employs a dual testing approach combining unit tests and property-based tests to ensure comprehensive coverage and correctness validation.

### Unit Testing Approach

Unit tests focus on specific examples, edge cases, and integration points:

**Core Component Tests:**
- Image preprocessing with various formats and sizes
- Embedding generation with different model types
- Statistical calculations for age group modeling
- Threshold computation and application
- Error handling for specific failure scenarios

**Integration Tests:**
- End-to-end pipeline from upload to results
- Model loading and switching between different algorithms
- Configuration changes and their effects
- Batch processing workflows

### Property-Based Testing Approach

**Framework Selection:** The system uses **Hypothesis** (Python) for property-based testing, configured to run a minimum of 100 iterations per property test.

**Property Test Implementation Requirements:**
- Each correctness property must be implemented by a single property-based test
- Tests must be tagged with comments referencing the design document property
- Tag format: `**Feature: children-drawing-anomaly-detection, Property {number}: {property_text}**`
- Tests should generate diverse, realistic input data to thoroughly exercise the system

**Key Property Test Categories:**
- **Mathematical Correctness**: Statistical calculations, distance metrics, normalization
- **Data Consistency**: Embedding dimensions, metadata preservation, score ranges
- **Behavioral Invariants**: Threshold-based decisions, error handling responses
- **System Properties**: Configuration changes, batch vs individual processing consistency

**Test Data Generation:**
- Synthetic drawing images with controlled characteristics
- Random age values within and outside valid ranges
- Various metadata combinations and edge cases
- Simulated failure conditions for error handling validation

### Testing Infrastructure

**Test Environment Setup:**
- Isolated test databases and model storage
- Mock external dependencies where appropriate
- Configurable test data generators
- Performance benchmarking for critical paths

**Continuous Validation:**
- Automated test execution on code changes
- Property test result analysis and failure investigation
- Performance regression detection
- Model accuracy validation with known datasets

The combination of unit and property-based testing ensures both concrete functionality validation and general correctness guarantees across the wide input space typical in machine learning applications.