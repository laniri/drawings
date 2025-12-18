# Design Document: Children's Drawing Anomaly Detection System

## Overview

The Children's Drawing Anomaly Detection System is a machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected and subject-specific norms. The system employs deep learning vision models to extract meaningful features from drawings, subject-aware statistical methods to model age-appropriate and subject-specific patterns, and enhanced interpretability techniques to explain anomaly decisions with contextual attribution.

The system follows a modular architecture with clear separation between data processing, machine learning inference, and user interface components. This design enables researchers and clinicians to efficiently process individual drawings or batches while providing interpretable insights into potential developmental, emotional, or perceptual anomalies. 

**Subject-Aware Architecture:** The system uses a unified subject-aware modeling approach where all drawings are processed using hybrid embeddings that combine visual features (768-dim from Vision Transformer) and subject category encodings (64-dim one-hot vector), resulting in 832-dimensional embeddings. Age information is used to **select the appropriate age-group model** rather than being embedded in the feature vector. This architecture supports up to 64 distinct subject categories and provides anomaly attribution capabilities that distinguish between subject-related and visual anomalies. When subject information is unavailable, the system uses a default "unspecified" category to maintain consistent processing across all drawings.

## Architecture

The system follows a modern layered architecture with the following technology stack:

```
┌─────────────────────────────────────────────────────────────┐
│           React Frontend (Vite + TypeScript)                │
│         Material-UI, React Query, Zustand                   │
├─────────────────────────────────────────────────────────────┤
│              FastAPI Gateway (Python 3.11+)                │
│           Pydantic, Uvicorn, Python-Multipart              │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline  │  Embedding    │  Model        │  Interp.  │
│  Service        │  Service      │  Manager      │  Engine   │
│  (Pillow,       │  (PyTorch,    │  (Scikit-     │  (Captum, │
│   OpenCV)       │   Transformers)│   learn)      │   Grad-CAM)│
├─────────────────────────────────────────────────────────────┤
│              Configuration Manager (Pydantic)               │
├─────────────────────────────────────────────────────────────┤
│    SQLite Database + File Storage (Models, Images, Cache)   │
│              SQLAlchemy ORM + Alembic Migrations            │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Frontend (React Ecosystem):**
- **React 18** with TypeScript for type safety
- **Vite** for fast development and optimized builds
- **Material-UI (MUI) v5** for modern, accessible UI components
- **React Query (TanStack Query)** for server state management
- **Zustand** for client-side state management
- **React Hook Form** with Zod validation for form handling
- **React Dropzone** for drag-and-drop file uploads
- **Recharts** for data visualization and charts

**Backend (Python Ecosystem):**
- **Python 3.11+** for latest performance improvements
- **FastAPI** for high-performance async API with automatic OpenAPI docs
- **Pydantic v2** for data validation and serialization
- **SQLAlchemy 2.0** with async support for database ORM
- **Alembic** for database migrations
- **Uvicorn** with Gunicorn for production ASGI server
- **Python-Multipart** for file upload handling

**Machine Learning Stack:**
- **PyTorch 2.0+** with CUDA support for deep learning
- **Transformers (Hugging Face)** for Vision Transformer models
- **OpenCV** and **Pillow** for image processing
- **NumPy** and **Pandas** for numerical computing
- **Captum** for model interpretability and explainability
- **Scikit-learn** for data splitting and evaluation metrics
- **Matplotlib** and **Seaborn** for training visualization and reporting

**Training Environment Stack:**
- **Amazon SageMaker** for cloud-based model training
- **Boto3** for AWS SDK integration
- **Weights & Biases (wandb)** for experiment tracking and visualization
- **Hydra** for configuration management and parameter sweeps
- **MLflow** for model versioning and deployment tracking

**Database & Storage:**
- **SQLite** with WAL mode for concurrent access
- **SQLAlchemy** async engine for database operations
- Local file system for image and model storage
- **Redis** (optional) for caching embeddings and results

**Development & Deployment:**
- **venv** for Python virtual environment management
- **pip** with **requirements.txt** for dependency management
- **Black**, **isort**, **flake8** for code formatting and linting
- **Pytest** with **pytest-asyncio** for testing
- **Docker** and **Docker Compose** for containerization
- **GitHub Actions** for CI/CD pipeline

**Key Architectural Principles:**
- **Modularity**: Each component has a single responsibility and clear interfaces
- **Extensibility**: New vision models and anomaly detection methods can be easily added
- **Scalability**: Components can be scaled independently based on load
- **Configurability**: Model selection, thresholds, and parameters are externally configurable
- **Type Safety**: Full TypeScript frontend and Pydantic backend for runtime validation

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
**Responsibility**: Converting drawings into numerical feature vectors with subject-aware augmentation

**Key Methods:**
- `load_vision_model(model_type: str, config: Dict) -> VisionModel`
- `generate_hybrid_embedding(image_tensor: Tensor, subject: str) -> HybridEmbedding`
  - Always generates 832-dimensional hybrid embeddings
  - Combines visual features (768-dim) and subject encoding (64-dim)
  - Uses "unspecified" as default subject when not provided
  - Age is NOT part of the embedding - it's used for model selection
- `encode_subject_category(subject: str) -> SubjectEncoding`
  - Returns 64-dimension one-hot vector for subject categories
  - Supports up to 64 distinct categories (currently ~50 defined)
  - Maps "unspecified" to a dedicated position in the one-hot vector
- `batch_embed(images: List[Tensor], subjects: List[str]) -> List[HybridEmbedding]`
  - Processes multiple drawings efficiently
  - Ensures consistent hybrid embedding generation
- `separate_embedding_components(hybrid_embedding: HybridEmbedding) -> Tuple[VisualComponent, SubjectComponent]`
  - Extracts individual components for analysis and attribution
  - Visual: dimensions 0-767
  - Subject: dimensions 768-831

**Supported Models:**
- Vision Transformer (ViT) - primary and only model for global spatial relationships and drawing analysis

**Embedding Architecture:**
- **Unified Hybrid Embeddings**: All drawings use 832-dimensional hybrid embeddings (768 visual + 64 subject)
- **Subject Encoding**: 64-dimension one-hot vector for extensible subject category support
- **Age-Based Model Selection**: Age determines which autoencoder model to use, not part of embedding
- **Consistent Dimensionality**: Same embedding structure regardless of subject availability
- **Component Separability**: Ability to isolate visual and subject contributions for attribution analysis

### Model Manager
**Responsibility**: Subject-aware age-based modeling and anomaly detection with attribution

**Key Methods:**
- `train_subject_aware_model(hybrid_embeddings: List[HybridEmbedding], age_group: str) -> SubjectAwareModel`
  - Trains autoencoder on 832-dimensional hybrid embeddings
  - Ensures balanced representation across subject categories within age group
  - Validates sufficient data for age-subject combinations
- `compute_subject_aware_score(hybrid_embedding: HybridEmbedding, model: SubjectAwareModel) -> SubjectAwareScore`
  - Computes reconstruction loss for full hybrid embedding
  - Returns overall anomaly score and component-specific scores
- `attribute_anomaly(hybrid_embedding: HybridEmbedding, reconstructed: HybridEmbedding) -> AnomalyAttribution`
  - Calculates reconstruction loss for visual component (dims 0-767)
  - Calculates reconstruction loss for subject component (dims 768-831)
  - Determines primary anomaly source: "subject" or "visual"
  - Note: "age" attribution comes from comparing across age-group models, not from embedding components
- `update_thresholds(validation_scores: List[float], percentile: float) -> Dict[str, float]`
  - Recalculates thresholds based on validation data
  - Supports arbitrary percentile values (50.0-99.9)
- `select_appropriate_model(age: float) -> SubjectAwareModel`
  - Selects age-appropriate subject-aware model based on age value
  - All models use unified subject-aware architecture

**Anomaly Detection Method:**
- **Autoencoder Reconstruction Loss**: Primary and only method
  - Measures difference between original and reconstructed hybrid embeddings
  - Provides component-level attribution (visual, subject)
  - Age-related anomalies detected by comparing scores across age-group models

**Model Architecture:**
- **Unified Subject-Aware Autoencoders**: One model per age group (e.g., 2-3, 3-4, 4-5 years)
- **Input**: 832-dimensional hybrid embeddings (768 visual + 64 subject)
- **Architecture**: Encoder-decoder with bottleneck layer
- **Training**: Subject-stratified sampling ensures balanced representation
- **Output**: Reconstructed hybrid embedding with same dimensionality
- **Consistency**: All age groups use identical architecture for uniform behavior
- **Age Handling**: Age determines which model to use, not part of the embedding

**Anomaly Attribution Logic:**
- **Visual-dominant**: High visual reconstruction loss, low subject loss
- **Subject-related**: High subject component loss relative to visual
- **Combined**: Both visual and subject components show elevated reconstruction loss
- **Age-related**: Determined by comparing the drawing's score against multiple age-group models (if score is normal for a different age group, it's age-related)
- **Threshold-based**: Component losses compared to component-specific thresholds

### Interpretability Engine
**Responsibility**: Generating explanations for anomaly decisions with subject-aware attribution

**Key Methods:**
- `generate_saliency_map(image: Tensor, model: VisionModel, score: float) -> SaliencyMap`
  - Creates visual heatmap highlighting anomalous regions
  - Uses simplified gradient-based approach for guaranteed availability
- `create_attention_visualization(image: Tensor, attention_weights: Tensor) -> AttentionMap`
  - Visualizes Vision Transformer attention patterns
  - Shows which image patches contributed to the embedding
- `explain_anomaly(drawing_data: DrawingData, result: SubjectAwareAnomalyResult) -> Explanation`
  - Generates human-readable explanation of anomaly
  - Includes anomaly attribution (age, subject, both, visual)
  - Provides context about age-subject norms
- `explain_subject_aware_anomaly(attribution: AnomalyAttribution, age_group: str, subject: str) -> DetailedExplanation`
  - Describes whether anomaly is age-related, subject-related, or both
  - Provides subject-specific context (e.g., "unusual for a 5-year-old's drawing of a house")
  - Compares to typical patterns for the age-subject combination
- `generate_comparative_examples(age_group: str, subject: str, anomaly_type: str) -> List[ComparisonExample]`
  - Retrieves normal examples from same age-subject category
  - Shows typical drawings for context
  - Highlights differences between normal and anomalous patterns

### Training Environment
**Responsibility**: Offline model development, experimentation, and optimization with subject-aware stratification

**Key Methods:**
- `prepare_dataset(data_folder: str, metadata_file: str, config: TrainingConfig) -> DatasetSplits`
  - Loads drawings with age and subject metadata
  - Performs subject-stratified splitting to ensure balanced representation
  - Validates sufficient data for age-subject combinations
  - Generates warnings for insufficient data scenarios
- `stratify_by_age_and_subject(data: List[Drawing], splits: Tuple[float, float, float]) -> DatasetSplits`
  - Ensures balanced representation across age groups and subject categories
  - Handles edge cases where specific age-subject combinations have limited data
  - Suggests category merging or age group consolidation when needed
- `train_subject_aware_model(config: TrainingConfig, environment: str) -> TrainingResult`
  - Trains autoencoder on hybrid embeddings (832-dim)
  - Uses subject-stratified batching during training
  - Validates model performance across subject categories
- `validate_cross_subject_performance(model: SubjectAwareModel, test_data: DatasetSplits) -> CrossSubjectMetrics`
  - Evaluates model performance for each subject category
  - Identifies subject-specific biases or weaknesses
  - Provides per-subject reconstruction loss statistics
- `generate_training_report(results: TrainingResult, metrics: Dict) -> TrainingReport`
  - Includes subject-stratified performance metrics
  - Shows per-subject category reconstruction losses
  - Highlights data sufficiency warnings
- `export_model_parameters(trained_model: SubjectAwareModel, format: str) -> ModelExport`
  - Exports subject-aware autoencoder parameters
  - Includes metadata about supported subject categories
- `deploy_model_to_production(model_export: ModelExport, deployment_config: Dict) -> DeploymentResult`
  - Validates model compatibility with production system
  - Ensures subject category consistency

**Training Environments:**
- Local training with GPU/CPU support
- Amazon SageMaker cloud training with configurable instance types
- Distributed training support for large datasets

**Configuration Management:**
- Hydra-based configuration with parameter sweeps
- Support for learning rate, batch size, epochs, architecture parameters
- Configurable train/validation/test split ratios with subject stratification
- Model architecture hyperparameter optimization
- Subject category management and merging strategies

**Data Sufficiency Handling:**
- Minimum sample thresholds per age-subject combination
- Automatic warnings for insufficient data
- Suggestions for category merging (e.g., combining similar subjects)
- Age group consolidation recommendations
- Fallback to "unspecified" category when appropriate

### API Endpoints (FastAPI)

**Drawing Management:**
- `POST /api/drawings/upload` - Upload drawing with metadata
- `GET /api/drawings/{id}` - Retrieve drawing details
- `GET /api/drawings` - List drawings with filtering
- `DELETE /api/drawings/{id}` - Delete drawing and associated data

**Analysis Operations:**
- `POST /api/analysis/analyze/{drawing_id}` - Analyze specific drawing
- `POST /api/analysis/batch` - Batch analyze multiple drawings
- `GET /api/analysis/{analysis_id}` - Get analysis results
- `GET /api/analysis/drawing/{drawing_id}` - Get all analyses for a drawing

**Model Management:**
- `GET /api/models/age-groups` - List available age group models
- `POST /api/models/train` - Train new age group model
- `PUT /api/models/{model_id}/threshold` - Update model threshold
- `GET /api/models/status` - Get model training status
- `POST /api/models/deploy` - Deploy trained model parameters to production
- `GET /api/models/training-reports/{report_id}` - Get training summary report
- `POST /api/models/export` - Export model parameters for deployment

**Configuration:**
- `GET /api/config` - Get current system configuration
- `PUT /api/config/threshold` - Update global threshold settings
- `PUT /api/config/age-grouping` - Modify age grouping strategy

**Enhanced Interpretability:**
- `GET /api/interpretability/{analysis_id}/interactive` - Get interactive saliency data with region mappings
- `GET /api/interpretability/{analysis_id}/simplified` - Get simplified explanations for non-technical users
- `GET /api/interpretability/{analysis_id}/confidence` - Get confidence metrics and reliability scores
- `POST /api/interpretability/{analysis_id}/export` - Export interpretability results in multiple formats
- `GET /api/interpretability/examples/{age_group}` - Get comparison examples for educational purposes
- `GET /api/interpretability/examples/{age_group}/{subject}` - Get subject-specific comparison examples
- `POST /api/interpretability/{analysis_id}/annotate` - Add user annotations to interpretability results
- `GET /api/interpretability/{analysis_id}/attribution` - Get detailed anomaly attribution breakdown (age vs subject vs visual)

### Frontend Components (React + TypeScript)

**Core Pages:**
- `UploadPage` - Drag-and-drop drawing upload with metadata form
- `AnalysisPage` - Display analysis results with interpretability
- `DashboardPage` - Overview of recent analyses and system status
- `ConfigurationPage` - System settings and model management
- `BatchProcessingPage` - Bulk upload and analysis management

**Shared Components:**
- `DrawingViewer` - Display drawings with zoom and pan capabilities
- `SaliencyOverlay` - Overlay saliency maps on original drawings
- `AnomalyScoreCard` - Visual representation of anomaly scores
- `AgeGroupChart` - Visualization of age group distributions
- `ModelStatusCard` - Display Vision Transformer model status and performance
- `ThresholdSlider` - Interactive threshold adjustment
- `ProgressTracker` - Real-time progress for batch operations

**Enhanced Interpretability Components:**
- `InteractiveInterpretabilityViewer` - Hoverable saliency maps with click-to-zoom and region explanations
- `ExplanationLevelToggle` - Switch between technical and simplified explanations
- `ConfidenceIndicator` - Visual confidence meters with reliability warnings
- `ComparativeAnalysisPanel` - Side-by-side normal vs anomalous example comparisons
- `InterpretabilityTutorial` - Interactive onboarding and contextual help system
- `ExportToolbar` - Multiple format export options with annotation capabilities
- `AnnotationTools` - User annotation and note-taking interface
- `ExampleGallery` - Library of interpretation examples and patterns

**State Management (Zustand):**
```typescript
interface AppState {
  // Drawing management
  drawings: Drawing[]
  selectedDrawing: Drawing | null
  uploadProgress: number
  
  // Analysis state
  currentAnalysis: AnalysisResult | null
  analysisHistory: AnalysisResult[]
  isAnalyzing: boolean
  
  // Configuration
  systemConfig: SystemConfig
  modelStatus: ModelStatus
  
  // UI state
  sidebarOpen: boolean
  currentView: 'upload' | 'analysis' | 'dashboard' | 'config'
}
```

## Data Models

### Database Schema (SQLAlchemy Models)

```python
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, LargeBinary, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Drawing(Base):
    __tablename__ = "drawings"
    
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    age_years = Column(Float, nullable=False)
    subject = Column(String, nullable=True)
    expert_label = Column(String, nullable=True)  # "normal", "concern", "severe"
    drawing_tool = Column(String, nullable=True)
    prompt = Column(Text, nullable=True)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    embeddings = relationship("DrawingEmbedding", back_populates="drawing")
    analyses = relationship("AnomalyAnalysis", back_populates="drawing")

class DrawingEmbedding(Base):
    __tablename__ = "drawing_embeddings"
    
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey("drawings.id"))
    model_type = Column(String, nullable=False)  # "vit"
    embedding_type = Column(String, nullable=False, default="hybrid")  # Always "hybrid" for subject-aware system
    embedding_vector = Column(LargeBinary, nullable=False)  # Serialized numpy array (832 dimensions)
    visual_component = Column(LargeBinary, nullable=True)  # Visual embedding component (768 dimensions)
    subject_component = Column(LargeBinary, nullable=True)  # Subject encoding component (64 dimensions)
    vector_dimension = Column(Integer, nullable=False, default=832)  # Always 832 for hybrid embeddings
    created_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drawing = relationship("Drawing", back_populates="embeddings")

class AgeGroupModel(Base):
    __tablename__ = "age_group_models"
    
    id = Column(Integer, primary_key=True)
    age_min = Column(Float, nullable=False)
    age_max = Column(Float, nullable=False)
    model_type = Column(String, nullable=False)  # "autoencoder"
    vision_model = Column(String, nullable=False)  # "vit"
    supports_subjects = Column(Boolean, nullable=False, default=True)  # Always True for subject-aware system
    subject_categories = Column(Text, nullable=True)  # JSON array of supported subject categories
    embedding_type = Column(String, nullable=False, default="hybrid")  # Always "hybrid" for subject-aware system
    parameters = Column(Text, nullable=False)  # JSON serialized parameters
    sample_count = Column(Integer, nullable=False)
    threshold = Column(Float, nullable=False)
    created_timestamp = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class AnomalyAnalysis(Base):
    __tablename__ = "anomaly_analyses"
    
    id = Column(Integer, primary_key=True)
    drawing_id = Column(Integer, ForeignKey("drawings.id"))
    age_group_model_id = Column(Integer, ForeignKey("age_group_models.id"))
    anomaly_score = Column(Float, nullable=False)  # Overall reconstruction loss
    normalized_score = Column(Float, nullable=False)  # Normalized to 0-1 range
    visual_anomaly_score = Column(Float, nullable=True)  # Visual component reconstruction loss (dims 0-767)
    subject_anomaly_score = Column(Float, nullable=True)  # Subject component reconstruction loss (dims 768-831)
    anomaly_attribution = Column(String, nullable=True)  # "subject", "visual", "both", or "age" (from cross-model comparison)
    analysis_type = Column(String, nullable=False, default="subject_aware")  # Always "subject_aware"
    is_anomaly = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    analysis_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    drawing = relationship("Drawing", back_populates="analyses")
    age_group_model = relationship("AgeGroupModel")
    interpretability = relationship("InterpretabilityResult", back_populates="analysis")

class InterpretabilityResult(Base):
    __tablename__ = "interpretability_results"
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("anomaly_analyses.id"))
    saliency_map_path = Column(String, nullable=False)
    overlay_image_path = Column(String, nullable=False)
    explanation_text = Column(Text, nullable=True)
    importance_regions = Column(Text, nullable=True)  # JSON serialized bounding boxes
    created_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("AnomalyAnalysis", back_populates="interpretability")

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True)
    job_name = Column(String, nullable=False)
    environment = Column(String, nullable=False)  # "local", "sagemaker"
    config_parameters = Column(Text, nullable=False)  # JSON serialized training config
    dataset_path = Column(String, nullable=False)
    status = Column(String, nullable=False)  # "pending", "running", "completed", "failed"
    start_timestamp = Column(DateTime, nullable=True)
    end_timestamp = Column(DateTime, nullable=True)
    sagemaker_job_arn = Column(String, nullable=True)  # For SageMaker jobs
    
    # Relationships
    training_reports = relationship("TrainingReport", back_populates="training_job")

class TrainingReport(Base):
    __tablename__ = "training_reports"
    
    id = Column(Integer, primary_key=True)
    training_job_id = Column(Integer, ForeignKey("training_jobs.id"))
    final_loss = Column(Float, nullable=False)
    validation_accuracy = Column(Float, nullable=False)
    best_epoch = Column(Integer, nullable=False)
    training_time_seconds = Column(Float, nullable=False)
    model_parameters_path = Column(String, nullable=False)
    metrics_summary = Column(Text, nullable=False)  # JSON serialized metrics
    report_file_path = Column(String, nullable=False)
    created_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    training_job = relationship("TrainingJob", back_populates="training_reports")
```

### API Models (Pydantic Schemas)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

class ExpertLabel(str, Enum):
    NORMAL = "normal"
    CONCERN = "concern"
    SEVERE = "severe"

class SubjectCategory(str, Enum):
    # Current subjects from database
    TV = "TV"
    AIRPLANE = "airplane"
    APPLE = "apple"
    BEAR = "bear"
    BED = "bed"
    BEE = "bee"
    BIKE = "bike"
    BIRD = "bird"
    BOAT = "boat"
    BOOK = "book"
    BOTTLE = "bottle"
    BOWL = "bowl"
    CACTUS = "cactus"
    CAMEL = "camel"
    CAR = "car"
    CAT = "cat"
    CHAIR = "chair"
    CLOCK = "clock"
    COUCH = "couch"
    COW = "cow"
    CUP = "cup"
    DOG = "dog"
    ELEPHANT = "elephant"
    FACE = "face"
    FISH = "fish"
    FROG = "frog"
    HAND = "hand"
    HAT = "hat"
    HORSE = "horse"
    HOUSE = "house"
    ICE_CREAM = "ice cream"
    KEY = "key"
    LAMP = "lamp"
    MUSHROOM = "mushroom"
    OCTOPUS = "octopus"
    PERSON = "person"
    PHONE = "phone"
    PIANO = "piano"
    RABBIT = "rabbit"
    SCISSORS = "scissors"
    SHEEP = "sheep"
    SNAIL = "snail"
    SPIDER = "spider"
    TIGER = "tiger"
    TRAIN = "train"
    TREE = "tree"
    WATCH = "watch"
    WHALE = "whale"
    
    # Reserved for future expansion (up to 64 total)
    UNSPECIFIED = "unspecified"

class DrawingUploadRequest(BaseModel):
    age_years: float = Field(..., ge=2.0, le=18.0, description="Child's age in years")
    subject: Optional[SubjectCategory] = Field(None, description="Drawing subject category")
    expert_label: Optional[ExpertLabel] = None
    drawing_tool: Optional[str] = None
    prompt: Optional[str] = None

class DrawingResponse(BaseModel):
    id: int
    filename: str
    age_years: float
    subject: Optional[str]
    expert_label: Optional[str]
    upload_timestamp: datetime
    
    class Config:
        from_attributes = True

class AnomalyAnalysisResponse(BaseModel):
    id: int
    anomaly_score: float  # Overall reconstruction loss
    normalized_score: float  # Normalized to 0-1 range
    visual_anomaly_score: Optional[float] = None  # Visual component loss (dims 0-767)
    subject_anomaly_score: Optional[float] = None  # Subject component loss (dims 768-831)
    anomaly_attribution: Optional[str] = None  # "subject", "visual", "both", or "age" (from cross-model comparison)
    analysis_type: str = "subject_aware"  # Always "subject_aware"
    is_anomaly: bool
    confidence: float
    age_group: str
    subject_category: Optional[str] = None  # Subject used in analysis
    method_used: str  # Always "autoencoder"
    analysis_timestamp: datetime
    
    class Config:
        from_attributes = True

class InterpretabilityResponse(BaseModel):
    saliency_map_url: str
    overlay_image_url: str
    explanation_text: Optional[str]
    importance_regions: List[dict]
    
    class Config:
        from_attributes = True

class AnalysisResultResponse(BaseModel):
    drawing: DrawingResponse
    analysis: AnomalyAnalysisResponse
    interpretability: Optional[InterpretabilityResponse]

class ConfigurationResponse(BaseModel):
    vision_model: str  # Always "vit"
    threshold_percentile: float
    age_grouping_strategy: str
    anomaly_detection_method: str  # Always "autoencoder"

class TrainingEnvironment(str, Enum):
    LOCAL = "local"
    SAGEMAKER = "sagemaker"

class TrainingConfigRequest(BaseModel):
    job_name: str = Field(..., description="Unique name for the training job")
    environment: TrainingEnvironment
    dataset_folder: str = Field(..., description="Path to folder containing drawings")
    metadata_file: str = Field(..., description="Path to metadata CSV/JSON file")
    
    # Training parameters
    learning_rate: float = Field(0.001, ge=1e-6, le=1.0)
    batch_size: int = Field(32, ge=1, le=512)
    epochs: int = Field(100, ge=1, le=1000)
    train_split: float = Field(0.7, ge=0.1, le=0.9)
    validation_split: float = Field(0.2, ge=0.1, le=0.5)
    test_split: float = Field(0.1, ge=0.05, le=0.3)
    
    # SageMaker specific parameters
    instance_type: Optional[str] = Field("ml.m5.large", description="SageMaker instance type")
    instance_count: int = Field(1, ge=1, le=10)

class TrainingJobResponse(BaseModel):
    id: int
    job_name: str
    environment: str
    status: str
    start_timestamp: Optional[datetime]
    end_timestamp: Optional[datetime]
    sagemaker_job_arn: Optional[str]
    
    class Config:
        from_attributes = True

class TrainingReportResponse(BaseModel):
    id: int
    final_loss: float
    validation_accuracy: float
    best_epoch: int
    training_time_seconds: float
    model_parameters_path: str
    report_file_path: str
    created_timestamp: datetime
    
    class Config:
        from_attributes = True

class ModelDeploymentRequest(BaseModel):
    model_parameters_path: str = Field(..., description="Path to trained model parameters")
    age_group_min: float = Field(..., ge=2.0, le=18.0)
    age_group_max: float = Field(..., ge=2.0, le=18.0)
    replace_existing: bool = Field(False, description="Whether to replace existing model for age group")

class InteractiveInterpretabilityResponse(BaseModel):
    saliency_regions: List[dict] = Field(..., description="Interactive regions with hover explanations")
    attention_patches: List[dict] = Field(..., description="Vision Transformer attention patch data")
    region_explanations: Dict[str, str] = Field(..., description="Explanations for each interactive region")
    confidence_scores: Dict[str, float] = Field(..., description="Confidence scores for each explanation")
    
class SimplifiedExplanationResponse(BaseModel):
    summary: str = Field(..., description="Simple, non-technical explanation")
    key_findings: List[str] = Field(..., description="Main points in accessible language")
    visual_indicators: List[dict] = Field(..., description="Simple visual cues and their meanings")
    confidence_level: str = Field(..., description="High/Medium/Low confidence description")
    
class ConfidenceMetricsResponse(BaseModel):
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    explanation_reliability: float = Field(..., ge=0.0, le=1.0)
    model_certainty: float = Field(..., ge=0.0, le=1.0)
    data_sufficiency: str = Field(..., description="Sufficient/Limited/Insufficient")
    warnings: List[str] = Field(default=[], description="Confidence-related warnings")
    
class ExportRequest(BaseModel):
    format: str = Field(..., description="Export format: pdf, png, csv, json")
    include_annotations: bool = Field(True, description="Include user annotations")
    include_comparisons: bool = Field(True, description="Include comparison examples")
    simplified_version: bool = Field(False, description="Use simplified explanations")
    
class AnnotationRequest(BaseModel):
    region_id: str = Field(..., description="ID of the region being annotated")
    annotation_text: str = Field(..., description="User annotation text")
    annotation_type: str = Field(..., description="Type: note, question, concern, etc.")
    
class ComparisonExamplesResponse(BaseModel):
    normal_examples: List[dict] = Field(..., description="Normal drawings from same age group")
    anomalous_examples: List[dict] = Field(..., description="Other anomalous examples")
    explanation_context: str = Field(..., description="Context for the comparisons")
```

### Age Group Modeling

**Age Grouping Strategy:**
- Default: 1-year intervals (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years)
- Automatic merging when insufficient data (< 50 samples per age-subject combination)
- Configurable grouping intervals
- Subject-stratified training ensures balanced representation

**Model Storage:**
```python
@dataclass
class SubjectAwareAgeGroupModel:
    age_range: Tuple[float, float]
    method: str  # Always "autoencoder"
    architecture: str  # Always "subject_aware_hybrid"
    embedding_dimension: int  # Always 832
    parameters: Dict  # Autoencoder weights and biases
    training_stats: Dict  # Sample count, validation metrics, per-subject statistics
    threshold: float  # Overall reconstruction loss threshold
    component_thresholds: Dict[str, float]  # Thresholds for visual and subject components
    supported_subjects: List[str]  # Subject categories present in training data
    subject_statistics: Dict[str, Dict]  # Per-subject reconstruction loss statistics
```

### Subject-Aware Anomaly Attribution

**Attribution Methodology:**

The system determines anomaly attribution by analyzing component-specific reconstruction losses:

1. **Component Reconstruction Losses:**
   - Visual Loss: `||visual_original - visual_reconstructed||²` (dims 0-767)
   - Subject Loss: `||subject_original - subject_reconstructed||²` (dims 768-831)

2. **Attribution Logic:**
   ```python
   def determine_attribution(visual_loss, subject_loss, thresholds, age, drawing_age):
       visual_anomalous = visual_loss > thresholds['visual']
       subject_anomalous = subject_loss > thresholds['subject']
       
       # Check if drawing would be normal for a different age group
       age_related = check_cross_age_group_scores(drawing, age)
       
       if visual_anomalous and subject_anomalous:
           return "both"
       elif subject_anomalous:
           return "subject"
       elif visual_anomalous:
           return "visual"
       elif age_related:
           return "age"  # Normal for different age group
       else:
           return "normal"
   ```

3. **Component Threshold Calculation:**
   - Component thresholds calculated independently during validation
   - Based on 95th percentile (or configured percentile) of component-specific losses
   - Allows fine-grained attribution even when overall score is below threshold

4. **Age-Related Anomaly Detection:**
   - Age information determines which model to use for analysis
   - "Age-related" anomalies detected by comparing scores across age-group models
   - If a drawing scores as anomalous for its age but normal for a different age, it's age-related
   - Example: A 5-year-old's drawing that would be normal for a 3-year-old indicates developmental concerns

5. **Attribution Interpretation:**
   - **"age"**: Drawing deviates from age-expected patterns (detected via cross-model comparison)
   - **"subject"**: Drawing deviates from subject-specific norms (e.g., unusual representation of the subject)
   - **"both"**: Drawing shows anomalies in both visual and subject components
   - **"visual"**: Visual features are unusual but subject encoding is appropriate
   - **"normal"**: All components within expected ranges for the age group

**Design Rationale:**
- Component-level attribution provides actionable insights for clinicians and researchers
- Separating age and subject effects enables more nuanced understanding of developmental patterns
- Visual component attribution captures anomalies not explained by age or subject alone
- Threshold-based approach ensures consistent and interpretable attribution decisions

### Subject Category System

**Supported Categories:**

The system supports up to 64 distinct subject categories using a 64-dimensional one-hot encoding scheme. Currently defined categories include:

- **Objects**: TV, airplane, apple, bed, bike, boat, book, bottle, bowl, cactus, car, chair, clock, couch, cup, hat, house, ice cream, key, lamp, mushroom, phone, piano, scissors, train, tree, watch
- **Animals**: bear, bee, bird, camel, cat, cow, dog, elephant, fish, frog, horse, octopus, rabbit, sheep, snail, spider, tiger, whale
- **People**: face, hand, person
- **Special**: unspecified (default when subject not provided)

**Encoding Scheme:**
- 64-dimensional one-hot vector
- Each category assigned a unique position (0-63)
- "unspecified" category uses position 0
- Extensible to 64 total categories for future expansion

**Design Rationale:**
- **One-hot encoding** chosen over learned embeddings for interpretability and simplicity
- **64 dimensions** provides sufficient capacity for current categories (~50) plus future expansion
- **Fixed dimensionality** ensures consistent hybrid embedding size (832 dims) across all drawings
- **"unspecified" default** enables processing drawings without subject metadata while maintaining architecture consistency
- **Categorical approach** aligns with discrete nature of drawing subjects and facilitates subject-stratified training

**Subject Validation:**
- Input validation ensures only supported categories are accepted
- Case-insensitive matching with normalization
- Clear error messages for unsupported categories
- API documentation includes complete category list

**Future Extensibility:**
- Additional categories can be added up to 64 total
- Requires retraining models with updated subject encodings
- Backward compatibility maintained through versioned model metadata

## Key Design Decisions and Rationales

### 1. Unified Subject-Aware Architecture

**Decision:** Use hybrid embeddings (832-dim) for all drawings, combining visual features and subject information in a single unified architecture. Age is used for model selection, not as part of the embedding.

**Rationale:**
- **Consistency**: Same processing pipeline for all drawings regardless of subject availability
- **Simplicity**: Single model architecture per age group instead of multiple subject-specific models
- **Scalability**: Adding new subjects doesn't require new model architectures
- **Attribution**: Component separation enables clear attribution of anomaly sources
- **Efficiency**: Single forward pass through autoencoder provides all necessary information

**Alternatives Considered:**
- Separate age-only and subject-aware models: Rejected due to complexity and inconsistency
- Learned subject embeddings: Rejected in favor of interpretable one-hot encoding
- Subject-specific models: Rejected due to data requirements and maintenance burden

### 2. Autoencoder-Only Anomaly Detection

**Decision:** Use reconstruction loss from autoencoders as the sole anomaly detection method.

**Rationale:**
- **Interpretability**: Reconstruction loss directly shows what the model couldn't reproduce
- **Component Attribution**: Enables separate analysis of visual, age, and subject components
- **Proven Effectiveness**: Well-established method for anomaly detection in high-dimensional spaces
- **Training Stability**: More stable than adversarial or one-class approaches
- **Computational Efficiency**: Single forward pass for both training and inference

**Alternatives Considered:**
- One-class SVM: Rejected due to lack of component-level attribution
- VAE: Rejected due to complexity and difficulty in threshold interpretation
- Isolation Forest: Rejected due to inability to handle high-dimensional embeddings effectively

### 3. Vision Transformer (ViT) for Feature Extraction

**Decision:** Use Vision Transformer as the sole embedding model.

**Rationale:**
- **Global Context**: Attention mechanism captures relationships across entire drawing
- **Patch-Based Processing**: Natural alignment with children's drawing composition
- **Pre-trained Models**: Leverage transfer learning from large-scale image datasets
- **Interpretability**: Attention weights provide insight into model focus
- **State-of-the-Art**: Superior performance on visual tasks compared to CNNs

**Alternatives Considered:**
- ResNet/CNN: Rejected due to limited global context and attention visualization
- Multiple model support: Rejected to reduce complexity and ensure consistent behavior

### 4. Component-Level Anomaly Attribution

**Decision:** Calculate separate reconstruction losses for visual and subject components. Age-related anomalies are detected through cross-model comparison.

**Rationale:**
- **Clinical Utility**: Clinicians need to know whether anomaly is age-related, subject-related, or both
- **Actionable Insights**: Different interventions appropriate for different anomaly types
- **Research Value**: Enables study of age vs subject effects in drawing development
- **Transparency**: Clear explanation of what aspects of drawing are unusual
- **Validation**: Allows verification that attribution matches expert judgment

**Implementation:**
- Component thresholds calculated independently during validation
- Attribution logic uses threshold-based decision rules
- Multiple components can contribute to overall anomaly

### 5. Subject Stratification in Training

**Decision:** Ensure balanced representation across subject categories during training and validation.

**Rationale:**
- **Fairness**: Prevents model bias toward common subjects
- **Generalization**: Ensures model performs well across all subject categories
- **Data Efficiency**: Maximizes learning from limited data per subject
- **Validation Accuracy**: Stratified validation provides reliable performance estimates
- **Clinical Applicability**: Model should work equally well regardless of drawing subject

**Implementation:**
- Stratified splitting during dataset preparation
- Balanced batch sampling during training
- Per-subject performance metrics in validation
- Warnings when subject representation is insufficient

### 6. "Unspecified" Default Subject Category

**Decision:** Use "unspecified" as default subject category when subject information is unavailable.

**Rationale:**
- **Backward Compatibility**: Enables processing of legacy data without subject labels
- **Graceful Degradation**: System continues to function without subject metadata
- **Consistent Architecture**: Maintains 832-dimensional hybrid embeddings for all drawings
- **User Flexibility**: Users not required to provide subject information
- **Research Scenarios**: Supports exploratory analysis where subject is unknown

**Implementation:**
- "unspecified" assigned position 0 in one-hot encoding
- Models trained with "unspecified" examples from unlabeled data
- Attribution logic handles "unspecified" appropriately

### 7. Age as Model Selector, Not Embedding Component

**Decision:** Use age to select which autoencoder model to use, rather than including age as part of the embedding.

**Rationale:**
- **Architectural Clarity**: One model per age group is simpler than a single model trying to learn age-dependent patterns
- **Better Separation**: Age-specific models can specialize in patterns typical for that age range
- **Clearer Attribution**: Age-related anomalies detected by comparing across models, not from reconstruction loss
- **Training Efficiency**: Each model trains on homogeneous age data, improving convergence
- **Interpretability**: Easier to explain "this is unusual for a 5-year-old" when using a 5-year-old-specific model

**Implementation:**
- Age determines model selection: `select_model(age) -> age_group_model`
- Hybrid embeddings contain only visual (768-dim) + subject (64-dim) = 832 dimensions
- Age-related anomalies detected by testing drawing against multiple age-group models
- If drawing scores normal for a different age group, it's classified as age-related anomaly

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

### Property 6: Embedding Serialization Round Trip
*For any* generated embedding, serializing then deserializing should produce an equivalent embedding vector
**Validates: Requirements 2.6**

### Property 7: Training Data Split Consistency
*For any* dataset with specified split ratios, the training environment should create splits with correct proportions and no data overlap between splits
**Validates: Requirements 3.1**

### Property 8: Training Configuration Validation
*For any* training parameters within valid ranges, the training environment should accept the configuration, and reject parameters outside valid ranges
**Validates: Requirements 3.2**

### Property 9: Training Report Generation Completeness
*For any* completed training job, the system should generate a comprehensive report containing all required performance metrics and validation curves
**Validates: Requirements 3.4**

### Property 10: Model Export Compatibility
*For any* trained model that meets performance criteria, the exported parameters should be loadable by the production system
**Validates: Requirements 3.5**

### Property 11: Autoencoder Training Validity
*For any* set of training embeddings for an age group, the training process should produce a valid autoencoder model with reconstruction capabilities
**Validates: Requirements 3.7**

### Property 12: Insufficient Data Warning Generation
*For any* age group with below-threshold data counts, the training environment should generate appropriate warnings
**Validates: Requirements 3.8**

### Property 13: Anomaly Score Generation
*For any* processed drawing, the system should generate a valid anomaly score (finite number) using the appropriate age group model
**Validates: Requirements 4.1**

### Property 14: Reconstruction Loss Calculation Accuracy
*For any* drawing embedding and trained autoencoder, the reconstruction loss should be calculated as the mathematical difference between original and reconstructed embeddings
**Validates: Requirements 4.2, 4.3**

### Property 15: Threshold Calculation Accuracy
*For any* set of validation scores, the 95th percentile threshold should be mathematically correct and properly applied to subsequent analyses
**Validates: Requirements 5.1, 5.5**

### Property 16: Threshold-Based Anomaly Detection
*For any* drawing with computed anomaly score, the drawing should be flagged as anomalous if and only if its score exceeds the configured threshold
**Validates: Requirements 5.3**

### Property 17: Dynamic Threshold Updates
*For any* threshold configuration change, subsequent analyses should use the new threshold without requiring system restart
**Validates: Requirements 5.2, 5.4**

### Property 18: Interpretability Generation Completeness
*For any* drawing flagged as anomalous, the system should generate corresponding saliency maps and attention visualizations
**Validates: Requirements 6.1, 6.3, 6.4, 6.5**

### Property 19: Saliency Map Overlay Accuracy
*For any* generated saliency map, the visualization should be properly overlaid on the original drawing for clear interpretation
**Validates: Requirements 6.6, 7.4**

### Property 20: Comparison Example Provision
*For any* analysis result, the system should provide appropriate comparison examples from the same age group
**Validates: Requirements 7.3**

### Property 21: Batch Processing Consistency
*For any* batch of drawings, each drawing should receive the same analysis result whether processed individually or as part of the batch
**Validates: Requirements 7.5**

### Property 22: Error Handling Robustness
*For any* invalid input or system failure condition, the system should provide appropriate error messages and continue processing valid inputs
**Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**

### Property 23: Interactive Interpretability Consistency
*For any* saliency map region, hovering or clicking should consistently provide relevant explanations with appropriate confidence indicators
**Validates: Requirements 9.1, 9.2, 9.3**

### Property 24: Explanation Level Adaptation
*For any* interpretability result, the system should provide both technical and simplified explanations that are contextually appropriate for the target audience
**Validates: Requirements 9.2, 10.2, 10.5**

### Property 25: Confidence Metric Accuracy
*For any* interpretability result, confidence scores should accurately reflect the reliability and certainty of the explanations provided
**Validates: Requirements 9.3, 10.4**

### Property 26: Export Format Completeness
*For any* export request, the generated output should contain all requested components (annotations, comparisons, explanations) in the specified format
**Validates: Requirements 11.1, 11.2, 11.5**

### Property 27: Educational Guidance Effectiveness
*For any* first-time user or help request, the system should provide appropriate tutorials and contextual guidance for interpreting results
**Validates: Requirements 10.1, 10.2, 10.3**

### Property 28: Subject Category Validation
*For any* subject metadata provided during upload, the system should accept it if and only if it matches the supported subject categories (person, house, tree, animal, family, abstract, other, unspecified)
**Validates: Requirements 1.4**

### Property 29: Subject Encoding Consistency
*For any* subject category, the categorical encoding should be consistent and produce the same one-hot vector representation across all processing stages
**Validates: Requirements 2.4**

### Property 30: Hybrid Embedding Construction
*For any* drawing with visual features, age, and subject information, the hybrid embedding should contain all three components with consistent dimensionality
**Validates: Requirements 2.5**

### Property 31: Subject Fallback Handling
*For any* drawing without subject information, the system should use the default "unspecified" category encoding and process successfully
**Validates: Requirements 2.6**

### Property 32: Hybrid Embedding Dimensionality Consistency
*For any* set of drawings processed with hybrid embeddings, all embeddings should have identical vector dimensions regardless of subject category
**Validates: Requirements 2.7**

### Property 33: Hybrid Embedding Serialization Round Trip
*For any* hybrid embedding, serializing then deserializing should preserve all visual, age, and subject components without data loss
**Validates: Requirements 2.9**

### Property 34: Subject-Aware Model Training
*For any* training dataset with hybrid embeddings, the autoencoder training should successfully produce models capable of reconstructing the full hybrid embedding space
**Validates: Requirements 3.7**

### Property 35: Subject Stratification Balance
*For any* training dataset, the stratification process should ensure balanced representation across available age-subject combinations
**Validates: Requirements 3.8**

### Property 36: Insufficient Data Warning Generation
*For any* age-subject combination with below-threshold sample counts, the training environment should generate appropriate warnings and suggestions
**Validates: Requirements 3.9**

### Property 37: Subject-Aware Model Selection
*For any* drawing analysis, the system should use the appropriate age-subject aware model when subject information is available
**Validates: Requirements 4.1**

### Property 38: Subject-Aware Scoring Influence
*For any* drawing with subject information, the anomaly scoring should consider subject-specific patterns and differ from age-only scoring
**Validates: Requirements 4.2**

### Property 39: Subject-Missing Default Handling
*For any* drawing without subject information, the system should use "unspecified" subject category in the hybrid embedding for consistent analysis
**Validates: Requirements 4.3**

### Property 40: Anomaly Attribution Accuracy
*For any* anomaly analysis, the attribution should correctly identify whether the anomaly is age-related, subject-related, or both based on component reconstruction losses
**Validates: Requirements 4.6**

### Property 41: Subject-Aware Interpretability Attribution
*For any* interpretability explanation, the system should correctly identify whether anomalies are primarily age-related, subject-related, or combination-based
**Validates: Requirements 6.5**

### Property 42: Subject-Specific Comparison Provision
*For any* analysis result with subject information, comparison examples should come from the same age-subject category when available
**Validates: Requirements 6.7**

### Property 43: Unified Subject-Aware Architecture
*For any* model training, the system should use only subject-aware hybrid embedding architecture for consistent behavior across all age groups
**Validates: Requirements 12.2, 12.4**

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
- **Mathematical Correctness**: Reconstruction loss calculations, autoencoder training
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