# Implementation Plan

- [x] 1. Set up project structure and development environment
  - Create Python backend project with venv virtual environment
  - Set up requirements.txt for dependency management
  - Set up FastAPI application with proper project structure
  - Create React frontend with Vite and TypeScript
  - Configure development tools (Black, isort, flake8, ESLint, Prettier)
  - Set up Docker and Docker Compose for development
  - _Requirements: All system requirements_

- [x] 2. Implement database layer and core models
- [x] 2.1 Create SQLAlchemy database models and migrations
  - Implement Drawing, DrawingEmbedding, AgeGroupModel, AnomalyAnalysis, and InterpretabilityResult models
  - Add TrainingJob and TrainingReport models for training environment
  - Set up Alembic for database migrations
  - Configure SQLite database with WAL mode
  - _Requirements: 1.4, 1.5, 3.1, 3.4, 4.1, 6.1_

- [x] 2.2 Write property test for database model consistency
  - **Property 3: Metadata Persistence**
  - **Validates: Requirements 1.4, 1.5**

- [x] 2.3 Create Pydantic API schemas and validation
  - Implement request/response models for all API endpoints
  - Add training environment schemas (TrainingConfigRequest, TrainingJobResponse, etc.)
  - Add field validation for age ranges, file formats, and enum values
  - _Requirements: 1.1, 1.3, 3.2, 8.1, 8.2_

- [x] 2.4 Write property test for input validation
  - **Property 1: Image Format Validation Consistency**
  - **Validates: Requirements 1.1, 1.3**

- [x] 3. Implement data pipeline service
- [x] 3.1 Create image preprocessing and validation utilities
  - Implement image format validation (PNG, JPEG, BMP)
  - Create image resizing and normalization functions
  - Add error handling for corrupted or invalid images
  - _Requirements: 1.1, 1.2, 8.1_

- [x] 3.2 Write property test for image preprocessing consistency
  - **Property 2: Image Preprocessing Uniformity**
  - **Validates: Requirements 1.2**

- [x] 3.3 Implement file upload handling with FastAPI
  - Create multipart file upload endpoints
  - Add progress tracking for large file uploads
  - Implement file storage with proper naming and organization
  - _Requirements: 7.1, 7.5_

- [x] 3.4 Write property test for error handling robustness
  - **Property 13: Error Handling Robustness**
  - **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**

- [x] 4. Implement embedding service with vision models
- [x] 4.1 Set up PyTorch and Transformers integration
  - Configure Vision Transformer model from Hugging Face
  - Implement model loading and caching mechanisms
  - Add GPU/CPU detection and device management
  - _Requirements: 2.1, 2.2_

- [x] 4.2 Create embedding generation pipeline
  - Implement image-to-embedding conversion using Vision Transformer
  - Add age information concatenation option
  - Create batch processing capabilities for multiple images
  - _Requirements: 2.1, 2.3, 2.4_

- [x] 4.3 Write property test for embedding consistency
  - **Property 4: Embedding Dimensionality Consistency**
  - **Validates: Requirements 2.1, 2.4**

- [x] 4.4 Write property test for age-augmented embeddings
  - **Property 5: Age-Augmented Embedding Consistency**
  - **Validates: Requirements 2.3**

- [x] 4.5 Implement embedding serialization system
  - Create embedding serialization and deserialization utilities
  - Add support for numpy array storage in database
  - Implement embedding caching mechanisms
  - _Requirements: 2.6_

- [x] 4.6 Write property test for embedding serialization
  - **Property 6: Embedding Serialization Round Trip**
  - **Validates: Requirements 2.6**

- [x] 5. Implement offline training environment
- [x] 5.1 Create dataset preparation and splitting utilities
  - Implement folder-based dataset loading with metadata parsing
  - Create configurable train/validation/test split functionality
  - Add data validation and preprocessing for training pipeline
  - _Requirements: 3.1_

- [x] 5.2 Write property test for data splitting consistency
  - **Property 7: Training Data Split Consistency**
  - **Validates: Requirements 3.1**

- [x] 5.3 Implement training configuration management
  - Create Hydra-based configuration system for training parameters
  - Add validation for learning rate, batch size, epochs, and architecture parameters
  - Implement parameter sweep and optimization capabilities
  - _Requirements: 3.2_

- [x] 5.4 Write property test for training configuration validation
  - **Property 8: Training Configuration Validation**
  - **Validates: Requirements 3.2**

- [x] 5.5 Create local training environment
  - Implement local autoencoder training with PyTorch
  - Add GPU/CPU training support with device detection
  - Create training progress monitoring and logging
  - _Requirements: 3.7_

- [x] 5.6 Implement Amazon SageMaker integration
  - Set up SageMaker training job submission and monitoring
  - Create Docker containers for SageMaker training environment
  - Add Boto3 integration for job management and artifact retrieval
  - _Requirements: 3.3_

- [x] 5.7 Create training report generation system
  - Implement comprehensive training metrics collection
  - Add validation curve plotting and performance visualization
  - Create summary report generation with model performance analysis
  - _Requirements: 3.4_

- [x] 5.8 Write property test for training report completeness
  - **Property 9: Training Report Generation Completeness**
  - **Validates: Requirements 3.4**

- [x] 5.9 Implement model export and deployment system
  - Create model parameter export in production-compatible format
  - Add model validation and compatibility checking
  - Implement deployment API endpoints for model loading
  - _Requirements: 3.5, 3.6_

- [x] 5.10 Write property test for model export compatibility
  - **Property 10: Model Export Compatibility**
  - **Validates: Requirements 3.5**

- [x] 5.11 Add insufficient data handling and warnings
  - Implement data count validation for age groups
  - Create warning generation for insufficient training data
  - Add suggestions for data augmentation and age group merging
  - _Requirements: 3.8_

- [x] 5.12 Write property test for insufficient data warnings
  - **Property 12: Insufficient Data Warning Generation**
  - **Validates: Requirements 3.8**

- [x] 6. Implement age-based modeling and anomaly detection
- [x] 6.1 Create autoencoder architecture for reconstruction
  - Implement encoder-decoder architecture for embedding reconstruction
  - Create age-group specific autoencoder training pipeline
  - Add model saving and loading functionality
  - _Requirements: 3.7, 4.1_

- [x] 6.2 Write property test for autoencoder training
  - **Property 11: Autoencoder Training Validity**
  - **Validates: Requirements 3.7**

- [x] 6.3 Implement reconstruction loss calculation
  - Add reconstruction error computation between original and reconstructed embeddings
  - Create loss function optimization for training
  - Implement validation and early stopping mechanisms
  - _Requirements: 4.2, 4.3_

- [x] 6.4 Write property test for reconstruction loss calculation
  - **Property 14: Reconstruction Loss Calculation Accuracy**
  - **Validates: Requirements 4.2, 4.3**

- [x] 6.5 Create age group management system
  - Implement automatic age group creation and merging
  - Add insufficient data handling and warnings
  - Create age group autoencoder training and persistence
  - _Requirements: 3.8_

- [x] 7. Implement threshold management and scoring
- [x] 7.1 Create threshold calculation and management
  - Implement 95th percentile threshold calculation
  - Add configurable threshold adjustment capabilities
  - Create dynamic threshold updates without restart
  - _Requirements: 5.1, 5.2, 5.4_

- [x] 7.2 Write property test for threshold calculations
  - **Property 15: Threshold Calculation Accuracy**
  - **Validates: Requirements 5.1, 5.5**

- [x] 7.3 Write property test for threshold-based detection
  - **Property 16: Threshold-Based Anomaly Detection**
  - **Validates: Requirements 5.3**

- [x] 7.4 Implement reconstruction loss normalization system
  - Create cross-age-group reconstruction loss normalization
  - Add confidence calculation for anomaly decisions based on reconstruction error
  - Implement score comparison and ranking utilities
  - _Requirements: 4.2, 5.3_

- [x] 7.5 Write property test for dynamic threshold updates
  - **Property 17: Dynamic Threshold Updates**
  - **Validates: Requirements 5.2, 5.4**

- [x] 8. Checkpoint - Ensure all core ML components are working
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Implement interpretability engine
- [x] 9.1 Create saliency map generation for Vision Transformer
  - Implement attention visualization for Vision Transformers
  - Add attention rollout techniques for multi-layer attention
  - Create patch-level importance scoring
  - _Requirements: 6.1, 6.3_

- [x] 9.2 Implement explanation generation system
  - Create visual feature identification algorithms
  - Add human-readable explanation text generation
  - Implement importance region detection and highlighting
  - _Requirements: 6.4, 6.5_

- [x] 9.3 Write property test for interpretability completeness
  - **Property 18: Interpretability Generation Completeness**
  - **Validates: Requirements 6.1, 6.3, 6.4, 6.5**

- [x] 9.4 Create saliency overlay and visualization utilities
  - Implement image overlay generation for saliency maps
  - Add bounding box detection for important regions
  - Create exportable visualization formats
  - _Requirements: 6.6, 7.4_

- [x] 9.5 Write property test for saliency map overlay accuracy
  - **Property 19: Saliency Map Overlay Accuracy**
  - **Validates: Requirements 6.6, 7.4**

- [x] 10. Implement FastAPI backend endpoints
- [x] 10.1 Create drawing management API endpoints
  - Implement upload, retrieve, list, and delete endpoints
  - Add metadata filtering and search capabilities
  - Create proper error handling and validation
  - _Requirements: 7.1, 8.1, 8.2_

- [x] 10.2 Implement analysis operation endpoints
  - Create single drawing analysis endpoint
  - Add batch analysis capabilities with progress tracking
  - Implement analysis result retrieval and history
  - _Requirements: 4.1, 7.2, 7.5_

- [x] 10.3 Create model management and configuration endpoints
  - Implement age group autoencoder listing and training endpoints
  - Add threshold and configuration update endpoints
  - Create system status and health check endpoints
  - _Requirements: 5.2, 5.4_

- [x] 10.4 Add training environment API endpoints
  - Implement training job submission and monitoring endpoints
  - Add training report retrieval and model deployment endpoints
  - Create SageMaker job management and status tracking
  - _Requirements: 3.4, 3.5, 3.6_

- [x] 10.5 Write property test for batch processing consistency
  - **Property 21: Batch Processing Consistency**
  - **Validates: Requirements 7.5**

- [x] 10.6 Write property test for comparison example provision
  - **Property 20: Comparison Example Provision**
  - **Validates: Requirements 7.3**

- [x] 10.7 Implement enhanced interpretability API endpoints
  - Add GET /api/interpretability/{analysis_id}/interactive for interactive saliency data
  - Create GET /api/interpretability/{analysis_id}/simplified for simplified explanations
  - Implement GET /api/interpretability/{analysis_id}/confidence for confidence metrics
  - Add POST /api/interpretability/{analysis_id}/export for result export functionality
  - Create GET /api/interpretability/examples/{age_group} for comparison examples
  - _Requirements: 9.1, 9.2, 9.3, 10.3, 11.1_

- [x] 10.8 Enhance interpretability engine with interactive features
  - Add region-specific explanation generation for click interactions
  - Implement confidence scoring for interpretability results
  - Create comparative analysis utilities for normal vs anomalous examples
  - Add export functionality for multiple formats (PDF, PNG, CSV)
  - _Requirements: 9.1, 9.4, 9.5, 11.1, 11.2_

- [x] 11. Implement React frontend application
- [x] 11.1 Set up React project structure and routing
  - Create main application structure with React Router
  - Set up Material-UI theme and component library
  - Configure Zustand for state management
  - Add React Query for server state management
  - _Requirements: 7.1_

- [x] 11.2 Create drawing upload interface
  - Implement drag-and-drop file upload with React Dropzone
  - Create metadata input form with validation
  - Add upload progress tracking and error handling
  - _Requirements: 7.1, 7.5_

- [x] 11.3 Implement analysis results display
  - Create anomaly score visualization components
  - Add saliency map overlay functionality
  - Implement comparison with normal examples display
  - _Requirements: 7.2, 7.3, 7.4_

- [x] 11.4 Create dashboard and configuration interfaces
  - Implement system overview dashboard
  - Add threshold configuration and autoencoder status display
  - Create batch processing management interface
  - _Requirements: 7.1, 5.2_

- [ ] 11.5 Add training environment management interface
  - Create training job submission and monitoring interface
  - Add training report visualization and model deployment controls
  - Implement SageMaker job status tracking and management
  - _Requirements: 3.4, 3.5, 3.6_

- [x] 11.7 Implement interactive interpretability components
  - Create InteractiveInterpretabilityViewer with hoverable saliency map overlays
  - Add click-to-zoom functionality and region-specific explanation panels
  - Implement ExplanationLevelToggle for technical vs simplified explanations
  - Add ConfidenceIndicator with visual confidence meters and reliability warnings
  - _Requirements: 9.1, 9.2, 9.3, 9.6_

- [x] 11.8 Create educational interpretability guidance system
  - Implement InterpretabilityTutorial with interactive onboarding flow
  - Add contextual help system with role-based explanations
  - Create example gallery component showing common interpretation patterns
  - Add adaptive explanation complexity based on user expertise level
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 11.9 Implement comparative analysis and export features
  - Create ComparativeAnalysisPanel with side-by-side normal vs anomalous examples
  - Add ExportToolbar with multiple format options (PDF, PNG, CSV)
  - Implement annotation tools for user notes and context
  - Create historical interpretation tracking and longitudinal analysis
  - _Requirements: 9.4, 11.1, 11.2, 11.3, 11.4, 11.5_

- [x] 11.6 Write integration tests for frontend components
  - Create tests for upload workflow
  - Add tests for analysis result display
  - Test configuration and settings functionality
  - _Requirements: 7.1, 7.2, 7.3_

- [x] 12. Implement system integration and error handling
- [x] 12.1 Create comprehensive error handling middleware
  - Add global exception handling for FastAPI
  - Implement proper HTTP status codes and error messages
  - Create error logging and monitoring system
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 12.2 Add system monitoring and health checks
  - Implement health check endpoints for all services
  - Add resource usage monitoring and alerts
  - Create request queuing for resource management
  - _Requirements: 8.4_

- [x] 12.3 Implement data persistence and backup systems
  - Add automatic database backup mechanisms
  - Create data export and import utilities
  - Implement file storage organization and cleanup
  - _Requirements: 1.4, 1.5_

- [x] 12.4 Write property test for error handling robustness
  - **Property 22: Error Handling Robustness**
  - **Validates: Requirements 2.5, 4.5, 8.1, 8.2, 8.3, 8.5**

- [ ] 12.5 Write property tests for enhanced interpretability features
  - **Property 23: Interactive Interpretability Consistency**
  - **Property 24: Explanation Level Adaptation**
  - **Property 25: Confidence Metric Accuracy**
  - **Property 26: Export Format Completeness**
  - **Property 27: Educational Guidance Effectiveness**
  - **Validates: Requirements 9.1, 9.2, 9.3, 10.1, 10.2, 10.3, 10.4, 10.5, 11.1, 11.2, 11.5**

- [x] 13. Final integration and testing
- [x] 13.1 Create end-to-end integration tests
  - Test complete workflow from upload to analysis
  - Verify all API endpoints work correctly
  - Test error scenarios and recovery mechanisms
  - _Requirements: All requirements_

- [x] 13.2 Write comprehensive unit tests for remaining components
  - Add unit tests for utility functions
  - Test edge cases and boundary conditions
  - Verify mathematical calculations and algorithms
  - _Requirements: All requirements_

- [x] 13.3 Set up deployment configuration
  - Create Docker containers for production deployment
  - Add environment configuration management
  - Set up database initialization and migration scripts
  - _Requirements: System deployment_

- [ ] 13.4 Add training environment deployment configuration
  - Create SageMaker training container configurations
  - Set up AWS IAM roles and permissions for training jobs
  - Add training environment Docker containers and deployment scripts
  - _Requirements: 3.3, 3.6_

- [x] 14. Final Checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 15. Implement subject-aware architecture foundation
- [x] 15.1 Add subject category enumeration and validation
  - Create SubjectCategory enum with all 50+ supported categories
  - Add "unspecified" as default category
  - Implement subject category validation in API schemas
  - _Requirements: 1.4, 1.7_

- [x] 15.2 Update database models for subject-aware architecture
  - Add embedding_type, visual_component, subject_component columns to DrawingEmbedding
  - Add supports_subjects, subject_categories, embedding_type columns to AgeGroupModel
  - Add visual_anomaly_score, subject_anomaly_score, anomaly_attribution, analysis_type columns to AnomalyAnalysis
  - Create Alembic migration for schema changes
  - _Requirements: 1.4, 2.4, 4.6_

- [x] 15.3 Write property test for subject category validation
  - **Property 28: Subject Category Validation**
  - **Validates: Requirements 1.4**

- [x] 16. Implement hybrid embedding generation
- [x] 16.1 Create subject encoding utilities
  - Implement 64-dimensional one-hot encoding for subject categories
  - Add subject category to index mapping
  - Create encode_subject_category() function
  - Handle "unspecified" default category
  - _Requirements: 2.4, 2.6_

- [x] 16.2 Write property test for subject encoding consistency
  - **Property 29: Subject Encoding Consistency**
  - **Validates: Requirements 2.4**

- [x] 16.3 Update embedding service for hybrid embeddings
  - Modify generate_embedding() to create 832-dimensional hybrid embeddings
  - Combine visual features (768-dim) with subject encoding (64-dim)
  - Add separate_embedding_components() function
  - Update batch_embed() for hybrid embeddings
  - _Requirements: 2.5, 2.7_

- [x] 16.4 Write property test for hybrid embedding construction
  - **Property 30: Hybrid Embedding Construction**
  - **Validates: Requirements 2.5**

- [x] 16.5 Write property test for subject fallback handling
  - **Property 31: Subject Fallback Handling**
  - **Validates: Requirements 2.6**

- [x] 16.6 Write property test for hybrid embedding dimensionality
  - **Property 32: Hybrid Embedding Dimensionality Consistency**
  - **Validates: Requirements 2.7**

- [x] 16.7 Update embedding serialization for hybrid embeddings
  - Modify serialization to preserve visual and subject components
  - Update deserialization to reconstruct hybrid embeddings
  - Add component separation in storage
  - _Requirements: 2.9_

- [x] 16.8 Write property test for hybrid embedding serialization
  - **Property 33: Hybrid Embedding Serialization Round Trip**
  - **Validates: Requirements 2.9**

- [-] 17. Implement subject-aware model training
- [x] 17.1 Update dataset preparation for subject stratification
  - Modify stratify_by_age() to include subject stratification
  - Implement balanced sampling across age-subject combinations
  - Add data sufficiency validation for age-subject pairs
  - Generate warnings for insufficient age-subject data
  - _Requirements: 3.8, 3.9_

- [x] 17.2 Write property test for subject stratification
  - **Property 35: Subject Stratification Balance**
  - **Validates: Requirements 3.8**

- [x] 17.3 Write property test for insufficient data warnings
  - **Property 36: Insufficient Data Warning Generation**
  - **Validates: Requirements 3.9**

- [x] 17.4 Update model manager for subject-aware autoencoders
  - Modify autoencoder architecture to handle 832-dimensional input
  - Update train_age_group_model() for hybrid embeddings
  - Add subject_categories metadata to trained models
  - Ensure all models use unified subject-aware architecture
  - _Requirements: 3.7, 12.2, 12.4_

- [x] 17.5 Write property test for subject-aware model training
  - **Property 34: Subject-Aware Model Training**
  - **Validates: Requirements 3.7**

- [x] 17.6 Write property test for unified architecture
  - **Property 43: Unified Subject-Aware Architecture**
  - **Validates: Requirements 12.2, 12.4**

- [x] 18. Implement subject-aware anomaly detection and attribution
- [x] 18.1 Update anomaly scoring for subject-aware analysis
  - Modify compute_anomaly_score() to use hybrid embeddings
  - Calculate overall reconstruction loss on full 832-dimensional embedding
  - Add component-specific loss calculation (visual: dims 0-767, subject: dims 768-831)
  - Store visual_anomaly_score and subject_anomaly_score
  - _Requirements: 4.1, 4.2, 4.3_

- [x] 18.2 Write property test for subject-aware model selection
  - **Property 37: Subject-Aware Model Selection**
  - **Validates: Requirements 4.1**

- [x] 18.3 Write property test for subject-aware scoring
  - **Property 38: Subject-Aware Scoring Influence**
  - **Validates: Requirements 4.2**

- [x] 18.4 Write property test for subject-missing handling
  - **Property 39: Subject-Missing Default Handling**
  - **Validates: Requirements 4.3**

- [x] 18.5 Implement anomaly attribution logic
  - Create determine_attribution() function
  - Calculate component-specific thresholds (visual, subject)
  - Implement attribution decision rules (age, subject, visual, both)
  - Add cross-age-group comparison for age-related detection
  - Store anomaly_attribution in analysis results
  - _Requirements: 4.6_

- [x] 18.6 Write property test for anomaly attribution
  - **Property 40: Anomaly Attribution Accuracy**
  - **Validates: Requirements 4.6**

- [x] 19. Update interpretability for subject-aware attribution
- [x] 19.1 Enhance interpretability engine with attribution context
  - Update explain_anomaly() to include attribution information
  - Add explain_subject_aware_anomaly() function
  - Generate attribution-specific explanations (age vs subject vs visual)
  - Update saliency map generation to highlight attribution-relevant regions
  - _Requirements: 6.5, 6.6_

- [x] 19.2 Write property test for subject-aware interpretability
  - **Property 41: Subject-Aware Interpretability Attribution**
  - **Validates: Requirements 6.5**

- [x] 19.3 Update comparison service for subject-specific examples
  - Modify get_comparison_examples() to filter by age AND subject
  - Implement fallback when age-subject examples unavailable
  - Add subject-specific context to comparisons
  - _Requirements: 6.7, 6.8_

- [x] 19.4 Write property test for subject-specific comparisons
  - **Property 42: Subject-Specific Comparison Provision**
  - **Validates: Requirements 6.7**

- [x] 20. Update API endpoints for subject-aware features
- [x] 20.1 Update analysis endpoints with subject-aware responses
  - Add visual_anomaly_score, subject_anomaly_score to AnomalyAnalysisResponse
  - Add anomaly_attribution field
  - Add subject_category field
  - Update analysis_type to "subject_aware"
  - _Requirements: 4.1, 4.6_

- [x] 20.2 Update interpretability endpoints for attribution
  - Add attribution information to interactive interpretability
  - Include subject-specific context in explanations
  - Update comparison examples endpoint to support subject filtering
  - Add GET /api/interpretability/{analysis_id}/attribution endpoint
  - _Requirements: 6.5, 6.7_

- [x] 20.3 Update configuration endpoints for subject management
  - Add endpoint to list supported subject categories
  - Add endpoint to get subject-specific statistics
  - Update model status to show subject-aware capabilities
  - _Requirements: 1.4, 12.4_

- [x] 21. Update frontend for subject-aware features
- [x] 21.1 Add subject category selection to upload interface
  - Create SubjectCategorySelect component with all 50+ categories
  - Add visual examples/icons for common subjects
  - Implement search/filter for subject selection
  - Make subject optional with "unspecified" default
  - _Requirements: 7.2, 7.9_

- [x] 21.2 Update analysis results display for attribution
  - Show anomaly attribution (age, subject, visual, both)
  - Display component-specific scores (visual, subject)
  - Add subject-specific context to explanations
  - Update comparison examples to show subject-matched examples
  - _Requirements: 7.5, 7.8_

- [x] 21.3 Add subject filtering to analysis history
  - Add subject filter to dashboard and history views
  - Group results by subject category
  - Show subject-specific statistics
  - _Requirements: 7.8_

- [x] 21.4 Update configuration interface for subject management
  - Add subject category management section
  - Show subject-specific model statistics
  - Display data sufficiency warnings per age-subject combination
  - Add toggle for subject-aware analysis (always enabled)
  - _Requirements: 7.9, 12.4_

- [ ] 22. Database migration and model retraining
- [x] 22.1 Create database migration script
  - Generate Alembic migration for all schema changes
  - Add data migration for existing records (set subject to "unspecified")
  - Update existing embeddings to hybrid format (pad with unspecified encoding)
  - Migrate existing analyses to subject-aware format
  - _Requirements: 12.1_

- [x] 22.2 Retrain all age-group models with subject-aware architecture
  - Regenerate all embeddings as hybrid embeddings
  - Retrain autoencoders on 832-dimensional hybrid embeddings
  - Calculate component-specific thresholds
  - Validate cross-subject performance
  - Deploy new models to production
  - _Requirements: 12.2, 12.3_

- [x] 22.3 Update training scripts for subject-aware workflow
  - Modify train_models.py for hybrid embeddings
  - Update train_models_offline.py for subject stratification
  - Add subject-aware validation to training pipeline
  - Update training reports to include subject statistics
  - _Requirements: 12.2_

- [ ] 23. Integration testing and validation
- [ ] 23.1 Write end-to-end tests for subject-aware workflow
  - Test upload with subject category
  - Test hybrid embedding generation
  - Test subject-aware anomaly detection
  - Test attribution accuracy
  - Test subject-specific comparisons
  - _Requirements: All subject-aware requirements_

- [ ] 23.2 Validate subject-aware system performance
  - Compare subject-aware vs age-only detection accuracy
  - Validate attribution correctness with expert labels
  - Test data sufficiency warnings
  - Verify cross-subject model performance
  - _Requirements: 3.10, 4.6_

- [ ] 23.3 Update documentation for subject-aware features
  - Document subject category system
  - Explain hybrid embedding architecture
  - Describe anomaly attribution logic
  - Add subject-aware API examples
  - Update user guides for subject selection
  - _Requirements: All subject-aware requirements_

- [ ] 24. Final subject-aware system checkpoint
  - Ensure all subject-aware tests pass
  - Validate hybrid embeddings are generated correctly
  - Verify attribution logic works as expected
  - Confirm subject-specific comparisons are accurate
  - Test with diverse subject categories
  - Ask the user if questions arise