# Requirements Document

## Introduction

The Children's Drawing Anomaly Detection System is a machine learning-based tool designed to analyze children's drawings and identify patterns that significantly deviate from age-expected and subject-specific visual norms. The system aims to flag potential developmental, emotional, or perceptual anomalies by comparing individual drawings against established age-based and subject-aware baselines, providing interpretable insights for researchers, educators, and child development professionals. The system now incorporates subject-based modeling alongside age-based modeling to provide more nuanced and contextually appropriate anomaly detection.

## Glossary

- **Drawing_Analysis_System**: The complete software system that processes children's drawings and detects anomalies
- **Age_Group_Model**: Statistical or machine learning model trained on drawings from children within a specific age range
- **Subject_Aware_Model**: Machine learning model trained on drawings that considers both age and subject category information
- **Subject_Category**: The type of object or scene depicted in a drawing (e.g., "person", "house", "tree", "animal", "family", "abstract")
- **Anomaly_Score**: Numerical value indicating how much a drawing deviates from age-expected and subject-specific patterns
- **Drawing_Embedding**: Fixed-length numerical vector representation of a drawing's visual features
- **Hybrid_Embedding**: Combined feature vector containing visual embeddings, age information, and subject category encoding
- **Subject_Embedding**: Categorical encoding of the drawing subject integrated into the feature vector
- **Interpretability_Engine**: Component that generates explanations for why a drawing was flagged as anomalous
- **Threshold_Manager**: Component that determines the cutoff point for flagging drawings as anomalous
- **Vision_Encoder**: Deep learning model that converts images into feature vectors
- **Saliency_Map**: Visual representation highlighting which parts of a drawing contribute to anomaly detection
- **Reconstruction_Loss**: Numerical measure of difference between original and autoencoder-reconstructed embeddings
- **Batch_Processing**: System capability to analyze multiple drawings simultaneously with progress tracking
- **Training_Environment**: Offline system for model development, experimentation, and optimization
- **Model_Deployment**: Process of loading trained model parameters into the production system
- **SageMaker_Training**: Amazon Web Services cloud-based machine learning training service
- **Categorical_Encoding**: One-hot or embedding-based representation of discrete subject categories
- **Subject_Stratified_Training**: Training approach that ensures balanced representation across subject categories within age groups
- **Cross_Subject_Validation**: Evaluation method that tests model performance across different subject categories

## Requirements

### Requirement 1

**User Story:** As a child development researcher, I want to upload children's drawings with age and subject information, so that I can identify drawings that deviate from typical developmental and subject-specific patterns.

#### Acceptance Criteria

1. WHEN a user uploads a drawing image with age metadata, THE Drawing_Analysis_System SHALL accept common image formats (PNG, JPEG, BMP) and validate the age input
2. WHEN processing uploaded data, THE Drawing_Analysis_System SHALL resize images to standardized dimensions and normalize pixel values
3. WHEN age information is provided, THE Drawing_Analysis_System SHALL validate that age is within acceptable range (2-18 years)
4. WHEN subject metadata is provided, THE Drawing_Analysis_System SHALL validate the subject category against supported categories (up to 64 predefined categories including person, house, tree, animal, family, abstract, other) and store this information for enhanced analysis
5. WHEN subject information is missing, THE Drawing_Analysis_System SHALL allow optional subject classification or default to "unspecified" category
6. WHERE expert annotations are available, THE Drawing_Analysis_System SHALL incorporate these labels for model validation
7. WHEN processing drawings without subject information, THE Drawing_Analysis_System SHALL use "unspecified" as the default subject category

### Requirement 2

**User Story:** As a system administrator, I want the system to convert drawings into numerical representations that include subject information, so that mathematical analysis can be performed on visual content with subject-aware context.

#### Acceptance Criteria

1. WHEN a drawing image is processed, THE Vision_Encoder SHALL generate a fixed-length feature vector representation using Vision Transformer architecture
2. WHEN embedding generation occurs, THE Drawing_Analysis_System SHALL use Vision Transformer as the primary and only encoder model
3. WHERE age information is available, THE Drawing_Analysis_System SHALL optionally concatenate age data to the drawing embedding
4. WHEN subject information is available, THE Drawing_Analysis_System SHALL create 64-dimension one-hot categorical encodings for subject categories and concatenate with visual embeddings
5. WHEN combining features, THE Drawing_Analysis_System SHALL create hybrid embeddings containing visual features, age information, and subject category encodings
6. WHEN subject information is unavailable, THE Drawing_Analysis_System SHALL use a default "unspecified" category encoding
7. WHEN embeddings are created, THE Drawing_Analysis_System SHALL ensure consistent dimensionality across all processed drawings including hybrid embeddings
8. WHEN processing fails, THE Drawing_Analysis_System SHALL log errors and provide meaningful error messages
9. WHEN serializing embeddings to storage, THE Drawing_Analysis_System SHALL encode and decode embedding vectors without data loss while maintaining the ability to separate visual, age, and subject components

### Requirement 3

**User Story:** As a data scientist, I want an offline training environment to develop and optimize age-appropriate and subject-aware drawing pattern models, so that I can experiment with different parameters and deploy the best performing models to the production system.

#### Acceptance Criteria

1. WHEN provided with a drawing folder and metadata file, THE Training_Environment SHALL automatically organize data into train/validation/test splits with configurable ratios
2. WHEN training parameters are specified, THE Training_Environment SHALL accept configuration for learning rate, number of epochs, batch size, and model architecture parameters
3. WHEN training is initiated, THE Training_Environment SHALL support both local training and Amazon SageMaker cloud training environments
4. WHEN training completes, THE Training_Environment SHALL generate a comprehensive summary report including model performance metrics, validation curves, and anomaly detection accuracy
5. WHEN a trained model meets performance criteria, THE Training_Environment SHALL export model parameters in a format compatible with the production system
6. WHEN model deployment is requested, THE Drawing_Analysis_System SHALL provide API endpoints and scripts to load new model parameters without system downtime
7. WHEN training data is available for an age group, THE Subject_Aware_Model SHALL train autoencoder models using hybrid embeddings containing visual features, age information, and subject category encodings
8. WHEN organizing training data, THE Training_Environment SHALL stratify samples to ensure balanced representation across age groups and subject categories
9. WHEN insufficient data exists for a specific age-subject combination, THE Training_Environment SHALL provide warnings and suggest data collection, category merging, or age group merging strategies
10. WHEN validating models, THE Training_Environment SHALL evaluate performance across different subject categories within each age group
11. WHEN retraining models, THE Training_Environment SHALL use only subject-aware hybrid embedding architecture for consistent model behavior

### Requirement 4

**User Story:** As a clinician, I want the system to generate anomaly scores that consider both age and subject context, so that I can prioritize which cases need further attention with more nuanced understanding.

#### Acceptance Criteria

1. WHEN a new drawing is analyzed, THE Drawing_Analysis_System SHALL compute reconstruction loss using the appropriate age-subject aware autoencoder model
2. WHEN subject information is provided, THE Drawing_Analysis_System SHALL use subject-aware scoring that considers subject-specific patterns within the age group
3. WHEN subject information is missing, THE Drawing_Analysis_System SHALL use "unspecified" subject category in the hybrid embedding for consistent analysis
4. WHEN scoring occurs, THE Drawing_Analysis_System SHALL use reconstruction error as the anomaly score for consistent interpretation across age groups and subject categories
5. WHEN reconstruction loss is calculated, THE Drawing_Analysis_System SHALL measure the difference between original and reconstructed hybrid embeddings
6. WHEN generating anomaly scores, THE Drawing_Analysis_System SHALL indicate whether the anomaly is age-related, subject-related, or both
7. WHEN processing a drawing, THE Drawing_Analysis_System SHALL complete scoring within 30 seconds for standard image sizes
8. WHEN scoring fails, THE Drawing_Analysis_System SHALL provide clear error reporting and retry mechanisms

### Requirement 5

**User Story:** As a school psychologist, I want configurable thresholds for flagging drawings, so that I can adjust sensitivity based on my specific use case.

#### Acceptance Criteria

1. WHEN establishing thresholds, THE Threshold_Manager SHALL calculate default cutoffs using the 95th percentile of validation scores
2. WHEN users require different sensitivity levels, THE Threshold_Manager SHALL allow threshold adjustment through configuration parameters
3. WHEN a drawing score exceeds the threshold, THE Drawing_Analysis_System SHALL flag it as a potential anomaly
4. WHEN threshold changes occur, THE Drawing_Analysis_System SHALL apply new thresholds to subsequent analyses without requiring system restart
5. WHEN validation data is updated, THE Threshold_Manager SHALL recalculate appropriate threshold values

### Requirement 6

**User Story:** As a researcher, I want to understand why specific drawings were flagged as anomalous in both age and subject context, so that I can validate the system's decisions and gain insights.

#### Acceptance Criteria

1. WHEN a drawing is flagged as anomalous, THE Interpretability_Engine SHALL generate saliency maps highlighting contributing regions
2. WHEN using CNN-based encoders, THE Interpretability_Engine SHALL implement Grad-CAM visualization techniques
3. WHERE Vision Transformer models are used, THE Interpretability_Engine SHALL provide attention map visualizations
4. WHEN explanations are requested, THE Drawing_Analysis_System SHALL identify specific visual features that contributed to the anomaly score
5. WHEN generating explanations, THE Interpretability_Engine SHALL identify whether anomalies are primarily age-related, subject-related, or combination-based
6. WHEN creating saliency maps, THE Interpretability_Engine SHALL highlight features that deviate from both age and subject norms
7. WHEN providing comparisons, THE Drawing_Analysis_System SHALL show examples from the same age-subject category when available
8. WHEN explaining decisions, THE Drawing_Analysis_System SHALL describe how the drawing compares to typical patterns for that specific subject and age combination
9. WHEN confidence is low, THE Interpretability_Engine SHALL indicate whether uncertainty stems from age modeling, subject modeling, or insufficient training data
10. WHEN interpretability analysis completes, THE Drawing_Analysis_System SHALL present results in human-readable format
11. WHEN generating saliency maps, THE Interpretability_Engine SHALL overlay visualization data on original drawings for clear interpretation

### Requirement 7

**User Story:** As an end user, I want an intuitive interface to upload drawings with subject information and view subject-aware results, so that I can efficiently analyze multiple drawings without technical expertise.

#### Acceptance Criteria

1. WHEN users access the system, THE Drawing_Analysis_System SHALL provide a web-based interface for drawing upload and analysis
2. WHEN uploading drawings, THE Drawing_Analysis_System SHALL provide intuitive subject category selection with visual examples
3. WHEN analysis completes, THE Drawing_Analysis_System SHALL display anomaly scores with clear visual indicators
4. WHEN results are shown, THE Drawing_Analysis_System SHALL provide comparison with similar normal examples from the same age-subject category when available
5. WHEN displaying results, THE Drawing_Analysis_System SHALL show subject-specific context and comparisons
6. WHEN interpretability is available, THE Drawing_Analysis_System SHALL overlay saliency information on the original drawing
7. WHERE batch processing is needed, THE Drawing_Analysis_System SHALL support multiple drawing uploads with progress tracking
8. WHEN viewing analysis history, THE Drawing_Analysis_System SHALL filter and group results by subject category
9. WHEN configuring the system, THE Drawing_Analysis_System SHALL allow users to enable or disable subject-aware analysis
10. WHEN displaying analysis results, THE Drawing_Analysis_System SHALL present information in an accessible format suitable for non-technical users
11. WHEN exporting results, THE Drawing_Analysis_System SHALL include subject-specific analysis information in reports

### Requirement 8

**User Story:** As a system maintainer, I want robust data validation and error handling, so that the system operates reliably with diverse input data.

#### Acceptance Criteria

1. WHEN invalid image formats are uploaded, THE Drawing_Analysis_System SHALL reject the input and provide clear error messages
2. WHEN age information is missing or invalid, THE Drawing_Analysis_System SHALL prompt for correction or use default handling
3. WHEN model inference fails, THE Drawing_Analysis_System SHALL log detailed error information and attempt graceful recovery
4. WHEN system resources are insufficient, THE Drawing_Analysis_System SHALL queue requests and provide estimated processing times
5. WHEN data corruption is detected, THE Drawing_Analysis_System SHALL isolate affected components and continue processing valid inputs

### Requirement 9

**User Story:** As a clinician reviewing anomaly results, I want to interactively explore why a drawing was flagged, so that I can understand and communicate the findings effectively.

#### Acceptance Criteria

1. WHEN viewing analysis results, THE Drawing_Analysis_System SHALL provide interactive saliency map regions that users can hover over to see detailed explanations of what the model detected
2. WHEN explanations are displayed, THE Drawing_Analysis_System SHALL provide both technical explanations for researchers and simplified explanations for educators and parents
3. WHEN interpretability results are shown, THE Drawing_Analysis_System SHALL include confidence levels and reliability scores to help users assess the trustworthiness of the interpretation
4. WHEN anomaly results are presented, THE Drawing_Analysis_System SHALL show side-by-side comparisons demonstrating how the flagged drawing differs from typical drawings in the same age group
5. WHEN users click on specific areas of the drawing, THE Drawing_Analysis_System SHALL provide targeted explanations about what makes that region unusual
6. WHEN saliency maps are generated, THE Drawing_Analysis_System SHALL ensure proper overlay alignment and visual clarity for interpretation

### Requirement 10

**User Story:** As an educator without ML expertise, I want clear guidance on interpreting saliency maps, so that I can understand what the system is telling me about a child's drawing.

#### Acceptance Criteria

1. WHEN first-time users access interpretability features, THE Drawing_Analysis_System SHALL provide guided tours explaining how to read saliency maps and attention visualizations
2. WHEN users interact with technical features, THE Drawing_Analysis_System SHALL display contextual help tooltips that explain technical concepts in accessible language
3. WHEN users need reference examples, THE Drawing_Analysis_System SHALL provide access to a library of example interpretations showing common patterns and what they might indicate
4. WHEN interpretations are uncertain, THE Drawing_Analysis_System SHALL clearly communicate interpretation confidence levels and when more data is needed
5. WHEN educational content is displayed, THE Drawing_Analysis_System SHALL adapt explanation complexity based on user role and expertise level

### Requirement 11

**User Story:** As a researcher, I want to export interpretability results for reports and presentations, so that I can share findings with colleagues and include them in publications.

#### Acceptance Criteria

1. WHEN users need to export results, THE Drawing_Analysis_System SHALL provide multiple export formats including PDF reports, PNG images, and CSV data files
2. WHEN comprehensive reports are requested, THE Drawing_Analysis_System SHALL generate complete interpretation reports combining visual and textual explanations
3. WHEN users want to add context, THE Drawing_Analysis_System SHALL provide annotation tools allowing users to add their own notes and annotations to interpretability results
4. WHEN longitudinal analysis is needed, THE Drawing_Analysis_System SHALL maintain a history of interpretations for tracking changes over time
5. WHEN sharing results, THE Drawing_Analysis_System SHALL ensure exported materials include all necessary context and metadata for proper interpretation

### Requirement 12

**User Story:** As a system administrator, I want to transition to subject-aware modeling with a clean database reset, so that the system operates with consistent subject-aware models from the start.

#### Acceptance Criteria

1. WHEN transitioning to subject-aware modeling, THE Drawing_Analysis_System SHALL support database reset and clean retraining of all models
2. WHEN retraining models, THE Drawing_Analysis_System SHALL use only subject-aware hybrid embedding architecture for all new models
3. WHEN insufficient data exists for specific age-subject combinations, THE Drawing_Analysis_System SHALL automatically merge similar categories or age ranges
4. WHEN managing models, THE Drawing_Analysis_System SHALL use unified subject-aware model architecture for all age groups
5. WHEN analyzing drawings, THE Drawing_Analysis_System SHALL always use subject-aware analysis (with "unspecified" for missing subject information)