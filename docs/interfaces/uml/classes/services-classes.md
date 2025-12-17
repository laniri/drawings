# Services Class Diagram

## Overview
Class diagram showing the structure and relationships of services classes.

## Classes Overview

### ScoreNormalizationError
**Source**: `app/services/score_normalizer.py`

**Inherits from**: Exception

### NormalizationConfig
**Source**: `app/services/score_normalizer.py`

**Attributes**:
- `normalization_method: str`
- `confidence_method: str`
- `min_samples_for_stats: int`
- `outlier_threshold: float`

### ScoreNormalizer
**Source**: `app/services/score_normalizer.py`

**Public Methods**:
- `normalize_score()`
- `calculate_confidence()`
- `compare_scores()`
- `get_normalization_summary()`
- `update_normalization_config()`
- `clear_cache()`
- `detect_outliers()`

### LocalTrainingError
**Source**: `app/services/local_training_environment.py`

**Inherits from**: Exception

### DeviceDetectionError
**Source**: `app/services/local_training_environment.py`

**Inherits from**: LocalTrainingError

### TrainingProgressError
**Source**: `app/services/local_training_environment.py`

**Inherits from**: LocalTrainingError

### TrainingProgress
**Source**: `app/services/local_training_environment.py`

**Attributes**:
- `job_id: int`
- `epoch: int`
- `total_epochs: int`
- `batch: int`
- `total_batches: int`
- `train_loss: float`
- `val_loss: Optional[float]`
- `learning_rate: float`
- `elapsed_time: float`
- `estimated_remaining: Optional[float]`
- `memory_usage: Optional[Dict]`

**Public Methods**:
- `epoch_progress()`
- `batch_progress()`

### DeviceManager
**Source**: `app/services/local_training_environment.py`

**Public Methods**:
- `device()`
- `device_info()`
- `get_memory_usage()`
- `clear_cache()`
- `optimize_for_training()`

### TrainingProgressMonitor
**Source**: `app/services/local_training_environment.py`

**Public Methods**:
- `add_callback()`
- `start_epoch()`
- `update_batch()`
- `update_epoch()`
- `get_latest_progress()`
- `get_history()`
- `stop()`

### LocalTrainingEnvironment
**Source**: `app/services/local_training_environment.py`

**Public Methods**:
- `get_environment_info()`
- `prepare_training_data()`
- `start_training_job()`
- `get_job_status()`
- `list_training_jobs()`
- `cancel_training_job()`

### EnhancedAutoencoderTrainer
**Source**: `app/services/local_training_environment.py`

**Inherits from**: AutoencoderTrainer

**Public Methods**:
- `train()`

### ComparisonService
**Source**: `app/services/comparison_service.py`

**Public Methods**:
- `find_similar_normal_examples()`
- `get_comparison_statistics()`

### ValidationResult
**Source**: `app/services/data_pipeline.py`

**Inherits from**: BaseModel

**Attributes**:
- `is_valid: bool`
- `error_message: Optional[str]`
- `image_format: Optional[str]`
- `dimensions: <ast.Subscript object at 0x11046d250>`
- `file_size: Optional[int]`

### DrawingMetadata
**Source**: `app/services/data_pipeline.py`

**Inherits from**: BaseModel

**Attributes**:
- `age_years: float`
- `subject: Optional[str]`
- `expert_label: Optional[str]`
- `drawing_tool: Optional[str]`
- `prompt: Optional[str]`

**Public Methods**:
- `validate_age()`

### ImagePreprocessingError
**Source**: `app/services/data_pipeline.py`

**Inherits from**: ImageProcessingError

### DataPipelineService
**Source**: `app/services/data_pipeline.py`

**Public Methods**:
- `validate_image()`
- `preprocess_image()`
- `extract_metadata()`
- `validate_and_preprocess()`

### MetadataFormat
**Source**: `app/services/dataset_preparation.py`

**Inherits from**: str, Enum

### DatasetSplit
**Source**: `app/services/dataset_preparation.py`

**Attributes**:
- `train_files: List[Path]`
- `train_metadata: List[DrawingMetadata]`
- `validation_files: List[Path]`
- `validation_metadata: List[DrawingMetadata]`
- `test_files: List[Path]`
- `test_metadata: List[DrawingMetadata]`

**Public Methods**:
- `train_count()`
- `validation_count()`
- `test_count()`
- `total_count()`

### SplitConfig
**Source**: `app/services/dataset_preparation.py`

**Attributes**:
- `train_ratio: float`
- `validation_ratio: float`
- `test_ratio: float`
- `random_seed: int`
- `stratify_by_age: bool`
- `age_group_size: float`

### DatasetPreparationService
**Source**: `app/services/dataset_preparation.py`

**Public Methods**:
- `load_dataset_from_folder()`
- `create_dataset_splits()`
- `prepare_dataset()`
- `validate_dataset_for_training()`

### InterpretabilityError
**Source**: `app/services/interpretability_engine.py`

**Inherits from**: Exception

### SaliencyGenerationError
**Source**: `app/services/interpretability_engine.py`

**Inherits from**: InterpretabilityError

### AttentionVisualizationError
**Source**: `app/services/interpretability_engine.py`

**Inherits from**: InterpretabilityError

### AttentionRollout
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `generate_rollout()`

### GradCAMViT
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `generate_cam()`

### PatchImportanceScorer
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `compute_attention_importance()`
- `compute_gradient_importance()`
- `reshape_to_spatial()`

### SaliencyMapGenerator
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `generate_saliency_map()`

### VisualFeatureIdentifier
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `identify_important_regions()`
- `analyze_drawing_content()`

### ExplanationGenerator
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `generate_explanation()`

### ImportanceRegionDetector
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `detect_bounding_boxes()`
- `create_region_highlights()`

### SaliencyOverlayGenerator
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `create_heatmap_overlay()`
- `create_contour_overlay()`
- `create_masked_overlay()`
- `create_side_by_side_comparison()`
- `save_overlay()`

### VisualizationExporter
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `export_comprehensive_report()`
- `export_interactive_data()`
- `export_presentation_slides()`

### InterpretabilityPipeline
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `generate_complete_analysis()`

### SaliencyOverlayGenerator
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `create_heatmap_overlay()`
- `create_contour_overlay()`
- `create_masked_overlay()`
- `create_side_by_side_comparison()`

### VisualizationExporter
**Source**: `app/services/interpretability_engine.py`

**Public Methods**:
- `export_visualization_set()`
- `create_interactive_html_report()`

### BackupService
**Source**: `app/services/backup_service.py`

### TrainingReportError
**Source**: `app/services/training_report_service.py`

**Inherits from**: Exception

### MetricsCalculationError
**Source**: `app/services/training_report_service.py`

**Inherits from**: TrainingReportError

### VisualizationError
**Source**: `app/services/training_report_service.py`

**Inherits from**: TrainingReportError

### TrainingMetrics
**Source**: `app/services/training_report_service.py`

**Attributes**:
- `final_train_loss: float`
- `final_val_loss: float`
- `best_val_loss: float`
- `best_epoch: int`
- `total_epochs: int`
- `training_time_seconds: float`
- `train_loss_mean: float`
- `train_loss_std: float`
- `val_loss_mean: float`
- `val_loss_std: float`
- `convergence_epoch: Optional[int]`
- `early_stopping_triggered: bool`
- `overfitting_detected: bool`
- `generalization_gap: float`
- `reconstruction_error_stats: Dict[str, float]`
- `anomaly_detection_threshold: float`
- `validation_accuracy_estimate: float`
- `loss_variance: float`
- `gradient_norm_stats: <ast.Subscript object at 0x1104e86d0>`
- `learning_rate_schedule: List[float]`

**Public Methods**:
- `to_dict()`

### ModelArchitectureInfo
**Source**: `app/services/training_report_service.py`

**Attributes**:
- `model_type: str`
- `input_dimension: int`
- `hidden_dimensions: List[int]`
- `latent_dimension: int`
- `total_parameters: int`
- `trainable_parameters: int`
- `model_size_mb: float`
- `activation_functions: List[str]`
- `dropout_rate: float`

**Public Methods**:
- `to_dict()`

### TrainingConfiguration
**Source**: `app/services/training_report_service.py`

**Attributes**:
- `learning_rate: float`
- `batch_size: int`
- `epochs: int`
- `optimizer: str`
- `loss_function: str`
- `early_stopping_patience: int`
- `min_delta: float`
- `data_split_ratios: Dict[str, float]`
- `device_used: str`
- `random_seed: Optional[int]`

**Public Methods**:
- `to_dict()`

### TrainingReportGenerator
**Source**: `app/services/training_report_service.py`

**Public Methods**:
- `generate_comprehensive_report()`

### AgeGroupManagerError
**Source**: `app/services/age_group_manager.py`

**Inherits from**: Exception

### InsufficientDataError
**Source**: `app/services/age_group_manager.py`

**Inherits from**: AgeGroupManagerError

### AgeGroupConfig
**Source**: `app/services/age_group_manager.py`

**Attributes**:
- `min_samples_per_group: int`
- `default_age_span: float`
- `max_age_span: float`
- `merge_threshold: int`

### AgeGroupManager
**Source**: `app/services/age_group_manager.py`

**Public Methods**:
- `analyze_age_distribution()`
- `suggest_age_groups()`
- `create_age_groups()`
- `find_appropriate_model()`
- `get_age_group_coverage()`
- `validate_age_group_data()`

### EmbeddingServiceError
**Source**: `app/services/embedding_service.py`

**Inherits from**: Exception

### ModelLoadingError
**Source**: `app/services/embedding_service.py`

**Inherits from**: EmbeddingServiceError

### EmbeddingGenerationError
**Source**: `app/services/embedding_service.py`

**Inherits from**: EmbeddingServiceError

### DeviceManager
**Source**: `app/services/embedding_service.py`

**Public Methods**:
- `device()`
- `device_info()`
- `get_memory_usage()`

### VisionTransformerWrapper
**Source**: `app/services/embedding_service.py`

**Public Methods**:
- `load_model()`
- `is_loaded()`
- `get_model_info()`

### EmbeddingService
**Source**: `app/services/embedding_service.py`

**Public Methods**:
- `initialize()`
- `is_ready()`
- `get_service_info()`
- `generate_embedding()`
- `generate_batch_embeddings()`
- `get_embedding_dimension()`
- `clear_cache()`
- `store_embedding_for_db()`
- `retrieve_embedding_from_db()`
- `invalidate_embedding_cache()`
- `get_storage_stats()`

### EmbeddingPipeline
**Source**: `app/services/embedding_service.py`

**Public Methods**:
- `process_drawing()`
- `process_batch()`
- `get_pipeline_stats()`
- `reset_stats()`

### HealthStatus
**Source**: `app/services/health_monitor.py`

**Attributes**:
- `name: str`
- `status: str`
- `message: str`
- `details: Dict[str, Any]`
- `last_check: datetime`
- `response_time_ms: Optional[float]`

### SystemMetrics
**Source**: `app/services/health_monitor.py`

**Attributes**:
- `timestamp: datetime`
- `cpu_percent: float`
- `memory_percent: float`
- `memory_available_gb: float`
- `memory_total_gb: float`
- `disk_percent: float`
- `disk_free_gb: float`
- `disk_total_gb: float`
- `active_connections: int`
- `process_count: int`

### HealthMonitor
**Source**: `app/services/health_monitor.py`

**Public Methods**:
- `get_metrics_history()`
- `get_overall_status()`
- `get_alerts()`
- `update_alert_thresholds()`

### ModelManagerError
**Source**: `app/services/model_manager.py`

**Inherits from**: Exception

### AutoencoderTrainingError
**Source**: `app/services/model_manager.py`

**Inherits from**: ModelManagerError

### ModelLoadingError
**Source**: `app/services/model_manager.py`

**Inherits from**: ModelManagerError

### TrainingConfig
**Source**: `app/services/model_manager.py`

**Attributes**:
- `hidden_dims: List[int]`
- `learning_rate: float`
- `batch_size: int`
- `epochs: int`
- `validation_split: float`
- `early_stopping_patience: int`
- `min_delta: float`
- `device: str`

### AutoencoderModel
**Source**: `app/services/model_manager.py`

**Public Methods**:
- `forward()`
- `encode()`
- `decode()`
- `get_architecture_info()`

### EarlyStopping
**Source**: `app/services/model_manager.py`

### AutoencoderTrainer
**Source**: `app/services/model_manager.py`

**Public Methods**:
- `train()`

### ModelManager
**Source**: `app/services/model_manager.py`

**Public Methods**:
- `train_age_group_model()`
- `load_model()`
- `compute_reconstruction_loss()`
- `get_model_info()`
- `list_models()`
- `clear_model_cache()`

### SageMakerError
**Source**: `app/services/sagemaker_training_service.py`

**Inherits from**: Exception

### SageMakerConfigurationError
**Source**: `app/services/sagemaker_training_service.py`

**Inherits from**: SageMakerError

### SageMakerJobError
**Source**: `app/services/sagemaker_training_service.py`

**Inherits from**: SageMakerError

### DockerContainerError
**Source**: `app/services/sagemaker_training_service.py`

**Inherits from**: SageMakerError

### SageMakerJobConfig
**Source**: `app/services/sagemaker_training_service.py`

**Attributes**:
- `job_name: str`
- `role_arn: str`
- `image_uri: str`
- `instance_type: str`
- `instance_count: int`
- `volume_size_gb: int`
- `max_runtime_seconds: int`
- `input_data_s3_uri: str`
- `output_s3_uri: str`
- `hyperparameters: Dict[str, str]`
- `environment_variables: Dict[str, str]`

**Public Methods**:
- `to_sagemaker_config()`

### SageMakerContainerBuilder
**Source**: `app/services/sagemaker_training_service.py`

**Public Methods**:
- `build_training_container()`
- `push_to_ecr()`

### SageMakerTrainingService
**Source**: `app/services/sagemaker_training_service.py`

**Public Methods**:
- `validate_configuration()`
- `create_execution_role()`
- `upload_training_data()`
- `submit_training_job()`
- `get_job_status()`
- `cancel_training_job()`
- `list_training_jobs()`

### FileStorageError
**Source**: `app/services/file_storage.py`

**Inherits from**: StorageError

### FileStorageService
**Source**: `app/services/file_storage.py`

**Public Methods**:
- `generate_unique_filename()`
- `get_file_url()`
- `delete_file()`
- `get_file_info()`
- `cleanup_old_files()`
- `get_storage_stats()`

### TrainingEnvironment
**Source**: `app/services/training_config.py`

**Inherits from**: str, Enum

### OptimizerType
**Source**: `app/services/training_config.py`

**Inherits from**: str, Enum

### SchedulerType
**Source**: `app/services/training_config.py`

**Inherits from**: str, Enum

### ModelConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `encoder_dim: int`
- `hidden_dims: List[int]`
- `latent_dim: int`
- `dropout_rate: float`
- `activation: str`
- `batch_norm: bool`

### OptimizerConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `type: OptimizerType`
- `learning_rate: float`
- `weight_decay: float`
- `momentum: float`
- `betas: tuple`

### SchedulerConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `type: SchedulerType`
- `step_size: int`
- `gamma: float`
- `T_max: int`

### DataConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `batch_size: int`
- `num_workers: int`
- `pin_memory: bool`
- `train_split: float`
- `validation_split: float`
- `test_split: float`
- `stratify_by_age: bool`
- `age_group_size: float`

### TrainingConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `job_name: str`
- `experiment_name: str`
- `environment: TrainingEnvironment`
- `device: str`
- `mixed_precision: bool`
- `dataset_folder: str`
- `metadata_file: str`
- `data: DataConfig`
- `model: ModelConfig`
- `epochs: int`
- `early_stopping_patience: int`
- `early_stopping_min_delta: float`
- `gradient_clip_norm: Optional[float]`
- `optimizer: OptimizerConfig`
- `scheduler: SchedulerConfig`
- `validation_frequency: int`
- `checkpoint_frequency: int`
- `save_best_only: bool`
- `log_frequency: int`
- `save_plots: bool`
- `plot_frequency: int`
- `sagemaker_instance_type: str`
- `sagemaker_instance_count: int`
- `sagemaker_volume_size: int`
- `sagemaker_max_runtime: int`
- `output_dir: str`
- `model_save_path: str`
- `log_save_path: str`

### ParameterSweepConfig
**Source**: `app/services/training_config.py`

**Attributes**:
- `learning_rates: List[float]`
- `batch_sizes: List[int]`
- `hidden_dims: <ast.Subscript object at 0x110447fd0>`
- `latent_dims: List[int]`
- `dropout_rates: List[float]`
- `max_trials: int`
- `optimization_metric: str`
- `optimization_direction: str`

### TrainingConfigManager
**Source**: `app/services/training_config.py`

**Public Methods**:
- `create_default_configs()`
- `load_config()`
- `save_config()`
- `validate_config()`
- `create_parameter_sweep()`
- `get_hydra_config()`

### ModelDeploymentError
**Source**: `app/services/model_deployment_service.py`

**Inherits from**: Exception

### ModelExportError
**Source**: `app/services/model_deployment_service.py`

**Inherits from**: ModelDeploymentError

### ModelValidationError
**Source**: `app/services/model_deployment_service.py`

**Inherits from**: ModelDeploymentError

### ModelCompatibilityError
**Source**: `app/services/model_deployment_service.py`

**Inherits from**: ModelDeploymentError

### ModelExportMetadata
**Source**: `app/services/model_deployment_service.py`

**Attributes**:
- `model_id: str`
- `export_timestamp: datetime`
- `training_job_id: int`
- `model_type: str`
- `model_version: str`
- `architecture_hash: str`
- `parameter_count: int`
- `input_dimension: int`
- `output_dimension: int`
- `age_group_min: float`
- `age_group_max: float`
- `training_metrics: Dict[str, Any]`
- `compatibility_version: str`
- `export_format: str`
- `file_size_bytes: int`
- `checksum: str`

**Public Methods**:
- `to_dict()`

### ModelDeploymentConfig
**Source**: `app/services/model_deployment_service.py`

**Attributes**:
- `model_export_path: str`
- `age_group_min: float`
- `age_group_max: float`
- `replace_existing: bool`
- `validate_before_deployment: bool`
- `backup_existing: bool`
- `deployment_environment: str`

**Public Methods**:
- `to_dict()`

### ModelExporter
**Source**: `app/services/model_deployment_service.py`

**Public Methods**:
- `export_model_from_training_job()`
- `export_model_direct()`
- `list_exported_models()`

### ModelValidator
**Source**: `app/services/model_deployment_service.py`

**Public Methods**:
- `validate_exported_model()`

### ModelDeploymentService
**Source**: `app/services/model_deployment_service.py`

**Public Methods**:
- `deploy_model()`
- `list_deployed_models()`
- `undeploy_model()`

### ThresholdManagerError
**Source**: `app/services/threshold_manager.py`

**Inherits from**: Exception

### ThresholdCalculationError
**Source**: `app/services/threshold_manager.py`

**Inherits from**: ThresholdManagerError

### ThresholdConfig
**Source**: `app/services/threshold_manager.py`

**Attributes**:
- `default_percentile: float`
- `min_samples_for_calculation: int`
- `confidence_levels: List[float]`

### ThresholdManager
**Source**: `app/services/threshold_manager.py`

**Public Methods**:
- `calculate_percentile_threshold()`
- `calculate_model_threshold()`
- `update_model_threshold()`
- `recalculate_all_thresholds()`
- `get_threshold_for_age()`
- `is_anomaly()`
- `get_current_percentile()`
- `set_current_percentile()`
- `get_threshold_statistics()`
- `clear_threshold_cache()`

### DataSufficiencyError
**Source**: `app/services/data_sufficiency_service.py`

**Inherits from**: Exception

### InsufficientDataError
**Source**: `app/services/data_sufficiency_service.py`

**Inherits from**: DataSufficiencyError

### AgeGroupDataInfo
**Source**: `app/services/data_sufficiency_service.py`

**Attributes**:
- `age_min: float`
- `age_max: float`
- `sample_count: int`
- `is_sufficient: bool`
- `recommended_min_samples: int`
- `data_quality_score: float`
- `subjects_distribution: Dict[str, int]`
- `age_distribution: List[float]`

**Public Methods**:
- `to_dict()`

### DataSufficiencyWarning
**Source**: `app/services/data_sufficiency_service.py`

**Attributes**:
- `warning_type: str`
- `severity: str`
- `age_group_min: float`
- `age_group_max: float`
- `current_samples: int`
- `recommended_samples: int`
- `message: str`
- `suggestions: List[str]`

**Public Methods**:
- `to_dict()`

### AgeGroupMergingSuggestion
**Source**: `app/services/data_sufficiency_service.py`

**Attributes**:
- `original_groups: <ast.Subscript object at 0x1105abad0>`
- `merged_group: Tuple[float, float]`
- `combined_sample_count: int`
- `improvement_score: float`
- `rationale: str`

**Public Methods**:
- `to_dict()`

### DataSufficiencyAnalyzer
**Source**: `app/services/data_sufficiency_service.py`

**Public Methods**:
- `analyze_age_group_data()`
- `generate_data_warnings()`
- `suggest_age_group_merging()`

### DataAugmentationSuggester
**Source**: `app/services/data_sufficiency_service.py`

**Public Methods**:
- `suggest_augmentation_strategies()`

## Class Diagram

```mermaid
classDiagram
    class ScoreNormalizationError {
    }

    class NormalizationConfig {
        +str normalization_method
        +str confidence_method
        +int min_samples_for_stats
        +float outlier_threshold
    }

    class ScoreNormalizer {
        +normalize_score()
        +calculate_confidence()
        +compare_scores()
        +get_normalization_summary()
        +update_normalization_config()
        +clear_cache()
        +detect_outliers()
    }

    class LocalTrainingError {
    }

    class DeviceDetectionError {
    }

    class TrainingProgressError {
    }

    class TrainingProgress {
        +int job_id
        +int epoch
        +int total_epochs
        +int batch
        +int total_batches
        +float train_loss
        +Optional[float] val_loss
        +float learning_rate
        +float elapsed_time
        +Optional[float] estimated_remaining
        +Optional[Dict] memory_usage
        +epoch_progress()
        +batch_progress()
    }

    class DeviceManager {
        +device()
        +device_info()
        +get_memory_usage()
        +clear_cache()
        +optimize_for_training()
    }

    class TrainingProgressMonitor {
        +add_callback()
        +start_epoch()
        +update_batch()
        +update_epoch()
        +get_latest_progress()
        +get_history()
        +stop()
    }

    class LocalTrainingEnvironment {
        +get_environment_info()
        +prepare_training_data()
        +start_training_job()
        +get_job_status()
        +list_training_jobs()
        +cancel_training_job()
    }

    class EnhancedAutoencoderTrainer {
        +train()
    }

    class ComparisonService {
        +find_similar_normal_examples()
        +get_comparison_statistics()
    }

    class ValidationResult {
        +bool is_valid
        +Optional[str] error_message
        +Optional[str] image_format
        +<ast.Subscript object at 0x11046d250> dimensions
        +Optional[int] file_size
    }

    class DrawingMetadata {
        +float age_years
        +Optional[str] subject
        +Optional[str] expert_label
        +Optional[str] drawing_tool
        +Optional[str] prompt
        +validate_age()
    }

    class ImagePreprocessingError {
    }

    class DataPipelineService {
        +validate_image()
        +preprocess_image()
        +extract_metadata()
        +validate_and_preprocess()
    }

    class MetadataFormat {
    }

    class DatasetSplit {
        +List[Path] train_files
        +List[DrawingMetadata] train_metadata
        +List[Path] validation_files
        +List[DrawingMetadata] validation_metadata
        +List[Path] test_files
        +List[DrawingMetadata] test_metadata
        +train_count()
        +validation_count()
        +test_count()
        +total_count()
    }

    class SplitConfig {
        +float train_ratio
        +float validation_ratio
        +float test_ratio
        +int random_seed
        +bool stratify_by_age
        +float age_group_size
    }

    class DatasetPreparationService {
        +load_dataset_from_folder()
        +create_dataset_splits()
        +prepare_dataset()
        +validate_dataset_for_training()
    }

    class InterpretabilityError {
    }

    class SaliencyGenerationError {
    }

    class AttentionVisualizationError {
    }

    class AttentionRollout {
        +generate_rollout()
    }

    class GradCAMViT {
        +generate_cam()
    }

    class PatchImportanceScorer {
        +compute_attention_importance()
        +compute_gradient_importance()
        +reshape_to_spatial()
    }

    class SaliencyMapGenerator {
        +generate_saliency_map()
    }

    class VisualFeatureIdentifier {
        +identify_important_regions()
        +analyze_drawing_content()
    }

    class ExplanationGenerator {
        +generate_explanation()
    }

    class ImportanceRegionDetector {
        +detect_bounding_boxes()
        +create_region_highlights()
    }

    class SaliencyOverlayGenerator {
        +create_heatmap_overlay()
        +create_contour_overlay()
        +create_masked_overlay()
        +create_side_by_side_comparison()
        +save_overlay()
    }

    class VisualizationExporter {
        +export_comprehensive_report()
        +export_interactive_data()
        +export_presentation_slides()
    }

    class InterpretabilityPipeline {
        +generate_complete_analysis()
    }

    class SaliencyOverlayGenerator {
        +create_heatmap_overlay()
        +create_contour_overlay()
        +create_masked_overlay()
        +create_side_by_side_comparison()
    }

    class VisualizationExporter {
        +export_visualization_set()
        +create_interactive_html_report()
    }

    class BackupService {
    }

    class TrainingReportError {
    }

    class MetricsCalculationError {
    }

    class VisualizationError {
    }

    class TrainingMetrics {
        +float final_train_loss
        +float final_val_loss
        +float best_val_loss
        +int best_epoch
        +int total_epochs
        +float training_time_seconds
        +float train_loss_mean
        +float train_loss_std
        +float val_loss_mean
        +float val_loss_std
        +Optional[int] convergence_epoch
        +bool early_stopping_triggered
        +bool overfitting_detected
        +float generalization_gap
        +Dict[str, float] reconstruction_error_stats
        +float anomaly_detection_threshold
        +float validation_accuracy_estimate
        +float loss_variance
        +<ast.Subscript object at 0x1104e86d0> gradient_norm_stats
        +List[float] learning_rate_schedule
        +to_dict()
    }

    class ModelArchitectureInfo {
        +str model_type
        +int input_dimension
        +List[int] hidden_dimensions
        +int latent_dimension
        +int total_parameters
        +int trainable_parameters
        +float model_size_mb
        +List[str] activation_functions
        +float dropout_rate
        +to_dict()
    }

    class TrainingConfiguration {
        +float learning_rate
        +int batch_size
        +int epochs
        +str optimizer
        +str loss_function
        +int early_stopping_patience
        +float min_delta
        +Dict[str, float] data_split_ratios
        +str device_used
        +Optional[int] random_seed
        +to_dict()
    }

    class TrainingReportGenerator {
        +generate_comprehensive_report()
    }

    class AgeGroupManagerError {
    }

    class InsufficientDataError {
    }

    class AgeGroupConfig {
        +int min_samples_per_group
        +float default_age_span
        +float max_age_span
        +int merge_threshold
    }

    class AgeGroupManager {
        +analyze_age_distribution()
        +suggest_age_groups()
        +create_age_groups()
        +find_appropriate_model()
        +get_age_group_coverage()
        +validate_age_group_data()
    }

    class EmbeddingServiceError {
    }

    class ModelLoadingError {
    }

    class EmbeddingGenerationError {
    }

    class DeviceManager {
        +device()
        +device_info()
        +get_memory_usage()
    }

    class VisionTransformerWrapper {
        +load_model()
        +is_loaded()
        +get_model_info()
    }

    class EmbeddingService {
        +initialize()
        +is_ready()
        +get_service_info()
        +generate_embedding()
        +generate_batch_embeddings()
        +get_embedding_dimension()
        +clear_cache()
        +store_embedding_for_db()
        +retrieve_embedding_from_db()
        +invalidate_embedding_cache()
        +get_storage_stats()
    }

    class EmbeddingPipeline {
        +process_drawing()
        +process_batch()
        +get_pipeline_stats()
        +reset_stats()
    }

    class HealthStatus {
        +str name
        +str status
        +str message
        +Dict[str, Any] details
        +datetime last_check
        +Optional[float] response_time_ms
    }

    class SystemMetrics {
        +datetime timestamp
        +float cpu_percent
        +float memory_percent
        +float memory_available_gb
        +float memory_total_gb
        +float disk_percent
        +float disk_free_gb
        +float disk_total_gb
        +int active_connections
        +int process_count
    }

    class HealthMonitor {
        +get_metrics_history()
        +get_overall_status()
        +get_alerts()
        +update_alert_thresholds()
    }

    class ModelManagerError {
    }

    class AutoencoderTrainingError {
    }

    class ModelLoadingError {
    }

    class TrainingConfig {
        +List[int] hidden_dims
        +float learning_rate
        +int batch_size
        +int epochs
        +float validation_split
        +int early_stopping_patience
        +float min_delta
        +str device
    }

    class AutoencoderModel {
        +forward()
        +encode()
        +decode()
        +get_architecture_info()
    }

    class EarlyStopping {
    }

    class AutoencoderTrainer {
        +train()
    }

    class ModelManager {
        +train_age_group_model()
        +load_model()
        +compute_reconstruction_loss()
        +get_model_info()
        +list_models()
        +clear_model_cache()
    }

    class SageMakerError {
    }

    class SageMakerConfigurationError {
    }

    class SageMakerJobError {
    }

    class DockerContainerError {
    }

    class SageMakerJobConfig {
        +str job_name
        +str role_arn
        +str image_uri
        +str instance_type
        +int instance_count
        +int volume_size_gb
        +int max_runtime_seconds
        +str input_data_s3_uri
        +str output_s3_uri
        +Dict[str, str] hyperparameters
        +Dict[str, str] environment_variables
        +to_sagemaker_config()
    }

    class SageMakerContainerBuilder {
        +build_training_container()
        +push_to_ecr()
    }

    class SageMakerTrainingService {
        +validate_configuration()
        +create_execution_role()
        +upload_training_data()
        +submit_training_job()
        +get_job_status()
        +cancel_training_job()
        +list_training_jobs()
    }

    class FileStorageError {
    }

    class FileStorageService {
        +generate_unique_filename()
        +get_file_url()
        +delete_file()
        +get_file_info()
        +cleanup_old_files()
        +get_storage_stats()
    }

    class TrainingEnvironment {
    }

    class OptimizerType {
    }

    class SchedulerType {
    }

    class ModelConfig {
        +int encoder_dim
        +List[int] hidden_dims
        +int latent_dim
        +float dropout_rate
        +str activation
        +bool batch_norm
    }

    class OptimizerConfig {
        +OptimizerType type
        +float learning_rate
        +float weight_decay
        +float momentum
        +tuple betas
    }

    class SchedulerConfig {
        +SchedulerType type
        +int step_size
        +float gamma
        +int T_max
    }

    class DataConfig {
        +int batch_size
        +int num_workers
        +bool pin_memory
        +float train_split
        +float validation_split
        +float test_split
        +bool stratify_by_age
        +float age_group_size
    }

    class TrainingConfig {
        +str job_name
        +str experiment_name
        +TrainingEnvironment environment
        +str device
        +bool mixed_precision
        +str dataset_folder
        +str metadata_file
        +DataConfig data
        +ModelConfig model
        +int epochs
        +int early_stopping_patience
        +float early_stopping_min_delta
        +Optional[float] gradient_clip_norm
        +OptimizerConfig optimizer
        +SchedulerConfig scheduler
        +int validation_frequency
        +int checkpoint_frequency
        +bool save_best_only
        +int log_frequency
        +bool save_plots
        +int plot_frequency
        +str sagemaker_instance_type
        +int sagemaker_instance_count
        +int sagemaker_volume_size
        +int sagemaker_max_runtime
        +str output_dir
        +str model_save_path
        +str log_save_path
    }

    class ParameterSweepConfig {
        +List[float] learning_rates
        +List[int] batch_sizes
        +<ast.Subscript object at 0x110447fd0> hidden_dims
        +List[int] latent_dims
        +List[float] dropout_rates
        +int max_trials
        +str optimization_metric
        +str optimization_direction
    }

    class TrainingConfigManager {
        +create_default_configs()
        +load_config()
        +save_config()
        +validate_config()
        +create_parameter_sweep()
        +get_hydra_config()
    }

    class ModelDeploymentError {
    }

    class ModelExportError {
    }

    class ModelValidationError {
    }

    class ModelCompatibilityError {
    }

    class ModelExportMetadata {
        +str model_id
        +datetime export_timestamp
        +int training_job_id
        +str model_type
        +str model_version
        +str architecture_hash
        +int parameter_count
        +int input_dimension
        +int output_dimension
        +float age_group_min
        +float age_group_max
        +Dict[str, Any] training_metrics
        +str compatibility_version
        +str export_format
        +int file_size_bytes
        +str checksum
        +to_dict()
    }

    class ModelDeploymentConfig {
        +str model_export_path
        +float age_group_min
        +float age_group_max
        +bool replace_existing
        +bool validate_before_deployment
        +bool backup_existing
        +str deployment_environment
        +to_dict()
    }

    class ModelExporter {
        +export_model_from_training_job()
        +export_model_direct()
        +list_exported_models()
    }

    class ModelValidator {
        +validate_exported_model()
    }

    class ModelDeploymentService {
        +deploy_model()
        +list_deployed_models()
        +undeploy_model()
    }

    class ThresholdManagerError {
    }

    class ThresholdCalculationError {
    }

    class ThresholdConfig {
        +float default_percentile
        +int min_samples_for_calculation
        +List[float] confidence_levels
    }

    class ThresholdManager {
        +calculate_percentile_threshold()
        +calculate_model_threshold()
        +update_model_threshold()
        +recalculate_all_thresholds()
        +get_threshold_for_age()
        +is_anomaly()
        +get_current_percentile()
        +set_current_percentile()
        +get_threshold_statistics()
        +clear_threshold_cache()
    }

    class DataSufficiencyError {
    }

    class InsufficientDataError {
    }

    class AgeGroupDataInfo {
        +float age_min
        +float age_max
        +int sample_count
        +bool is_sufficient
        +int recommended_min_samples
        +float data_quality_score
        +Dict[str, int] subjects_distribution
        +List[float] age_distribution
        +to_dict()
    }

    class DataSufficiencyWarning {
        +str warning_type
        +str severity
        +float age_group_min
        +float age_group_max
        +int current_samples
        +int recommended_samples
        +str message
        +List[str] suggestions
        +to_dict()
    }

    class AgeGroupMergingSuggestion {
        +<ast.Subscript object at 0x1105abad0> original_groups
        +Tuple[float, float] merged_group
        +int combined_sample_count
        +float improvement_score
        +str rationale
        +to_dict()
    }

    class DataSufficiencyAnalyzer {
        +analyze_age_group_data()
        +generate_data_warnings()
        +suggest_age_group_merging()
    }

    class DataAugmentationSuggester {
        +suggest_augmentation_strategies()
    }

    Exception <|-- ScoreNormalizationError
    Exception <|-- LocalTrainingError
    LocalTrainingError <|-- DeviceDetectionError
    LocalTrainingError <|-- TrainingProgressError
    AutoencoderTrainer <|-- EnhancedAutoencoderTrainer
    BaseModel <|-- ValidationResult
    BaseModel <|-- DrawingMetadata
    ImageProcessingError <|-- ImagePreprocessingError
    str <|-- MetadataFormat
    Enum <|-- MetadataFormat
    Exception <|-- InterpretabilityError
    InterpretabilityError <|-- SaliencyGenerationError
    InterpretabilityError <|-- AttentionVisualizationError
    Exception <|-- TrainingReportError
    TrainingReportError <|-- MetricsCalculationError
    TrainingReportError <|-- VisualizationError
    Exception <|-- AgeGroupManagerError
    AgeGroupManagerError <|-- InsufficientDataError
    Exception <|-- EmbeddingServiceError
    EmbeddingServiceError <|-- ModelLoadingError
    EmbeddingServiceError <|-- EmbeddingGenerationError
    Exception <|-- ModelManagerError
    ModelManagerError <|-- AutoencoderTrainingError
    ModelManagerError <|-- ModelLoadingError
    Exception <|-- SageMakerError
    SageMakerError <|-- SageMakerConfigurationError
    SageMakerError <|-- SageMakerJobError
    SageMakerError <|-- DockerContainerError
    StorageError <|-- FileStorageError
    str <|-- TrainingEnvironment
    Enum <|-- TrainingEnvironment
    str <|-- OptimizerType
    Enum <|-- OptimizerType
    str <|-- SchedulerType
    Enum <|-- SchedulerType
    Exception <|-- ModelDeploymentError
    ModelDeploymentError <|-- ModelExportError
    ModelDeploymentError <|-- ModelValidationError
    ModelDeploymentError <|-- ModelCompatibilityError
    Exception <|-- ThresholdManagerError
    ThresholdManagerError <|-- ThresholdCalculationError
    Exception <|-- DataSufficiencyError
    DataSufficiencyError <|-- InsufficientDataError
```

## Relationships

| From | To | Type | Description |
|------|----|----- |-------------|
| ScoreNormalizationError | Exception | inheritance | ScoreNormalizationError inherits from Exception |
| LocalTrainingError | Exception | inheritance | LocalTrainingError inherits from Exception |
| DeviceDetectionError | LocalTrainingError | inheritance | DeviceDetectionError inherits from LocalTrainingError |
| TrainingProgressError | LocalTrainingError | inheritance | TrainingProgressError inherits from LocalTrainingError |
| EnhancedAutoencoderTrainer | AutoencoderTrainer | inheritance | EnhancedAutoencoderTrainer inherits from AutoencoderTrainer |
| ValidationResult | BaseModel | inheritance | ValidationResult inherits from BaseModel |
| DrawingMetadata | BaseModel | inheritance | DrawingMetadata inherits from BaseModel |
| ImagePreprocessingError | ImageProcessingError | inheritance | ImagePreprocessingError inherits from ImageProcessingError |
| MetadataFormat | str | inheritance | MetadataFormat inherits from str |
| MetadataFormat | Enum | inheritance | MetadataFormat inherits from Enum |
| InterpretabilityError | Exception | inheritance | InterpretabilityError inherits from Exception |
| SaliencyGenerationError | InterpretabilityError | inheritance | SaliencyGenerationError inherits from InterpretabilityError |
| AttentionVisualizationError | InterpretabilityError | inheritance | AttentionVisualizationError inherits from InterpretabilityError |
| TrainingReportError | Exception | inheritance | TrainingReportError inherits from Exception |
| MetricsCalculationError | TrainingReportError | inheritance | MetricsCalculationError inherits from TrainingReportError |
| VisualizationError | TrainingReportError | inheritance | VisualizationError inherits from TrainingReportError |
| AgeGroupManagerError | Exception | inheritance | AgeGroupManagerError inherits from Exception |
| InsufficientDataError | AgeGroupManagerError | inheritance | InsufficientDataError inherits from AgeGroupManagerError |
| EmbeddingServiceError | Exception | inheritance | EmbeddingServiceError inherits from Exception |
| ModelLoadingError | EmbeddingServiceError | inheritance | ModelLoadingError inherits from EmbeddingServiceError |
| EmbeddingGenerationError | EmbeddingServiceError | inheritance | EmbeddingGenerationError inherits from EmbeddingServiceError |
| ModelManagerError | Exception | inheritance | ModelManagerError inherits from Exception |
| AutoencoderTrainingError | ModelManagerError | inheritance | AutoencoderTrainingError inherits from ModelManagerError |
| ModelLoadingError | ModelManagerError | inheritance | ModelLoadingError inherits from ModelManagerError |
| SageMakerError | Exception | inheritance | SageMakerError inherits from Exception |
| SageMakerConfigurationError | SageMakerError | inheritance | SageMakerConfigurationError inherits from SageMakerError |
| SageMakerJobError | SageMakerError | inheritance | SageMakerJobError inherits from SageMakerError |
| DockerContainerError | SageMakerError | inheritance | DockerContainerError inherits from SageMakerError |
| FileStorageError | StorageError | inheritance | FileStorageError inherits from StorageError |
| TrainingEnvironment | str | inheritance | TrainingEnvironment inherits from str |
| TrainingEnvironment | Enum | inheritance | TrainingEnvironment inherits from Enum |
| OptimizerType | str | inheritance | OptimizerType inherits from str |
| OptimizerType | Enum | inheritance | OptimizerType inherits from Enum |
| SchedulerType | str | inheritance | SchedulerType inherits from str |
| SchedulerType | Enum | inheritance | SchedulerType inherits from Enum |
| ModelDeploymentError | Exception | inheritance | ModelDeploymentError inherits from Exception |
| ModelExportError | ModelDeploymentError | inheritance | ModelExportError inherits from ModelDeploymentError |
| ModelValidationError | ModelDeploymentError | inheritance | ModelValidationError inherits from ModelDeploymentError |
| ModelCompatibilityError | ModelDeploymentError | inheritance | ModelCompatibilityError inherits from ModelDeploymentError |
| ThresholdManagerError | Exception | inheritance | ThresholdManagerError inherits from Exception |
| ThresholdCalculationError | ThresholdManagerError | inheritance | ThresholdCalculationError inherits from ThresholdManagerError |
| DataSufficiencyError | Exception | inheritance | DataSufficiencyError inherits from Exception |
| InsufficientDataError | DataSufficiencyError | inheritance | InsufficientDataError inherits from DataSufficiencyError |

