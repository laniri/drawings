# Training Report Service Service

Training Report Generation Service for Children's Drawing Anomaly Detection System

This module provides comprehensive training metrics collection, validation curve plotting,
performance visualization, and summary report generation with model performance analysis.

## Class: TrainingReportError

Base exception for training report generation errors.

## Class: MetricsCalculationError

Raised when metrics calculation fails.

## Class: VisualizationError

Raised when visualization generation fails.

## Class: TrainingMetrics

Container for comprehensive training metrics.

### to_dict

Convert metrics to dictionary.

**Signature**: `to_dict()`

## Class: ModelArchitectureInfo

Container for model architecture information.

### to_dict

Convert architecture info to dictionary.

**Signature**: `to_dict()`

## Class: TrainingConfiguration

Container for training configuration details.

### to_dict

Convert configuration to dictionary.

**Signature**: `to_dict()`

## Class: TrainingReportGenerator

Generator for comprehensive training reports with visualizations.

### generate_comprehensive_report

Generate comprehensive training report with all metrics and visualizations.

Args:
    training_job_id: ID of the training job
    training_result: Training results dictionary
    model_info: Optional model information
    db: Database session
    
Returns:
    Dictionary containing complete report information
    
Raises:
    TrainingReportError: If report generation fails

**Signature**: `generate_comprehensive_report(training_job_id, training_result, model_info, db)`

