# Model Deployment Service Service

Model Export and Deployment Service for Children's Drawing Anomaly Detection System

This module provides model parameter export in production-compatible format, model validation
and compatibility checking, and deployment API endpoints for model loading.

## Class: ModelDeploymentError

Base exception for model deployment errors.

## Class: ModelExportError

Raised when model export fails.

## Class: ModelValidationError

Raised when model validation fails.

## Class: ModelCompatibilityError

Raised when model compatibility check fails.

## Class: ModelExportMetadata

Metadata for exported model.

### to_dict

Convert to dictionary for serialization.

**Signature**: `to_dict()`

## Class: ModelDeploymentConfig

Configuration for model deployment.

### to_dict

Convert to dictionary.

**Signature**: `to_dict()`

## Class: ModelExporter

Service for exporting trained models in production-compatible format.

### export_model_from_training_job

Export model from completed training job.

Args:
    training_job_id: ID of the training job
    age_group_min: Minimum age for the age group
    age_group_max: Maximum age for the age group
    export_format: Export format (pytorch, onnx, pickle)
    db: Database session
    
Returns:
    Model export metadata
    
Raises:
    ModelExportError: If export fails

**Signature**: `export_model_from_training_job(training_job_id, age_group_min, age_group_max, export_format, db)`

### export_model_direct

Export model directly without training job reference.

Args:
    model: PyTorch model to export
    model_info: Model information dictionary
    age_group_min: Minimum age for the age group
    age_group_max: Maximum age for the age group
    export_format: Export format
    
Returns:
    Model export metadata

**Signature**: `export_model_direct(model, model_info, age_group_min, age_group_max, export_format)`

### list_exported_models

List all exported models.

**Signature**: `list_exported_models()`

## Class: ModelValidator

Service for validating model compatibility and integrity.

### validate_exported_model

Validate exported model for compatibility and integrity.

Args:
    export_metadata: Model export metadata
    
Returns:
    Validation results dictionary
    
Raises:
    ModelValidationError: If validation fails

**Signature**: `validate_exported_model(export_metadata)`

## Class: ModelDeploymentService

Service for deploying models to production environment.

### deploy_model

Deploy model to production environment.

Args:
    deployment_config: Deployment configuration
    db: Database session
    
Returns:
    Deployment result dictionary
    
Raises:
    ModelDeploymentError: If deployment fails

**Signature**: `deploy_model(deployment_config, db)`

### list_deployed_models

List all deployed models.

**Signature**: `list_deployed_models(db)`

### undeploy_model

Undeploy (deactivate) a model.

**Signature**: `undeploy_model(model_id, db)`

