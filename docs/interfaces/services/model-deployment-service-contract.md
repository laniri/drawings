# Model Deployment Service Contract

## Overview
Service contract for Model Deployment Service (service)

**Source File**: `app/services/model_deployment_service.py`

## Interface Specification

### Classes

#### ModelDeploymentError

Base exception for model deployment errors.

**Inherits from**: Exception

#### ModelExportError

Raised when model export fails.

**Inherits from**: ModelDeploymentError

#### ModelValidationError

Raised when model validation fails.

**Inherits from**: ModelDeploymentError

#### ModelCompatibilityError

Raised when model compatibility check fails.

**Inherits from**: ModelDeploymentError

#### ModelExportMetadata

Metadata for exported model.

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

#### ModelDeploymentConfig

Configuration for model deployment.

**Attributes**:

- `model_export_path: str`
- `age_group_min: float`
- `age_group_max: float`
- `replace_existing: bool`
- `validate_before_deployment: bool`
- `backup_existing: bool`
- `deployment_environment: str`

#### ModelExporter

Service for exporting trained models in production-compatible format.

#### ModelValidator

Service for validating model compatibility and integrity.

#### ModelDeploymentService

Service for deploying models to production environment.

## Methods

### to_dict

Convert to dictionary for serialization.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### to_dict

Convert to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

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

**Signature**: `export_model_from_training_job(training_job_id: int, age_group_min: float, age_group_max: float, export_format: str, db: Session) -> ModelExportMetadata`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `training_job_id` | `int` | Parameter description |
| `age_group_min` | `float` | Parameter description |
| `age_group_max` | `float` | Parameter description |
| `export_format` | `str` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `ModelExportMetadata`

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

**Signature**: `export_model_direct(model: nn.Module, model_info: Dict[str, Any], age_group_min: float, age_group_max: float, export_format: str) -> ModelExportMetadata`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `model` | `nn.Module` | Parameter description |
| `model_info` | `Dict[str, Any]` | Parameter description |
| `age_group_min` | `float` | Parameter description |
| `age_group_max` | `float` | Parameter description |
| `export_format` | `str` | Parameter description |

**Returns**: `ModelExportMetadata`

### list_exported_models

List all exported models.

**Signature**: `list_exported_models() -> <ast.Subscript object at 0x1105aec50>`

**Returns**: `<ast.Subscript object at 0x1105aec50>`

### validate_exported_model

Validate exported model for compatibility and integrity.

Args:
    export_metadata: Model export metadata
    
Returns:
    Validation results dictionary
    
Raises:
    ModelValidationError: If validation fails

**Signature**: `validate_exported_model(export_metadata: ModelExportMetadata) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `export_metadata` | `ModelExportMetadata` | Parameter description |

**Returns**: `Dict[str, Any]`

### deploy_model

Deploy model to production environment.

Args:
    deployment_config: Deployment configuration
    db: Database session
    
Returns:
    Deployment result dictionary
    
Raises:
    ModelDeploymentError: If deployment fails

**Signature**: `deploy_model(deployment_config: ModelDeploymentConfig, db: Session) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `deployment_config` | `ModelDeploymentConfig` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict[str, Any]`

### list_deployed_models

List all deployed models.

**Signature**: `list_deployed_models(db: Session) -> <ast.Subscript object at 0x11050da90>`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `<ast.Subscript object at 0x11050da90>`

### undeploy_model

Undeploy (deactivate) a model.

**Signature**: `undeploy_model(model_id: int, db: Session) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `bool`

## Dependencies

- `app.models.database.TrainingJob`
- `app.models.database.TrainingReport`
- `app.models.database.AgeGroupModel`
- `app.services.model_manager.AutoencoderModel`

## Defined Interfaces

### ModelExportMetadataInterface

**Type**: Protocol
**Implemented by**: ModelExportMetadata

**Methods**:

- `to_dict() -> Dict[str, Any]`

### ModelDeploymentConfigInterface

**Type**: Protocol
**Implemented by**: ModelDeploymentConfig

**Methods**:

- `to_dict() -> Dict[str, Any]`

### ModelExporterInterface

**Type**: Protocol
**Implemented by**: ModelExporter

**Methods**:

- `export_model_from_training_job(training_job_id: int, age_group_min: float, age_group_max: float, export_format: str, db: Session) -> ModelExportMetadata`
- `export_model_direct(model: nn.Module, model_info: Dict[str, Any], age_group_min: float, age_group_max: float, export_format: str) -> ModelExportMetadata`
- `list_exported_models() -> <ast.Subscript object at 0x1105aec50>`

### ModelValidatorInterface

**Type**: Protocol
**Implemented by**: ModelValidator

**Methods**:

- `validate_exported_model(export_metadata: ModelExportMetadata) -> Dict[str, Any]`

### ModelDeploymentServiceInterface

**Type**: Protocol
**Implemented by**: ModelDeploymentService

**Methods**:

- `deploy_model(deployment_config: ModelDeploymentConfig, db: Session) -> Dict[str, Any]`
- `list_deployed_models(db: Session) -> <ast.Subscript object at 0x11050da90>`
- `undeploy_model(model_id: int, db: Session) -> bool`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/model_deployment_service.py`
- Last validated: 2025-12-16 15:47:04

