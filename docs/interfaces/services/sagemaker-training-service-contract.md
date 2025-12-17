# Sagemaker Training Service Contract

## Overview
Service contract for Sagemaker Training Service (service)

**Source File**: `app/services/sagemaker_training_service.py`

## Interface Specification

### Classes

#### SageMakerError

Base exception for SageMaker training errors.

**Inherits from**: Exception

#### SageMakerConfigurationError

Raised when SageMaker configuration is invalid.

**Inherits from**: SageMakerError

#### SageMakerJobError

Raised when SageMaker job operations fail.

**Inherits from**: SageMakerError

#### DockerContainerError

Raised when Docker container operations fail.

**Inherits from**: SageMakerError

#### SageMakerJobConfig

Configuration for SageMaker training job.

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

#### SageMakerContainerBuilder

Builder for SageMaker training containers.

#### SageMakerTrainingService

Service for managing SageMaker training jobs.

## Methods

### to_sagemaker_config

Convert to SageMaker training job configuration.

**Signature**: `to_sagemaker_config() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### build_training_container

Build Docker container for SageMaker training.

Args:
    base_image: Base Docker image to use
    tag: Tag for the built image
    
Returns:
    Image URI of the built container
    
Raises:
    DockerContainerError: If container build fails

**Signature**: `build_training_container(base_image: str, tag: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `base_image` | `str` | Parameter description |
| `tag` | `str` | Parameter description |

**Returns**: `str`

### push_to_ecr

Push Docker image to Amazon ECR.

Args:
    image_tag: Local image tag
    ecr_repository_uri: ECR repository URI
    
Returns:
    Full ECR image URI
    
Raises:
    DockerContainerError: If push fails

**Signature**: `push_to_ecr(image_tag: str, ecr_repository_uri: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `image_tag` | `str` | Parameter description |
| `ecr_repository_uri` | `str` | Parameter description |

**Returns**: `str`

### validate_configuration

Validate SageMaker configuration and permissions.

Returns:
    Dictionary with validation results

**Signature**: `validate_configuration() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### create_execution_role

Create IAM execution role for SageMaker training.

Args:
    role_name: Name for the IAM role
    
Returns:
    ARN of the created role
    
Raises:
    SageMakerConfigurationError: If role creation fails

**Signature**: `create_execution_role(role_name: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `role_name` | `str` | Parameter description |

**Returns**: `str`

### upload_training_data

Upload training data to S3.

Args:
    dataset_folder: Local path to dataset folder
    metadata_file: Local path to metadata file
    s3_bucket: S3 bucket name
    s3_prefix: S3 prefix for uploaded data
    
Returns:
    S3 URI of uploaded data
    
Raises:
    SageMakerError: If upload fails

**Signature**: `upload_training_data(dataset_folder: str, metadata_file: str, s3_bucket: str, s3_prefix: str) -> str`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `dataset_folder` | `str` | Parameter description |
| `metadata_file` | `str` | Parameter description |
| `s3_bucket` | `str` | Parameter description |
| `s3_prefix` | `str` | Parameter description |

**Returns**: `str`

### submit_training_job

Submit SageMaker training job.

Args:
    config: Training configuration
    s3_input_uri: S3 URI for input data
    s3_output_uri: S3 URI for output data
    role_arn: IAM role ARN for execution
    image_uri: Docker image URI for training
    db: Database session
    
Returns:
    Training job ID
    
Raises:
    SageMakerJobError: If job submission fails

**Signature**: `submit_training_job(config: TrainingConfig, s3_input_uri: str, s3_output_uri: str, role_arn: str, image_uri: str, db: Session) -> int`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `config` | `TrainingConfig` | Parameter description |
| `s3_input_uri` | `str` | Parameter description |
| `s3_output_uri` | `str` | Parameter description |
| `role_arn` | `str` | Parameter description |
| `image_uri` | `str` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `int`

### get_job_status

Get status of SageMaker training job.

**Signature**: `get_job_status(job_id: int, db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `job_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### cancel_training_job

Cancel SageMaker training job.

**Signature**: `cancel_training_job(job_id: int, db: Session) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `job_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `bool`

### list_training_jobs

List SageMaker training jobs.

**Signature**: `list_training_jobs(db: Session) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `List[Dict]`

## Dependencies

- `app.models.database.TrainingJob`
- `app.models.database.TrainingReport`
- `app.services.training_config.TrainingConfig`
- `app.services.training_config.TrainingEnvironment`
- `app.services.dataset_preparation.DatasetPreparationService`

## Defined Interfaces

### SageMakerJobConfigInterface

**Type**: Protocol
**Implemented by**: SageMakerJobConfig

**Methods**:

- `to_sagemaker_config() -> Dict[str, Any]`

### SageMakerContainerBuilderInterface

**Type**: Protocol
**Implemented by**: SageMakerContainerBuilder

**Methods**:

- `build_training_container(base_image: str, tag: str) -> str`
- `push_to_ecr(image_tag: str, ecr_repository_uri: str) -> str`

### SageMakerTrainingServiceInterface

**Type**: Protocol
**Implemented by**: SageMakerTrainingService

**Methods**:

- `validate_configuration() -> Dict[str, Any]`
- `create_execution_role(role_name: str) -> str`
- `upload_training_data(dataset_folder: str, metadata_file: str, s3_bucket: str, s3_prefix: str) -> str`
- `submit_training_job(config: TrainingConfig, s3_input_uri: str, s3_output_uri: str, role_arn: str, image_uri: str, db: Session) -> int`
- `get_job_status(job_id: int, db: Session) -> Dict`
- `cancel_training_job(job_id: int, db: Session) -> bool`
- `list_training_jobs(db: Session) -> List[Dict]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/sagemaker_training_service.py`
- Last validated: 2025-12-16 15:47:04

