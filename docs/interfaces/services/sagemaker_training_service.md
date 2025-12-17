# Sagemaker Training Service Service

Amazon SageMaker Training Service for Children's Drawing Anomaly Detection System

This module provides SageMaker training job submission and monitoring, Docker container
management for SageMaker training environment, and Boto3 integration for job management
and artifact retrieval.

## Class: SageMakerError

Base exception for SageMaker training errors.

## Class: SageMakerConfigurationError

Raised when SageMaker configuration is invalid.

## Class: SageMakerJobError

Raised when SageMaker job operations fail.

## Class: DockerContainerError

Raised when Docker container operations fail.

## Class: SageMakerJobConfig

Configuration for SageMaker training job.

### to_sagemaker_config

Convert to SageMaker training job configuration.

**Signature**: `to_sagemaker_config()`

## Class: SageMakerContainerBuilder

Builder for SageMaker training containers.

### build_training_container

Build Docker container for SageMaker training.

Args:
    base_image: Base Docker image to use
    tag: Tag for the built image
    
Returns:
    Image URI of the built container
    
Raises:
    DockerContainerError: If container build fails

**Signature**: `build_training_container(base_image, tag)`

### push_to_ecr

Push Docker image to Amazon ECR.

Args:
    image_tag: Local image tag
    ecr_repository_uri: ECR repository URI
    
Returns:
    Full ECR image URI
    
Raises:
    DockerContainerError: If push fails

**Signature**: `push_to_ecr(image_tag, ecr_repository_uri)`

## Class: SageMakerTrainingService

Service for managing SageMaker training jobs.

### validate_configuration

Validate SageMaker configuration and permissions.

Returns:
    Dictionary with validation results

**Signature**: `validate_configuration()`

### create_execution_role

Create IAM execution role for SageMaker training.

Args:
    role_name: Name for the IAM role
    
Returns:
    ARN of the created role
    
Raises:
    SageMakerConfigurationError: If role creation fails

**Signature**: `create_execution_role(role_name)`

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

**Signature**: `upload_training_data(dataset_folder, metadata_file, s3_bucket, s3_prefix)`

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

**Signature**: `submit_training_job(config, s3_input_uri, s3_output_uri, role_arn, image_uri, db)`

### get_job_status

Get status of SageMaker training job.

**Signature**: `get_job_status(job_id, db)`

### cancel_training_job

Cancel SageMaker training job.

**Signature**: `cancel_training_job(job_id, db)`

### list_training_jobs

List SageMaker training jobs.

**Signature**: `list_training_jobs(db)`

