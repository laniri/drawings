"""
Amazon SageMaker Training Service for Children's Drawing Anomaly Detection System

This module provides SageMaker training job submission and monitoring, Docker container
management for SageMaker training environment, and Boto3 integration for job management
and artifact retrieval.
"""

import json
import logging
import os
import queue
import tarfile
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import docker
from botocore.exceptions import ClientError, NoCredentialsError
from docker.errors import DockerException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.core.exceptions import ValidationError
from app.models.database import TrainingJob, TrainingReport
from app.services.dataset_preparation import DatasetPreparationService
from app.services.training_config import TrainingConfig, TrainingEnvironment

logger = logging.getLogger(__name__)


class SageMakerError(Exception):
    """Base exception for SageMaker training errors."""

    pass


class SageMakerConfigurationError(SageMakerError):
    """Raised when SageMaker configuration is invalid."""

    pass


class SageMakerJobError(SageMakerError):
    """Raised when SageMaker job operations fail."""

    pass


class DockerContainerError(SageMakerError):
    """Raised when Docker container operations fail."""

    pass


@dataclass
class SageMakerJobConfig:
    """Configuration for SageMaker training job."""

    job_name: str
    role_arn: str
    image_uri: str
    instance_type: str
    instance_count: int
    volume_size_gb: int
    max_runtime_seconds: int
    input_data_s3_uri: str
    output_s3_uri: str
    hyperparameters: Dict[str, str]
    environment_variables: Dict[str, str]

    def to_sagemaker_config(self) -> Dict[str, Any]:
        """Convert to SageMaker training job configuration."""
        return {
            "TrainingJobName": self.job_name,
            "RoleArn": self.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
            },
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": self.input_data_s3_uri,
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                    "ContentType": "application/x-parquet",
                    "CompressionType": "None",
                }
            ],
            "OutputDataConfig": {"S3OutputPath": self.output_s3_uri},
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": self.instance_count,
                "VolumeSizeInGB": self.volume_size_gb,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": self.max_runtime_seconds},
            "HyperParameters": self.hyperparameters,
            "Environment": self.environment_variables,
        }


class SageMakerContainerBuilder:
    """Builder for SageMaker training containers."""

    def __init__(self):
        self.docker_client = None
        self._initialize_docker()

    def _initialize_docker(self):
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            # Test Docker connection
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise DockerContainerError(f"Docker initialization failed: {str(e)}")

    def build_training_container(
        self,
        base_image: str = "python:3.11-slim",
        tag: str = "drawing-anomaly-sagemaker:latest",
    ) -> str:
        """
        Build Docker container for SageMaker training.

        Args:
            base_image: Base Docker image to use
            tag: Tag for the built image

        Returns:
            Image URI of the built container

        Raises:
            DockerContainerError: If container build fails
        """
        try:
            logger.info(f"Building SageMaker training container with tag: {tag}")

            # Create Dockerfile content
            dockerfile_content = self._generate_dockerfile(base_image)

            # Create training script
            training_script = self._generate_training_script()

            # Create requirements file
            requirements_content = self._generate_requirements()

            # Build container
            with tempfile.TemporaryDirectory() as build_dir:
                build_path = Path(build_dir)

                # Write files
                (build_path / "Dockerfile").write_text(dockerfile_content)
                (build_path / "train.py").write_text(training_script)
                (build_path / "requirements.txt").write_text(requirements_content)

                # Copy additional files if needed
                self._copy_source_files(build_path)

                # Build image
                image, build_logs = self.docker_client.images.build(
                    path=str(build_path), tag=tag, rm=True, forcerm=True
                )

                # Log build output
                for log in build_logs:
                    if "stream" in log:
                        logger.debug(f"Docker build: {log['stream'].strip()}")

                logger.info(f"Successfully built container: {tag}")
                return tag

        except DockerException as e:
            logger.error(f"Failed to build SageMaker container: {str(e)}")
            raise DockerContainerError(f"Container build failed: {str(e)}")

    def _generate_dockerfile(self, base_image: str) -> str:
        """Generate Dockerfile content for SageMaker training."""
        return f"""
FROM {base_image}

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .
COPY app/ ./app/

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${{PATH}}"

# Set the training script as entrypoint
ENTRYPOINT ["python", "train.py"]
"""

    def _generate_training_script(self) -> str:
        """Generate the main training script for SageMaker."""
        return '''
#!/usr/bin/env python3
"""
SageMaker Training Script for Children's Drawing Anomaly Detection

This script runs autoencoder training in the SageMaker environment.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add app to path
sys.path.append('/opt/ml/code')

from app.services.embedding_service import EmbeddingService
from app.services.model_manager import AutoencoderModel, AutoencoderTrainer
from app.services.training_config import TrainingConfig

# SageMaker paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging for SageMaker environment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/opt/ml/output/training.log')
        ]
    )


def parse_hyperparameters():
    """Parse hyperparameters from SageMaker environment."""
    parser = argparse.ArgumentParser()
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dims', type=str, default='256,128,64')
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    
    # Data parameters
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    
    # Model parameters
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Parse hidden dimensions
    args.hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    return args


def load_training_data():
    """Load training data from SageMaker input channels."""
    training_path = Path(SM_CHANNEL_TRAINING)
    
    # Look for metadata file
    metadata_files = list(training_path.glob('*.csv')) + list(training_path.glob('*.json'))
    if not metadata_files:
        raise ValueError("No metadata file found in training data")
    
    metadata_file = metadata_files[0]
    logger.info(f"Loading metadata from: {metadata_file}")
    
    # Load metadata
    if metadata_file.suffix.lower() == '.csv':
        metadata_df = pd.read_csv(metadata_file)
    else:
        metadata_df = pd.read_json(metadata_file)
    
    # Find image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        image_files.extend(training_path.glob(f'*{ext}'))
        image_files.extend(training_path.glob(f'*{ext.upper()}'))
    
    logger.info(f"Found {len(image_files)} image files")
    
    return image_files, metadata_df


def generate_embeddings(image_files, metadata_df):
    """Generate embeddings for training data."""
    logger.info("Initializing embedding service")
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    embedding_service.initialize()
    
    embeddings = []
    ages = []
    
    for image_file in image_files:
        filename = image_file.name
        
        # Find metadata for this file
        metadata_row = metadata_df[metadata_df['filename'] == filename]
        if metadata_row.empty:
            # Try without extension
            stem = image_file.stem
            metadata_row = metadata_df[metadata_df['filename'].str.contains(stem)]
        
        if metadata_row.empty:
            logger.warning(f"No metadata found for {filename}, skipping")
            continue
        
        try:
            # Load image and generate embedding
            image = Image.open(image_file).convert('RGB')
            age = float(metadata_row.iloc[0]['age_years'])
            
            embedding = embedding_service.generate_embedding(image, age)
            embeddings.append(embedding)
            ages.append(age)
            
        except Exception as e:
            logger.warning(f"Failed to process {filename}: {str(e)}")
            continue
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    return np.array(embeddings), np.array(ages)


def split_data(embeddings, ages, train_split, val_split, test_split):
    """Split data into train/validation/test sets."""
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    if test_split > 0:
        train_val_embeddings, test_embeddings, train_val_ages, test_ages = train_test_split(
            embeddings, ages, test_size=test_split, random_state=42
        )
    else:
        train_val_embeddings, test_embeddings = embeddings, np.array([])
        train_val_ages, test_ages = ages, np.array([])
    
    # Second split: separate train and validation
    if val_split > 0:
        val_ratio_adjusted = val_split / (train_split + val_split)
        train_embeddings, val_embeddings, train_ages, val_ages = train_test_split(
            train_val_embeddings, train_val_ages, test_size=val_ratio_adjusted, random_state=42
        )
    else:
        train_embeddings, val_embeddings = train_val_embeddings, np.array([])
        train_ages, val_ages = train_val_ages, np.array([])
    
    logger.info(f"Data split: train={len(train_embeddings)}, val={len(val_embeddings)}, test={len(test_embeddings)}")
    
    return train_embeddings, val_embeddings, test_embeddings


def train_model(args, train_embeddings, val_embeddings):
    """Train autoencoder model."""
    logger.info("Starting model training")
    
    # Create training configuration
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta
    )
    
    # Initialize trainer
    trainer = AutoencoderTrainer(config)
    
    # Train model
    training_result = trainer.train(train_embeddings, val_embeddings)
    
    logger.info(f"Training completed. Best validation loss: {training_result['best_val_loss']:.6f}")
    
    return trainer.model, training_result


def save_model_and_results(model, training_result, args):
    """Save trained model and results."""
    model_dir = Path(SM_MODEL_DIR)
    output_dir = Path(SM_OUTPUT_DIR)
    
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model.get_architecture_info(),
        'training_config': vars(args)
    }, model_path)
    
    logger.info(f"Model saved to: {model_path}")
    
    # Save training results
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(training_result, f, indent=2, default=str)
    
    logger.info(f"Training results saved to: {results_path}")
    
    # Save metrics summary
    metrics_path = output_dir / "metrics.json"
    metrics = {
        'final_loss': training_result['best_val_loss'],
        'epochs_trained': training_result['epochs_trained'],
        'training_time': training_result.get('training_time', 0),
        'model_parameters': model.get_architecture_info()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_path}")


def main():
    """Main training function."""
    setup_logging()
    logger.info("Starting SageMaker training job")
    
    try:
        # Parse hyperparameters
        args = parse_hyperparameters()
        logger.info(f"Training configuration: {vars(args)}")
        
        # Load training data
        image_files, metadata_df = load_training_data()
        
        # Generate embeddings
        embeddings, ages = generate_embeddings(image_files, metadata_df)
        
        # Split data
        train_embeddings, val_embeddings, test_embeddings = split_data(
            embeddings, ages, args.train_split, args.validation_split, args.test_split
        )
        
        # Train model
        model, training_result = train_model(args, train_embeddings, val_embeddings)
        
        # Save results
        save_model_and_results(model, training_result, args)
        
        logger.info("SageMaker training job completed successfully")
        
    except Exception as e:
        logger.error(f"Training job failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
'''

    def _generate_requirements(self) -> str:
        """Generate requirements.txt for SageMaker container."""
        return """
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
pillow>=9.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
boto3>=1.26.0
"""

    def _copy_source_files(self, build_path: Path):
        """Copy necessary source files to build directory."""
        # This would copy the app directory structure
        # For now, we'll assume the training script is self-contained
        pass

    def push_to_ecr(self, image_tag: str, ecr_repository_uri: str) -> str:
        """
        Push Docker image to Amazon ECR.

        Args:
            image_tag: Local image tag
            ecr_repository_uri: ECR repository URI

        Returns:
            Full ECR image URI

        Raises:
            DockerContainerError: If push fails
        """
        try:
            logger.info(f"Pushing image {image_tag} to ECR: {ecr_repository_uri}")

            # Tag image for ECR
            ecr_tag = f"{ecr_repository_uri}:latest"
            image = self.docker_client.images.get(image_tag)
            image.tag(ecr_tag)

            # Push to ECR
            push_logs = self.docker_client.images.push(
                ecr_tag, stream=True, decode=True
            )

            for log in push_logs:
                if "status" in log:
                    logger.debug(f"ECR push: {log['status']}")
                if "error" in log:
                    raise DockerContainerError(f"ECR push failed: {log['error']}")

            logger.info(f"Successfully pushed image to ECR: {ecr_tag}")
            return ecr_tag

        except DockerException as e:
            logger.error(f"Failed to push image to ECR: {str(e)}")
            raise DockerContainerError(f"ECR push failed: {str(e)}")


class SageMakerTrainingService:
    """Service for managing SageMaker training jobs."""

    def __init__(self):
        self.settings = get_settings()
        self.sagemaker_client = None
        self.s3_client = None
        self.iam_client = None
        self.container_builder = SageMakerContainerBuilder()
        self.active_jobs = {}  # job_id -> job_info
        self._initialize_aws_clients()

    def _initialize_aws_clients(self):
        """Initialize AWS clients."""
        try:
            # Initialize SageMaker client
            self.sagemaker_client = boto3.client("sagemaker")

            # Initialize S3 client
            self.s3_client = boto3.client("s3")

            # Initialize IAM client
            self.iam_client = boto3.client("iam")

            # Test connection
            self.sagemaker_client.list_training_jobs(MaxResults=1)

            logger.info("AWS clients initialized successfully")

        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise SageMakerConfigurationError("AWS credentials not configured")
        except ClientError as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise SageMakerConfigurationError(
                f"AWS client initialization failed: {str(e)}"
            )

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate SageMaker configuration and permissions.

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "configuration": {},
        }

        try:
            # Check SageMaker permissions
            try:
                self.sagemaker_client.list_training_jobs(MaxResults=1)
                validation_result["configuration"]["sagemaker_access"] = True
            except ClientError as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"SageMaker access denied: {str(e)}")

            # Check S3 permissions
            try:
                self.s3_client.list_buckets()
                validation_result["configuration"]["s3_access"] = True
            except ClientError as e:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"S3 access denied: {str(e)}")

            # Check IAM permissions
            try:
                self.iam_client.get_user()
                validation_result["configuration"]["iam_access"] = True
            except ClientError as e:
                validation_result["warnings"].append(f"Limited IAM access: {str(e)}")

            # Check required environment variables
            required_vars = ["AWS_DEFAULT_REGION"]
            for var in required_vars:
                if not os.environ.get(var):
                    validation_result["warnings"].append(
                        f"Environment variable {var} not set"
                    )

        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Configuration validation failed: {str(e)}"
            )

        return validation_result

    def create_execution_role(self, role_name: str) -> str:
        """
        Create IAM execution role for SageMaker training.

        Args:
            role_name: Name for the IAM role

        Returns:
            ARN of the created role

        Raises:
            SageMakerConfigurationError: If role creation fails
        """
        try:
            # Define trust policy
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            # Create role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Execution role for SageMaker training jobs",
            )

            role_arn = response["Role"]["Arn"]

            # Attach required policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            ]

            for policy_arn in policies:
                self.iam_client.attach_role_policy(
                    RoleName=role_name, PolicyArn=policy_arn
                )

            logger.info(f"Created SageMaker execution role: {role_arn}")
            return role_arn

        except ClientError as e:
            if e.response["Error"]["Code"] == "EntityAlreadyExists":
                # Role already exists, get its ARN
                response = self.iam_client.get_role(RoleName=role_name)
                return response["Role"]["Arn"]
            else:
                logger.error(f"Failed to create execution role: {str(e)}")
                raise SageMakerConfigurationError(f"Role creation failed: {str(e)}")

    def upload_training_data(
        self, dataset_folder: str, metadata_file: str, s3_bucket: str, s3_prefix: str
    ) -> str:
        """
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
        """
        try:
            logger.info(f"Uploading training data to s3://{s3_bucket}/{s3_prefix}")

            dataset_path = Path(dataset_folder)
            metadata_path = Path(metadata_file)

            # Upload metadata file
            metadata_s3_key = f"{s3_prefix}/metadata{metadata_path.suffix}"
            self.s3_client.upload_file(str(metadata_path), s3_bucket, metadata_s3_key)

            # Upload image files
            image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
            uploaded_count = 0

            for image_file in dataset_path.iterdir():
                if image_file.suffix.lower() in image_extensions:
                    s3_key = f"{s3_prefix}/{image_file.name}"
                    self.s3_client.upload_file(str(image_file), s3_bucket, s3_key)
                    uploaded_count += 1

            s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
            logger.info(f"Uploaded {uploaded_count} files to {s3_uri}")

            return s3_uri

        except ClientError as e:
            logger.error(f"Failed to upload training data: {str(e)}")
            raise SageMakerError(f"Data upload failed: {str(e)}")

    def submit_training_job(
        self,
        config: TrainingConfig,
        s3_input_uri: str,
        s3_output_uri: str,
        role_arn: str,
        image_uri: str,
        db: Session,
    ) -> int:
        """
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
        """
        try:
            # Create unique job name
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            sagemaker_job_name = f"{config.job_name}-{timestamp}"

            # Prepare hyperparameters
            hyperparameters = {
                "learning_rate": str(config.optimizer.learning_rate),
                "batch_size": str(config.data.batch_size),
                "epochs": str(config.epochs),
                "hidden_dims": ",".join(map(str, config.model.hidden_dims)),
                "latent_dim": str(config.model.latent_dim),
                "dropout_rate": str(config.model.dropout_rate),
                "train_split": str(config.data.train_split),
                "validation_split": str(config.data.validation_split),
                "test_split": str(config.data.test_split),
                "early_stopping_patience": str(config.early_stopping_patience),
            }

            # Create SageMaker job configuration
            sagemaker_config = SageMakerJobConfig(
                job_name=sagemaker_job_name,
                role_arn=role_arn,
                image_uri=image_uri,
                instance_type=config.sagemaker_instance_type,
                instance_count=config.sagemaker_instance_count,
                volume_size_gb=config.sagemaker_volume_size,
                max_runtime_seconds=config.sagemaker_max_runtime,
                input_data_s3_uri=s3_input_uri,
                output_s3_uri=s3_output_uri,
                hyperparameters=hyperparameters,
                environment_variables={},
            )

            # Submit training job
            job_config = sagemaker_config.to_sagemaker_config()
            response = self.sagemaker_client.create_training_job(**job_config)

            sagemaker_job_arn = response["TrainingJobArn"]

            # Create database record
            training_job = TrainingJob(
                job_name=config.job_name,
                environment="sagemaker",
                config_parameters=json.dumps(asdict(config)),
                dataset_path=config.dataset_folder,
                status="pending",
                start_timestamp=datetime.utcnow(),
                sagemaker_job_arn=sagemaker_job_arn,
            )

            db.add(training_job)
            db.commit()
            db.refresh(training_job)

            # Store job info for monitoring
            self.active_jobs[training_job.id] = {
                "sagemaker_job_name": sagemaker_job_name,
                "sagemaker_job_arn": sagemaker_job_arn,
                "config": config,
                "start_time": datetime.utcnow(),
            }

            logger.info(
                f"Submitted SageMaker training job: {sagemaker_job_name} (ID: {training_job.id})"
            )

            # Start monitoring in background
            monitoring_thread = threading.Thread(
                target=self._monitor_training_job,
                args=(training_job.id, sagemaker_job_name, db),
                daemon=True,
            )
            monitoring_thread.start()

            return training_job.id

        except ClientError as e:
            logger.error(f"Failed to submit SageMaker training job: {str(e)}")
            raise SageMakerJobError(f"Job submission failed: {str(e)}")

    def _monitor_training_job(self, job_id: int, sagemaker_job_name: str, db: Session):
        """Monitor SageMaker training job in background."""
        try:
            logger.info(f"Starting monitoring for SageMaker job: {sagemaker_job_name}")

            while True:
                try:
                    # Get job status
                    response = self.sagemaker_client.describe_training_job(
                        TrainingJobName=sagemaker_job_name
                    )

                    status = response["TrainingJobStatus"]

                    # Update database
                    training_job = (
                        db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
                    )
                    if training_job:
                        training_job.status = status.lower()

                        if status in ["Completed", "Failed", "Stopped"]:
                            training_job.end_timestamp = datetime.utcnow()

                            if status == "Completed":
                                # Download and process results
                                self._process_completed_job(job_id, response, db)

                            db.commit()
                            break

                        db.commit()

                    # Wait before next check
                    time.sleep(30)  # Check every 30 seconds

                except ClientError as e:
                    logger.error(f"Error monitoring job {sagemaker_job_name}: {str(e)}")
                    break
                except Exception as e:
                    logger.error(
                        f"Unexpected error monitoring job {sagemaker_job_name}: {str(e)}"
                    )
                    break

        finally:
            # Clean up
            if job_id in self.active_jobs:
                del self.active_jobs[job_id]

    def _process_completed_job(self, job_id: int, job_description: Dict, db: Session):
        """Process completed SageMaker training job."""
        try:
            # Extract job information
            output_s3_uri = job_description.get("OutputDataConfig", {}).get(
                "S3OutputPath", ""
            )
            training_time = job_description.get("TrainingTimeInSeconds", 0)

            # Download training artifacts
            if output_s3_uri:
                artifacts_path = self._download_training_artifacts(
                    job_id, output_s3_uri
                )
            else:
                artifacts_path = None

            # Create training report
            training_report = TrainingReport(
                training_job_id=job_id,
                final_loss=0.0,  # Will be updated from artifacts
                validation_accuracy=0.0,  # Will be updated from artifacts
                best_epoch=0,  # Will be updated from artifacts
                training_time_seconds=training_time,
                model_parameters_path=(
                    str(artifacts_path / "model.pth") if artifacts_path else ""
                ),
                metrics_summary="{}",  # Will be updated from artifacts
                report_file_path=(
                    str(artifacts_path / "training_results.json")
                    if artifacts_path
                    else ""
                ),
            )

            # Load metrics from artifacts if available
            if artifacts_path and (artifacts_path / "metrics.json").exists():
                with open(artifacts_path / "metrics.json", "r") as f:
                    metrics = json.load(f)
                    training_report.final_loss = metrics.get("final_loss", 0.0)
                    training_report.validation_accuracy = 1.0 - metrics.get(
                        "final_loss", 1.0
                    )
                    training_report.best_epoch = metrics.get("epochs_trained", 0)
                    training_report.metrics_summary = json.dumps(metrics)

            db.add(training_report)
            db.commit()

            logger.info(f"Processed completed SageMaker job: {job_id}")

        except Exception as e:
            logger.error(f"Failed to process completed job {job_id}: {str(e)}")

    def _download_training_artifacts(
        self, job_id: int, s3_output_uri: str
    ) -> Optional[Path]:
        """Download training artifacts from S3."""
        try:
            # Parse S3 URI
            s3_parts = s3_output_uri.replace("s3://", "").split("/", 1)
            bucket = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ""

            # Create local directory
            artifacts_dir = Path(f"outputs/sagemaker_job_{job_id}")
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            # List and download artifacts
            response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    filename = Path(key).name
                    local_path = artifacts_dir / filename

                    self.s3_client.download_file(bucket, key, str(local_path))
                    logger.info(f"Downloaded artifact: {filename}")

            return artifacts_dir

        except ClientError as e:
            logger.error(f"Failed to download artifacts: {str(e)}")
            return None

    def get_job_status(self, job_id: int, db: Session) -> Dict:
        """Get status of SageMaker training job."""
        training_job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()

        if not training_job:
            raise SageMakerJobError(f"Training job {job_id} not found")

        status = {
            "job_id": job_id,
            "job_name": training_job.job_name,
            "status": training_job.status,
            "start_timestamp": training_job.start_timestamp,
            "end_timestamp": training_job.end_timestamp,
            "environment": training_job.environment,
            "sagemaker_job_arn": training_job.sagemaker_job_arn,
        }

        # Get detailed SageMaker status if job is active
        if job_id in self.active_jobs and training_job.sagemaker_job_arn:
            try:
                job_name = self.active_jobs[job_id]["sagemaker_job_name"]
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )

                status.update(
                    {
                        "sagemaker_status": response.get("TrainingJobStatus"),
                        "instance_type": response.get("ResourceConfig", {}).get(
                            "InstanceType"
                        ),
                        "instance_count": response.get("ResourceConfig", {}).get(
                            "InstanceCount"
                        ),
                        "training_time_seconds": response.get(
                            "TrainingTimeInSeconds", 0
                        ),
                        "billable_time_seconds": response.get(
                            "BillableTimeInSeconds", 0
                        ),
                    }
                )

            except ClientError as e:
                logger.warning(f"Failed to get SageMaker job details: {str(e)}")

        return status

    def cancel_training_job(self, job_id: int, db: Session) -> bool:
        """Cancel SageMaker training job."""
        if job_id not in self.active_jobs:
            return False

        try:
            job_name = self.active_jobs[job_id]["sagemaker_job_name"]

            # Stop SageMaker job
            self.sagemaker_client.stop_training_job(TrainingJobName=job_name)

            # Update database
            training_job = (
                db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
            )
            if training_job:
                training_job.status = "stopped"
                training_job.end_timestamp = datetime.utcnow()
                db.commit()

            logger.info(f"Cancelled SageMaker training job: {job_name}")
            return True

        except ClientError as e:
            logger.error(f"Failed to cancel SageMaker job: {str(e)}")
            return False

    def list_training_jobs(self, db: Session) -> List[Dict]:
        """List SageMaker training jobs."""
        jobs = (
            db.query(TrainingJob)
            .filter(TrainingJob.environment == "sagemaker")
            .order_by(TrainingJob.start_timestamp.desc())
            .all()
        )

        return [
            {
                "id": job.id,
                "job_name": job.job_name,
                "status": job.status,
                "start_timestamp": job.start_timestamp,
                "end_timestamp": job.end_timestamp,
                "sagemaker_job_arn": job.sagemaker_job_arn,
            }
            for job in jobs
        ]


# Global service instance
_sagemaker_training_service = None


def get_sagemaker_training_service() -> Optional[SageMakerTrainingService]:
    """Get the global SageMaker training service instance."""
    global _sagemaker_training_service
    if _sagemaker_training_service is None:
        try:
            _sagemaker_training_service = SageMakerTrainingService()
        except Exception as e:
            logger.warning(f"SageMaker service initialization failed: {str(e)}")
            logger.info(
                "SageMaker functionality will be disabled. Local training is still available."
            )
            _sagemaker_training_service = None
    return _sagemaker_training_service
