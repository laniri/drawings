"""
Application configuration settings using Pydantic.
"""

import os
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.environment import (
    EnvironmentType,
    StorageBackend,
    get_current_environment,
)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    PROJECT_NAME: str = "Children's Drawing Anomaly Detection System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Environment-aware configuration
    _env_config = None

    @property
    def env_config(self):
        """Get environment configuration with lazy loading"""
        if self._env_config is None:
            self._env_config = get_current_environment()
        return self._env_config

    @property
    def DATABASE_URL(self) -> str:
        """Get database URL from environment configuration"""
        return self.env_config.database_url

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.env_config.environment == EnvironmentType.PRODUCTION

    @property
    def is_local(self) -> bool:
        """Check if running in local environment"""
        return self.env_config.environment == EnvironmentType.LOCAL

    @property
    def storage_backend(self) -> StorageBackend:
        """Get storage backend type"""
        return self.env_config.storage_backend

    @property
    def s3_bucket_name(self) -> Optional[str]:
        """Get S3 bucket name for production"""
        return self.env_config.s3_bucket_name

    @property
    def aws_region(self) -> Optional[str]:
        """Get AWS region"""
        return self.env_config.aws_region

    # CORS
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]

    # File storage - environment aware
    @property
    def UPLOAD_DIR(self) -> str:
        """Get upload directory from environment configuration"""
        return self.env_config.upload_dir

    @property
    def STATIC_DIR(self) -> str:
        """Get static directory from environment configuration"""
        return self.env_config.static_dir

    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB

    # ML Models
    VISION_MODEL: str = "vit"
    DEFAULT_THRESHOLD_PERCENTILE: float = 95.0
    MIN_SAMPLES_PER_AGE_GROUP: int = 50

    # Development
    DEBUG: bool = False

    # AWS Configuration
    AWS_REGION: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # Authentication Configuration
    ADMIN_PASSWORD: Optional[str] = None

    # SageMaker Configuration
    SAGEMAKER_EXECUTION_ROLE_ARN: Optional[str] = None
    SAGEMAKER_DEFAULT_INSTANCE_TYPE: str = "ml.m5.large"
    SAGEMAKER_DEFAULT_INSTANCE_COUNT: int = 1
    SAGEMAKER_DEFAULT_VOLUME_SIZE: int = 30
    SAGEMAKER_MAX_RUNTIME_SECONDS: int = 86400  # 24 hours

    # S3 Configuration
    S3_TRAINING_BUCKET: Optional[str] = None
    S3_MODEL_ARTIFACTS_BUCKET: Optional[str] = None

    # ECR Configuration
    ECR_REPOSITORY_URI: Optional[str] = None

    # Monitoring Configuration
    SNS_ALERT_TOPIC_ARN: Optional[str] = None
    CLOUDWATCH_LOG_GROUP: str = "/aws/ecs/drawing-analysis"
    MONITORING_ENABLED: bool = True
    COST_THRESHOLD: float = 40.0
    PERFORMANCE_THRESHOLDS: dict = {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "response_time": 5.0,
        "error_rate": 5.0,
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


settings = Settings()
