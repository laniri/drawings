"""
Application configuration settings using Pydantic.
"""

from typing import List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    PROJECT_NAME: str = "Children's Drawing Anomaly Detection System"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./drawings.db"
    
    # CORS
    BACKEND_CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS.split(",")]
    
    # File storage
    UPLOAD_DIR: str = "uploads"
    STATIC_DIR: str = "static"
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


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


settings = Settings()