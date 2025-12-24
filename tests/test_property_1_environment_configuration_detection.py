"""
Property-based test for environment configuration detection.

**Feature: aws-production-deployment, Property 1: Environment Configuration Detection**
**Validates: Requirements 1.1, 1.2, 1.3**

This test validates that the system automatically detects and uses the appropriate
configuration for local vs production environments based on environment variables.
"""

import os
import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import patch
from typing import Dict, Any

from app.core.environment import (
    EnvironmentDetector,
    EnvironmentType,
    StorageBackend,
    EnvironmentConfig,
    reset_environment_config
)


class TestEnvironmentConfigurationDetection:
    """Property-based tests for environment configuration detection"""
    
    def setup_method(self):
        """Reset environment configuration before each test"""
        reset_environment_config()
    
    def teardown_method(self):
        """Clean up after each test"""
        reset_environment_config()
    
    @given(
        app_environment=st.one_of(
            st.just("production"),
            st.just("prod"),
            st.just("local"),
            st.just("development"),
            st.just("dev"),
            st.just(""),
            st.just("invalid")
        ),
        aws_region=st.one_of(
            st.just("eu-west-1"),
            st.just("us-east-1"),
            st.just("us-west-2"),
            st.just(""),
            st.none()
        ),
        s3_bucket=st.one_of(
            st.text(min_size=3, max_size=63, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters='-')),
            st.just(""),
            st.none()
        )
    )
    def test_environment_detection_consistency(self, app_environment: str, aws_region: str, s3_bucket: str):
        """
        **Feature: aws-production-deployment, Property 1: Environment Configuration Detection**
        **Validates: Requirements 1.1, 1.2, 1.3**
        
        For any environment variable configuration, the system should automatically
        detect and use the appropriate storage backend (SQLite+local for development,
        SQLite+S3 for production).
        """
        # Filter out invalid S3 bucket names
        if s3_bucket and (s3_bucket.startswith('-') or s3_bucket.endswith('-') or '--' in s3_bucket):
            assume(False)
        
        # Set up environment variables
        env_vars = {}
        if app_environment:
            env_vars[EnvironmentDetector.ENV_VAR_ENVIRONMENT] = app_environment
        if aws_region:
            env_vars[EnvironmentDetector.ENV_VAR_AWS_REGION] = aws_region
        if s3_bucket:
            env_vars[EnvironmentDetector.ENV_VAR_S3_BUCKET] = s3_bucket
        
        with patch.dict(os.environ, env_vars, clear=False):
            # Test environment detection
            detected_env = EnvironmentDetector.detect_environment()
            
            # Verify detection logic
            if app_environment.lower() in ["production", "prod"]:
                assert detected_env == EnvironmentType.PRODUCTION
            elif app_environment.lower() in ["local", "development", "dev"]:
                assert detected_env == EnvironmentType.LOCAL
            elif aws_region:  # Implicit production detection
                assert detected_env == EnvironmentType.PRODUCTION
            else:  # Default to local
                assert detected_env == EnvironmentType.LOCAL
            
            # Test storage backend selection
            storage_backend = EnvironmentDetector.get_storage_backend(detected_env)
            if detected_env == EnvironmentType.PRODUCTION:
                assert storage_backend == StorageBackend.S3
            else:
                assert storage_backend == StorageBackend.LOCAL
            
            # Test database URL generation
            db_url = EnvironmentDetector.get_database_url(detected_env)
            assert db_url.startswith("sqlite:///")
            if detected_env == EnvironmentType.PRODUCTION:
                assert "production" in db_url
            else:
                assert "drawings.db" in db_url
    
    @given(
        explicit_db_url=st.one_of(
            st.just("sqlite:///./custom.db"),
            st.just("sqlite:///./test_drawings.db"),
            st.just("sqlite:///./production_custom.db"),
            st.none()
        )
    )
    def test_database_url_override(self, explicit_db_url: str):
        """
        Test that explicit DATABASE_URL environment variable overrides default behavior.
        """
        env_vars = {}
        if explicit_db_url:
            env_vars[EnvironmentDetector.ENV_VAR_DATABASE_URL] = explicit_db_url
        
        with patch.dict(os.environ, env_vars, clear=False):
            for env_type in [EnvironmentType.LOCAL, EnvironmentType.PRODUCTION]:
                db_url = EnvironmentDetector.get_database_url(env_type)
                
                if explicit_db_url:
                    assert db_url == explicit_db_url
                else:
                    assert db_url.startswith("sqlite:///")
    
    @given(
        environment=st.sampled_from([EnvironmentType.LOCAL, EnvironmentType.PRODUCTION]),
        aws_region=st.one_of(
            st.just("eu-west-1"),
            st.just("us-east-1"),
            st.just("ap-southeast-1"),
            st.none()
        ),
        s3_bucket=st.one_of(
            st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'))),
            st.none()
        ),
        metrics_enabled=st.booleans(),
        secrets_enabled=st.booleans()
    )
    def test_configuration_creation_validation(
        self, 
        environment: EnvironmentType, 
        aws_region: str, 
        s3_bucket: str,
        metrics_enabled: bool,
        secrets_enabled: bool
    ):
        """
        Test that configuration creation properly validates required fields
        and applies appropriate defaults.
        """
        env_vars = {
            EnvironmentDetector.ENV_VAR_ENVIRONMENT: environment.value,
            EnvironmentDetector.ENV_VAR_ENABLE_METRICS: str(metrics_enabled).lower(),
            EnvironmentDetector.ENV_VAR_ENABLE_SECRETS_MANAGER: str(secrets_enabled).lower()
        }
        
        if aws_region:
            env_vars[EnvironmentDetector.ENV_VAR_AWS_REGION] = aws_region
        if s3_bucket:
            env_vars[EnvironmentDetector.ENV_VAR_S3_BUCKET] = s3_bucket
        
        with patch.dict(os.environ, env_vars, clear=False):
            if environment == EnvironmentType.PRODUCTION and not s3_bucket:
                # Production without S3 bucket should fail validation
                with pytest.raises(ValueError, match="s3_bucket_name is required"):
                    EnvironmentDetector.create_config()
            else:
                # Should create valid configuration
                config = EnvironmentDetector.create_config()
                
                assert config.environment == environment
                assert config.database_url.startswith("sqlite:///")
                
                if environment == EnvironmentType.PRODUCTION:
                    assert config.storage_backend == StorageBackend.S3
                    assert config.s3_bucket_name == s3_bucket
                    assert config.aws_region == (aws_region or "eu-west-1")  # Default region
                    assert config.metrics_enabled is True  # Always enabled in production
                else:
                    assert config.storage_backend == StorageBackend.LOCAL
                    assert config.metrics_enabled == metrics_enabled
                
                assert config.secret_manager_enabled == secrets_enabled
    
    def test_environment_isolation_property(self):
        """
        Test that different environment configurations maintain proper isolation.
        """
        # Test local environment
        with patch.dict(os.environ, {EnvironmentDetector.ENV_VAR_ENVIRONMENT: "local"}, clear=False):
            local_config = EnvironmentDetector.create_config()
            
            assert local_config.environment == EnvironmentType.LOCAL
            assert local_config.storage_backend == StorageBackend.LOCAL
            assert "drawings.db" in local_config.database_url
        
        # Test production environment
        with patch.dict(os.environ, {
            EnvironmentDetector.ENV_VAR_ENVIRONMENT: "production",
            EnvironmentDetector.ENV_VAR_S3_BUCKET: "test-production-bucket",
            EnvironmentDetector.ENV_VAR_AWS_REGION: "eu-west-1"
        }, clear=False):
            prod_config = EnvironmentDetector.create_config()
            
            assert prod_config.environment == EnvironmentType.PRODUCTION
            assert prod_config.storage_backend == StorageBackend.S3
            assert "production" in prod_config.database_url
            assert prod_config.s3_bucket_name == "test-production-bucket"
        
        # Verify configurations are different
        assert local_config.environment != prod_config.environment
        assert local_config.storage_backend != prod_config.storage_backend
        assert local_config.database_url != prod_config.database_url
    
    @given(
        upload_dir=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-')),
        static_dir=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='_-'))
    )
    def test_directory_configuration_consistency(self, upload_dir: str, static_dir: str):
        """
        Test that directory configurations are properly handled across environments.
        """
        env_vars = {
            "UPLOAD_DIR": upload_dir,
            "STATIC_DIR": static_dir
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            for env_type in ["local", "production"]:
                test_env_vars = {EnvironmentDetector.ENV_VAR_ENVIRONMENT: env_type}
                if env_type == "production":
                    test_env_vars.update({
                        EnvironmentDetector.ENV_VAR_S3_BUCKET: "test-bucket",
                        EnvironmentDetector.ENV_VAR_AWS_REGION: "eu-west-1"
                    })
                
                with patch.dict(os.environ, test_env_vars, clear=False):
                    config = EnvironmentDetector.create_config()
                    
                    assert config.upload_dir == upload_dir
                    assert config.static_dir == static_dir