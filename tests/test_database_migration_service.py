"""
Unit tests for the database migration service.
"""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from app.services.database_migration_service import DatabaseMigrationService
from app.core.environment import EnvironmentType


class TestDatabaseMigrationService:
    """Unit tests for DatabaseMigrationService"""
    
    def test_service_initialization_local(self):
        """Test service initialization in local environment"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.LOCAL
            mock_env.return_value.database_url = "sqlite:///test.db"
            mock_env.return_value.aws_region = None
            
            with patch('app.services.database_migration_service.BackupService'):
                service = DatabaseMigrationService()
                
                assert service.env_config.environment == EnvironmentType.LOCAL
                assert service.s3_client is None
    
    def test_service_initialization_production(self):
        """Test service initialization in production environment"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.PRODUCTION
            mock_env.return_value.database_url = "sqlite:///production.db"
            mock_env.return_value.aws_region = "eu-west-1"
            mock_env.return_value.s3_bucket_name = "test-bucket"
            
            with patch('app.services.database_migration_service.BackupService'):
                # Mock HAS_AWS to be True and provide a mock boto3
                with patch('app.services.database_migration_service.HAS_AWS', True):
                    with patch('app.services.database_migration_service.boto3') as mock_boto3_module:
                        mock_s3_client = MagicMock()
                        mock_boto3_module.client.return_value = mock_s3_client
                        
                        service = DatabaseMigrationService()
                        
                        assert service.env_config.environment == EnvironmentType.PRODUCTION
                        assert service.s3_client == mock_s3_client
                        mock_boto3_module.client.assert_called_once_with('s3', region_name='eu-west-1')
    
    @pytest.mark.asyncio
    async def test_create_automated_backup_local(self):
        """Test automated backup creation in local environment"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.LOCAL
            mock_env.return_value.database_url = "sqlite:///test.db"
            
            with patch('app.services.database_migration_service.BackupService') as mock_backup_service_class:
                mock_backup_service = MagicMock()
                mock_backup_service.create_database_backup = AsyncMock(return_value={
                    'backup_name': 'test_backup.db',
                    'backup_path': '/tmp/test_backup.db',
                    'status': 'completed'
                })
                mock_backup_service_class.return_value = mock_backup_service
                
                service = DatabaseMigrationService()
                service._get_migration_info = AsyncMock(return_value={
                    'current_revision': 'abc123',
                    'head_revision': 'abc123',
                    'is_up_to_date': True
                })
                
                result = await service.create_automated_backup()
                
                assert result['backup_name'] == 'test_backup.db'
                assert result['status'] == 'completed'
                assert result['environment'] == 'local'
                assert 'migration_info' in result
                assert 's3_uploaded' not in result  # No S3 upload in local
    
    @pytest.mark.asyncio
    async def test_create_automated_backup_production_with_s3(self):
        """Test automated backup creation in production with S3 upload"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.PRODUCTION
            mock_env.return_value.database_url = "sqlite:///production.db"
            mock_env.return_value.aws_region = "eu-west-1"
            mock_env.return_value.s3_bucket_name = "test-bucket"
            
            with patch('app.services.database_migration_service.BackupService') as mock_backup_service_class:
                mock_backup_service = MagicMock()
                mock_backup_service.create_database_backup = AsyncMock(return_value={
                    'backup_name': 'test_backup.db',
                    'backup_path': '/tmp/test_backup.db',
                    'status': 'completed'
                })
                mock_backup_service_class.return_value = mock_backup_service
                
                with patch('app.services.database_migration_service.boto3.client') as mock_boto3:
                    mock_s3_client = MagicMock()
                    mock_boto3.return_value = mock_s3_client
                    
                    service = DatabaseMigrationService()
                    service._get_migration_info = AsyncMock(return_value={
                        'current_revision': 'abc123',
                        'head_revision': 'abc123',
                        'is_up_to_date': True
                    })
                    service._upload_backup_to_s3 = AsyncMock(return_value={
                        's3_uploaded': True,
                        's3_bucket': 'test-bucket',
                        's3_key': 'database-backups/production/test_backup.db',
                        's3_url': 's3://test-bucket/database-backups/production/test_backup.db'
                    })
                    
                    result = await service.create_automated_backup()
                    
                    assert result['backup_name'] == 'test_backup.db'
                    assert result['status'] == 'completed'
                    assert result['environment'] == 'production'
                    assert result['s3_uploaded'] is True
                    assert result['s3_bucket'] == 'test-bucket'
    
    def test_schema_comparison_identical(self):
        """Test schema comparison with identical schemas"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.LOCAL
            mock_env.return_value.database_url = "sqlite:///test.db"
            
            with patch('app.services.database_migration_service.BackupService'):
                service = DatabaseMigrationService()
                
                schema1 = {
                    'tables': {
                        'users': {
                            'columns': [
                                {'name': 'id', 'type': 'INTEGER', 'not_null': True, 'default_value': None, 'primary_key': True},
                                {'name': 'name', 'type': 'TEXT', 'not_null': False, 'default_value': None, 'primary_key': False}
                            ]
                        }
                    },
                    'indexes': {}
                }
                
                schema2 = {
                    'tables': {
                        'users': {
                            'columns': [
                                {'name': 'id', 'type': 'INTEGER', 'not_null': True, 'default_value': None, 'primary_key': True},
                                {'name': 'name', 'type': 'TEXT', 'not_null': False, 'default_value': None, 'primary_key': False}
                            ]
                        }
                    },
                    'indexes': {}
                }
                
                result = service._compare_schemas(schema1, schema2)
                
                assert result['is_identical'] is True
                assert len(result['differences']) == 0
    
    def test_schema_comparison_different(self):
        """Test schema comparison with different schemas"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.LOCAL
            mock_env.return_value.database_url = "sqlite:///test.db"
            
            with patch('app.services.database_migration_service.BackupService'):
                service = DatabaseMigrationService()
                
                schema1 = {
                    'tables': {
                        'users': {
                            'columns': [
                                {'name': 'id', 'type': 'INTEGER', 'not_null': True, 'default_value': None, 'primary_key': True},
                                {'name': 'name', 'type': 'TEXT', 'not_null': False, 'default_value': None, 'primary_key': False}
                            ]
                        }
                    },
                    'indexes': {}
                }
                
                schema2 = {
                    'tables': {
                        'users': {
                            'columns': [
                                {'name': 'id', 'type': 'INTEGER', 'not_null': True, 'default_value': None, 'primary_key': True},
                                {'name': 'email', 'type': 'TEXT', 'not_null': False, 'default_value': None, 'primary_key': False}
                            ]
                        }
                    },
                    'indexes': {}
                }
                
                result = service._compare_schemas(schema1, schema2)
                
                assert result['is_identical'] is False
                assert len(result['differences']) == 2  # name missing in schema2, email missing in schema1
                assert any('name' in diff for diff in result['differences'])
                assert any('email' in diff for diff in result['differences'])
    
    @pytest.mark.asyncio
    async def test_migration_info_retrieval(self):
        """Test migration information retrieval"""
        with patch('app.services.database_migration_service.get_current_environment') as mock_env:
            mock_env.return_value.environment = EnvironmentType.LOCAL
            mock_env.return_value.database_url = "sqlite:///test.db"
            
            with patch('app.services.database_migration_service.BackupService'):
                with patch('app.services.database_migration_service.create_engine') as mock_create_engine:
                    mock_engine = MagicMock()
                    mock_conn = MagicMock()
                    mock_engine.connect.return_value.__enter__.return_value = mock_conn
                    mock_create_engine.return_value = mock_engine
                    
                    with patch('app.services.database_migration_service.MigrationContext') as mock_migration_context:
                        mock_context = MagicMock()
                        mock_context.get_current_revision.return_value = 'abc123'
                        mock_migration_context.configure.return_value = mock_context
                        
                        with patch('app.services.database_migration_service.ScriptDirectory') as mock_script_dir:
                            mock_script = MagicMock()
                            mock_script.get_current_head.return_value = 'def456'
                            mock_script_dir.from_config.return_value = mock_script
                            
                            service = DatabaseMigrationService()
                            
                            result = await service._get_migration_info()
                            
                            assert result['current_revision'] == 'abc123'
                            assert result['head_revision'] == 'def456'
                            assert result['is_up_to_date'] is False