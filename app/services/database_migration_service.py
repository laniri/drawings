"""
Database migration and backup service for AWS production deployment.

This service provides automated SQLite backup with S3 integration,
Alembic migration consistency across environments, and migration rollback capabilities.
"""

import asyncio
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.environment import get_current_environment, EnvironmentType
from app.core.exceptions import StorageError, ConfigurationError
from app.services.backup_service import BackupService

logger = logging.getLogger(__name__)


class DatabaseMigrationService:
    """
    Service for managing database migrations and backups with S3 integration.
    
    This service extends the existing BackupService with production-ready features:
    - Automated S3 backup for production environments
    - Migration consistency validation across environments
    - Migration rollback capabilities
    - Database consistency checks
    """
    
    def __init__(self):
        self.env_config = get_current_environment()
        self.backup_service = BackupService()
        
        # S3 configuration for production
        self.s3_client = None
        if self.env_config.environment == EnvironmentType.PRODUCTION:
            try:
                self.s3_client = boto3.client(
                    's3',
                    region_name=self.env_config.aws_region
                )
                logger.info(f"S3 client initialized for region: {self.env_config.aws_region}")
            except (NoCredentialsError, Exception) as e:
                logger.warning(f"S3 client initialization failed: {str(e)}")
        
        # Migration configuration
        self.alembic_cfg = self._get_alembic_config()
        
        logger.info(f"DatabaseMigrationService initialized for {self.env_config.environment.value} environment")
    
    def _get_alembic_config(self) -> Config:
        """Get Alembic configuration for the current environment"""
        alembic_ini_path = Path("alembic.ini")
        
        if not alembic_ini_path.exists():
            raise ConfigurationError("alembic.ini not found")
        
        alembic_cfg = Config(str(alembic_ini_path))
        
        # Set database URL for current environment
        alembic_cfg.set_main_option("sqlalchemy.url", self.env_config.database_url)
        
        return alembic_cfg
    
    async def create_automated_backup(self, upload_to_s3: bool = None) -> Dict[str, Any]:
        """
        Create automated database backup with optional S3 upload.
        
        Args:
            upload_to_s3: Whether to upload to S3. If None, uses environment default.
            
        Returns:
            Backup information dictionary
        """
        try:
            # Determine S3 upload based on environment
            if upload_to_s3 is None:
                upload_to_s3 = (
                    self.env_config.environment == EnvironmentType.PRODUCTION and 
                    self.s3_client is not None
                )
            
            logger.info(f"Creating automated backup (S3 upload: {upload_to_s3})")
            
            # Create local backup
            backup_info = await self.backup_service.create_database_backup()
            
            # Upload to S3 if configured
            if upload_to_s3 and self.s3_client:
                s3_info = await self._upload_backup_to_s3(backup_info['backup_path'])
                backup_info.update(s3_info)
            
            # Add migration metadata
            backup_info['migration_info'] = await self._get_migration_info()
            backup_info['environment'] = self.env_config.environment.value
            
            logger.info(f"Automated backup completed: {backup_info['backup_name']}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Automated backup failed: {str(e)}")
            raise StorageError(f"Automated backup failed: {str(e)}")
    
    async def _upload_backup_to_s3(self, backup_path: str) -> Dict[str, Any]:
        """Upload backup file to S3"""
        try:
            backup_file = Path(backup_path)
            
            # Generate S3 key with timestamp and environment
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            s3_key = f"database-backups/{self.env_config.environment.value}/{timestamp}_{backup_file.name}"
            
            # Upload to S3
            self.s3_client.upload_file(
                str(backup_file),
                self.env_config.s3_bucket_name,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'environment': self.env_config.environment.value,
                        'backup_type': 'database',
                        'created_timestamp': timestamp
                    }
                }
            )
            
            # Generate S3 URL
            s3_url = f"s3://{self.env_config.s3_bucket_name}/{s3_key}"
            
            logger.info(f"Backup uploaded to S3: {s3_url}")
            
            return {
                's3_uploaded': True,
                's3_bucket': self.env_config.s3_bucket_name,
                's3_key': s3_key,
                's3_url': s3_url
            }
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {str(e)}")
            raise StorageError(f"S3 upload failed: {str(e)}")
    
    async def _get_migration_info(self) -> Dict[str, Any]:
        """Get current migration information"""
        try:
            engine = create_engine(self.env_config.database_url)
            
            with engine.connect() as conn:
                context = MigrationContext.configure(conn)
                current_rev = context.get_current_revision()
            
            # Get script directory info
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            head_rev = script_dir.get_current_head()
            
            return {
                'current_revision': current_rev,
                'head_revision': head_rev,
                'is_up_to_date': current_rev == head_rev
            }
            
        except Exception as e:
            logger.warning(f"Failed to get migration info: {str(e)}")
            return {
                'current_revision': None,
                'head_revision': None,
                'is_up_to_date': None,
                'error': str(e)
            }
    
    async def run_migrations(self, target_revision: str = "head") -> Dict[str, Any]:
        """
        Run database migrations with consistency validation.
        
        Args:
            target_revision: Target revision to migrate to
            
        Returns:
            Migration result information
        """
        try:
            logger.info(f"Running migrations to {target_revision}")
            
            # Get pre-migration info
            pre_migration_info = await self._get_migration_info()
            
            # Create backup before migration
            backup_info = await self.create_automated_backup()
            
            # Run migration
            command.upgrade(self.alembic_cfg, target_revision)
            
            # Get post-migration info
            post_migration_info = await self._get_migration_info()
            
            # Validate migration consistency
            consistency_check = await self._validate_migration_consistency()
            
            migration_result = {
                'status': 'completed',
                'target_revision': target_revision,
                'pre_migration': pre_migration_info,
                'post_migration': post_migration_info,
                'backup_info': backup_info,
                'consistency_check': consistency_check,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Migration completed successfully: {post_migration_info['current_revision']}")
            return migration_result
            
        except Exception as e:
            logger.error(f"Migration failed: {str(e)}")
            
            # Attempt rollback if possible
            rollback_info = None
            try:
                rollback_info = await self._attempt_migration_rollback(pre_migration_info)
            except Exception as rollback_error:
                logger.error(f"Rollback also failed: {str(rollback_error)}")
            
            raise StorageError(f"Migration failed: {str(e)}", {
                'rollback_info': rollback_info,
                'backup_info': backup_info if 'backup_info' in locals() else None
            })
    
    async def _validate_migration_consistency(self) -> Dict[str, Any]:
        """Validate database consistency after migration"""
        try:
            engine = create_engine(self.env_config.database_url)
            
            consistency_results = {
                'foreign_keys_enabled': False,
                'integrity_check_passed': False,
                'table_count': 0,
                'errors': []
            }
            
            with engine.connect() as conn:
                # Check foreign key enforcement
                result = conn.execute(text("PRAGMA foreign_keys"))
                fk_enabled = result.fetchone()[0] == 1
                consistency_results['foreign_keys_enabled'] = fk_enabled
                
                # Run integrity check
                result = conn.execute(text("PRAGMA integrity_check"))
                integrity_result = result.fetchone()[0]
                consistency_results['integrity_check_passed'] = integrity_result == "ok"
                
                if integrity_result != "ok":
                    consistency_results['errors'].append(f"Integrity check failed: {integrity_result}")
                
                # Count tables
                result = conn.execute(text("SELECT COUNT(*) FROM sqlite_master WHERE type='table'"))
                table_count = result.fetchone()[0]
                consistency_results['table_count'] = table_count
            
            consistency_results['status'] = 'passed' if not consistency_results['errors'] else 'failed'
            
            return consistency_results
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'foreign_keys_enabled': None,
                'integrity_check_passed': None,
                'table_count': None
            }
    
    async def _attempt_migration_rollback(self, target_info: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to rollback migration to previous state"""
        try:
            target_revision = target_info.get('current_revision')
            
            if not target_revision:
                raise ConfigurationError("No target revision for rollback")
            
            logger.info(f"Attempting migration rollback to {target_revision}")
            
            # Run downgrade
            command.downgrade(self.alembic_cfg, target_revision)
            
            # Validate rollback
            post_rollback_info = await self._get_migration_info()
            
            rollback_result = {
                'status': 'completed',
                'target_revision': target_revision,
                'post_rollback': post_rollback_info,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Migration rollback completed: {target_revision}")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Migration rollback failed: {str(e)}")
            raise StorageError(f"Migration rollback failed: {str(e)}")
    
    async def validate_cross_environment_consistency(self, other_db_url: str) -> Dict[str, Any]:
        """
        Validate that migrations produce consistent schemas across environments.
        
        Args:
            other_db_url: Database URL of other environment to compare
            
        Returns:
            Consistency validation results
        """
        try:
            logger.info("Validating cross-environment migration consistency")
            
            # Get schema from current environment
            current_schema = await self._get_database_schema(self.env_config.database_url)
            
            # Get schema from other environment
            other_schema = await self._get_database_schema(other_db_url)
            
            # Compare schemas
            comparison_result = self._compare_schemas(current_schema, other_schema)
            
            validation_result = {
                'status': 'consistent' if comparison_result['is_identical'] else 'inconsistent',
                'current_environment': self.env_config.environment.value,
                'current_schema': current_schema,
                'other_schema': other_schema,
                'differences': comparison_result['differences'],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Cross-environment consistency check: {validation_result['status']}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Cross-environment validation failed: {str(e)}")
            raise StorageError(f"Cross-environment validation failed: {str(e)}")
    
    async def _get_database_schema(self, database_url: str) -> Dict[str, Any]:
        """Extract database schema information"""
        try:
            engine = create_engine(database_url)
            
            schema_info = {
                'tables': {},
                'indexes': {},
                'triggers': {}
            }
            
            with engine.connect() as conn:
                # Get table information
                result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                table_names = [row[0] for row in result.fetchall()]
                
                for table_name in table_names:
                    if table_name.startswith('sqlite_'):
                        continue
                    
                    # Get table schema
                    result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                    columns = []
                    for row in result.fetchall():
                        columns.append({
                            'name': row[1],
                            'type': row[2],
                            'not_null': bool(row[3]),
                            'default_value': row[4],
                            'primary_key': bool(row[5])
                        })
                    
                    schema_info['tables'][table_name] = {
                        'columns': columns
                    }
                
                # Get index information
                result = conn.execute(text("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'"))
                for row in result.fetchall():
                    if row[0] and not row[0].startswith('sqlite_'):
                        schema_info['indexes'][row[0]] = {
                            'table': row[1],
                            'sql': row[2]
                        }
            
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            raise StorageError(f"Failed to get database schema: {str(e)}")
    
    def _compare_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two database schemas for differences"""
        differences = []
        
        # Compare tables
        tables1 = set(schema1['tables'].keys())
        tables2 = set(schema2['tables'].keys())
        
        # Tables only in schema1
        for table in tables1 - tables2:
            differences.append(f"Table '{table}' exists in first schema but not second")
        
        # Tables only in schema2
        for table in tables2 - tables1:
            differences.append(f"Table '{table}' exists in second schema but not first")
        
        # Compare common tables
        for table in tables1 & tables2:
            table1_cols = {col['name']: col for col in schema1['tables'][table]['columns']}
            table2_cols = {col['name']: col for col in schema2['tables'][table]['columns']}
            
            # Compare columns
            cols1 = set(table1_cols.keys())
            cols2 = set(table2_cols.keys())
            
            for col in cols1 - cols2:
                differences.append(f"Column '{table}.{col}' exists in first schema but not second")
            
            for col in cols2 - cols1:
                differences.append(f"Column '{table}.{col}' exists in second schema but not first")
            
            # Compare column definitions
            for col in cols1 & cols2:
                col1_def = table1_cols[col]
                col2_def = table2_cols[col]
                
                if col1_def != col2_def:
                    differences.append(f"Column '{table}.{col}' has different definitions")
        
        return {
            'is_identical': len(differences) == 0,
            'differences': differences
        }
    
    async def schedule_automated_backups(self, interval_hours: int = 6) -> None:
        """
        Schedule automated backups with S3 upload for production.
        
        Args:
            interval_hours: Backup interval in hours
        """
        logger.info(f"Scheduling automated backups every {interval_hours} hours")
        
        while True:
            try:
                await asyncio.sleep(interval_hours * 3600)
                
                logger.info("Running scheduled automated backup")
                backup_info = await self.create_automated_backup()
                logger.info(f"Scheduled backup completed: {backup_info['backup_name']}")
                
                # Clean up old backups
                await self._cleanup_old_s3_backups()
                
            except Exception as e:
                logger.error(f"Scheduled backup failed: {str(e)}")
                continue
    
    async def _cleanup_old_s3_backups(self, retention_days: int = 30) -> None:
        """Clean up old S3 backups based on retention policy"""
        if not self.s3_client or not self.env_config.s3_bucket_name:
            return
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # List objects in backup prefix
            prefix = f"database-backups/{self.env_config.environment.value}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.env_config.s3_bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return
            
            # Delete old backups
            deleted_count = 0
            for obj in response['Contents']:
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.env_config.s3_bucket_name,
                        Key=obj['Key']
                    )
                    deleted_count += 1
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old S3 backups")
                
        except Exception as e:
            logger.warning(f"S3 backup cleanup failed: {str(e)}")


# Global service instance
database_migration_service = DatabaseMigrationService()