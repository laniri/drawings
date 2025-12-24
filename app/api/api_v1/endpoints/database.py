"""
Database management API endpoints.

This module provides API endpoints for database backup, migration,
and consistency validation operations.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.database_migration_service import database_migration_service
from app.core.exceptions import StorageError, ConfigurationError

router = APIRouter()


class BackupRequest(BaseModel):
    """Request model for database backup operations"""
    upload_to_s3: Optional[bool] = None
    include_files: bool = False


class MigrationRequest(BaseModel):
    """Request model for database migration operations"""
    target_revision: str = "head"


class ConsistencyCheckRequest(BaseModel):
    """Request model for cross-environment consistency checks"""
    other_db_url: str


@router.post("/backup", response_model=Dict[str, Any])
async def create_database_backup(request: BackupRequest):
    """
    Create a database backup with optional S3 upload.
    
    - **upload_to_s3**: Whether to upload to S3 (defaults to environment setting)
    - **include_files**: Whether to include uploaded files and static content
    """
    try:
        if request.include_files:
            # Use the backup service for full backup
            backup_info = await database_migration_service.backup_service.create_full_backup(
                include_files=True
            )
            
            # Add S3 upload if requested
            if request.upload_to_s3:
                s3_info = await database_migration_service._upload_backup_to_s3(
                    backup_info['backup_path']
                )
                backup_info.update(s3_info)
        else:
            # Use the migration service for database-only backup
            backup_info = await database_migration_service.create_automated_backup(
                upload_to_s3=request.upload_to_s3
            )
        
        return {
            "status": "success",
            "message": "Database backup created successfully",
            "backup_info": backup_info
        }
        
    except (StorageError, ConfigurationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@router.post("/migrate", response_model=Dict[str, Any])
async def run_database_migration(request: MigrationRequest):
    """
    Run database migrations to the specified revision.
    
    - **target_revision**: Target migration revision (defaults to "head")
    """
    try:
        migration_result = await database_migration_service.run_migrations(
            target_revision=request.target_revision
        )
        
        return {
            "status": "success",
            "message": "Database migration completed successfully",
            "migration_result": migration_result
        }
        
    except (StorageError, ConfigurationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


@router.get("/migration-info", response_model=Dict[str, Any])
async def get_migration_info():
    """
    Get current database migration information.
    """
    try:
        migration_info = await database_migration_service._get_migration_info()
        
        return {
            "status": "success",
            "migration_info": migration_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get migration info: {str(e)}")


@router.post("/validate-consistency", response_model=Dict[str, Any])
async def validate_cross_environment_consistency(request: ConsistencyCheckRequest):
    """
    Validate database schema consistency across environments.
    
    - **other_db_url**: Database URL of the other environment to compare
    """
    try:
        validation_result = await database_migration_service.validate_cross_environment_consistency(
            other_db_url=request.other_db_url
        )
        
        return {
            "status": "success",
            "message": "Cross-environment consistency validation completed",
            "validation_result": validation_result
        }
        
    except (StorageError, ConfigurationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/backup-list", response_model=Dict[str, Any])
async def list_backups():
    """
    Get list of available database backups.
    """
    try:
        backup_list = await database_migration_service.backup_service.get_backup_list()
        
        return {
            "status": "success",
            "backups": backup_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


@router.post("/schedule-backups", response_model=Dict[str, Any])
async def schedule_automated_backups(
    background_tasks: BackgroundTasks,
    interval_hours: int = 6
):
    """
    Schedule automated database backups.
    
    - **interval_hours**: Backup interval in hours (default: 6)
    """
    try:
        # Add the backup scheduling as a background task
        background_tasks.add_task(
            database_migration_service.schedule_automated_backups,
            interval_hours=interval_hours
        )
        
        return {
            "status": "success",
            "message": f"Automated backups scheduled every {interval_hours} hours"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule backups: {str(e)}")


@router.post("/consistency-check", response_model=Dict[str, Any])
async def run_consistency_check():
    """
    Run database consistency validation.
    """
    try:
        consistency_result = await database_migration_service._validate_migration_consistency()
        
        return {
            "status": "success",
            "message": "Database consistency check completed",
            "consistency_result": consistency_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consistency check failed: {str(e)}")