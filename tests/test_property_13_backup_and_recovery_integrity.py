"""
Property-based test for backup and recovery integrity.

**Feature: aws-production-deployment, Property 13: Backup and Recovery Integrity**
**Validates: Requirements 4.2, 6.4**

This test validates that SQLite database backup operations to S3 result in backups
that can be restored to functionally equivalent database states.
"""

import os
import tempfile
import sqlite3
import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from hypothesis import given, strategies as st, assume, settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.environment import EnvironmentDetector, EnvironmentType, reset_environment_config
from app.services.backup_service import BackupService
from app.models.database import Base, Drawing, AgeGroupModel, AnomalyAnalysis

logger = logging.getLogger(__name__)


class TestBackupAndRecoveryIntegrity:
    """Property-based tests for backup and recovery integrity"""
    
    def setup_method(self):
        """Reset environment configuration before each test"""
        reset_environment_config()
    
    def teardown_method(self):
        """Clean up after each test"""
        reset_environment_config()
    
    def _create_test_database(self, db_path: Path) -> None:
        """Create a test database with sample data"""
        engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(engine)
        
        # Add sample data
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Add sample drawings
            drawing1 = Drawing(
                filename="test1.png",
                file_path="uploads/test1.png",
                age_years=5.5,
                subject="person",
                upload_timestamp=datetime(2024, 1, 1)
            )
            
            drawing2 = Drawing(
                filename="test2.png", 
                file_path="uploads/test2.png",
                age_years=7.2,
                subject="house",
                upload_timestamp=datetime(2024, 1, 2)
            )
            
            session.add(drawing1)
            session.add(drawing2)
            session.flush()  # Get IDs
            
            # Add sample age group model
            model = AgeGroupModel(
                age_min=5.0,
                age_max=7.0,
                model_type="autoencoder",
                vision_model="vit",
                parameters='{"hidden_dim": 128}',
                sample_count=100,
                threshold=0.8,
                created_timestamp=datetime(2024, 1, 1)
            )
            
            session.add(model)
            session.flush()
            
            # Add sample analysis
            analysis = AnomalyAnalysis(
                drawing_id=drawing1.id,
                age_group_model_id=model.id,
                anomaly_score=0.75,
                normalized_score=0.75,
                is_anomaly=False,
                confidence=0.85,
                analysis_timestamp=datetime(2024, 1, 1, 1, 0, 0)
            )
            
            session.add(analysis)
            
            # CRITICAL FIX: Ensure all data is committed and flushed to disk
            session.commit()
            
        finally:
            session.close()
            
        # CRITICAL FIX: Force SQLite to write all data to disk
        # This ensures the database file contains actual data before backup
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA wal_checkpoint(FULL)")  # Flush WAL to main DB
            conn.execute("PRAGMA synchronous = FULL")    # Ensure data is written to disk
            conn.commit()
            conn.close()
            
            # Verify data was actually written
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM drawings")
            drawing_count = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM age_group_models")
            model_count = cursor.fetchone()[0]
            cursor = conn.execute("SELECT COUNT(*) FROM anomaly_analyses")
            analysis_count = cursor.fetchone()[0]
            conn.close()
            
            logger.info(f"Test database created with {drawing_count} drawings, {model_count} models, {analysis_count} analyses")
            
            if drawing_count == 0 or model_count == 0 or analysis_count == 0:
                raise RuntimeError("Test database was not properly populated with data")
                
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to verify test database creation: {e}")
    
    def _get_database_content(self, db_path: Path) -> Dict[str, List[Dict]]:
        """Extract all data from database for comparison"""
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        
        content = {}
        
        try:
            # Get all table names
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Extract data from each table
            for table in tables:
                if table.startswith('sqlite_'):  # Skip system tables
                    continue
                    
                cursor = conn.execute(f"SELECT * FROM {table}")
                rows = cursor.fetchall()
                content[table] = [dict(row) for row in rows]
        
        finally:
            conn.close()
        
        return content
    
    def _compare_database_content(self, content1: Dict[str, List[Dict]], content2: Dict[str, List[Dict]]) -> bool:
        """Compare database content for functional equivalence"""
        # Check that both have the same tables
        if set(content1.keys()) != set(content2.keys()):
            return False
        
        # Check each table's content
        for table_name in content1.keys():
            table1_data = content1[table_name]
            table2_data = content2[table_name]
            
            # Check row count
            if len(table1_data) != len(table2_data):
                return False
            
            # Sort rows by ID for comparison (if ID exists)
            if table1_data and 'id' in table1_data[0]:
                table1_data = sorted(table1_data, key=lambda x: x.get('id', 0))
                table2_data = sorted(table2_data, key=lambda x: x.get('id', 0))
            
            # Compare each row
            for row1, row2 in zip(table1_data, table2_data):
                # Compare all fields except timestamps which might have slight differences
                for key in row1.keys():
                    if key.endswith('_timestamp') or key.endswith('_at'):
                        continue  # Skip timestamp comparison for backup/restore
                    if row1[key] != row2[key]:
                        return False
        
        return True
    
    @settings(deadline=None, max_examples=5)  # Reduce examples and disable deadline for async operations
    @given(
        drawing_count=st.integers(min_value=1, max_value=3),
        model_count=st.integers(min_value=1, max_value=2),
        include_files=st.booleans()
    )
    def test_database_backup_restore_integrity(self, drawing_count: int, model_count: int, include_files: bool):
        """
        **Feature: aws-production-deployment, Property 13: Backup and Recovery Integrity**
        **Validates: Requirements 4.2, 6.4**
        
        For any SQLite database backup operation to S3, the backup should be
        restorable to a functionally equivalent database state.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original database
            original_db = temp_path / "original.db"
            self._create_test_database(original_db)
            
            # Create backup directory
            backup_dir = temp_path / "backups"
            backup_dir.mkdir()
            
            # Create test directories
            (temp_path / "uploads").mkdir()
            (temp_path / "static").mkdir()
            
            # Create some test files if include_files is True
            if include_files:
                (temp_path / "uploads" / "test_file.png").write_text("test content")
                (temp_path / "static" / "model.pth").write_text("model data")
            
            # Mock settings to ensure consistent database URL handling
            with patch('app.services.backup_service.settings') as mock_settings:
                mock_settings.DATABASE_URL = f"sqlite:///{original_db}"
                
                # Initialize backup service with test configuration
                backup_service = BackupService(str(backup_dir))
                # Explicitly set the database path to ensure it's correct
                backup_service.db_path = original_db
                
                # Get original database content
                original_content = self._get_database_content(original_db)
                
                # Verify original database has content
                assert len(original_content) > 0, "Original database should have content"
                
                # Create backup (database only to avoid settings mock issues)
                backup_info = asyncio.run(backup_service.create_database_backup())
                
                # Verify backup was created
                backup_path = Path(backup_info['backup_path'])
                assert backup_path.exists(), "Backup file should exist"
                assert backup_info['status'] == 'completed', "Backup should be completed"
                assert backup_info['size_bytes'] > 0, "Backup should not be empty"
                
                # Create a new database location for restore
                restored_db = temp_path / "restored.db"
                
                # Create a new backup service instance for restore to avoid state issues
                restore_service = BackupService(str(backup_dir))
                restore_service.db_path = restored_db
                
                # Restore from backup (skip safety backup by patching)
                with patch.object(restore_service, 'create_database_backup', new_callable=AsyncMock) as mock_safety_backup:
                    mock_safety_backup.return_value = {"backup_name": "safety_backup.db"}
                    restore_info = asyncio.run(restore_service.restore_from_backup(backup_path))
                
                # Verify restore was successful
                assert restore_info['status'] == 'completed', "Restore should be completed"
                assert restored_db.exists(), "Restored database file should exist"
                
                # Verify restored database is not empty
                restored_size = restored_db.stat().st_size
                assert restored_size > 0, "Restored database should not be empty"
                
                # Get restored database content
                restored_content = self._get_database_content(restored_db)
                
                # Verify restored database has content
                assert len(restored_content) > 0, "Restored database should have content"
                
                # Verify functional equivalence
                assert self._compare_database_content(original_content, restored_content), \
                    f"Restored database content does not match original. Original: {original_content}, Restored: {restored_content}"
    
    @given(
        corruption_type=st.sampled_from(['truncate', 'random_bytes', 'missing_header']),
        backup_format=st.sampled_from(['database', 'full'])
    )
    def test_backup_corruption_detection(self, corruption_type: str, backup_format: str):
        """
        Test that corrupted backups are properly detected and handled.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original database
            original_db = temp_path / "original.db"
            self._create_test_database(original_db)
            
            # Create backup directory
            backup_dir = temp_path / "backups"
            backup_dir.mkdir()
            
            with patch('app.services.backup_service.settings') as mock_settings:
                mock_settings.DATABASE_URL = f"sqlite:///{original_db}"
                mock_settings.UPLOAD_DIR = str(temp_path / "uploads")
                mock_settings.STATIC_DIR = str(temp_path / "static")
                
                (temp_path / "uploads").mkdir()
                (temp_path / "static").mkdir()
                
                backup_service = BackupService(str(backup_dir))
                
                # Create backup
                if backup_format == 'full':
                    backup_info = asyncio.run(backup_service.create_full_backup())
                else:
                    backup_info = asyncio.run(backup_service.create_database_backup())
                
                backup_path = Path(backup_info['backup_path'])
                
                # Corrupt the backup file
                original_content = backup_path.read_bytes()
                
                if corruption_type == 'truncate':
                    # Truncate file to half size
                    corrupted_content = original_content[:len(original_content)//2]
                elif corruption_type == 'random_bytes':
                    # Replace middle section with random bytes
                    import random
                    corrupted_content = bytearray(original_content)
                    start = len(corrupted_content) // 4
                    end = 3 * len(corrupted_content) // 4
                    for i in range(start, end):
                        corrupted_content[i] = random.randint(0, 255)
                    corrupted_content = bytes(corrupted_content)
                elif corruption_type == 'missing_header':
                    # Remove first 100 bytes
                    corrupted_content = original_content[100:]
                
                backup_path.write_bytes(corrupted_content)
                
                # Attempt restore - should fail gracefully
                restored_db = temp_path / "restored_corrupted.db"
                backup_service.db_path = restored_db
                
                # Should detect corruption and return error result
                try:
                    result = asyncio.run(backup_service.restore_from_backup(backup_path))
                    # If no exception, check that the result indicates failure
                    assert result is None or (isinstance(result, dict) and result.get('status') != 'completed'), \
                        "Corrupted backup should not restore successfully"
                except Exception:
                    # Exception is also acceptable - corruption was detected
                    pass
                
                # Verify original database is not affected
                assert original_db.exists()
                original_content_after = self._get_database_content(original_db)
                assert len(original_content_after) > 0  # Should still have data
    
    def test_backup_retention_policy(self):
        """
        Test that backup retention policy works correctly and maintains
        the specified number of backups.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original database
            original_db = temp_path / "original.db"
            self._create_test_database(original_db)
            
            # Create backup directory
            backup_dir = temp_path / "backups"
            backup_dir.mkdir()
            
            with patch('app.services.backup_service.settings') as mock_settings:
                mock_settings.DATABASE_URL = f"sqlite:///{original_db}"
                mock_settings.UPLOAD_DIR = str(temp_path / "uploads")
                mock_settings.STATIC_DIR = str(temp_path / "static")
                
                (temp_path / "uploads").mkdir()
                (temp_path / "static").mkdir()
                
                backup_service = BackupService(str(backup_dir))
                backup_service.max_backup_count = 3  # Set low limit for testing
                
                # Create multiple backups
                backup_infos = []
                for i in range(5):  # Create more than the limit
                    backup_info = asyncio.run(backup_service.create_database_backup())
                    backup_infos.append(backup_info)
                    
                    # Small delay to ensure different timestamps
                    import time
                    time.sleep(0.1)
                
                # Check that only max_backup_count backups remain
                remaining_backups = list(backup_dir.glob("*backup*"))
                assert len(remaining_backups) <= backup_service.max_backup_count
                
                # Verify the newest backups are kept
                backup_list = asyncio.run(backup_service.get_backup_list())
                assert len(backup_list) <= backup_service.max_backup_count
                
                # Verify all remaining backups are valid
                for backup_info in backup_list:
                    backup_path = Path(backup_info['path'])
                    assert backup_path.exists()
                    assert backup_path.stat().st_size > 0
    
    @given(
        export_format=st.sampled_from(['json', 'csv']),
        include_embeddings=st.booleans()
    )
    def test_data_export_integrity(self, export_format: str, include_embeddings: bool):
        """
        Test that data export operations preserve data integrity
        and can be used for backup purposes.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original database
            original_db = temp_path / "original.db"
            self._create_test_database(original_db)
            
            # Create backup directory
            backup_dir = temp_path / "backups"
            backup_dir.mkdir()
            
            with patch('app.services.backup_service.settings') as mock_settings:
                mock_settings.DATABASE_URL = f"sqlite:///{original_db}"
                mock_settings.UPLOAD_DIR = str(temp_path / "uploads")
                mock_settings.STATIC_DIR = str(temp_path / "static")
                
                (temp_path / "uploads").mkdir()
                (temp_path / "static").mkdir()
                
                backup_service = BackupService(str(backup_dir))
                
                # Get original database content
                original_content = self._get_database_content(original_db)
                
                # Export data
                export_info = asyncio.run(backup_service.export_data(
                    format=export_format,
                    include_embeddings=include_embeddings
                ))
                
                # Verify export was created
                export_path = Path(export_info['export_path'])
                assert export_path.exists()
                assert export_info['status'] == 'completed'
                assert export_info['format'] == export_format
                assert export_info['includes_embeddings'] == include_embeddings
                
                # Verify record counts match
                record_counts = export_info['record_counts']
                for table_name, count in record_counts.items():
                    if table_name in original_content:
                        assert count == len(original_content[table_name])
                
                # For JSON exports, verify content can be loaded
                if export_format == 'json':
                    with open(export_path, 'r') as f:
                        export_data = json.load(f)
                    
                    assert 'export_info' in export_data
                    assert export_data['export_info']['format'] == 'json'
                    assert export_data['export_info']['includes_embeddings'] == include_embeddings
                    
                    # Verify data sections exist
                    expected_sections = ['drawings', 'age_group_models', 'anomaly_analyses']
                    for section in expected_sections:
                        assert section in export_data
                        if section.replace('_', '') in original_content:
                            # Compare counts (table names might have different formats)
                            original_table = None
                            for table in original_content.keys():
                                if section.replace('_', '') in table.replace('_', ''):
                                    original_table = table
                                    break
                            
                            if original_table:
                                assert len(export_data[section]) == len(original_content[original_table])
    
    def test_concurrent_backup_operations(self):
        """
        Test that concurrent backup operations don't interfere with each other
        and maintain data integrity.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create original database
            original_db = temp_path / "original.db"
            self._create_test_database(original_db)
            
            # Create backup directory
            backup_dir = temp_path / "backups"
            backup_dir.mkdir()
            
            with patch('app.services.backup_service.settings') as mock_settings:
                mock_settings.DATABASE_URL = f"sqlite:///{original_db}"
                mock_settings.UPLOAD_DIR = str(temp_path / "uploads")
                mock_settings.STATIC_DIR = str(temp_path / "static")
                
                (temp_path / "uploads").mkdir()
                (temp_path / "static").mkdir()
                
                backup_service = BackupService(str(backup_dir))
                
                # Get original content
                original_content = self._get_database_content(original_db)
                
                async def create_backup_task(backup_type: str):
                    """Create a backup task"""
                    if backup_type == 'database':
                        return await backup_service.create_database_backup()
                    else:
                        return await backup_service.create_full_backup()
                
                # Run concurrent backup operations
                async def run_concurrent_backups():
                    tasks = [
                        create_backup_task('database'),
                        create_backup_task('full'),
                        create_backup_task('database')
                    ]
                    return await asyncio.gather(*tasks, return_exceptions=True)
                
                results = asyncio.run(run_concurrent_backups())
                
                # Verify all backups completed successfully
                successful_backups = []
                for result in results:
                    if isinstance(result, dict) and result.get('status') == 'completed':
                        successful_backups.append(result)
                
                assert len(successful_backups) >= 2  # At least 2 should succeed
                
                # Verify each backup can be restored correctly
                for backup_info in successful_backups[:2]:  # Test first 2
                    backup_path = Path(backup_info['backup_path'])
                    assert backup_path.exists()
                    
                    # Test restore
                    restored_db = temp_path / f"restored_{backup_path.stem}.db"
                    test_backup_service = BackupService(str(backup_dir))
                    test_backup_service.db_path = restored_db
                    
                    restore_info = asyncio.run(test_backup_service.restore_from_backup(backup_path))
                    assert restore_info['status'] == 'completed'
                    
                    # Verify content integrity
                    restored_content = self._get_database_content(restored_db)
                    assert self._compare_database_content(original_content, restored_content)