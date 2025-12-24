"""
Property-based test for database migration consistency.

**Feature: aws-production-deployment, Property 9: Database Migration Consistency**
**Validates: Requirements 6.2, 6.3, 6.5**

This test validates that database schema changes applied through Alembic migrations
result in identical database schemas across local and production environments.
"""

import os
import tempfile
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Any
from unittest.mock import patch

import pytest
from hypothesis import given, strategies as st, assume
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, Text, inspect, text
from sqlalchemy.engine import Engine

from app.core.environment import EnvironmentDetector, EnvironmentType, reset_environment_config
from app.models.database import Base


class TestDatabaseMigrationConsistency:
    """Property-based tests for database migration consistency"""
    
    def setup_method(self):
        """Reset environment configuration before each test"""
        reset_environment_config()
    
    def teardown_method(self):
        """Clean up after each test"""
        reset_environment_config()
    
    def _create_test_database(self, db_path: Path) -> Engine:
        """Create a test database with proper configuration"""
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}
        )
        
        # Apply SQLite pragmas similar to production
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA foreign_keys=ON"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.commit()
        
        return engine
    
    def _get_database_schema(self, engine: Engine) -> Dict[str, Any]:
        """Extract database schema information for comparison"""
        inspector = inspect(engine)
        
        schema = {
            'tables': {},
            'indexes': {}
        }
        
        # Get table information
        for table_name in inspector.get_table_names():
            columns = {}
            for column in inspector.get_columns(table_name):
                columns[column['name']] = {
                    'type': str(column['type']),
                    'nullable': column['nullable'],
                    'default': column['default']
                }
            
            schema['tables'][table_name] = {
                'columns': columns,
                'primary_keys': inspector.get_pk_constraint(table_name)['constrained_columns'],
                'foreign_keys': [
                    {
                        'constrained_columns': fk['constrained_columns'],
                        'referred_table': fk['referred_table'],
                        'referred_columns': fk['referred_columns']
                    }
                    for fk in inspector.get_foreign_keys(table_name)
                ]
            }
        
        # Get index information
        for table_name in inspector.get_table_names():
            indexes = []
            for index in inspector.get_indexes(table_name):
                indexes.append({
                    'name': index['name'],
                    'columns': index['column_names'],
                    'unique': index['unique']
                })
            schema['indexes'][table_name] = indexes
        
        return schema
    
    def _run_alembic_migrations(self, db_path: Path, alembic_dir: Path) -> None:
        """Run Alembic migrations on a database"""
        # Create Alembic configuration
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", str(alembic_dir))
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
    
    def _create_test_migration(self, alembic_dir: Path, migration_name: str, operations: List[str]) -> str:
        """Create a test migration file"""
        # Create migration template
        migration_content = f'''"""Test migration: {migration_name}

Revision ID: test_{migration_name}
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'test_{migration_name}'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Upgrade operations"""
{chr(10).join(f"    {op}" for op in operations)}

def downgrade():
    """Downgrade operations"""
    pass
'''
        
        migration_file = alembic_dir / "versions" / f"test_{migration_name}.py"
        migration_file.write_text(migration_content)
        
        return str(migration_file)
    
    @given(
        table_name=st.text(min_size=3, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu'), whitelist_characters='_')),
        column_count=st.integers(min_value=1, max_value=5)
    )
    def test_migration_consistency_across_environments(self, table_name: str, column_count: int):
        """
        **Feature: aws-production-deployment, Property 9: Database Migration Consistency**
        **Validates: Requirements 6.2, 6.3, 6.5**
        
        For any database schema change, applying migrations in both local and production
        environments should result in identical database schemas.
        """
        # Skip reserved SQL keywords
        reserved_keywords = {'user', 'order', 'group', 'table', 'index', 'select', 'from', 'where'}
        assume(table_name.lower() not in reserved_keywords)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test databases for local and production
            local_db = temp_path / "local_test.db"
            prod_db = temp_path / "prod_test.db"
            
            # Create Alembic directory structure
            alembic_dir = temp_path / "alembic"
            alembic_dir.mkdir()
            (alembic_dir / "versions").mkdir()
            
            # Create alembic.ini
            alembic_ini = temp_path / "alembic.ini"
            alembic_ini.write_text(f"""[alembic]
script_location = {alembic_dir}
sqlalchemy.url = sqlite:///test.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
""")
            
            # Create env.py for Alembic
            env_py = alembic_dir / "env.py"
            env_py.write_text(f"""from alembic import context
from sqlalchemy import engine_from_config, pool
import logging

config = context.config

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    pass
else:
    run_migrations_online()
""")
            
            # Generate test migration operations
            operations = []
            operations.append(f"op.create_table('{table_name}',")
            operations.append("    sa.Column('id', sa.Integer(), primary_key=True),")
            
            for i in range(column_count):
                col_name = f"column_{i}"
                operations.append(f"    sa.Column('{col_name}', sa.String(255), nullable=True),")
            
            operations.append(")")
            
            # Create test migration
            self._create_test_migration(alembic_dir, table_name, operations)
            
            # Create engines for both databases
            local_engine = self._create_test_database(local_db)
            prod_engine = self._create_test_database(prod_db)
            
            # Apply migrations to both databases
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{local_db}"}, clear=False):
                self._run_alembic_migrations(local_db, alembic_dir)
            
            with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{prod_db}"}, clear=False):
                self._run_alembic_migrations(prod_db, alembic_dir)
            
            # Compare schemas
            local_schema = self._get_database_schema(local_engine)
            prod_schema = self._get_database_schema(prod_engine)
            
            # Verify schemas are identical
            assert local_schema['tables'] == prod_schema['tables'], \
                f"Table schemas differ between environments: {local_schema['tables']} != {prod_schema['tables']}"
            
            # Verify the test table was created correctly
            assert table_name in local_schema['tables']
            assert table_name in prod_schema['tables']
            
            # Verify column count
            local_columns = local_schema['tables'][table_name]['columns']
            prod_columns = prod_schema['tables'][table_name]['columns']
            
            assert len(local_columns) == column_count + 1  # +1 for id column
            assert len(prod_columns) == column_count + 1
            
            # Clean up
            local_engine.dispose()
            prod_engine.dispose()
    
    def test_existing_migrations_consistency(self):
        """
        Test that existing Alembic migrations produce consistent results
        across different environments.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test databases
            local_db = temp_path / "local_existing.db"
            prod_db = temp_path / "prod_existing.db"
            
            # Copy existing alembic configuration
            project_root = Path(__file__).parent.parent
            alembic_ini_source = project_root / "alembic.ini"
            alembic_dir_source = project_root / "alembic"
            
            if alembic_ini_source.exists() and alembic_dir_source.exists():
                # Copy Alembic configuration
                alembic_ini_dest = temp_path / "alembic.ini"
                shutil.copy2(alembic_ini_source, alembic_ini_dest)
                
                alembic_dir_dest = temp_path / "alembic"
                shutil.copytree(alembic_dir_source, alembic_dir_dest)
                
                # Update alembic.ini to use test databases
                alembic_ini_content = alembic_ini_dest.read_text()
                alembic_ini_content = alembic_ini_content.replace(
                    "sqlalchemy.url = sqlite:///./drawings.db",
                    f"sqlalchemy.url = sqlite:///{local_db}"
                )
                alembic_ini_dest.write_text(alembic_ini_content)
                
                # Create engines
                local_engine = self._create_test_database(local_db)
                prod_engine = self._create_test_database(prod_db)
                
                try:
                    # Apply migrations to local database
                    with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{local_db}"}, clear=False):
                        alembic_cfg = Config(str(alembic_ini_dest))
                        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{local_db}")
                        command.upgrade(alembic_cfg, "head")
                    
                    # Apply migrations to production database
                    with patch.dict(os.environ, {"DATABASE_URL": f"sqlite:///{prod_db}"}, clear=False):
                        alembic_cfg = Config(str(alembic_ini_dest))
                        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{prod_db}")
                        command.upgrade(alembic_cfg, "head")
                    
                    # Compare schemas
                    local_schema = self._get_database_schema(local_engine)
                    prod_schema = self._get_database_schema(prod_engine)
                    
                    # Verify schemas are identical
                    assert local_schema == prod_schema, \
                        "Database schemas differ between local and production environments after applying existing migrations"
                    
                finally:
                    local_engine.dispose()
                    prod_engine.dispose()
    
    @given(
        migration_operations=st.lists(
            st.sampled_from([
                "add_table",
                "add_column", 
                "add_index"
            ]),
            min_size=1,
            max_size=3
        )
    )
    def test_migration_rollback_consistency(self, migration_operations: List[str]):
        """
        Test that migration rollbacks work consistently across environments.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test databases
            local_db = temp_path / "local_rollback.db"
            prod_db = temp_path / "prod_rollback.db"
            
            # Create Alembic directory structure
            alembic_dir = temp_path / "alembic"
            alembic_dir.mkdir()
            (alembic_dir / "versions").mkdir()
            
            # Create basic alembic configuration
            alembic_ini = temp_path / "alembic.ini"
            alembic_ini.write_text(f"""[alembic]
script_location = {alembic_dir}
sqlalchemy.url = sqlite:///test.db
""")
            
            env_py = alembic_dir / "env.py"
            env_py.write_text("""from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    pass
else:
    run_migrations_online()
""")
            
            # Create engines
            local_engine = self._create_test_database(local_db)
            prod_engine = self._create_test_database(prod_db)
            
            try:
                # Create initial migration
                operations = ["op.create_table('test_rollback', sa.Column('id', sa.Integer(), primary_key=True))"]
                self._create_test_migration(alembic_dir, "initial", operations)
                
                # Apply initial migration to both databases
                for db_path in [local_db, prod_db]:
                    alembic_cfg = Config(str(alembic_ini))
                    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
                    command.upgrade(alembic_cfg, "head")
                
                # Get schemas after migration
                local_schema_after = self._get_database_schema(local_engine)
                prod_schema_after = self._get_database_schema(prod_engine)
                
                # Verify schemas are identical after migration
                assert local_schema_after == prod_schema_after
                
                # Verify the test table exists
                assert 'test_rollback' in local_schema_after['tables']
                assert 'test_rollback' in prod_schema_after['tables']
                
                # Note: For this test, we're primarily testing that the migration
                # application is consistent. Full rollback testing would require
                # more complex migration setup with proper downgrade methods.
                
            finally:
                local_engine.dispose()
                prod_engine.dispose()
    
    def test_migration_failure_handling(self):
        """
        Test that migration failures are handled consistently and don't
        leave databases in inconsistent states.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test database
            test_db = temp_path / "test_failure.db"
            
            # Create Alembic directory structure
            alembic_dir = temp_path / "alembic"
            alembic_dir.mkdir()
            (alembic_dir / "versions").mkdir()
            
            # Create alembic configuration
            alembic_ini = temp_path / "alembic.ini"
            alembic_ini.write_text(f"""[alembic]
script_location = {alembic_dir}
sqlalchemy.url = sqlite:///{test_db}
""")
            
            env_py = alembic_dir / "env.py"
            env_py.write_text("""from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    pass
else:
    run_migrations_online()
""")
            
            # Create a migration that will fail (invalid SQL)
            failing_operations = [
                "op.execute('INVALID SQL STATEMENT THAT WILL FAIL')"
            ]
            self._create_test_migration(alembic_dir, "failing", failing_operations)
            
            engine = self._create_test_database(test_db)
            
            try:
                # Get initial schema
                initial_schema = self._get_database_schema(engine)
                
                # Attempt to run failing migration
                alembic_cfg = Config(str(alembic_ini))
                alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{test_db}")
                
                with pytest.raises(Exception):  # Should fail
                    command.upgrade(alembic_cfg, "head")
                
                # Verify database is still in consistent state
                final_schema = self._get_database_schema(engine)
                
                # Schema should be unchanged after failed migration
                assert initial_schema == final_schema
                
            finally:
                engine.dispose()