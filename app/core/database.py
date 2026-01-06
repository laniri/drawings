"""
Database configuration and session management.

This module provides database engine configuration with SQLite WAL mode
and session management for the application.
"""

import os
import sqlite3
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.models.database import Base


def get_database_url() -> str:
    """Get database URL from environment configuration"""
    from app.core.config import settings

    return settings.DATABASE_URL


# Create engine with SQLite-specific configurations
def create_database_engine():
    """Create database engine with environment-aware configuration"""
    database_url = get_database_url()

    return create_engine(
        database_url,
        connect_args={
            "check_same_thread": False,  # Allow SQLite to be used with multiple threads
            "timeout": 20,  # Connection timeout in seconds
        },
        echo=False,  # Set to True for SQL query logging during development
    )


# Initialize engine
engine = create_database_engine()


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for optimal performance and WAL mode."""
    cursor = dbapi_connection.cursor()
    # Enable WAL mode for better concurrent access
    cursor.execute("PRAGMA journal_mode=WAL")
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys=ON")
    # Set synchronous mode for better performance with WAL
    cursor.execute("PRAGMA synchronous=NORMAL")
    # Set cache size (negative value means KB, positive means pages)
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    # Set temp store to memory for better performance
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_exists():
    """Check if database exists locally (non-blocking)."""
    import os

    db_path = "drawings.db"
    
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"✅ Database exists at {db_path} ({file_size} bytes)")
        return True, file_size
    else:
        print(f"📄 Database not found at {db_path} - will create empty database")
        return False, 0


def start_background_database_sync():
    """Start background sync of database from S3 after startup."""
    import os
    import threading
    import time

    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    def background_sync():
        """Background thread to sync database from S3."""
        # Wait for startup to complete
        time.sleep(10)  # Reduced from 30 to 10 seconds

        db_path = "drawings.db"
        temp_db_path = "drawings_sync.db"

        # Only sync in production
        if os.getenv("APP_ENVIRONMENT") != "production":
            print("🔧 Development environment - skipping background database sync")
            return

        # Skip if we already have a large database (likely already synced)
        if (
            os.path.exists(db_path) and os.path.getsize(db_path) > 50 * 1024 * 1024
        ):  # 50MB (reduced from 100MB)
            print(
                f"✅ Large database already present ({os.path.getsize(db_path)} bytes) - skipping sync"
            )
            return

        try:
            print("🔄 Starting background database sync from S3...")

            s3_client = boto3.client("s3", region_name="eu-west-1")

            # Download to temporary file first
            s3_client.download_file(
                "children-drawing-production-drawings-921400262514",
                "database/drawings.db",
                temp_db_path,
            )

            # Verify download
            if os.path.exists(temp_db_path) and os.path.getsize(temp_db_path) > 0:
                # Atomic replace
                if os.path.exists(db_path):
                    os.replace(temp_db_path, db_path)
                else:
                    os.rename(temp_db_path, db_path)

                print(
                    f"✅ Background database sync completed ({os.path.getsize(db_path)} bytes)"
                )
                print("📊 Historical data is now available in dashboard and analysis")
            else:
                print("⚠️  Downloaded file appears to be empty or invalid")
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)

        except (ClientError, NoCredentialsError) as e:
            print(f"⚠️  Background database sync failed: {e}")
            print("🚀 Service continues with local database - all features available")
        except Exception as e:
            print(f"⚠️  Unexpected error in background sync: {e}")
            print("🚀 Service continues with local database - all features available")
        finally:
            # Clean up temp file if it exists
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except:
                    pass

    # Start background thread
    sync_thread = threading.Thread(target=background_sync, daemon=True)
    sync_thread.start()
    print(
        "🔄 Background database sync started - historical data will be available in ~10 seconds"
    )


def init_db():
    """Initialize database tables and start background sync."""
    # Quick check if database exists (non-blocking)
    db_exists, file_size = check_database_exists()

    # Import models to ensure they're registered with Base.metadata
    # This is critical - SQLAlchemy only creates tables for imported models
    from app.models import database  # noqa: F401

    print(
        f"Database models imported. Available tables: {list(Base.metadata.tables.keys())}"
    )

    # Get database URL for path extraction
    database_url = get_database_url()
    print(f"Database URL: {database_url}")

    # Ensure the database directory exists
    db_path = database_url.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        print(f"Creating database directory: {db_dir}")
        os.makedirs(db_dir)

    # Always start background sync for production (non-blocking)
    if os.getenv("APP_ENVIRONMENT") == "production":
        # Start background sync if database is small or doesn't exist
        if not db_exists or file_size < 50 * 1024 * 1024:  # Less than 50MB
            print(
                "🔄 Starting background sync for historical data (non-blocking)"
            )
            start_background_database_sync()
        else:
            print(
                f"✅ Large database present ({file_size} bytes) - skipping background sync"
            )

    # Create all tables (works with empty or existing database)
    print("Creating database tables...")
    create_tables()

    # Verify tables were created
    try:
        # Test database connection and table existence
        from sqlalchemy import inspect

        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"Tables created successfully: {existing_tables}")

        if not existing_tables:
            raise Exception(
                "No tables were created - this indicates a problem with model registration"
            )

        print("✅ Database initialization completed successfully")
        print("🚀 Service is ready - health checks will pass immediately")

    except Exception as e:
        print(f"Error verifying database tables: {e}")
        raise
