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


def download_database_from_s3():
    """Download database from S3 if not present locally (optional for startup)."""
    import os

    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    db_path = "drawings.db"

    # Skip if database already exists
    if os.path.exists(db_path):
        print(f"✅ Database already exists at {db_path} ({os.path.getsize(db_path)} bytes)")
        return True

    # Only download in production
    if os.getenv("APP_ENVIRONMENT") != "production":
        print("🔧 Development environment - skipping S3 database download")
        return True

    try:
        print("📥 Database not found locally - attempting S3 download...")

        s3_client = boto3.client("s3", region_name="eu-west-1")
        s3_client.download_file(
            "children-drawing-production-drawings-921400262514",
            "database/drawings.db",
            db_path,
        )

        print(f"✅ Database downloaded from S3 successfully ({os.path.getsize(db_path)} bytes)")
        return True

    except (ClientError, NoCredentialsError) as e:
        print(f"⚠️  Could not download database from S3: {e}")
        print("🚀 Continuing with empty database - service will be fully functional")
        return True  # Changed: Don't fail startup, continue with empty DB
    except Exception as e:
        print(f"⚠️  Unexpected error downloading database: {e}")
        print("🚀 Continuing with empty database - service will be fully functional")
        return True  # Changed: Don't fail startup, continue with empty DB


def start_background_database_sync():
    """Start background sync of database from S3 after startup."""
    import os
    import threading
    import time

    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    def background_sync():
        """Background thread to sync database from S3."""
        # Wait a bit for startup to complete
        time.sleep(30)
        
        db_path = "drawings.db"
        temp_db_path = "drawings_sync.db"
        
        # Only sync in production
        if os.getenv("APP_ENVIRONMENT") != "production":
            print("🔧 Development environment - skipping background database sync")
            return

        # Skip if we already have a large database (likely already synced)
        if os.path.exists(db_path) and os.path.getsize(db_path) > 100 * 1024 * 1024:  # 100MB
            print(f"✅ Large database already present ({os.path.getsize(db_path)} bytes) - skipping sync")
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
                    
                print(f"✅ Background database sync completed ({os.path.getsize(db_path)} bytes)")
                print("📊 Historical data is now available in dashboard and analysis")
            else:
                print("⚠️  Downloaded file appears to be empty or invalid")
                if os.path.exists(temp_db_path):
                    os.remove(temp_db_path)
                    
        except (ClientError, NoCredentialsError) as e:
            print(f"⚠️  Background database sync failed: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error in background sync: {e}")
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
    print("🔄 Background database sync started - historical data will be available shortly")


def init_db():
    """Initialize database tables and start background sync."""
    # Try to download database from S3 (non-blocking, returns True even if fails)
    download_success = download_database_from_s3()
    
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

    # Check if database file exists and its size
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"Database file exists at: {db_path}")
        print(f"Database file size: {file_size} bytes")
        
        # If database is small (likely empty), start background sync
        if file_size < 100 * 1024 * 1024:  # Less than 100MB
            print("🔄 Small database detected - starting background sync for historical data")
            start_background_database_sync()
    else:
        print(f"Database file does not exist at: {db_path} - will create empty database")
        # Start background sync to get historical data
        start_background_database_sync()

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
        if os.getenv("APP_ENVIRONMENT") == "production":
            print("🔄 Background database sync is running - historical data will be available shortly")

    except Exception as e:
        print(f"Error verifying database tables: {e}")
        raise
