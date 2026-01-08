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

    # Use the same path as DATABASE_URL to ensure consistency
    db_path = "./drawings.db"  # Match sqlite:///./drawings.db

    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path)
        print(f"✅ Database exists at {db_path} ({file_size} bytes)")
        return True, file_size
    else:
        print(f"📄 Database not found at {db_path} - will create empty database")
        return False, 0


def _sync_database_blocking():
    """
    Synchronously sync database from S3 (blocking).

    This function downloads the production database from S3 and replaces
    the local database file. It blocks until complete to ensure data
    is available before the application starts accepting requests.
    """
    import os
    import sqlite3
    import time
    import traceback

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError

    # Use the same path as DATABASE_URL to ensure consistency
    db_path = "./drawings.db"
    temp_db_path = "./drawings_sync.db"

    # Only sync in production
    if os.getenv("APP_ENVIRONMENT") != "production":
        print("🔧 Development environment - skipping database sync")
        return

    try:
        print("🔄 Starting BLOCKING database sync from S3...")
        print(
            f"📍 Source: s3://children-drawing-production-drawings-921400262514/database/drawings.db"
        )
        print(f"📍 Target: {os.path.abspath(db_path)}")

        # Configure boto3 with increased timeouts for large file
        boto_config = Config(
            connect_timeout=300,
            read_timeout=300,
            retries={"max_attempts": 3, "mode": "standard"},
        )

        s3_client = boto3.client("s3", region_name="eu-west-1", config=boto_config)

        # Check if file exists in S3 first
        print("🔍 Checking S3 file existence...")
        try:
            head_response = s3_client.head_object(
                Bucket="children-drawing-production-drawings-921400262514",
                Key="database/drawings.db",
            )
            s3_file_size = head_response["ContentLength"]
            print(
                f"✅ S3 file found: {s3_file_size:,} bytes ({s3_file_size / (1024*1024):.1f} MB)"
            )
        except ClientError as e:
            print(f"❌ S3 file not found or not accessible: {e}")
            return

        # Download to temporary file first
        print(f"📥 Downloading database to {temp_db_path}...")
        download_start = time.time()

        s3_client.download_file(
            "children-drawing-production-drawings-921400262514",
            "database/drawings.db",
            temp_db_path,
        )

        download_duration = time.time() - download_start
        print(f"⏱️  Download completed in {download_duration:.1f} seconds")

        # Verify download
        if not os.path.exists(temp_db_path):
            print("❌ Downloaded file does not exist")
            return

        temp_size = os.path.getsize(temp_db_path)
        print(
            f"✅ Downloaded file size: {temp_size:,} bytes ({temp_size / (1024*1024):.1f} MB)"
        )

        if temp_size == 0:
            print("❌ Downloaded file is empty")
            os.remove(temp_db_path)
            return

        # Verify it's a valid SQLite database
        try:
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            if integrity_result != "ok":
                print(f"⚠️  Database integrity check failed: {integrity_result}")
                conn.close()
                os.remove(temp_db_path)
                return
            print("✅ Database integrity check passed")

            # Force WAL checkpoint
            cursor.execute("PRAGMA wal_checkpoint(FULL)")
            conn.commit()

            # Get table list and counts
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"✅ Valid SQLite database with {len(tables)} tables")

            # Count records
            cursor.execute("SELECT COUNT(*) FROM drawings")
            drawing_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM anomaly_analyses")
            analysis_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM age_group_models")
            model_count = cursor.fetchone()[0]

            conn.close()
            print(
                f"📊 S3 database contains {drawing_count:,} drawings, {analysis_count:,} analyses, {model_count} models"
            )

        except Exception as e:
            print(f"⚠️  Database validation failed: {e}")
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
            return

        # Remove existing database files (including WAL and SHM)
        for ext in ["", "-wal", "-shm"]:
            path = db_path + ext
            if os.path.exists(path):
                print(f"🗑️  Removing existing file: {path}")
                os.remove(path)

        # Move downloaded database to final location
        print(f"🔄 Moving database to {db_path}...")
        os.rename(temp_db_path, db_path)

        final_size = os.path.getsize(db_path)
        print(
            f"✅ BLOCKING database sync completed: {final_size:,} bytes ({final_size / (1024*1024):.1f} MB)"
        )

        # Dispose existing engine connections so they reconnect to new file
        global engine
        engine.dispose()
        print(
            "✅ Database engine connections disposed - will reconnect to new database"
        )

    except (ClientError, NoCredentialsError) as e:
        print(f"❌ Database sync failed (AWS error): {e}")
        print("🚀 Service will continue with empty database")
    except Exception as e:
        print(f"❌ Unexpected error in database sync: {e}")
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        print("🚀 Service will continue with empty database")
    finally:
        # Clean up temp file if it exists
        if os.path.exists(temp_db_path):
            try:
                os.remove(temp_db_path)
            except Exception:
                pass


def start_background_database_sync():
    """Start background sync of database from S3 after startup."""
    import threading

    def background_sync():
        """Background thread to sync database from S3."""
        # Import required modules at function scope
        import os
        import sqlite3
        import time
        import traceback

        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError, NoCredentialsError

        # Start immediately - no delay needed
        # The FastAPI app will be ready to serve requests while sync happens
        # Use the same path as DATABASE_URL to ensure consistency
        db_path = "./drawings.db"  # Match sqlite:///./drawings.db
        temp_db_path = "./drawings_sync.db"

        # Only sync in production
        if os.getenv("APP_ENVIRONMENT") != "production":
            print("🔧 Development environment - skipping background database sync")
            return

        # Check current database status
        current_size = 0
        if os.path.exists(db_path):
            current_size = os.path.getsize(db_path)
            print(
                f"📊 Current database size: {current_size:,} bytes ({current_size / (1024*1024):.1f} MB)"
            )

        # Skip if we already have a large database (likely already synced)
        if current_size > 50 * 1024 * 1024:  # 50MB
            print(
                f"✅ Large database already present ({current_size:,} bytes) - skipping sync"
            )
            return

        try:
            print("🔄 Starting background database sync from S3...")
            print(
                f"📍 Source: s3://children-drawing-production-drawings-921400262514/database/drawings.db"
            )
            print(f"📍 Target: {os.path.abspath(db_path)}")

            # Configure boto3 with increased timeouts for large file
            boto_config = Config(
                connect_timeout=300,  # 5 minutes connection timeout
                read_timeout=300,  # 5 minutes read timeout
                retries={"max_attempts": 3, "mode": "standard"},
            )

            s3_client = boto3.client("s3", region_name="eu-west-1", config=boto_config)

            # Check if file exists in S3 first
            print("🔍 Checking S3 file existence...")
            try:
                head_response = s3_client.head_object(
                    Bucket="children-drawing-production-drawings-921400262514",
                    Key="database/drawings.db",
                )
                s3_file_size = head_response["ContentLength"]
                print(
                    f"✅ S3 file found: {s3_file_size:,} bytes ({s3_file_size / (1024*1024):.1f} MB)"
                )
                print(
                    f"⏱️  Estimated download time: {s3_file_size / (1024*1024) / 10:.0f}-{s3_file_size / (1024*1024) / 5:.0f} seconds"
                )
            except ClientError as e:
                print(f"❌ S3 file not found or not accessible: {e}")
                print(f"📋 Error details: {e.response.get('Error', {})}")
                return

            # Download to temporary file first
            print(f"📥 Downloading database to {temp_db_path}...")
            download_start = time.time()

            s3_client.download_file(
                "children-drawing-production-drawings-921400262514",
                "database/drawings.db",
                temp_db_path,
            )

            download_duration = time.time() - download_start
            print(f"⏱️  Download completed in {download_duration:.1f} seconds")

            # Verify download
            if os.path.exists(temp_db_path):
                temp_size = os.path.getsize(temp_db_path)
                print(
                    f"✅ Downloaded file size: {temp_size:,} bytes ({temp_size / (1024*1024):.1f} MB)"
                )

                if temp_size > 0:
                    # Verify it's a valid SQLite database
                    try:
                        conn = sqlite3.connect(temp_db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                        tables = cursor.fetchall()
                        conn.close()
                        print(
                            f"✅ Valid SQLite database with {len(tables)} tables: {[t[0] for t in tables]}"
                        )
                    except Exception as e:
                        print(f"⚠️  Database validation failed: {e}")
                        if os.path.exists(temp_db_path):
                            os.remove(temp_db_path)
                        return

                    # Atomic replace
                    print(f"🔄 Replacing database file...")
                    if os.path.exists(db_path):
                        os.replace(temp_db_path, db_path)
                    else:
                        os.rename(temp_db_path, db_path)

                    final_size = os.path.getsize(db_path)
                    print(
                        f"✅ Background database sync completed: {final_size:,} bytes ({final_size / (1024*1024):.1f} MB)"
                    )
                    print(
                        "📊 Historical data is now available in dashboard and analysis"
                    )

                    # Force SQLAlchemy to close all existing connections
                    # This ensures fresh connections will read the new database file
                    try:
                        print("🔄 Refreshing database connections...")
                        from app.core.database import engine

                        engine.dispose()
                        print("✅ Database connections refreshed")
                    except Exception as e:
                        print(f"⚠️  Could not refresh connections: {e}")

                    # Log record counts with fresh connection
                    try:
                        # Debug: Print working directory and file paths
                        print(f"🔍 Working directory: {os.getcwd()}")
                        print(f"🔍 Database path: {db_path}")
                        print(f"🔍 Absolute path: {os.path.abspath(db_path)}")
                        print(f"🔍 File exists: {os.path.exists(db_path)}")
                        if os.path.exists(db_path):
                            print(f"🔍 File size: {os.path.getsize(db_path):,} bytes")

                        # Use a fresh connection to the replaced database
                        conn = sqlite3.connect(db_path)

                        # Force FULL checkpoint to ensure all WAL data is in main database
                        print("🔄 Forcing WAL checkpoint...")
                        conn.execute("PRAGMA wal_checkpoint(FULL)")
                        conn.commit()

                        # Verify database integrity
                        print("🔍 Checking database integrity...")
                        cursor = conn.cursor()
                        cursor.execute("PRAGMA integrity_check")
                        integrity_result = cursor.fetchone()[0]
                        if integrity_result != "ok":
                            print(
                                f"⚠️  Database integrity check failed: {integrity_result}"
                            )
                            conn.close()
                            return
                        print("✅ Database integrity check passed")

                        # Check table structure
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                        )
                        tables = [row[0] for row in cursor.fetchall()]
                        print(f"📋 Tables in database: {tables}")

                        # Count records
                        cursor.execute("SELECT COUNT(*) FROM drawings")
                        drawing_count = cursor.fetchone()[0]
                        cursor.execute("SELECT COUNT(*) FROM anomaly_analyses")
                        analysis_count = cursor.fetchone()[0]
                        cursor.execute("SELECT COUNT(*) FROM age_group_models")
                        model_count = cursor.fetchone()[0]

                        conn.close()
                        print(
                            f"📊 Database contains {drawing_count:,} drawings, {analysis_count:,} analyses, and {model_count} models"
                        )
                    except Exception as e:
                        print(f"⚠️  Could not query database: {e}")
                        print(f"📋 Error details: {traceback.format_exc()}")
                else:
                    print("❌ Downloaded file is empty")
                    if os.path.exists(temp_db_path):
                        os.remove(temp_db_path)
            else:
                print("❌ Downloaded file does not exist")

        except (ClientError, NoCredentialsError) as e:
            print(f"❌ Background database sync failed (AWS error): {e}")
            print(f"📋 Error type: {type(e).__name__}")
            print(f"📋 Error details: {str(e)}")
            if hasattr(e, "response"):
                print(f"📋 Response: {e.response}")
            print("🚀 Service continues with local database - all features available")
        except Exception as e:
            print(f"❌ Unexpected error in background sync: {e}")
            print(f"📋 Error type: {type(e).__name__}")
            print(f"📋 Traceback:\n{traceback.format_exc()}")
            print("🚀 Service continues with local database - all features available")
        finally:
            # Clean up temp file if it exists
            if os.path.exists(temp_db_path):
                try:
                    print(f"🧹 Cleaning up temporary file: {temp_db_path}")
                    os.remove(temp_db_path)
                except Exception as e:
                    print(f"⚠️  Could not remove temp file: {e}")

    # Start background thread
    sync_thread = threading.Thread(target=background_sync, daemon=True)
    sync_thread.start()
    print(
        "🔄 Background database sync started - historical data will be available in ~10-60 seconds"
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

    # In production, we need to sync from S3 BEFORE creating tables
    # Otherwise SQLAlchemy will create an empty database that overwrites the sync
    sync_needed = False
    if os.getenv("APP_ENVIRONMENT") == "production":
        # Start background sync if database is small or doesn't exist
        if not db_exists or file_size < 50 * 1024 * 1024:  # Less than 50MB
            print("🔄 Starting BLOCKING database sync for historical data...")
            print(
                "⏳ This ensures data is available before service starts accepting requests"
            )
            sync_needed = True
            # Do a BLOCKING sync instead of background sync
            _sync_database_blocking()
        else:
            print(f"✅ Large database present ({file_size} bytes) - skipping sync")

    # Create all tables (works with empty or existing database)
    # This will be a no-op if tables already exist from synced database
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

        # Verify we have data if we synced
        if sync_needed or (db_exists and file_size > 50 * 1024 * 1024):
            from sqlalchemy.orm import Session

            from app.models.database import AgeGroupModel, Drawing

            with Session(engine) as session:
                drawing_count = session.query(Drawing).count()
                model_count = session.query(AgeGroupModel).count()
                print(
                    f"📊 Database verification: {drawing_count:,} drawings, {model_count} models"
                )
                if drawing_count == 0 and sync_needed:
                    print("⚠️  WARNING: Database sync completed but no drawings found!")

        print("✅ Database initialization completed successfully")
        print("🚀 Service is ready - health checks will pass immediately")

    except Exception as e:
        print(f"Error verifying database tables: {e}")
        raise
