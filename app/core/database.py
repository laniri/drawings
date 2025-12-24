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


def init_db():
    """Initialize the database by creating all tables."""
    # Get database URL for path extraction
    database_url = get_database_url()

    # Ensure the database directory exists
    db_path = database_url.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)

    # Create all tables
    create_tables()
