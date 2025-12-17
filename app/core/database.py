"""
Database configuration and session management.

This module provides database engine configuration with SQLite WAL mode
and session management for the application.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
import sqlite3
import os
from typing import Generator

from app.models.database import Base

# Database configuration
DATABASE_URL = "sqlite:///./drawings.db"

# Create engine with SQLite-specific configurations
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Allow SQLite to be used with multiple threads
        "timeout": 20,  # Connection timeout in seconds
    },
    echo=False,  # Set to True for SQL query logging during development
)


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
    # Ensure the database directory exists
    db_path = DATABASE_URL.replace("sqlite:///", "")
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Create all tables
    create_tables()