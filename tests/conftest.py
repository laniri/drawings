"""
Pytest configuration and fixtures for the children's drawing anomaly detection system.

This module provides shared test fixtures and configuration for all tests,
including database setup, test client configuration, and common test utilities.
"""

import os
import tempfile
import pytest
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

# Import application components
from app.main import app
from app.models.database import Base
from app.core.database import get_db, set_sqlite_pragma
from app.core.config import settings


@pytest.fixture(scope="session")
def test_engine():
    """
    Create a test database engine for the entire test session.
    
    This uses an in-memory SQLite database for fast, isolated testing.
    """
    # Use in-memory database for tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )
    
    # Apply SQLite pragmas for testing
    event.listen(engine, "connect", set_sqlite_pragma)
    
    return engine


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    """Create a session factory for the test database."""
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session(test_engine, test_session_factory) -> Generator[Session, None, None]:
    """
    Create a fresh database session for each test function.
    
    This fixture:
    1. Creates all tables before each test
    2. Provides a clean database session
    3. Rolls back any changes after the test
    4. Drops all tables to ensure isolation
    """
    # Create all tables
    Base.metadata.create_all(bind=test_engine)
    
    # Create session
    session = test_session_factory()
    
    try:
        yield session
    finally:
        # Clean up
        session.rollback()
        session.close()
        
        # Drop all tables to ensure test isolation
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def test_client(db_session: Session) -> TestClient:
    """
    Create a test client with database dependency override.
    
    This fixture provides a FastAPI test client that uses the test database
    instead of the production database.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # Session cleanup is handled by db_session fixture
    
    # Override the database dependency
    app.dependency_overrides[get_db] = override_get_db
    
    # Create test client
    client = TestClient(app)
    
    yield client
    
    # Clean up dependency override
    app.dependency_overrides.clear()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment variables and configuration.
    
    This fixture runs automatically for all tests and ensures
    proper test environment configuration.
    """
    # Set test environment variables
    os.environ["SKIP_MODEL_LOADING"] = "true"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["TESTING"] = "true"
    
    # Ensure uploads directory exists for tests
    test_uploads_dir = "test_uploads"
    if not os.path.exists(test_uploads_dir):
        os.makedirs(test_uploads_dir)
    
    yield
    
    # Cleanup test uploads directory
    import shutil
    if os.path.exists(test_uploads_dir):
        shutil.rmtree(test_uploads_dir)


@pytest.fixture
def temp_file():
    """
    Create a temporary file for testing file operations.
    
    Returns the file path and automatically cleans up after the test.
    """
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_directory():
    """
    Create a temporary directory for testing.
    
    Returns the directory path and automatically cleans up after the test.
    """
    temp_dir = tempfile.mkdtemp()
    
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_drawing_data():
    """
    Provide sample drawing data for tests.
    
    Returns a dictionary with valid drawing metadata.
    """
    return {
        "filename": "test_drawing.png",
        "file_path": "test_uploads/test_drawing.png",
        "age_years": 5.5,
        "subject": "house",
        "expert_label": None,
        "drawing_tool": "crayon",
        "prompt": "Draw your house"
    }


@pytest.fixture
def sample_embedding_data():
    """
    Provide sample embedding data for tests.
    
    Returns a dictionary with valid embedding data.
    """
    import numpy as np
    
    return {
        "model_type": "vit",
        "embedding_vector": np.random.randn(768).astype(np.float32),
        "vector_dimension": 768
    }


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "ci_skip: marks tests to skip in CI environment"
    )


# Skip certain tests in CI environment
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    if os.getenv("CI"):
        # Skip tests marked with ci_skip in CI environment
        skip_ci = pytest.mark.skip(reason="Skipped in CI environment")
        for item in items:
            if "ci_skip" in item.keywords:
                item.add_marker(skip_ci)
    
    # Skip slow tests if not explicitly requested
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )