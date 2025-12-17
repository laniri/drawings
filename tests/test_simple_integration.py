"""
Simple integration test to debug database setup.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.main import app
from app.core.database import get_db, Base
from app.models.database import Drawing


def test_database_setup():
    """Test that database tables can be created."""
    # Create in-memory SQLite database
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Check that tables exist
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print(f"Created tables: {tables}")
    assert "drawings" in tables
    assert "anomaly_analyses" in tables
    assert "drawing_embeddings" in tables


def test_simple_api_call():
    """Test a simple API call without database."""
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200


if __name__ == "__main__":
    test_database_setup()
    test_simple_api_call()
    print("All tests passed!")