# Testing Documentation

**Children's Drawing Anomaly Detection System**  
**Version**: 2.0.0 (Subject-Aware)  
**Last Updated**: December 24, 2025

## Overview

The Children's Drawing Anomaly Detection System uses a comprehensive testing strategy that combines unit testing, integration testing, and property-based testing to ensure system reliability and correctness. The test infrastructure is designed to be robust, fast, and maintainable.

## Test Architecture

### Core Testing Principles

1. **Isolation**: Each test runs in complete isolation with fresh database state
2. **Reproducibility**: Tests produce consistent results across different environments
3. **Performance**: Fast execution using in-memory databases and optimized fixtures
4. **Robustness**: Graceful handling of missing dependencies and import failures
5. **Comprehensiveness**: Coverage of unit, integration, and property-based scenarios

### Test Infrastructure Components

```
tests/
├── conftest.py                    # Core test configuration and fixtures
├── test_*.py                      # Unit and integration tests
├── test_property_*.py             # Property-based tests using Hypothesis
└── __pycache__/                   # Compiled test files
```

## Test Configuration (`conftest.py`)

The `conftest.py` file provides the foundation for all testing with robust import handling and comprehensive fixtures.

### Key Features

#### Delayed Import System
```python
def _import_app_modules():
    """Import app modules after ensuring proper path setup."""
    try:
        # Import all required modules
        from sqlalchemy import create_engine, event
        from app.main import app
        # ... other imports
        return module_dict
    except ImportError as e:
        pytest.skip(f"Could not import required modules: {e}")
```

**Benefits**:
- Handles missing dependencies gracefully
- Ensures proper Python path setup before imports
- Provides clear error messages for import failures
- Allows tests to skip rather than fail when dependencies are unavailable

#### Fixture Hierarchy

```
app_modules (session)
    ↓
test_engine (session)
    ↓
test_session_factory (session)
    ↓
db_session (function) ← test_client (function)
```

### Core Fixtures

#### `app_modules` (Session Scope)
- **Purpose**: Provides all application modules with safe import handling
- **Scope**: Session-wide (shared across all tests)
- **Features**: 
  - Delayed imports after path setup
  - Graceful failure handling
  - Module dependency injection

#### `test_engine` (Session Scope)
- **Purpose**: SQLite database engine for testing
- **Configuration**: In-memory database with SQLite pragmas
- **Features**:
  - Fast in-memory execution
  - Proper SQLite configuration for testing
  - Session-wide reuse for performance

#### `db_session` (Function Scope)
- **Purpose**: Isolated database session for each test
- **Lifecycle**:
  1. Creates all database tables
  2. Provides clean session to test
  3. Rolls back all changes
  4. Drops all tables for complete isolation
- **Benefits**:
  - Complete test isolation
  - No data persistence between tests
  - Automatic cleanup

#### `test_client` (Function Scope)
- **Purpose**: FastAPI test client with database override
- **Features**:
  - Uses test database instead of production
  - Automatic dependency injection
  - Proper cleanup of overrides

### Utility Fixtures

#### File Management
- `temp_file`: Creates temporary file with automatic cleanup
- `temp_directory`: Creates temporary directory with automatic cleanup

#### Sample Data
- `sample_drawing_data`: Valid drawing metadata for testing
- `sample_embedding_data`: Sample embedding vectors for testing

#### Environment Setup
- `setup_test_environment`: Configures test environment variables
  - Sets `SKIP_MODEL_LOADING=true` for faster execution
  - Configures test database URL
  - Creates and manages test upload directories

## Test Categories

### Unit Tests

**Purpose**: Test individual components in isolation

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies
- Mock external services
- Focus on single function/class behavior

**Examples**:
```python
def test_age_group_classification(db_session):
    """Test age group classification logic."""
    # Test specific age group assignment logic
    
def test_embedding_serialization():
    """Test embedding vector serialization."""
    # Test data serialization without database
```

### Integration Tests

**Purpose**: Test component interactions and system workflows

**Characteristics**:
- Moderate execution time (1-10 seconds per test)
- Use real database (in-memory)
- Test API endpoints end-to-end
- Verify component integration

**Examples**:
```python
def test_drawing_upload_and_analysis(test_client, db_session):
    """Test complete drawing upload and analysis workflow."""
    # Upload drawing → Generate embedding → Analyze → Return results
    
def test_model_training_pipeline(test_client, db_session):
    """Test model training from start to finish."""
    # Prepare data → Train model → Validate results
```

### Property-Based Tests

**Purpose**: Test system properties across wide input ranges using Hypothesis

**Characteristics**:
- Generate random test inputs
- Verify universal properties
- Catch edge cases
- High confidence in correctness

**Examples**:
```python
@given(age=st.floats(min_value=2.0, max_value=18.0))
def test_age_validation_property(age):
    """Property: All valid ages should be accepted."""
    # Test that age validation works for any valid age
    
@given(drawing_data=drawing_metadata_strategy)
def test_metadata_persistence_property(db_session, drawing_data):
    """Property: All drawing metadata should persist correctly."""
    # Test metadata storage and retrieval
```

## Test Execution

### Running Tests

#### Basic Execution
```bash
# Run all tests
pytest

# Verbose output with test names
pytest -v

# Show test coverage
pytest --cov=app --cov-report=html
```

#### Selective Execution
```bash
# Skip slow tests (recommended for development)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Run only property-based tests
pytest tests/test_property_*.py

# Run specific test file
pytest tests/test_embedding_service.py

# Run specific test function
pytest tests/test_embedding_service.py::test_embedding_generation
```

#### Performance Analysis
```bash
# Show 10 slowest tests (default configuration)
pytest --durations=10

# Show all test durations
pytest --durations=0

# Fail fast after 5 failures (default configuration)
pytest --maxfail=5
```

#### CI/CD Execution
```bash
# CI mode (skips local-only tests)
CI=1 pytest

# Run with coverage for CI
pytest --cov=app --cov-report=xml --cov-report=term
```

### Test Markers

Tests are organized using pytest markers:

```python
@pytest.mark.unit
def test_individual_function():
    """Fast unit test."""
    pass

@pytest.mark.integration  
def test_system_workflow():
    """Integration test requiring full setup."""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Slow test (model training, large data processing)."""
    pass

@pytest.mark.ci_skip
def test_local_only_feature():
    """Test that only runs locally, not in CI."""
    pass
```

## Test Development Guidelines

### Writing Effective Tests

#### 1. Use Proper Fixtures
```python
def test_drawing_analysis(db_session, sample_drawing_data, app_modules):
    """Use fixtures for dependencies."""
    Drawing = app_modules['Drawing']  # Get model from fixture
    
    # Create test data using fixture
    drawing = Drawing(**sample_drawing_data)
    db_session.add(drawing)
    db_session.commit()
    
    # Test logic here
```

#### 2. Handle Imports Safely
```python
# DON'T: Import at module level
# from app.models.database import Drawing  # Can fail in CI

# DO: Use app_modules fixture
def test_something(app_modules):
    Drawing = app_modules['Drawing']  # Safe import
```

#### 3. Ensure Test Isolation
```python
def test_independent_operation(db_session):
    """Each test should be completely independent."""
    # Don't rely on data from other tests
    # Don't leave side effects
    # Use fresh database session
```

#### 4. Use Appropriate Markers
```python
@pytest.mark.slow
def test_model_training():
    """Mark expensive tests appropriately."""
    pass

@pytest.mark.unit
def test_fast_calculation():
    """Mark fast tests for selective execution."""
    pass
```

#### 5. Property-Based Test Design
```python
from hypothesis import given, strategies as st

@given(age=st.floats(min_value=2.0, max_value=18.0))
def test_age_property(age):
    """Test properties that should hold for all valid inputs."""
    result = validate_age(age)
    assert result.is_valid  # Should be true for all valid ages
```

### Test Data Management

#### Sample Data Strategy
```python
# Use fixtures for consistent test data
def test_with_sample_data(sample_drawing_data, sample_embedding_data):
    # Fixtures provide realistic, consistent test data
    drawing = create_drawing(sample_drawing_data)
    embedding = create_embedding(sample_embedding_data)
```

#### Temporary Resources
```python
def test_file_operations(temp_file, temp_directory):
    # Fixtures handle cleanup automatically
    with open(temp_file, 'w') as f:
        f.write("test data")
    # File automatically cleaned up after test
```

## Performance Considerations

### Test Execution Speed

#### Fast Tests (< 1 second)
- Unit tests with mocked dependencies
- Simple calculations and validations
- In-memory operations only

#### Medium Tests (1-10 seconds)
- Integration tests with database
- API endpoint testing
- Small model operations

#### Slow Tests (> 10 seconds)
- Model training and evaluation
- Large data processing
- External service integration

### Optimization Strategies

#### Database Performance
- Use in-memory SQLite for speed
- Minimize database operations in loops
- Use bulk operations where possible

#### Model Loading
- Set `SKIP_MODEL_LOADING=true` in test environment
- Mock model operations for unit tests
- Load models only when actually needed

#### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto  # Use all available CPU cores
pytest -n 4     # Use 4 parallel workers
```

## Continuous Integration

### GitHub Actions Integration

The test suite integrates with GitHub Actions for automated testing:

```yaml
# .github/workflows/deploy-production.yml
- name: Run unit tests
  env:
    SKIP_MODEL_LOADING: "true"
    DATABASE_URL: "sqlite:///:memory:"
  run: |
    pytest tests/ -v --cov=app --cov-report=xml -x --tb=short

- name: Run property-based tests
  run: |
    pytest tests/test_property_*.py -v --tb=short
```

### CI Optimizations

1. **Disk Space Management**: Removes unnecessary packages to free space
2. **Dependency Caching**: Uses pip cache for faster installs
3. **Parallel Execution**: Runs different test categories in parallel
4. **Fail Fast**: Stops on first failure for quick feedback
5. **Coverage Reporting**: Uploads coverage to external services

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Check that virtual environment is activated
source venv/bin/activate
pip install -e .
```

#### Database Errors
```bash
# Error: Database locked or connection issues
# Solution: Tests use in-memory database, check conftest.py configuration
```

#### Slow Test Execution
```bash
# Skip slow tests during development
pytest -m "not slow"

# Check which tests are slowest
pytest --durations=10
```

#### Missing Dependencies
```bash
# Error: Missing test dependencies
# Solution: Install development requirements
pip install -r requirements-dev.txt
```

#### AWS Dependencies in Tests
```bash
# AWS-dependent tests are automatically skipped when boto3/botocore are unavailable
# Tests use @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")

# To run AWS-dependent tests, install AWS dependencies:
pip install boto3 botocore

# Or install all enhanced features:
pip install -r requirements-enhanced.txt
```

### Debugging Tests

#### Verbose Output
```bash
# See detailed test output
pytest -v -s

# Show local variables on failure
pytest --tb=long

# Drop into debugger on failure
pytest --pdb
```

#### Selective Debugging
```bash
# Run single test with debugging
pytest tests/test_specific.py::test_function -v -s --pdb
```

## Best Practices Summary

1. **Use Fixtures**: Leverage provided fixtures for database, client, and sample data
2. **Safe Imports**: Use `app_modules` fixture instead of direct imports
3. **Mark Tests**: Use appropriate markers for test categorization
4. **Isolate Tests**: Ensure each test is independent and clean
5. **Performance**: Keep unit tests fast, mark slow tests appropriately
6. **Property Testing**: Use Hypothesis for testing universal properties
7. **Documentation**: Write clear test descriptions and docstrings
8. **Coverage**: Aim for high test coverage but focus on critical paths
9. **CI Integration**: Ensure tests run reliably in CI environment
10. **Maintenance**: Keep tests up-to-date with code changes

## Future Enhancements

### Planned Improvements

1. **Parallel Test Execution**: Add pytest-xdist for faster CI runs
2. **Test Data Factories**: Implement factory pattern for complex test data
3. **Visual Regression Testing**: Add screenshot comparison for frontend
4. **Performance Benchmarking**: Add performance regression detection
5. **Mutation Testing**: Add mutation testing for test quality assessment
6. **Contract Testing**: Add API contract testing with external services
7. **Load Testing**: Add load testing for performance validation
8. **Security Testing**: Add security-focused test scenarios

### Monitoring and Metrics

1. **Test Coverage Tracking**: Monitor coverage trends over time
2. **Test Performance Monitoring**: Track test execution time trends
3. **Flaky Test Detection**: Identify and fix unreliable tests
4. **Test Quality Metrics**: Measure test effectiveness and maintainability