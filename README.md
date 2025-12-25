# Children's Drawing Anomaly Detection System

A machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected norms. The system uses Vision Transformer (ViT) embeddings and autoencoder models trained on age-specific drawing patterns to detect anomalies through reconstruction loss analysis.

## ✅ System Status

**Current Status**: Fully functional and trained (v2.0.0 - Subject-Aware)
- **37,778+ drawings** uploaded and processed
- **8 trained subject-aware autoencoder models** for age groups (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years)
- **Hybrid embeddings** (832-dimensional: 768 visual + 64 subject) generated for all drawings
- **Subject-aware anomaly detection** with 64 predefined subject categories
- **Interactive interpretability** with guaranteed saliency maps and subject-specific comparisons
- **Advanced export system** with multi-format support (PNG, PDF, JSON, CSV, HTML)
- **Web interface** with 6 interpretability tabs and real-time features

## Features

- **Drawing Upload & Analysis**: Support for PNG, JPEG, and BMP formats with metadata and subject categorization
- **Subject-Aware Modeling**: 64 predefined subject categories (objects, living beings, nature, abstract concepts)
- **Hybrid Embeddings**: 832-dimensional vectors combining visual features (768-dim ViT) and subject encoding (64-dim)
- **Age-Based Modeling**: Separate subject-aware autoencoder models trained for different age groups
- **Anomaly Detection**: Reconstruction loss-based scoring with subject-contextualized thresholds
- **Interactive Interpretability**: Subject-aware saliency maps with hoverable regions, zoom/pan, and subject-specific comparisons
- **Export System**: Multi-format exports (PNG, PDF, JSON, CSV, HTML) with subject-aware comprehensive reports
- **Web Interface**: Modern React frontend with Material-UI components and subject category selection
- **REST API**: FastAPI backend with automatic OpenAPI documentation and subject-aware endpoints
- **Real-time Dashboard**: System statistics, age distribution, subject distribution, and analysis results

## Technology Stack

### Backend
- **Python 3.11+** with FastAPI web framework
- **PyTorch** for deep learning models and autoencoder training
- **Vision Transformer (ViT)** for visual feature extraction (768-dimensional)
- **Subject Encoding System** for categorical features (64-dimensional one-hot encoding)
- **Hybrid Embeddings** combining visual and subject features (832-dimensional total)
- **SQLAlchemy** with SQLite database for data persistence
- **Alembic** for database migrations
- **Pydantic** for data validation and settings management
- **ReportLab** for PDF generation and comprehensive export reports (optional)
- **Pillow** for core image processing and saliency map generation
- **OpenCV** for advanced image processing (optional, with PIL fallback)
- **NumPy 1.26.4** (downgraded for PyTorch compatibility)
- **Boto3** for AWS services integration (optional for local development)

### Frontend
- **React 18** with TypeScript
- **Material-UI (MUI)** for component library
- **Vite** for build tool and dev server with API proxy
- **React Query** for server state management
- **Zustand** for client state management
- **React Hook Form + Zod** for form handling and validation
- **Recharts** for data visualization

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd children-drawing-anomaly-detection
   ```

2. **Backend Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   # venv\Scripts\activate
   
   # Install Python dependencies
   pip install -r requirements-dev.txt
   
   # Optional: Install enhanced features (OpenCV, ReportLab)
   pip install -r requirements-enhanced.txt
   
   # Install the package in development mode for proper imports
   pip install -e .
   
   # Copy environment file
   cp .env.example .env
   
   # Run database migrations (when implemented)
   # alembic upgrade head
   
   # Start backend server
   uvicorn app.main:app --reload
   ```

**Note**: AWS dependencies (boto3, botocore) are included in requirements.txt but are optional for local development. The system will work without AWS services and gracefully handle missing AWS dependencies. All AWS-dependent services (cost optimization, monitoring, security validation, database migration) include fallback behavior for local development.

3. **Frontend Setup**
   ```bash
   cd frontend
   
   # Install dependencies
   npm install
   
   # Start development server
   npm run dev
   ```

4. **Initialize the database**
   ```bash
   # Run database migrations
   alembic upgrade head
   ```

5. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Model Training

The system comes with pre-trained models, but you can retrain them:

### Quick Training with Sample Data

1. **Generate and upload sample drawings**
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   
   # Generate 95 sample drawings
   python create_sample_drawings.py
   
   # Upload them to the system
   python upload_sample_drawings.py
   ```

2. **Train the models**
   ```bash
   # Complete training workflow
   python train_models.py
   ```

This will:
- Generate hybrid ViT embeddings for all drawings (832-dimensional vectors: 768 visual + 64 subject)
- Train subject-aware autoencoder models for 3 age groups:
  - Early childhood (3.0-6.0 years)
  - Middle childhood (6.0-9.0 years)  
  - Late childhood (9.0-12.0 years)
- Set up subject-contextualized anomaly detection thresholds

### Manual Training Steps

If you prefer manual control:

1. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Generate hybrid embeddings for existing drawings**
   ```bash
   # For each drawing, generate hybrid embeddings (visual + subject)
   curl -X POST "http://localhost:8000/api/v1/analysis/embeddings/{drawing_id}"
   ```

3. **Train subject-aware age group models**
   ```bash
   # Train models via API with subject-aware architecture
   curl -X POST "http://localhost:8000/api/v1/models/train" \
        -H "Content-Type: application/json" \
        -d '{"age_min": 3.0, "age_max": 6.0, "min_samples": 10}'
   ```

4. **Check training status**
   ```bash
   curl "http://localhost:8000/api/v1/models/status"
   ```

### Docker Development

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up --build

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## Project Structure

```
├── app/                          # Python backend
│   ├── api/                      # API endpoints
│   ├── core/                     # Core configuration
│   ├── models/                   # Database models
│   ├── schemas/                  # Pydantic schemas
│   ├── services/                 # Business logic
│   └── utils/                    # Utilities
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── components/           # React components
│   │   ├── pages/                # Page components
│   │   ├── store/                # State management
│   │   └── theme/                # Material-UI theme
├── alembic/                      # Database migrations
├── uploads/                      # Uploaded drawings
├── static/                       # Static files
└── docker-compose.yml            # Docker configuration
```

## Testing

The project uses pytest for comprehensive testing with property-based testing via Hypothesis. The test suite includes automated fixtures, database isolation, and comprehensive test utilities with robust import handling.

### Test Infrastructure

The testing infrastructure provides:
- **Isolated Test Database**: In-memory SQLite database for each test function
- **Robust Import Handling**: Delayed imports with graceful failure handling for missing dependencies
- **Automatic Fixtures**: Database sessions, test clients, and sample data with dependency injection
- **Environment Isolation**: Separate test environment with proper configuration
- **Cleanup Management**: Automatic cleanup of test data and temporary files

### Test Configuration

The pytest configuration in `pytest.ini` includes:
- **Test discovery**: Automatically finds `test_*.py` files in the `tests/` directory
- **Fail-fast**: Stops after 5 test failures (`--maxfail=5`)
- **Performance monitoring**: Shows 10 slowest tests (`--durations=10`)
- **Async support**: Automatic asyncio mode for async tests
- **Strict markers**: Ensures all test markers are properly defined

### Test Fixtures

The `tests/conftest.py` provides comprehensive fixtures with robust error handling:

#### Core Infrastructure
- `app_modules`: Session-scoped fixture that handles delayed imports and provides all required modules
- **Import Safety**: Gracefully handles missing dependencies with `pytest.skip()` for unavailable modules
- **Path Management**: Ensures proper Python path setup before importing application modules

#### Database Fixtures
- `test_engine`: Session-scoped SQLite engine for testing with proper pragma configuration
- `test_session_factory`: Session factory for test database with dependency injection
- `db_session`: Function-scoped database session with automatic cleanup and table management
- `test_client`: FastAPI test client with database dependency override and proper cleanup

#### Utility Fixtures
- `temp_file`: Temporary file with automatic cleanup
- `temp_directory`: Temporary directory with automatic cleanup
- `sample_drawing_data`: Valid drawing metadata for tests
- `sample_embedding_data`: Sample embedding vectors for tests

#### Environment Setup
- `setup_test_environment`: Automatic test environment configuration
- Sets `SKIP_MODEL_LOADING=true` for faster test execution
- Creates and cleans up test upload directories
- Configures test-specific environment variables

### Test Markers

Tests are organized using markers:
- `slow`: Marks tests as slow (skip with `-m "not slow"`)
- `integration`: Integration tests that require full system setup
- `unit`: Fast unit tests for individual components
- `ci_skip`: Tests to skip in CI environment (for local-only tests)

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Skip slow tests (recommended for development)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_input_validation.py

# Run tests with coverage
pytest --cov=app --cov-report=html

# Show test durations (10 slowest by default)
pytest --durations=0  # Show all test durations

# Run slow tests (when needed)
pytest --runslow

# Run tests in CI mode (skips ci_skip marked tests)
CI=1 pytest
```

### Test Database Isolation

Each test function gets a fresh database with robust setup:
1. **Module Loading**: Delayed import of all required modules with error handling
2. **Setup**: Creates all tables in in-memory SQLite database with proper configuration
3. **Execution**: Test runs with isolated database session and dependency injection
4. **Cleanup**: Rolls back changes and drops all tables for complete isolation
5. **Error Handling**: Graceful handling of import failures and missing dependencies

This ensures:
- **Fast execution**: In-memory database for speed
- **Complete isolation**: No test interference between functions
- **Consistent state**: Each test starts with clean database and fresh imports
- **Automatic cleanup**: No manual database management needed
- **Robust imports**: Handles missing dependencies gracefully with proper error messages

### Import Safety and Error Handling

The test infrastructure includes robust import handling:
- **Delayed Imports**: Modules are imported only when needed, after proper path setup
- **Graceful Failures**: Missing dependencies result in test skips rather than failures
- **Dependency Injection**: All fixtures receive required modules through the `app_modules` fixture
- **Path Management**: Ensures project root is in Python path before any imports
- **Error Messages**: Clear error messages when imports fail with specific module information

### Property-Based Testing

The system includes extensive property-based tests using Hypothesis:
- Input validation consistency tests
- Data sufficiency warning generation tests
- Subject encoding and embedding tests
- Authentication and access control tests
- Infrastructure deployment reproducibility tests

Run property-based tests specifically:
```bash
pytest tests/test_property_*.py -v
```

### Test Development Guidelines

When writing tests:

1. **Use provided fixtures**: Leverage `db_session`, `test_client`, and utility fixtures
2. **Handle imports properly**: Use the `app_modules` fixture for accessing application modules
3. **Mark appropriately**: Use `@pytest.mark.slow`, `@pytest.mark.unit`, etc.
4. **Isolate tests**: Each test should be independent and not rely on others
5. **Use sample data**: Leverage `sample_drawing_data` and `sample_embedding_data` fixtures
6. **Clean up**: Fixtures handle cleanup automatically, but clean up any external resources
7. **Import safety**: Don't import application modules at module level; use fixtures instead

## Development Commands

### Backend
```bash
# Activate virtual environment first
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Format code
black app/
isort app/

# Lint code
flake8 app/

# Run tests
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest -m "not slow"      # Skip slow tests
pytest -m unit            # Run only unit tests
pytest -m integration     # Run only integration tests
pytest --durations=10     # Show 10 slowest tests (configured by default)

# Type checking (relaxed configuration for development)
mypy app/
```

### Frontend
```bash
# Format code
npm run format

# Lint code
npm run lint

# Type checking
npm run type-check

# Build for production
npm run build
```

## Usage

### Web Interface

1. **Dashboard** (http://localhost:5173)
   - View system statistics and model status
   - See age distribution of drawings
   - Monitor recent analyses and anomaly detection results

2. **Upload Drawings** 
   - Upload individual drawings with age, subject, and metadata
   - Supported formats: PNG, JPEG, BMP (max 10MB)
   - Subject categories: 64 predefined categories including objects, living beings, nature, abstract concepts

3. **Analysis Results**
   - View subject-aware anomaly scores and confidence levels with 6 interactive tabs:
     - **Interactive Analysis**: Hoverable saliency regions with click-to-zoom and subject-specific insights
     - **Saliency Map**: Original + saliency overlays with adjustable opacity and subject context
     - **Confidence**: Detailed confidence metrics with subject-aware reliability warnings
     - **Comparison**: Similar examples from same age group and subject category
     - **History**: Historical analysis tracking and subject-aware trends
     - **Annotations**: User annotation tools for regions with subject context
   - Export results in multiple formats with subject-aware comprehensive reports (PNG, PDF, JSON, CSV, HTML)
   - Browse analysis history with subject-contextualized interpretability

4. **Configuration**
   - View trained subject-aware models and their statistics
   - Adjust system thresholds and subject-specific parameters
   - Monitor model performance across different subject categories

### API Usage

The system provides a comprehensive REST API:

```bash
# Get system statistics
curl "http://localhost:8000/api/v1/analysis/stats"

# Analyze a drawing
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/1"

# Get all drawings
curl "http://localhost:8000/api/v1/drawings/"

# Get model information
curl "http://localhost:8000/api/v1/models/age-groups"

# Get subject-aware interactive interpretability data
curl "http://localhost:8000/api/v1/interpretability/522/interactive"

# Export subject-aware analysis results
curl -X POST "http://localhost:8000/api/v1/interpretability/522/export" \
     -H "Content-Type: application/json" \
     -d '{"format": "pdf", "export_options": {"include_subject_context": true}}'
```

## API Documentation

The API documentation is automatically generated and available at:
- Development: http://localhost:8000/docs
- Interactive API explorer with request/response examples
- Complete endpoint documentation with schemas

## Optional Dependencies

The system is designed to work with minimal dependencies, but offers enhanced functionality with optional packages. See [OPTIONAL_DEPENDENCIES.md](OPTIONAL_DEPENDENCIES.md) for detailed information.

### Enhanced Image Processing (OpenCV)
```bash
# Install OpenCV for advanced image processing
pip install opencv-python>=4.8.0,<4.10.0

# Or install all enhanced features
pip install -r requirements-enhanced.txt
```

**Benefits of OpenCV:**
- Advanced contour detection with precise boundary algorithms
- High-quality image resizing with cubic interpolation
- Canny edge detection for drawing complexity analysis
- Enhanced saliency map overlays with accurate contour drawing

**Fallback without OpenCV:**
- PIL-based image resizing (Lanczos interpolation)
- Simple gradient-based edge detection
- PIL-based contour approximation using edge pixel detection
- All core functionality remains available with consistent visual results

### PDF Generation (ReportLab)
```bash
# Install ReportLab for comprehensive PDF reports
pip install reportlab>=4.0.0
```

**Benefits of ReportLab:**
- Professional PDF export reports
- Multi-page analysis summaries
- Embedded charts and visualizations
- Subject-aware comprehensive documentation

**Fallback without ReportLab:**
- PNG, JSON, CSV, and HTML exports remain available
- Web-based report viewing through the interface

## Troubleshooting

### Common Issues

1. **NumPy Compatibility Error**
   ```bash
   # If you see NumPy 2.x compatibility issues:
   pip install "numpy>=1.25.2,<2.0.0"
   ```

2. **AWS Dependencies Missing (Local Development)**
   ```bash
   # AWS services are optional for local development
   # The system will work without boto3/botocore
   # For production deployment with AWS features:
   pip install boto3 botocore
   ```

3. **OpenCV Import Errors**
   ```bash
   # OpenCV is optional - the system will work without it using PIL fallbacks
   # Common OpenCV issues and solutions:
   
   # Issue: "No module named 'cv2'"
   # Solution: Install OpenCV for enhanced functionality
   pip install opencv-python>=4.8.0,<4.10.0
   
   # Issue: "ImportError: libGL.so.1: cannot open shared object file"
   # Solution: Install system graphics libraries (Linux)
   sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
   
   # Issue: OpenCV import fails with other system library errors
   # Solution: The system will automatically fall back to PIL-based processing
   # Check logs for specific error details if needed
   
   # Install all enhanced features including OpenCV:
   pip install -r requirements-enhanced.txt
   ```

4. **MyPy Type Checking Issues**
   ```bash
   # The project uses relaxed MyPy configuration for development
   # If you encounter type checking errors, they are likely ignored by default
   # To enable strict type checking (not recommended for development):
   # Edit pyproject.toml and set warn_return_any = true, disallow_untyped_defs = true
   ```

5. **Frontend Shows 0 Drawings**
   - Check if backend is running on port 8000
   - Verify Vite proxy configuration in `frontend/vite.config.ts`
   - Ensure API endpoints are accessible

6. **Model Training Fails**
   - Ensure sufficient drawings are uploaded (minimum 10 per age group)
   - Check that hybrid embeddings are generated before training
   - Verify database connectivity and subject category data

7. **Vision Transformer Issues**
   - Ensure PyTorch and transformers are properly installed
   - Check that the embedding service initializes correctly
   - Verify image preprocessing pipeline

### Development Tips

- Use `python train_models.py` for complete automated training
- Check logs in `backend.log` for debugging
- Monitor training progress via API endpoints
- Use the web interface to verify system status

## Contributing

1. Install pre-commit hooks: `pre-commit install`
2. Follow the existing code style and conventions
3. Write tests for new functionality
4. Update documentation as needed

## License

[Add your license information here]