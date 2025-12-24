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
- **ReportLab** for PDF generation and comprehensive export reports
- **Pillow & OpenCV** for image processing and simplified saliency map generation
- **NumPy 1.26.4** (downgraded for PyTorch compatibility)

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

### Development Setup

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
   
   # Copy environment file
   cp .env.example .env
   
   # Run database migrations (when implemented)
   # alembic upgrade head
   
   # Start backend server
   uvicorn app.main:app --reload
   ```

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
pytest

# Type checking
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

## Troubleshooting

### Common Issues

1. **NumPy Compatibility Error**
   ```bash
   # If you see NumPy 2.x compatibility issues:
   pip install "numpy>=1.25.2,<2.0.0"
   ```

2. **Frontend Shows 0 Drawings**
   - Check if backend is running on port 8000
   - Verify Vite proxy configuration in `frontend/vite.config.ts`
   - Ensure API endpoints are accessible

3. **Model Training Fails**
   - Ensure sufficient drawings are uploaded (minimum 10 per age group)
   - Check that hybrid embeddings are generated before training
   - Verify database connectivity and subject category data

4. **Vision Transformer Issues**
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