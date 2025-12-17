# Children's Drawing Anomaly Detection System

A machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected norms. The system uses Vision Transformer (ViT) embeddings and autoencoder models trained on age-specific drawing patterns to detect anomalies through reconstruction loss analysis.

## ✅ System Status

**Current Status**: Fully functional and trained
- **95 sample drawings** uploaded and processed
- **3 trained autoencoder models** for age groups (3-6, 6-9, 9-12 years)
- **Vision Transformer embeddings** generated for all drawings
- **Web interface** connected and displaying data
- **Anomaly detection** active and working

## Features

- **Drawing Upload & Analysis**: Support for PNG, JPEG, and BMP formats with metadata
- **Age-Based Modeling**: Separate autoencoder models trained for different age groups
- **Anomaly Detection**: Reconstruction loss-based scoring with configurable thresholds
- **Interpretability**: Saliency maps and attention visualizations for model explanations
- **Web Interface**: Modern React frontend with Material-UI components
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Real-time Dashboard**: System statistics, age distribution, and analysis results

## Technology Stack

### Backend
- **Python 3.11+** with FastAPI web framework
- **PyTorch** for deep learning models and autoencoder training
- **Vision Transformer (ViT)** for feature extraction from drawings
- **SQLAlchemy** with SQLite database for data persistence
- **Alembic** for database migrations
- **Pydantic** for data validation and settings management
- **Captum** for model interpretability (saliency maps)
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
- Generate ViT embeddings for all drawings (769-dimensional vectors)
- Train autoencoder models for 3 age groups:
  - Early childhood (3.0-6.0 years)
  - Middle childhood (6.0-9.0 years)  
  - Late childhood (9.0-12.0 years)
- Set up anomaly detection thresholds

### Manual Training Steps

If you prefer manual control:

1. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Generate embeddings for existing drawings**
   ```bash
   # For each drawing, generate embeddings
   curl -X POST "http://localhost:8000/api/v1/analysis/embeddings/{drawing_id}"
   ```

3. **Train age group models**
   ```bash
   # Train models via API
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
   - Upload individual drawings with age and subject metadata
   - Supported formats: PNG, JPEG, BMP (max 10MB)

3. **Analysis Results**
   - View anomaly scores and confidence levels
   - See interpretability visualizations for anomalous drawings
   - Browse analysis history

4. **Configuration**
   - View trained models and their statistics
   - Adjust system thresholds and parameters
   - Monitor model performance

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
   - Check that embeddings are generated before training
   - Verify database connectivity

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