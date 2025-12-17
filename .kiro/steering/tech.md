# Technology Stack & Build System

## Backend Stack

- **Python 3.11+** - Core runtime
- **FastAPI** - Web framework with automatic OpenAPI docs
- **PyTorch** - Deep learning framework
- **Vision Transformer (ViT)** - Feature extraction from drawings
- **SQLAlchemy** - ORM with SQLite database
- **Alembic** - Database migrations
- **Pydantic** - Data validation and settings management
- **Captum** - Model interpretability (saliency maps)
- **Pillow & OpenCV** - Image processing
- **NumPy 1.26.4** - Downgraded for PyTorch compatibility (avoid 2.x)
- **Transformers** - Hugging Face library for ViT models
- **Scikit-learn** - Machine learning utilities
- **Pandas** - Data manipulation and analysis
- **Boto3** - AWS SDK for SageMaker integration
- **Docker** - Containerization support

## Frontend Stack

- **React 18** with TypeScript
- **Vite** - Build tool and dev server with API proxy
- **Material-UI (MUI)** - Component library
- **React Query (@tanstack/react-query)** - Server state management
- **Zustand** - Client state management
- **React Hook Form + Zod** - Form handling and validation
- **Recharts** - Data visualization
- **React Router** - Client-side routing
- **React Dropzone** - File upload interface
- **Axios** - HTTP client for API calls

## Development Tools

- **Black** - Code formatting (line length: 88)
- **isort** - Import sorting
- **Flake8** - Linting
- **MyPy** - Type checking
- **Prettier** - Frontend formatting
- **ESLint** - Frontend linting
- **Pre-commit** - Git hooks
- **Pytest** - Backend testing framework
- **Vitest** - Frontend testing framework
- **Testing Library** - React component testing utilities

## Common Commands

### Backend Development
```bash
# Setup
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements-dev.txt
- **always use python with the virtual enviroment to the project

# Development server
uvicorn app.main:app --reload

# Code quality
black app/
isort app/
flake8 app/
mypy app/

# Testing
pytest

# Model training (automated)
python train_models.py

# Offline training with verbose progress
python train_models_offline.py

# Generate sample data
python create_sample_drawings.py
python upload_sample_drawings.py
```

### Frontend Development
```bash
# Setup
cd frontend
npm install

# Development server
npm run dev

# Code quality
npm run format
npm run lint
npm run type-check

# Build
npm run build

# Testing
npm run test          # Run tests once
npm run test:watch    # Run tests in watch mode
npm run test:ui       # Run tests with UI
```

### Docker Development
```bash
# Start all services
docker-compose -f docker-compose.dev.yml up --build

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## Configuration

- **Environment**: `.env` file (copy from `.env.example`)
- **Database**: SQLite with Alembic migrations
- **CORS**: Configured for localhost:3000 and localhost:5173
- **File uploads**: Max 10MB, stored in `uploads/` directory
- **API Proxy**: Vite dev server proxy routes `/api/*` to backend at `localhost:8000/api/v1/*`

## Recent Improvements

- **Real-time Dashboard Updates**: Dashboard stats now recalculate anomaly classifications dynamically
- **Optimized Threshold Management**: Fast threshold recalculation using existing analysis results
- **Robust Configuration**: Support for arbitrary percentile values with proper error handling
- **Cache Invalidation**: Frontend properly refreshes when configuration changes

## Troubleshooting

### Dashboard Not Updating After Configuration Changes
- **Fixed**: Dashboard now uses dynamic anomaly classification instead of stored flags
- **Verification**: Change threshold percentile in configuration, return to dashboard to see updated counts

### Slow Threshold Recalculation
- **Fixed**: Optimized to use existing analysis results instead of recalculating reconstruction losses
- **Performance**: Threshold updates now complete in seconds instead of minutes

### Configuration Errors with Custom Percentiles
- **Fixed**: Robust handling of arbitrary percentile values (not just 90%, 95%, 99%)
- **Support**: Any percentile between 50.0 and 99.9 is now supported