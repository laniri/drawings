# Technology Stack & Build System

## Backend Stack

- **Python 3.11+** - Core runtime
- **FastAPI** - Web framework with automatic OpenAPI docs
- **PyTorch 2.2.2+** - Deep learning framework for subject-aware autoencoder models
- **Vision Transformer (ViT)** - Visual feature extraction from drawings (768-dimensional)
- **Subject Encoding System** - One-hot encoding for 64 predefined subject categories (64-dimensional)
- **Hybrid Embeddings** - Concatenated visual and subject features (832-dimensional total)
- **SQLAlchemy** - ORM with SQLite database
- **Alembic** - Database migrations
- **Pydantic** - Data validation and settings management
- **ReportLab** - PDF generation for comprehensive export reports (optional)
- **Pillow** - Core image processing and saliency map generation
- **OpenCV** - Advanced image processing (optional, with PIL fallback)
- **NumPy 1.26.4** - Compatible with PyTorch 2.2.2+ (avoid NumPy 2.x for stability)
- **Transformers** - Hugging Face library for ViT models
- **Scikit-learn** - Machine learning utilities
- **Pandas** - Data manipulation and analysis
- **Boto3** - AWS SDK for production deployment (optional for local development, graceful fallback)
- **Docker** - Containerization support

## Optional Dependencies

### AWS Services (Production Only)
- **Boto3/Botocore** - AWS SDK for cloud services integration
- **Graceful Fallback** - All AWS-dependent services work without AWS clients in local development
- **Services Affected**: Cost optimization, monitoring, security validation, database migration
- **Local Behavior**: Provides local estimates and recommendations without AWS integration

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
  - **Path**: `sqlite:///./drawings.db` (relative path with `./` prefix)
  - **Critical**: All file operations must use `./drawings.db` to match DATABASE_URL
  - **Production**: 373MB database synced from S3 on container startup
- **CORS**: Configured for localhost:3000 and localhost:5173
- **File uploads**: Max 10MB, stored in `uploads/` directory
- **API Proxy**: Vite dev server proxy routes `/api/*` to backend at `localhost:8000/api/v1/*` and `/static/*` to `localhost:8000/static/*`

## Production Architecture

**Container**: `/app` working dir, nginx → FastAPI:8000, CloudFront (HTTPS) → ALB → HTTP

**File Storage**:
| Type | S3 | Synced? | Served By | URL |
|------|----|---------|-----------|----|
| Drawings | `drawings/` | ✅ `/app/uploads/` | nginx | `/uploads/file.png` |
| New Saliency | N/A | N/A (local) | nginx | `/static/saliency_maps/file.png` |
| Old Saliency | `saliency_maps/` | ❌ (37k+ files) | API | `/api/v1/files/s3/saliency_maps/file.png` |
| Models | `static/models/` | ✅ `/app/static/models/` | nginx | `/static/models/file.pt` |

**Auth**: Cookie `path="/"`, `secure` via `X-Forwarded-Proto: https`  
**Rate Limiting**: 100 req/min per IP, exempt `/demo/*`

## Recent Improvements

- **Real-time Dashboard Updates**: Dashboard stats now recalculate anomaly classifications dynamically
- **Optimized Threshold Management**: Fast threshold recalculation using existing analysis results
- **Robust Configuration**: Support for arbitrary percentile values with proper error handling
- **Cache Invalidation**: Frontend properly refreshes when configuration changes
- **Guaranteed Interpretability**: All drawings now receive interpretability analysis using simplified gradient-based saliency generation
- **Enhanced Export System**: Multi-format exports (PNG, PDF, JSON, CSV, HTML) with comprehensive reports and composite visualizations

## Troubleshooting

### Session Cookie Redirect Loop
**Symptom**: Login succeeds but redirects back to login  
**Fix**: Set `path="/"` and check `X-Forwarded-Proto` for `secure` flag

### Images Not Loading
**Symptom**: Placeholder images in demo/old drawings  
**Fix**: Check both relative and `/app/` paths; use API endpoint for non-synced S3 files

### Blank Screen After Deploy
**Symptom**: White screen on production  
**Fix**: Exempt `/demo/*` from rate limiting

### Wrong Password Error
**Symptom**: Login always fails  
**Fix**: Update IAM policy with correct Secrets Manager ARN pattern

### Database Sync Issues
**Symptom**: Queries return 0 records despite S3 sync  
**Fix**: Use `./drawings.db` (with `./` prefix) consistently

### AWS Dependencies Missing (Local)
**Expected**: AWS services optional for local dev, provide local functionality without AWS clients