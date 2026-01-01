# Environment Setup

## Development Environment

### Prerequisites
- Python 3.11+
- Node.js 18+
- Git

### Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

## Production Environment

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 50GB+ for models and data
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Installation
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv nodejs npm

# Clone and setup application
git clone <repository-url>
cd children-drawing-anomaly-detection
./setup.sh
```

### Configuration
1. Copy environment configuration: `cp .env.example .env`
2. Edit configuration file: `nano .env`
3. Configure database settings
4. Set up file storage paths
5. Configure ML model paths

#### Environment Variables

The system uses automatic environment detection with the following priority:

1. **APP_ENVIRONMENT** variable (highest priority) - Explicit environment specification (`production`, `local`, etc.)
2. **TESTING** variable - When set to `true`, `1`, or `yes` in CI environments, forces LOCAL storage backend for test isolation
3. **AWS_REGION** presence - Implies production environment when set
4. **Default** - Falls back to local environment

For production deployment, set:
```bash
APP_ENVIRONMENT=production
AWS_REGION=eu-west-1
```

For testing (automatically set by test infrastructure):
```bash
TESTING=true
CI=true  # Forces LOCAL storage backend when combined with TESTING=true
SKIP_MODEL_LOADING=true
DATABASE_URL=sqlite:///:memory:
```

**Storage Backend Selection**: The system automatically selects the appropriate storage backend:
- **Production**: Uses S3 storage when `APP_ENVIRONMENT=production` or `AWS_REGION` is set
- **Testing**: Uses LOCAL storage when `TESTING=true` and `CI=true` are both set, **except when `APP_ENVIRONMENT=production` is explicitly set**
- **Local Development**: Uses LOCAL storage by default
- **Container Deployment**: Docker containers default to LOCAL storage via `STORAGE_BACKEND=local` and `S3_BUCKET_NAME=""`

**Production Environment Precedence**: When `APP_ENVIRONMENT=production` is explicitly set, it takes precedence over testing overrides. This ensures production deployments work correctly even in CI/testing contexts, which is essential for production deployment pipelines.

**Container Storage Configuration**: Docker containers are preconfigured with local storage settings:
```bash
STORAGE_BACKEND=local
DATABASE_URL=sqlite:///./drawings.db
ENVIRONMENT=production
S3_BUCKET_NAME=""
```

To enable S3 storage in containers, override these environment variables:
```bash
# For S3 storage in containers
docker run -e STORAGE_BACKEND=s3 -e S3_BUCKET_NAME=your-bucket-name <image>
```

#### Database Configuration
The system supports flexible SQLite database URL formats:
```bash
# Standard absolute path (recommended for production)
DATABASE_URL=sqlite:///./drawings.db

# Alternative format for relative paths
DATABASE_URL=sqlite://drawings.db

# In-memory database (testing only - limited backup support)
DATABASE_URL=sqlite://:memory:
```

**Note**: The backup service automatically detects the database URL format and handles different SQLite configurations. In-memory databases have limited backup capabilities and will log appropriate warnings.

### Service Management
```bash
# Start services
sudo systemctl start cdads-backend
sudo systemctl start cdads-frontend

# Enable auto-start
sudo systemctl enable cdads-backend
sudo systemctl enable cdads-frontend

# Check status
sudo systemctl status cdads-backend
```
