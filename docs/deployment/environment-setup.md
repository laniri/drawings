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
