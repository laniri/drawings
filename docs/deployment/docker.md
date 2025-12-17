# Docker Deployment

## Overview
This application can be deployed using Docker and Docker Compose for easy setup and consistent environments.

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd children-drawing-anomaly-detection

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services
The Docker Compose configuration includes the following services:

### Backend Service
- **Image**: Custom Python application
- **Port**: 8000
- **Dependencies**: Database, file storage

### Frontend Service  
- **Image**: Custom React application
- **Port**: 3000
- **Dependencies**: Backend service

### Database Service
- **Image**: SQLite (file-based)
- **Storage**: Persistent volume

## Configuration
Environment variables can be configured in `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

## Production Deployment
For production deployment, consider:

1. **Security**: Enable HTTPS, configure firewalls
2. **Scaling**: Use container orchestration (Kubernetes)
3. **Monitoring**: Add logging and monitoring solutions
4. **Backup**: Implement data backup strategies
