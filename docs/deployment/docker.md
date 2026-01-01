## Docker Deployment

## Overview
This application can be deployed using Docker and Docker Compose for easy setup and consistent environments. The production deployment uses supervisord for robust process management and nginx for frontend serving.

## Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

## Architecture

### Production Container Architecture
The production Docker image (`Dockerfile.prod`) uses a multi-stage build approach:

1. **Frontend Build Stage**: 
   - Node.js 18-alpine base image
   - Builds React application with Vite
   - Generates optimized production bundle

2. **Backend Runtime Stage**:
   - Python 3.11-slim base image
   - Installs nginx and supervisord for process management
   - Combines frontend assets with backend application

### Process Management with Supervisord
The production container uses supervisord to manage multiple processes:

- **nginx**: Serves React frontend and proxies API requests
- **uvicorn**: Runs FastAPI backend with standard ASGI server
- **Automatic Restart**: Both processes restart automatically on failure
- **Centralized Logging**: All process logs are managed through supervisord
- **Log Rotation**: Automatic log rotation at 50MB to prevent disk space issues
- **Environment Management**: Preconfigured environment variables for container deployment

### Service Configuration

#### Nginx Configuration
- **Primary Frontend Serving**: Serves React app from `/var/www/html` at root path (`/`)
- **API Proxying**: Proxies `/api/*` requests to backend on port 8000
- **Static File Handling**: Serves `/static/*`, `/uploads/*`, `/auth/*`, `/demo/*` endpoints
- **Health & Documentation**: Proxies `/health`, `/docs`, `/openapi.json` to backend
- **Client-Side Routing**: Handles React Router with `try_files` fallback to `index.html`
- **Performance**: Includes gzip compression and security headers
- **Reliability**: Configurable proxy timeouts (30s connect/send/read)
- **Frontend Integration**: Root path (`/`) serves React frontend when available

#### Uvicorn Configuration
- Binds to `0.0.0.0:8000` (accessible within container network)
- Single worker process for development/small deployments
- **Debug logging enabled** for detailed troubleshooting
- **Logs redirected to Docker stdout/stderr** for better integration
- **Log rotation**: Automatic rotation at 50MB for both stdout and stderr logs
- **Environment**: Preconfigured with `STORAGE_BACKEND="local"`, `DATABASE_URL="sqlite:///./drawings.db"`, and `ENVIRONMENT="production"` for container deployment
- Access logging enabled for request monitoring
- Runs as non-root `appuser` for security

## Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd children-drawing-anomaly-detection

# Development deployment
docker-compose up -d

# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services
The Docker Compose configuration includes the following services:

### Backend Service (Development)
- **Image**: Custom Python application
- **Port**: 8000
- **Dependencies**: Database, file storage
- **Hot Reload**: Enabled for development

### Frontend Service (Development)
- **Image**: Custom React application  
- **Port**: 3000
- **Dependencies**: Backend service
- **Hot Reload**: Enabled with Vite dev server

### Production Service
- **Image**: Combined frontend + backend container
- **Port**: 80
- **Process Manager**: Supervisord
- **Services**: nginx + uvicorn
- **Health Checks**: Built-in endpoint monitoring

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

### Key Environment Variables
- `PYTHONPATH=/app` - Python module path
- `PYTHONDONTWRITEBYTECODE=1` - Disable .pyc files
- `PYTHONUNBUFFERED=1` - Unbuffered output for logging
- `STORAGE_BACKEND=local` - Storage backend configuration (default for containers)
- `DATABASE_URL=sqlite:///./drawings.db` - Database connection URL (default for containers)
- `ENVIRONMENT=production` - Environment setting (default for containers)
- `S3_BUCKET_NAME=""` - S3 bucket name (empty for local storage mode)
- `HF_HOME=/app/.cache/huggingface` - Hugging Face cache directory
- `TRANSFORMERS_CACHE=/app/.cache/huggingface` - Transformers model cache
- `MPLCONFIGDIR=/app/.cache/matplotlib` - Matplotlib configuration directory

## Health Checks
The production container includes comprehensive health monitoring:

```bash
# Container health check
curl -f http://localhost:80/health

# Check supervisord status
docker exec <container> supervisorctl status

# View process logs
docker exec <container> supervisorctl tail -f nginx
docker exec <container> supervisorctl tail -f uvicorn
```

## Troubleshooting

### Process Management Issues
```bash
# Check supervisord status
docker exec <container> supervisorctl status

# Restart individual services
docker exec <container> supervisorctl restart nginx
docker exec <container> supervisorctl restart uvicorn

# View supervisord logs
docker exec <container> cat /var/log/supervisor/supervisord.log

# View uvicorn logs via Docker (recommended)
docker logs -f <container>
```

### Common Issues

1. **Container Won't Start**
   - Check Docker logs: `docker logs <container>`
   - Verify port availability: `netstat -tulpn | grep :80`
   - Check supervisord configuration

2. **Frontend Not Loading**
   - Verify nginx is running: `supervisorctl status nginx`
   - Check nginx logs: `tail -f /var/log/supervisor/nginx.err.log`
   - Ensure frontend build completed successfully: `ls -la /var/www/html/`
   - Verify nginx configuration serves React app at root path
   - Test nginx directly: `curl http://localhost:80/` (should return HTML, not JSON)
   - **Backend Integration**: FastAPI now serves frontend when `frontend_build` exists
   - **Fallback Behavior**: Without frontend, root path returns API information JSON

3. **API Requests Failing**
   - Check uvicorn status: `supervisorctl status uvicorn`
   - Verify backend logs: `docker logs <container>` (uvicorn logs now via Docker)
   - Test direct backend access: `curl http://127.0.0.1:8000/health`
   - Check log file sizes: `ls -lh /var/log/supervisor/` (logs rotate at 50MB)

4. **Log Management Issues**
   - **Log Rotation**: Supervisord automatically rotates logs at 50MB
   - **Disk Space**: Log rotation prevents disk space exhaustion
   - **Log Access**: Use `docker logs <container>` for uvicorn logs
   - **Historical Logs**: Check `/var/log/supervisor/` for rotated log files

5. **Storage Configuration Issues**
   - **Default**: Container uses local storage (`STORAGE_BACKEND=local`)
   - **Database**: Container uses SQLite database (`DATABASE_URL=sqlite:///./drawings.db`)
   - **Environment**: Container sets production environment (`ENVIRONMENT=production`)
   - **S3 Override**: Set environment variables to enable S3 storage
   - **Configuration**: Override via Docker environment variables or .env file

4. **Permission Issues**
   - Verify directory permissions: `ls -la /app`
   - Check user configuration: `id appuser`
   - Ensure proper ownership: `chown -R appuser:appuser /app`
   - Check log directory permissions: `ls -la /var/log/supervisor/`

## Production Deployment
For production deployment, consider:

1. **Security**: 
   - Enable HTTPS with proper SSL certificates
   - Configure firewalls and security groups
   - Use secrets management for sensitive data
   - Regular security updates

2. **Scaling**: 
   - Use container orchestration (Kubernetes, ECS)
   - Configure load balancers
   - Implement horizontal pod autoscaling

3. **Monitoring**: 
   - Add logging aggregation (ELK stack)
   - Implement metrics collection (Prometheus)
   - Set up alerting and notifications
   - Monitor supervisord process health

4. **Backup**: 
   - Implement data backup strategies
   - Database backup automation
   - Configuration backup procedures
   - Disaster recovery planning

## Performance Optimization

### Container Optimization
- Multi-stage builds reduce image size
- Layer caching improves build times
- Non-root execution enhances security
- Proper resource limits prevent resource exhaustion

### Process Optimization
- Supervisord provides reliable process management
- Nginx handles static files efficiently
- Uvicorn worker configuration optimized for workload
- Health checks ensure service availability

### Resource Management
```yaml
# Example resource limits in docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```
