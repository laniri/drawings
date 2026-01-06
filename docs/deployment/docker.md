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

### Static File Sync Architecture

The production container includes intelligent static file synchronization with truly non-blocking startup:

1. **Critical ML Models Sync**: 
   - Syncs ML models from S3 first in detached background process
   - Essential for application functionality but non-blocking for startup
   - Proper permissions set automatically (755 + www-data ownership)
   - Location: `s3://bucket/static/models/` → `/app/static/models/`
   - Uses `--quiet` flag for cleaner logs

2. **Background Static File Sync**:
   - Completely detached background process with comprehensive logging
   - Uploads: `s3://bucket/uploads/` → `/app/uploads/` (medium priority)
   - Saliency maps: Skipped during startup (too many files, generated on-demand)
   - All operations logged to `/tmp/sync.log` for debugging

3. **Enhanced Sync Process**:
   - **Detached Process**: Uses subshell with output redirection for true non-blocking operation
   - **Priority-Based**: ML models first, then uploads, saliency maps skipped
   - **Permission Management**: Automatic chmod 755 and chown www-data:www-data
   - **Error Handling**: Graceful handling with detailed status messages
   - **Development Mode**: Automatically skips S3 sync when not in production

### Service Configuration
#### Nginx Configuration
- **Primary Frontend Serving**: Serves React app from `/var/www/html` at root path (`/`)
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

# Production deployment (PostgreSQL)
docker-compose -f docker-compose.prod.yml up -d

# Production deployment (SQLite - simplified)
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Deployment Options

### Development Deployment
```bash
# Standard development with hot reload
docker-compose up -d
```

### Production Deployment Options

#### Option 1: PostgreSQL Production (Full-featured)
```bash
# Production with PostgreSQL database
docker-compose -f docker-compose.prod.yml up -d
```
- **Database**: PostgreSQL with persistent volumes
- **Scalability**: Designed for high-traffic production environments
- **Features**: Full database features, connection pooling, advanced queries
- **Use Case**: Large-scale deployments, multiple concurrent users

#### Option 2: SQLite Production (Simplified)
```bash
# Production with SQLite database (simplified deployment)
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml up -d
```
- **Database**: SQLite with local file storage
- **Simplicity**: Single-file database, no separate database container
- **Performance**: Excellent for small to medium workloads
- **Use Case**: Single-server deployments, development staging, cost-effective production

## Services Configuration

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

### Production Service (PostgreSQL)
- **Image**: Combined frontend + backend container
- **Port**: 80
- **Process Manager**: Supervisord
- **Services**: nginx + uvicorn
- **Database**: PostgreSQL container
- **Health Checks**: Built-in endpoint monitoring

### Production Service (SQLite)
- **Image**: Combined frontend + backend container
- **Port**: 80 (backend), 80/443 (frontend)
- **Database**: SQLite file mounted from host
- **Storage**: Local directories mounted from host
- **Services**: Backend, Frontend, Redis, Nginx
- **Resource Limits**: Optimized for single-server deployment

## Configuration
Environment variables can be configured in `.env` file:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env
```

### SQLite Production Configuration
The SQLite production deployment uses simplified configuration:

```yaml
# Key environment variables for SQLite production
environment:
  - DATABASE_URL=sqlite:///./drawings.db
  - DEBUG=false
  - LOG_LEVEL=info
  - MAX_FILE_SIZE=52428800  # 50MB
  - CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
  - SECRET_KEY=${SECRET_KEY}
```

### Volume Mounts (SQLite Production)
```yaml
volumes:
  # Database file (SQLite)
  - ./drawings.db:/app/drawings.db
  # Application data directories
  - ./uploads:/app/uploads
  - ./static:/app/static
  - ./backups:/app/backups
  # Logs (Docker volume)
  - logs_data:/app/logs
```

### Key Environment Variables
- `PYTHONPATH=/app` - Python module path
- `PYTHONDONTWRITEBYTECODE=1` - Disable .pyc files
- `PYTHONUNBUFFERED=1` - Unbuffered output for logging
- `STORAGE_BACKEND=local` - Storage backend configuration (default for containers)
- `DATABASE_URL=sqlite:///./drawings.db` - Database connection URL (SQLite production)
- `DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/drawings` - Database URL (PostgreSQL production)
- `ENVIRONMENT=production` - Environment setting (default for containers)
- `S3_BUCKET_NAME=""` - S3 bucket name (empty for local storage mode)
- `HF_HOME=/app/.cache/huggingface` - Hugging Face cache directory
- `TRANSFORMERS_CACHE=/app/.cache/huggingface` - Transformers model cache
- `MPLCONFIGDIR=/app/.cache/matplotlib` - Matplotlib configuration directory

## Health Checks
The production container includes comprehensive health monitoring with multiple endpoints:

```bash
# Ultra-lightweight health check (recommended for ALB)
curl -f http://localhost:80/health/simple

# Standard health check with environment info
curl -f http://localhost:80/health

# Detailed health check with system metrics
curl -f http://localhost:80/health/detailed

# Check supervisord status (if using multi-process container)
docker exec <container> supervisorctl status

# View process logs
docker exec <container> supervisorctl tail -f nginx
docker exec <container> supervisorctl tail -f uvicorn
```

### Health Check Endpoints

- **`/health/simple`**: Ultra-lightweight endpoint returning `{"status": "ok"}` - ideal for load balancer health checks
- **`/health`**: Standard endpoint with timestamp, environment, and storage backend information
- **`/health/detailed`**: Comprehensive endpoint with system metrics, middleware stats, and resource usage

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
   - **Simplified Container**: FastAPI StaticFiles mount serves frontend directly
   - **Architecture Change**: Root path now handled by StaticFiles mount, no JSON fallback
   - **API Information**: Use `/api` endpoint for JSON API information

3. **API Requests Failing**
   - Check uvicorn status: `supervisorctl status uvicorn`
   - Verify backend logs: `docker logs <container>` (uvicorn logs now via Docker)
   - Test direct backend access: `curl http://127.0.0.1:8000/health`
   - Check log file sizes: `ls -lh /var/log/supervisor/` (logs rotate at 50MB)

4. **SQLite Database Issues (SQLite Production)**
   - **File Permissions**: Ensure `drawings.db` is writable by container user
   - **File Location**: Verify database file exists in project root: `ls -la ./drawings.db`
   - **Mount Issues**: Check volume mount is correct: `docker inspect <container> | grep Mounts`
   - **Database Corruption**: Restore from backup or recreate: `rm drawings.db && docker restart <container>`
   - **Concurrent Access**: SQLite handles concurrent reads but limited concurrent writes
   - **Backup Strategy**: Regular file-based backups: `cp drawings.db backups/drawings-$(date +%Y%m%d).db`

5. **PostgreSQL Database Issues (PostgreSQL Production)**
   - **Connection Issues**: Check database container status: `docker-compose logs db`
   - **Password Issues**: Verify `POSTGRES_PASSWORD` environment variable
   - **Network Issues**: Ensure containers are on same network
   - **Data Persistence**: Check PostgreSQL volume mount

6. **Log Management Issues**
   - **Log Rotation**: Supervisord automatically rotates logs at 50MB
   - **Disk Space**: Log rotation prevents disk space exhaustion
   - **Log Access**: Use `docker logs <container>` for uvicorn logs
   - **Historical Logs**: Check `/var/log/supervisor/` for rotated log files

7. **Storage Configuration Issues**
   - **Default**: Container uses local storage (`STORAGE_BACKEND=local`)
   - **Database**: Container uses SQLite database (`DATABASE_URL=sqlite:///./drawings.db`)
   - **Environment**: Container sets production environment (`ENVIRONMENT=production`)
   - **S3 Override**: Set environment variables to enable S3 storage
   - **Configuration**: Override via Docker environment variables or .env file

8. **ML Models Sync Issues (Production)**
   - **Detached Background Sync**: ML models sync runs in completely detached background process
   - **Non-blocking Startup**: Application starts immediately, models sync in background
   - **Expected Output**: `🔄 Starting background static file sync (non-blocking)...`
   - **Success Messages**: 
     - `📥 Syncing ML models from S3 (critical)...`
     - `✅ ML models sync completed`
     - `📥 Syncing uploads from S3 (background)...`
     - `✅ Uploads sync completed`
     - `⏭️ Skipping saliency maps sync during startup (too many files)`
     - `✅ Background sync process completed`
   - **Sync Logs**: Check `/tmp/sync.log` inside container for detailed sync output
   - **Manual Sync**: `aws s3 sync s3://bucket/static/models/ /app/static/models/ --region eu-west-1 --quiet`
   - **Troubleshooting**: Check AWS credentials, S3 bucket access, and network connectivity
   - **Local Development**: All sync operations skipped when `APP_ENVIRONMENT != production`
   - **Saliency Maps**: Skipped during startup (too many files), generated on-demand or synced later

9. **Permission Issues**
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
