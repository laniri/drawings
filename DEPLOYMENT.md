# Deployment Guide

This guide covers the deployment of the Children's Drawing Anomaly Detection System to production environments.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended)
- **Memory**: Minimum 4GB RAM (8GB+ recommended)
- **Storage**: Minimum 20GB free space (50GB+ recommended)
- **CPU**: 2+ cores (4+ cores recommended for better performance)
- **Network**: Stable internet connection for Docker image pulls

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Git
- OpenSSL (for SSL certificate generation)
- Curl (for health checks)

### Installation Commands (Ubuntu)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install other dependencies
sudo apt install -y git openssl curl bc

# Logout and login again for Docker group changes to take effect
```

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd children-drawing-anomaly-detection
```

### 2. Run Production Setup

```bash
chmod +x scripts/setup-production.sh
./scripts/setup-production.sh
```

This script will:
- Generate secure passwords and secret keys
- Create production environment configuration
- Set up SSL certificates (self-signed for development)
- Create nginx configuration
- Set up backup and monitoring scripts

### 3. Configure Environment

Edit the generated `.env` file:

```bash
nano .env
```

**Important**: Update the following values:
- `CORS_ORIGINS`: Replace with your actual domain(s)
- SSL certificates: Replace self-signed certificates with real ones

### 4. Deploy Application

```bash
chmod +x deploy.sh
./deploy.sh deploy
```

### 5. Verify Deployment

Check service status:
```bash
./deploy.sh status
```

Test the application:
```bash
curl -f https://your-domain.com/health
```

## Manual Deployment Steps

If you prefer manual deployment or need to customize the process:

### 1. Environment Configuration

Create `.env` file from template:
```bash
cp .env.production .env
```

Edit the file with your production values:
- Database passwords
- Secret keys
- Domain names
- SSL certificate paths

### 2. Choose Database Backend

#### Option A: PostgreSQL (Recommended for high-traffic production)
```bash
# Use PostgreSQL configuration
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/drawings
```

#### Option B: SQLite (Simplified single-server deployment)
```bash
# Use SQLite configuration
DATABASE_URL=sqlite:///./drawings.db
```

**SQLite Benefits:**
- Single file database (no separate container)
- Excellent performance for small to medium workloads
- Simplified backup and restore
- Lower resource requirements
- Perfect for single-server deployments

**PostgreSQL Benefits:**
- Better concurrent access handling
- Advanced database features
- Horizontal scaling capabilities
- Better for high-traffic applications

### 3. SSL Certificates

For production, obtain real SSL certificates:

#### Using Let's Encrypt (Recommended)
```bash
# Install certbot
sudo apt install certbot

# Obtain certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/server.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/server.key
```

#### Using Custom Certificates
```bash
# Copy your certificates
cp your-certificate.crt nginx/ssl/server.crt
cp your-private-key.key nginx/ssl/server.key
```

### 4. Database Setup

The system uses PostgreSQL in production. The database will be automatically initialized with the required schema.

### 5. Build and Deploy

The system provides multiple Docker container deployment options:

#### Standard Production Container (PostgreSQL - Recommended for most deployments)
```bash
# Build standard production image
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

**Features:**
- nginx + supervisord + uvicorn architecture
- Process management with automatic restart
- Pre-loaded Vision Transformer models
- Advanced nginx features (rate limiting, security headers)
- Comprehensive logging and monitoring
- PostgreSQL database with full SQL features

#### SQLite Production Container (Simplified single-server deployment)
```bash
# Build and start SQLite production services
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml up --build -d

# Run database migrations (if needed)
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml exec backend alembic upgrade head
```

**Features:**
- Single-file SQLite database (no separate database container)
- Local file storage with host directory mounts
- Excellent performance for small to medium workloads
- Simplified backup and restore (file-based)
- Lower resource requirements
- Perfect for single-server deployments
- Redis for caching and session management
- Nginx for load balancing and SSL termination

**SQLite Production Benefits:**
- **Simplicity**: No database container management
- **Performance**: Excellent for concurrent reads, good for moderate writes
- **Backup**: Simple file copy for backups
- **Cost**: Lower resource usage and hosting costs
- **Reliability**: Fewer moving parts, less complexity

**When to use SQLite production:**
- Single-server deployments
- Small to medium user base (< 100 concurrent users)
- Cost-effective production environments
- Development staging environments
- Simplified maintenance requirements

#### Simplified Production Container (For memory-constrained environments)
```bash
# Build simplified production image
docker build -f Dockerfile.prod.simplified -t children-drawing-app:simplified .

# Run simplified container
docker run -d -p 80:80 --name drawing-app children-drawing-app:simplified

# Run database migrations (if needed)
docker exec drawing-app alembic upgrade head
```

**Features:**
- Single uvicorn process architecture
- FastAPI serves frontend directly on port 80
- Lazy model loading (models load on first API request)
- ~570MB less memory usage during startup
- Faster container startup time
- Simpler debugging and troubleshooting

**When to use simplified container:**
- Memory-constrained environments (< 2GB available RAM)
- Container startup failures with exit code 137 (memory exhaustion)
- Faster deployment requirements
- Development or testing environments

**Note**: The production container uses supervisord to manage both nginx (frontend) and uvicorn (backend) processes within a single container. This provides better process management, automatic restarts, and centralized logging compared to separate containers.

**Important**: In production, nginx is configured to serve the React frontend at the root path (`/`). The FastAPI backend also has a root endpoint that returns API information. Depending on your nginx configuration, direct API access to the root endpoint may be intercepted by nginx to serve the frontend instead. To access the API information endpoint directly, you may need to configure nginx routing or access it through the API prefix.

### 6. Verify Deployment

#### PostgreSQL Production Verification
```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs

# Test health endpoint
curl -f http://localhost:8000/health
```

#### SQLite Production Verification
```bash
# Check service status
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml ps

# Check logs
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml logs

# Test health endpoint
curl -f http://localhost:8000/health

# Verify SQLite database
docker-compose -f tmp_files/docker-compose.prod.sqlite.yml exec backend sqlite3 /app/drawings.db ".tables"

# Check database file permissions
ls -la ./drawings.db
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | Database password | Generated |
| `SECRET_KEY` | Application secret key | Generated |
| `DEBUG` | Debug mode | `false` |
| `CORS_ORIGINS` | Allowed CORS origins | Update required |
| `MAX_FILE_SIZE` | Maximum upload size | `52428800` (50MB) |
| `LOG_LEVEL` | Logging level | `info` |

### Service Configuration

#### Backend (FastAPI)
- Runs on port 8000 (internal)
- Uses Uvicorn ASGI server with 1 worker
- **Comprehensive environment validation** with startup script
- **Automatic directory creation** with proper permissions
- **Database initialization** for SQLite with proper ownership
- **Python import validation** to catch configuration issues early (lightweight imports only, avoiding heavy model loading)
- **Logs redirected to Docker stdout/stderr** for better integration
- **Hugging Face cache configuration** to prevent model loading permission errors
- Managed by supervisord for automatic restart with process dependencies
- Includes health checks and monitoring
- Runs as non-root user for security

#### Frontend (React)
- Served through nginx on port 80
- Built with Vite for optimized production bundle
- Includes SSL termination and security headers
- Rate limiting and gzip compression enabled
- Managed by supervisord alongside backend

#### Process Management (Supervisord)
- Manages both nginx and uvicorn processes
- Automatic restart on process failure
- Centralized logging to `/var/log/supervisor/`
- Process monitoring and control via supervisorctl
- Configuration in `/etc/supervisor/conf.d/supervisord.conf`

### Nginx Configuration

The nginx configuration includes:
- SSL termination
- Rate limiting
- Security headers
- Gzip compression
- Static file serving

## Monitoring and Maintenance

### Health Checks

The system includes built-in health checks:

```bash
# Application health
curl -f https://your-domain.com/health

# Detailed health check
curl -f https://your-domain.com/health/detailed

# Service metrics
curl -f https://your-domain.com/metrics
```

### Logging

Logs are stored in the following locations:
- Application logs: `docker-compose logs backend`
- Supervisord logs: `/var/log/supervisor/supervisord.log`
- Nginx logs: `/var/log/supervisor/nginx.err.log` and `/var/log/supervisor/nginx.out.log`
- Uvicorn logs: **Redirected to Docker stdout/stderr** (accessible via `docker logs`)
- Docker logs: `docker-compose logs`

**Important**: The uvicorn process logs are now redirected to Docker's stdout/stderr instead of log files. This provides better integration with Docker logging and makes logs accessible through standard Docker commands.

Access process logs:
```bash
# View all supervisord managed processes
docker exec <container> supervisorctl status

# View specific process logs
docker exec <container> supervisorctl tail -f nginx
docker exec <container> supervisorctl tail -f uvicorn

# View supervisord main log
docker exec <container> cat /var/log/supervisor/supervisord.log

# View uvicorn logs via Docker (recommended)
docker logs <container>

# View real-time uvicorn logs via Docker
docker logs -f <container>
```

### Backup

Automated backups are configured:

```bash
# Manual backup
./scripts/backup.sh

# Restore from backup
./deploy.sh rollback
```

Backup schedule (configurable via cron):
```bash
# Add to crontab
0 2 * * * /path/to/scripts/backup.sh
```

### Monitoring

Set up monitoring with:

```bash
# Add to crontab for regular health checks
*/5 * * * * /path/to/scripts/monitor.sh
```

## Scaling and Performance

### Horizontal Scaling

To scale the backend:

```yaml
# In docker-compose.prod.yml
backend:
  deploy:
    replicas: 3
```

### Performance Tuning

#### Database Optimization
- Adjust PostgreSQL settings in `init-db.sql`
- Monitor query performance
- Set up connection pooling

#### Application Optimization
- Increase worker processes: `WORKER_PROCESSES=8`
- Adjust memory limits in Docker Compose
- Enable Redis caching

#### Load Balancing
- Use nginx upstream for multiple backend instances
- Configure session affinity if needed

## Security

### Security Measures Implemented

1. **SSL/TLS Encryption**: All traffic encrypted
2. **Rate Limiting**: API and upload endpoints protected
3. **Security Headers**: XSS, CSRF, and other protections
4. **Non-root Containers**: Services run as non-root users
5. **Network Isolation**: Services communicate through internal network
6. **Input Validation**: All inputs validated and sanitized

### Additional Security Recommendations

1. **Firewall Configuration**:
   ```bash
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 80/tcp    # HTTP
   sudo ufw allow 443/tcp   # HTTPS
   sudo ufw enable
   ```

2. **Regular Updates**:
   ```bash
   # Update system packages
   sudo apt update && sudo apt upgrade -y
   
   # Update Docker images
   docker-compose -f docker-compose.prod.yml pull
   ```

3. **Backup Encryption**:
   ```bash
   # Encrypt backups
   gpg --symmetric --cipher-algo AES256 backup_file.tar.gz
   ```

## Troubleshooting

### Docker Supervisord Issues

The production container uses supervisord to manage nginx and uvicorn processes. For detailed troubleshooting of supervisord-related issues, see the [Docker Supervisord Troubleshooting Guide](tmp_files/DOCKER_SUPERVISORD_TROUBLESHOOTING.md).

**Quick Diagnostics**:
```bash
# Check process status
docker exec <container> supervisorctl status

# View process logs
docker exec <container> supervisorctl tail -f nginx
docker exec <container> supervisorctl tail -f uvicorn

# Restart services
docker exec <container> supervisorctl restart nginx
docker exec <container> supervisorctl restart uvicorn

# Verify frontend build files are present (for debugging)
docker exec <container> ls -la /var/www/html/
```

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check supervisord status
docker exec <container> supervisorctl status

# Check system resources
df -h
free -h

# Restart individual services
docker exec <container> supervisorctl restart nginx
docker exec <container> supervisorctl restart uvicorn
```

#### Database Connection Issues
```bash
# Check database status
docker-compose -f docker-compose.prod.yml exec db pg_isready -U postgres

# Reset database
docker-compose -f docker-compose.prod.yml down
docker volume rm drawings_postgres_data
docker-compose -f docker-compose.prod.yml up -d
```

#### SSL Certificate Issues
```bash
# Check certificate validity
openssl x509 -in nginx/ssl/server.crt -text -noout

# Regenerate self-signed certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/server.crt
```

#### Performance Issues
```bash
# Check resource usage
docker stats

# Check application metrics
curl -f https://your-domain.com/metrics

# Analyze logs
tail -f logs/app.log
```

### Getting Help

1. Check application logs: `docker-compose logs backend`
2. Check system resources: `htop`, `df -h`
3. Verify network connectivity: `curl -f http://localhost:8000/health`
4. Review configuration: Check `.env` and `docker-compose.prod.yml`

## Maintenance Tasks

### Regular Maintenance

1. **Weekly**:
   - Check disk space
   - Review logs for errors
   - Verify backups

2. **Monthly**:
   - Update system packages
   - Update Docker images
   - Clean up old logs and backups

3. **Quarterly**:
   - Review security settings
   - Update SSL certificates
   - Performance optimization review

### Update Procedure

1. **Backup Current System**:
   ```bash
   ./scripts/backup.sh
   ```

2. **Pull Latest Code**:
   ```bash
   git pull origin main
   ```

3. **Deploy Updates**:
   ```bash
   ./deploy.sh deploy
   ```

4. **Verify Update**:
   ```bash
   ./deploy.sh status
   curl -f https://your-domain.com/health
   ```

## Support

For additional support:
- Check the application logs
- Review this documentation
- Consult the API documentation at `/docs`
- Check system requirements and prerequisites