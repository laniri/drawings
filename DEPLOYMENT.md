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

### 2. SSL Certificates

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

### 3. Database Setup

The system uses PostgreSQL in production. The database will be automatically initialized with the required schema.

### 4. Build and Deploy

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Run database migrations
docker-compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

### 5. Verify Deployment

```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Check logs
docker-compose -f docker-compose.prod.yml logs

# Test health endpoint
curl -f http://localhost:8000/health
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
- Runs on port 8000
- Uses Gunicorn with 4 workers
- Includes health checks and monitoring

#### Database (PostgreSQL)
- Runs on port 5432
- Includes automatic backups
- Optimized for performance

#### Frontend (React)
- Served through nginx
- Includes SSL termination
- Rate limiting enabled

#### Redis (Caching)
- Runs on port 6379
- Used for session storage and caching

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

Logs are stored in the `logs/` directory:
- `app.log`: Application logs
- `monitor.log`: Monitoring logs
- Docker logs: `docker-compose logs`

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

### Common Issues

#### Services Won't Start
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check system resources
df -h
free -h
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