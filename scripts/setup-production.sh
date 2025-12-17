#!/bin/bash

# Production Setup Script
# This script prepares the system for production deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Generate secure random passwords and keys
generate_secrets() {
    log "Generating secure secrets..."
    
    # Generate PostgreSQL password
    POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    
    # Generate Django secret key
    SECRET_KEY=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
    
    # Generate app user password
    APP_USER_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    
    success "Secrets generated successfully."
}

# Create production environment file
create_env_file() {
    log "Creating production environment file..."
    
    cat > .env << EOF
# Production Environment Configuration
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/drawings

# Security
SECRET_KEY=${SECRET_KEY}
DEBUG=false

# Application Settings
PROJECT_NAME=Children's Drawing Anomaly Detection System
VERSION=1.0.0
LOG_LEVEL=info

# File Upload Settings
MAX_FILE_SIZE=52428800
UPLOAD_DIR=uploads
STATIC_DIR=static

# CORS Settings (UPDATE WITH YOUR DOMAIN)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Model Settings
DEFAULT_MODEL_TYPE=vit
EMBEDDING_DIMENSION=768
DEFAULT_THRESHOLD_PERCENTILE=95.0

# Cache Settings
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600

# Monitoring and Logging
LOG_FORMAT=json
LOG_FILE=logs/app.log

# Backup Settings
BACKUP_SCHEDULE=0 2 * * *
BACKUP_RETENTION_DAYS=30
BACKUP_DIR=backups

# Performance Settings
WORKER_PROCESSES=4
MAX_REQUESTS_PER_WORKER=1000
WORKER_TIMEOUT=120

# Health Check Settings
HEALTH_CHECK_INTERVAL=30
HEALTH_CHECK_TIMEOUT=10
EOF

    success "Environment file created at .env"
    warning "Please update CORS_ORIGINS with your actual domain!"
}

# Set up SSL certificates (self-signed for development)
setup_ssl() {
    log "Setting up SSL certificates..."
    
    mkdir -p nginx/ssl
    
    # Generate self-signed certificate (replace with real certificates in production)
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key \
        -out nginx/ssl/server.crt \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
    
    success "SSL certificates created (self-signed for development)"
    warning "Replace with real SSL certificates for production!"
}

# Create nginx configuration
create_nginx_config() {
    log "Creating nginx configuration..."
    
    mkdir -p nginx
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }
    
    upstream frontend {
        server frontend:80;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload:10m rate=2r/s;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }
    
    # HTTPS server
    server {
        listen 443 ssl http2;
        server_name _;
        
        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        
        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Increase timeout for ML operations
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        # File upload endpoints
        location /api/v1/drawings/upload {
            limit_req zone=upload burst=5 nodelay;
            
            client_max_body_size 50M;
            
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
        
        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
        
        # Health check
        location /health {
            proxy_pass http://backend;
            access_log off;
        }
    }
}
EOF

    success "Nginx configuration created."
}

# Create systemd service file
create_systemd_service() {
    log "Creating systemd service file..."
    
    cat > drawing-anomaly-detection.service << EOF
[Unit]
Description=Children's Drawing Anomaly Detection System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

    success "Systemd service file created."
    log "To install: sudo cp drawing-anomaly-detection.service /etc/systemd/system/"
    log "To enable: sudo systemctl enable drawing-anomaly-detection"
}

# Create backup script
create_backup_script() {
    log "Creating backup script..."
    
    mkdir -p scripts
    
    cat > scripts/backup.sh << 'EOF'
#!/bin/bash

# Automated backup script for Children's Drawing Anomaly Detection System

BACKUP_DIR="backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"
RETENTION_DAYS=30

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup database
echo "Backing up database..."
docker-compose -f docker-compose.prod.yml exec -T db pg_dump -U postgres drawings > "$BACKUP_PATH/database.sql"

# Backup uploads
echo "Backing up uploads..."
if [ -d "uploads" ]; then
    tar -czf "$BACKUP_PATH/uploads.tar.gz" uploads/
fi

# Backup static files
echo "Backing up static files..."
if [ -d "static" ]; then
    tar -czf "$BACKUP_PATH/static.tar.gz" static/
fi

# Backup configuration
echo "Backing up configuration..."
cp .env "$BACKUP_PATH/"
cp docker-compose.prod.yml "$BACKUP_PATH/"

# Clean up old backups
echo "Cleaning up old backups..."
find "$BACKUP_DIR" -type d -name "backup_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \;

echo "Backup completed: $BACKUP_PATH"
EOF

    chmod +x scripts/backup.sh
    success "Backup script created at scripts/backup.sh"
}

# Create monitoring script
create_monitoring_script() {
    log "Creating monitoring script..."
    
    cat > scripts/monitor.sh << 'EOF'
#!/bin/bash

# System monitoring script

COMPOSE_FILE="docker-compose.prod.yml"
LOG_FILE="logs/monitor.log"

# Check service health
check_services() {
    echo "=== Service Status ===" >> "$LOG_FILE"
    docker-compose -f "$COMPOSE_FILE" ps >> "$LOG_FILE"
    
    # Check if all services are running
    if ! docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        echo "ERROR: Some services are not running" >> "$LOG_FILE"
        # Send alert (implement your alerting mechanism here)
    fi
}

# Check disk space
check_disk_space() {
    DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$DISK_USAGE" -gt 80 ]; then
        echo "WARNING: Disk usage is ${DISK_USAGE}%" >> "$LOG_FILE"
        # Send alert
    fi
}

# Check memory usage
check_memory() {
    MEMORY_USAGE=$(free | awk 'NR==2{printf "%.2f", $3*100/$2}')
    
    if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
        echo "WARNING: Memory usage is ${MEMORY_USAGE}%" >> "$LOG_FILE"
        # Send alert
    fi
}

# Main monitoring function
main() {
    echo "$(date): Starting health check" >> "$LOG_FILE"
    
    check_services
    check_disk_space
    check_memory
    
    echo "$(date): Health check completed" >> "$LOG_FILE"
}

main
EOF

    chmod +x scripts/monitor.sh
    success "Monitoring script created at scripts/monitor.sh"
}

# Main setup function
main() {
    log "Starting production setup..."
    
    # Create directories
    mkdir -p logs backups scripts nginx/ssl
    
    # Generate secrets and create environment
    generate_secrets
    create_env_file
    
    # Set up SSL and nginx
    setup_ssl
    create_nginx_config
    
    # Create service and scripts
    create_systemd_service
    create_backup_script
    create_monitoring_script
    
    success "Production setup completed!"
    
    echo ""
    echo "Next steps:"
    echo "1. Update CORS_ORIGINS in .env with your actual domain"
    echo "2. Replace SSL certificates in nginx/ssl/ with real certificates"
    echo "3. Review and customize nginx/nginx.conf if needed"
    echo "4. Install systemd service: sudo cp drawing-anomaly-detection.service /etc/systemd/system/"
    echo "5. Enable service: sudo systemctl enable drawing-anomaly-detection"
    echo "6. Set up cron job for backups: 0 2 * * * /path/to/scripts/backup.sh"
    echo "7. Set up cron job for monitoring: */5 * * * * /path/to/scripts/monitor.sh"
    echo "8. Deploy with: ./deploy.sh deploy"
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    warning "Running as root is not recommended for security reasons."
fi

# Run main function
main