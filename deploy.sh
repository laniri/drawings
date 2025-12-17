#!/bin/bash

# Production Deployment Script for Children's Drawing Anomaly Detection System
# This script handles the complete deployment process

set -e  # Exit on any error

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_FILE="deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
    fi
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        warning ".env file not found. Creating from template..."
        if [ -f ".env.production" ]; then
            cp .env.production "$ENV_FILE"
            warning "Please edit $ENV_FILE with your production values before continuing."
            exit 1
        else
            error ".env.production template not found."
        fi
    fi
    
    success "Prerequisites check completed."
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p "$BACKUP_DIR"
    mkdir -p logs
    mkdir -p uploads/drawings
    mkdir -p static/models
    mkdir -p static/saliency_maps
    mkdir -p static/overlays
    
    success "Directories created."
}

# Backup existing data
backup_data() {
    if [ "$1" = "--skip-backup" ]; then
        log "Skipping backup as requested."
        return
    fi
    
    log "Creating backup of existing data..."
    
    BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    BACKUP_PATH="$BACKUP_DIR/backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup database if it exists
    if docker-compose -f "$COMPOSE_FILE" ps db | grep -q "Up"; then
        log "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T db pg_dump -U postgres drawings > "$BACKUP_PATH/database.sql"
    fi
    
    # Backup uploads and static files
    if [ -d "uploads" ]; then
        cp -r uploads "$BACKUP_PATH/"
    fi
    
    if [ -d "static" ]; then
        cp -r static "$BACKUP_PATH/"
    fi
    
    success "Backup created at $BACKUP_PATH"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    log "Waiting for database to be ready..."
    sleep 10
    
    # Run Alembic migrations
    docker-compose -f "$COMPOSE_FILE" exec backend alembic upgrade head
    
    success "Database migrations completed."
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait for services to start
    sleep 30
    
    # Check backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        success "Backend health check passed."
    else
        error "Backend health check failed."
    fi
    
    # Check database connection
    if docker-compose -f "$COMPOSE_FILE" exec -T db pg_isready -U postgres > /dev/null 2>&1; then
        success "Database health check passed."
    else
        error "Database health check failed."
    fi
    
    success "All health checks passed."
}

# Deploy function
deploy() {
    log "Starting deployment..."
    
    # Pull latest images
    log "Pulling latest images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build application images
    log "Building application images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down
    
    # Start services
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Run migrations
    run_migrations
    
    # Health check
    health_check
    
    success "Deployment completed successfully!"
}

# Rollback function
rollback() {
    log "Starting rollback..."
    
    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Find latest backup
    LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -n1)
    
    if [ -z "$LATEST_BACKUP" ]; then
        error "No backup found for rollback."
    fi
    
    log "Rolling back to backup: $LATEST_BACKUP"
    
    # Restore files
    if [ -d "$BACKUP_DIR/$LATEST_BACKUP/uploads" ]; then
        rm -rf uploads
        cp -r "$BACKUP_DIR/$LATEST_BACKUP/uploads" .
    fi
    
    if [ -d "$BACKUP_DIR/$LATEST_BACKUP/static" ]; then
        rm -rf static
        cp -r "$BACKUP_DIR/$LATEST_BACKUP/static" .
    fi
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    # Restore database
    if [ -f "$BACKUP_DIR/$LATEST_BACKUP/database.sql" ]; then
        log "Restoring database..."
        sleep 10  # Wait for DB to start
        docker-compose -f "$COMPOSE_FILE" exec -T db psql -U postgres -d drawings < "$BACKUP_DIR/$LATEST_BACKUP/database.sql"
    fi
    
    success "Rollback completed."
}

# Show usage
usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "Commands:"
    echo "  deploy     Deploy the application"
    echo "  rollback   Rollback to previous version"
    echo "  status     Show service status"
    echo "  logs       Show service logs"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo ""
    echo "Options:"
    echo "  --skip-backup    Skip backup creation during deploy"
    echo "  --help          Show this help message"
}

# Show service status
show_status() {
    log "Service Status:"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Show logs
show_logs() {
    docker-compose -f "$COMPOSE_FILE" logs -f
}

# Stop services
stop_services() {
    log "Stopping services..."
    docker-compose -f "$COMPOSE_FILE" down
    success "Services stopped."
}

# Restart services
restart_services() {
    log "Restarting services..."
    docker-compose -f "$COMPOSE_FILE" restart
    success "Services restarted."
}

# Main script logic
main() {
    case "$1" in
        "deploy")
            check_prerequisites
            create_directories
            backup_data "$2"
            deploy
            ;;
        "rollback")
            rollback
            ;;
        "status")
            show_status
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "--help"|"help"|"")
            usage
            ;;
        *)
            error "Unknown command: $1"
            usage
            ;;
    esac
}

# Run main function with all arguments
main "$@"