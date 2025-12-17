"""
FastAPI application entry point for Children's Drawing Anomaly Detection System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.core.middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    ResourceMonitoringMiddleware,
    setup_error_monitoring
)
from app.api.api_v1.api import api_router

# Initialize error monitoring
setup_error_monitoring()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Machine learning system for detecting anomalies in children's drawings",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Add error handling middleware (first to catch all errors)
error_middleware = ErrorHandlingMiddleware(app)
app.add_middleware(ErrorHandlingMiddleware)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add resource monitoring middleware
resource_middleware = ResourceMonitoringMiddleware(app, max_concurrent_requests=10)
app.add_middleware(ResourceMonitoringMiddleware, max_concurrent_requests=10)

# Store middleware references in app state for metrics access
app.state.error_middleware = error_middleware
app.state.resource_middleware = resource_middleware

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Mount static files for serving uploaded images and results
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint providing basic system information."""
    return {
        "message": "Children's Drawing Anomaly Detection System",
        "version": settings.VERSION,
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "drawing-anomaly-detection"}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information."""
    import psutil
    import os
    from datetime import datetime
    
    # Get middleware instances for stats
    error_middleware = None
    resource_middleware = None
    
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, ErrorHandlingMiddleware):
            error_middleware = middleware
        elif isinstance(middleware.cls, type) and issubclass(middleware.cls, ResourceMonitoringMiddleware):
            resource_middleware = middleware
    
    health_info = {
        "status": "healthy",
        "service": "drawing-anomaly-detection",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_id": os.getpid(),
        },
        "database": {
            "url": settings.DATABASE_URL,
            "status": "connected"  # TODO: Add actual DB health check
        },
        "storage": {
            "upload_dir": settings.UPLOAD_DIR,
            "static_dir": settings.STATIC_DIR,
            "max_file_size": settings.MAX_FILE_SIZE
        }
    }
    
    # Add error statistics if available
    if hasattr(app.state, 'error_middleware'):
        health_info["errors"] = app.state.error_middleware.get_error_stats()
    
    # Add resource statistics if available
    if hasattr(app.state, 'resource_middleware'):
        health_info["resources"] = app.state.resource_middleware.get_resource_stats()
    
    return health_info


@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring."""
    import psutil
    from datetime import datetime
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available,
                "total": psutil.virtual_memory().total
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "free": psutil.disk_usage('/').free,
                "total": psutil.disk_usage('/').total
            }
        }
    }
    
    # Add middleware statistics if available
    if hasattr(app.state, 'error_middleware'):
        metrics["errors"] = app.state.error_middleware.get_error_stats()
    
    if hasattr(app.state, 'resource_middleware'):
        metrics["resources"] = app.state.resource_middleware.get_resource_stats()
    
    return metrics


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )