"""
FastAPI application entry point for Children's Drawing Anomaly Detection System.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.api_v1.api import api_router
from app.api.api_v1.endpoints.auth import router as auth_router
from app.core.auth_middleware import AuthenticationMiddleware
from app.core.config import settings
from app.core.metrics_middleware import MetricsCollectionMiddleware
from app.core.middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    ResourceMonitoringMiddleware,
    setup_error_monitoring,
)
from app.core.security_middleware import SecurityMiddleware

# Initialize error monitoring
setup_error_monitoring()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Machine learning system for detecting anomalies in children's drawings",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Add security middleware (first for rate limiting and security headers)
security_middleware = SecurityMiddleware(app)
app.add_middleware(SecurityMiddleware)

# Add metrics collection middleware
metrics_middleware = MetricsCollectionMiddleware(app)
app.add_middleware(MetricsCollectionMiddleware)

# Add authentication middleware (before other middleware)
app.add_middleware(AuthenticationMiddleware)

# Add error handling middleware (first to catch all errors)
error_middleware = ErrorHandlingMiddleware(app)
app.add_middleware(ErrorHandlingMiddleware)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add resource monitoring middleware
resource_middleware = ResourceMonitoringMiddleware(app, max_concurrent_requests=10)
app.add_middleware(ResourceMonitoringMiddleware, max_concurrent_requests=10)

# Store middleware references in app state for metrics access
app.state.security_middleware = security_middleware
app.state.metrics_middleware = metrics_middleware
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

# Include authentication router (without API prefix)
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

# Include demo router at root level for public access
from app.api.api_v1.endpoints.demo import router as demo_router

app.include_router(demo_router, prefix="/demo", tags=["demo"])

# Mount static files for serving uploaded images and results
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Root endpoint providing basic system information."""
    return {
        "message": "Children's Drawing Anomaly Detection System",
        "version": settings.VERSION,
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "service": "drawing-anomaly-detection"}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information."""
    import os
    from datetime import datetime

    import psutil

    # Get middleware instances for stats
    error_middleware = None
    resource_middleware = None

    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(
            middleware.cls, ErrorHandlingMiddleware
        ):
            error_middleware = middleware
        elif isinstance(middleware.cls, type) and issubclass(
            middleware.cls, ResourceMonitoringMiddleware
        ):
            resource_middleware = middleware

    health_info = {
        "status": "healthy",
        "service": "drawing-anomaly-detection",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "process_id": os.getpid(),
        },
        "database": {
            "url": settings.DATABASE_URL,
            "status": "connected",  # TODO: Add actual DB health check
        },
        "storage": {
            "upload_dir": settings.UPLOAD_DIR,
            "static_dir": settings.STATIC_DIR,
            "max_file_size": settings.MAX_FILE_SIZE,
        },
    }

    # Add security statistics if available
    if hasattr(app.state, "security_middleware"):
        health_info["security"] = app.state.security_middleware.get_rate_limit_stats()

    # Add error statistics if available
    if hasattr(app.state, "error_middleware"):
        health_info["errors"] = app.state.error_middleware.get_error_stats()

    # Add resource statistics if available
    if hasattr(app.state, "resource_middleware"):
        health_info["resources"] = app.state.resource_middleware.get_resource_stats()

    return health_info


@app.get("/metrics")
async def get_metrics():
    """Get system metrics for monitoring."""
    from datetime import datetime

    import psutil

    from app.services.monitoring_service import get_monitoring_service

    monitoring_service = get_monitoring_service()

    # Collect system metrics
    system_metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "available": psutil.virtual_memory().available,
                "total": psutil.virtual_memory().total,
            },
            "disk": {
                "percent": psutil.disk_usage("/").percent,
                "free": psutil.disk_usage("/").free,
                "total": psutil.disk_usage("/").total,
            },
        },
    }

    # Add middleware statistics if available
    if hasattr(app.state, "security_middleware"):
        system_metrics["security"] = (
            app.state.security_middleware.get_rate_limit_stats()
        )

    if hasattr(app.state, "error_middleware"):
        system_metrics["errors"] = app.state.error_middleware.get_error_stats()

    if hasattr(app.state, "resource_middleware"):
        system_metrics["resources"] = app.state.resource_middleware.get_resource_stats()

    if hasattr(app.state, "metrics_middleware"):
        system_metrics["application"] = (
            app.state.metrics_middleware.get_metrics_summary()
        )

    # Add monitoring service statistics
    system_metrics["monitoring"] = monitoring_service.get_service_stats()

    # Record these metrics to CloudWatch
    monitoring_service.record_performance_metrics(
        {
            "cpu_usage": system_metrics["system"]["cpu_percent"],
            "memory_usage": system_metrics["system"]["memory"]["percent"],
            "disk_usage": system_metrics["system"]["disk"]["percent"],
        }
    )

    return system_metrics


@app.get("/monitoring/logs")
async def get_recent_logs(limit: int = 100):
    """Get recent structured logs for monitoring."""
    from app.services.monitoring_service import get_monitoring_service

    monitoring_service = get_monitoring_service()

    # Get recent log entries
    recent_logs = list(monitoring_service._log_entries)[-limit:]

    return {
        "logs": [
            {
                "correlation_id": log.correlation_id,
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "message": log.message,
                "component": log.component,
                "operation": log.operation,
                "success": log.success,
                "error_message": log.error_message,
            }
            for log in recent_logs
        ],
        "total_logs": len(monitoring_service._log_entries),
        "limit": limit,
    }


@app.get("/monitoring/alerts")
async def get_recent_alerts(limit: int = 50):
    """Get recent alerts for monitoring."""
    from app.services.monitoring_service import get_monitoring_service

    monitoring_service = get_monitoring_service()

    # Get recent alerts
    recent_alerts = list(monitoring_service._alert_history)[-limit:]

    return {
        "alerts": [
            {
                "alert_id": alert.alert_id,
                "correlation_id": alert.correlation_id,
                "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                "success": alert.success,
                "error_message": alert.error_message,
            }
            for alert in recent_alerts
        ],
        "total_alerts": len(monitoring_service._alert_history),
        "limit": limit,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
