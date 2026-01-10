"""
FastAPI application entry point for Children's Drawing Anomaly Detection System.
"""

import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create minimal FastAPI app first for immediate health checks
app = FastAPI(
    title="Children's Drawing Anomaly Detection System",
    version="0.1.0",
    description="Machine learning system for detecting anomalies in children's drawings",
)


# Lightweight health endpoints available immediately
@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for load balancer."""
    return {
        "status": "healthy",
        "service": "drawing-anomaly-detection",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("APP_ENVIRONMENT", "unknown"),
        "storage": os.getenv("STORAGE_BACKEND", "unknown"),
    }


@app.get("/health/simple")
async def simple_health_check():
    """Ultra-lightweight health check for ALB - no dependencies."""
    return {"status": "ok"}


# Track service initialization status
SERVICES_INITIALIZED = False
INITIALIZATION_ERROR = None

# Now proceed with heavy imports and initialization
try:
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    from app.api.api_v1.api import api_router
    from app.api.api_v1.endpoints.auth import router as auth_router
    from app.core.auth_middleware import AuthenticationMiddleware
    from app.core.config import settings
    from app.core.database import init_db
    from app.core.metrics_middleware import MetricsCollectionMiddleware
    from app.core.middleware import (
        ErrorHandlingMiddleware,
        RequestLoggingMiddleware,
        ResourceMonitoringMiddleware,
        ResponseTimeMiddleware,
        SessionTrackingMiddleware,
        setup_error_monitoring,
    )
    from app.core.security_middleware import SecurityMiddleware

    # Initialize error monitoring
    setup_error_monitoring()

    # Update app configuration with settings
    app.title = settings.PROJECT_NAME
    app.version = settings.VERSION
    app.openapi_url = f"{settings.API_V1_STR}/openapi.json"

    SERVICES_INITIALIZED = True

except Exception as e:
    INITIALIZATION_ERROR = str(e)
    print(f"Warning: Service initialization failed: {e}")
    # Continue with minimal app functionality


# Database initialization on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on application startup."""
    try:
        print("=" * 50)
        print("STARTING DATABASE INITIALIZATION")
        print("=" * 50)
        init_db()
        print("=" * 50)
        print("DATABASE INITIALIZATION COMPLETED SUCCESSFULLY")
        print("=" * 50)

        # Log storage configuration for debugging
        try:
            from app.services.environment_storage import get_storage_service

            storage_service = get_storage_service()
            storage_info = storage_service.get_storage_info()
            print("=" * 50)
            print("STORAGE CONFIGURATION")
            print("=" * 50)
            for key, value in storage_info.items():
                print(f"{key}: {value}")
            print("=" * 50)
        except Exception as e:
            print(f"Warning: Failed to initialize storage service: {e}")
            print(
                "🚀 Continuing without storage service - core functionality available"
            )

    except Exception as e:
        print("=" * 50)
        print(f"ERROR: Database initialization failed: {e}")
        print("=" * 50)
        import traceback

        traceback.print_exc()

        # In production, try to continue with minimal functionality
        if os.getenv("APP_ENVIRONMENT") == "production":
            print(
                "🚨 PRODUCTION MODE: Attempting to continue with minimal functionality"
            )
            print(
                "⚠️  Some features may not be available until database issues are resolved"
            )
            # Don't raise - allow service to start with degraded functionality
        else:
            # In development, fail fast to catch issues early
            print("🔧 DEVELOPMENT MODE: Failing startup to catch database issues early")
            raise RuntimeError(f"Database initialization failed: {e}")


# Only add middleware and routes if services initialized successfully
if SERVICES_INITIALIZED:
    # Add security middleware (first for rate limiting and security headers)
    security_middleware = SecurityMiddleware(app)
    app.add_middleware(SecurityMiddleware)

    # Add session tracking middleware (early to track all requests)
    app.add_middleware(SessionTrackingMiddleware)

    # Add response time tracking middleware (early to track all requests)
    app.add_middleware(ResponseTimeMiddleware)

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

    # Add alias for /api/* to /api/v1/* for production compatibility
    # This handles the case where frontend calls /api/drawings/upload
    # but backend expects /api/v1/drawings/upload
    app.include_router(api_router, prefix="/api")

    # Include authentication router (without API prefix)
    app.include_router(auth_router, prefix="/auth", tags=["authentication"])

    # Include demo router at root level for public access
    from app.api.api_v1.endpoints.demo import router as demo_router

    app.include_router(demo_router, prefix="/demo", tags=["demo"])

    # Mount static files for serving uploaded images and results with better error handling
    try:
        if os.path.exists("static"):
            app.mount(
                "/static",
                StaticFiles(directory="static", check_dir=False),
                name="static",
            )
            print("✅ Static files mounted at /static")
        else:
            print("⚠️ Static directory not found, creating it...")
            os.makedirs("static", exist_ok=True)
            os.makedirs("static/models", exist_ok=True)
            os.makedirs("static/saliency_maps", exist_ok=True)
            os.makedirs("static/exports", exist_ok=True)
            app.mount(
                "/static",
                StaticFiles(directory="static", check_dir=False),
                name="static",
            )
            print("✅ Static files mounted at /static")

        if os.path.exists("uploads"):
            app.mount(
                "/uploads",
                StaticFiles(directory="uploads", check_dir=False),
                name="uploads",
            )
            print("✅ Uploads mounted at /uploads")
        else:
            print("⚠️ Uploads directory not found, creating it...")
            os.makedirs("uploads", exist_ok=True)
            app.mount(
                "/uploads",
                StaticFiles(directory="uploads", check_dir=False),
                name="uploads",
            )
            print("✅ Uploads mounted at /uploads")

        # Mount docs directory for documentation files
        if os.path.exists("docs"):
            app.mount(
                "/docs",
                StaticFiles(directory="docs", check_dir=False, html=True),
                name="docs",
            )
            print("✅ Documentation files mounted at /docs")
        else:
            print("⚠️ Docs directory not found, creating it...")
            os.makedirs("docs", exist_ok=True)
            app.mount(
                "/docs",
                StaticFiles(directory="docs", check_dir=False, html=True),
                name="docs",
            )
            print("✅ Documentation files mounted at /docs")
    except Exception as e:
        print(f"⚠️ Error mounting static files: {e}")

        # Create fallback endpoints for static files
        @app.get("/static/{file_path:path}")
        async def serve_static_fallback(file_path: str):
            return JSONResponse(
                status_code=404,
                content={"error": "Static file not found", "path": file_path},
            )

        @app.get("/uploads/{file_path:path}")
        async def serve_uploads_fallback(file_path: str):
            return JSONResponse(
                status_code=404,
                content={"error": "Upload file not found", "path": file_path},
            )

    # Mount React frontend (only if frontend_build directory exists)
    if os.path.exists("frontend_build"):
        app.mount(
            "/", StaticFiles(directory="frontend_build", html=True), name="frontend"
        )
    else:
        # Fallback root endpoint when frontend build doesn't exist (e.g., during testing)
        @app.get("/")
        async def root_fallback():
            """Fallback root endpoint when React frontend build is not available."""
            return {
                "message": "Children's Drawing Anomaly Detection System",
                "version": settings.VERSION,
                "docs_url": "/docs",
                "api_url": f"{settings.API_V1_STR}",
                "demo_url": "/demo",
                "status": "Frontend build not available - API only mode",
            }

else:
    # Minimal functionality when services failed to initialize
    @app.get("/")
    async def degraded_root():
        """Root endpoint in degraded mode."""
        return {
            "message": "Children's Drawing Anomaly Detection System",
            "status": "degraded",
            "error": "Services not fully initialized",
            "health_check": "/health",
        }


@app.get("/api")
async def api_root():
    """API root endpoint - returns basic API information."""
    if not SERVICES_INITIALIZED:
        return {
            "message": "Children's Drawing Anomaly Detection System API",
            "status": "degraded",
            "error": INITIALIZATION_ERROR,
            "health_check": "/health",
        }

    return {
        "message": "Children's Drawing Anomaly Detection System API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "api_url": f"{settings.API_V1_STR}",
        "demo_url": "/demo",
    }


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information."""
    import os
    from datetime import datetime, timezone

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    from datetime import datetime, timezone

    import psutil

    from app.services.monitoring_service import get_monitoring_service

    monitoring_service = get_monitoring_service()

    # Collect system metrics
    system_metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
