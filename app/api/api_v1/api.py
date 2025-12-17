"""
API router configuration for v1 endpoints.
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import drawings, analysis, models, config, health, backup, training, documentation, interpretability

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(drawings.router, prefix="/drawings", tags=["drawings"])
api_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
api_router.include_router(interpretability.router, prefix="/interpretability", tags=["interpretability"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(config.router, prefix="/config", tags=["configuration"])
api_router.include_router(documentation.router, prefix="/documentation", tags=["documentation"])
api_router.include_router(health.router, tags=["health"])
api_router.include_router(backup.router, tags=["backup"])