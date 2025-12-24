"""
API endpoints for usage metrics and monitoring.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.usage_metrics_service import get_metrics_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Get metrics service
metrics_service = get_metrics_service()


@router.get("/usage")
async def get_usage_metrics(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get comprehensive usage metrics for the dashboard.

    Returns metrics including:
    - Total analyses and drawings
    - Time-based analysis counts (daily, weekly, monthly)
    - Active user sessions and geographic distribution
    - System health and performance metrics
    - Processing time statistics
    """
    try:
        metrics = metrics_service.get_dashboard_metrics(db)
        return {"status": "success", "data": metrics}
    except Exception as e:
        logger.error(f"Failed to get usage metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve usage metrics: {str(e)}",
        )


@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """
    Get system health metrics including uptime, error rates, and resource usage.
    """
    try:
        # Get basic health metrics from the service
        health_data = metrics_service._get_health_metrics()

        # Add additional system information
        import os
        from datetime import datetime

        import psutil

        system_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "process_id": os.getpid(),
            "system": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_usage_percent": psutil.disk_usage("/").percent,
            },
            "service": {
                "name": "children-drawing-anomaly-detection",
                "version": "1.0.0",
                "environment": (
                    "production"
                    if metrics_service._cloudwatch_enabled
                    else "development"
                ),
            },
        }

        return {"status": "healthy", "health": health_data, "system": system_info}

    except Exception as e:
        logger.error(f"Failed to get system health: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/sessions")
async def get_session_metrics() -> Dict[str, Any]:
    """
    Get current user session metrics and geographic distribution.
    """
    try:
        session_metrics = metrics_service._get_session_metrics()
        geographic_metrics = metrics_service._get_geographic_distribution()

        return {
            "status": "success",
            "sessions": session_metrics,
            "geographic_distribution": geographic_metrics,
        }

    except Exception as e:
        logger.error(f"Failed to get session metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve session metrics: {str(e)}",
        )


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Get detailed performance metrics including processing times and system resources.
    """
    try:
        # Get performance data from the service
        with metrics_service._lock:
            analysis_metrics = {
                "total_analyses": metrics_service._analysis_metrics.total_analyses,
                "average_processing_time": metrics_service._analysis_metrics.average_processing_time,
                "recent_processing_times": list(
                    metrics_service._analysis_metrics.processing_times
                )[-10:],
                "anomaly_count": metrics_service._analysis_metrics.anomaly_count,
                "normal_count": metrics_service._analysis_metrics.normal_count,
            }

            system_metrics = {
                "total_requests": metrics_service._system_health.total_requests,
                "successful_requests": metrics_service._system_health.successful_requests,
                "failed_requests": metrics_service._system_health.failed_requests,
                "error_rate": metrics_service._system_health.error_rate,
                "average_response_time": metrics_service._system_health.average_response_time,
                "recent_response_times": list(
                    metrics_service._system_health.response_times
                )[-10:],
                "memory_usage_mb": metrics_service._system_health.memory_usage_mb,
                "cpu_usage_percent": metrics_service._system_health.cpu_usage_percent,
            }

        return {
            "status": "success",
            "analysis": analysis_metrics,
            "system": system_metrics,
        }

    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}",
        )


@router.post("/session/start")
async def start_user_session(request_info: Dict[str, str]) -> Dict[str, Any]:
    """
    Manually start a user session (alternative to automatic detection).

    Request body should contain:
    - ip_address: Client IP address
    - user_agent: User agent string
    """
    try:
        session_id = metrics_service.start_session(request_info)

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session started successfully",
        }

    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}",
        )


@router.post("/session/{session_id}/end")
async def end_user_session(session_id: str) -> Dict[str, Any]:
    """
    Manually end a user session.
    """
    try:
        metrics_service.end_session(session_id)

        return {"status": "success", "message": "Session ended successfully"}

    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end session: {str(e)}",
        )


@router.get("/cloudwatch/status")
async def get_cloudwatch_status() -> Dict[str, Any]:
    """
    Get CloudWatch integration status and configuration.
    """
    return {
        "cloudwatch_enabled": metrics_service._cloudwatch_enabled,
        "aws_region": (
            getattr(metrics_service._cloudwatch_client, "meta", {}).get("region_name")
            if metrics_service._cloudwatch_client
            else None
        ),
        "namespace": "ChildrenDrawingAnalysis",
        "metrics_sent": metrics_service._cloudwatch_enabled,
    }
