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
        metrics = metrics_service.get_dashboard_stats()
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
        from datetime import datetime, timezone

        import psutil

        system_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
            # Calculate metrics from deque
            total_analyses = len(metrics_service._analysis_metrics)

            if total_analyses > 0:
                processing_times = [
                    m.processing_time for m in metrics_service._analysis_metrics
                ]
                average_processing_time = sum(processing_times) / len(processing_times)
                recent_processing_times = processing_times[-10:]

                anomaly_count = sum(
                    1 for m in metrics_service._analysis_metrics if m.anomaly_detected
                )
                normal_count = total_analyses - anomaly_count
            else:
                average_processing_time = 0.0
                recent_processing_times = []
                anomaly_count = 0
                normal_count = 0

            analysis_metrics = {
                "total_analyses": total_analyses,
                "average_processing_time": round(average_processing_time, 3),
                "recent_processing_times": recent_processing_times,
                "anomaly_count": anomaly_count,
                "normal_count": normal_count,
            }

            # Calculate system metrics from response times
            if (
                hasattr(metrics_service, "_response_times")
                and metrics_service._response_times
            ):
                total_requests = len(metrics_service._response_times)
                successful_requests = sum(
                    1
                    for rt in metrics_service._response_times
                    if rt["status_code"] < 400
                )
                failed_requests = total_requests - successful_requests
                error_rate = (
                    (failed_requests / total_requests * 100)
                    if total_requests > 0
                    else 0.0
                )

                response_times = [
                    rt["duration"] for rt in metrics_service._response_times
                ]
                average_response_time = sum(response_times) / len(response_times)
                recent_response_times = response_times[-10:]
            else:
                total_requests = 0
                successful_requests = 0
                failed_requests = 0
                error_rate = 0.0
                average_response_time = 0.0
                recent_response_times = []

            # Get current system metrics
            import psutil

            memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu_usage_percent = psutil.cpu_percent(interval=0.1)

            system_metrics = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "error_rate": round(error_rate, 2),
                "average_response_time": round(average_response_time, 3),
                "recent_response_times": recent_response_times,
                "memory_usage_mb": round(memory_usage_mb, 2),
                "cpu_usage_percent": round(cpu_usage_percent, 2),
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
