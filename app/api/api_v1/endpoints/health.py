"""
Health check and monitoring endpoints.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from app.core.middleware import ErrorHandlingMiddleware, ResourceMonitoringMiddleware
from app.services.health_monitor import health_monitor

router = APIRouter()


@router.get("/health", summary="Basic health check")
async def basic_health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "drawing-anomaly-detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check():
    """Detailed health check with all system components."""
    try:
        # Run all health checks
        health_status = await health_monitor.check_all_components()
        overall_status = health_monitor.get_overall_status()
        alerts = health_monitor.get_alerts()

        return {
            "status": overall_status,
            "service": "drawing-anomaly-detection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "details": check.details,
                    "last_check": check.last_check.isoformat(),
                    "response_time_ms": check.response_time_ms,
                }
                for name, check in health_status.items()
            },
            "alerts": alerts,
            "alert_count": len(alerts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get(
    "/health/component/{component_name}", summary="Component-specific health check"
)
async def component_health_check(component_name: str):
    """Get health status for a specific component."""
    try:
        # Run all health checks to ensure we have current data
        await health_monitor.check_all_components()

        if component_name not in health_monitor.health_checks:
            raise HTTPException(
                status_code=404, detail=f"Component '{component_name}' not found"
            )

        check = health_monitor.health_checks[component_name]

        return {
            "component": component_name,
            "status": check.status,
            "message": check.message,
            "details": check.details,
            "last_check": check.last_check.isoformat(),
            "response_time_ms": check.response_time_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Component health check failed: {str(e)}"
        )


@router.get("/metrics", summary="System metrics")
async def get_system_metrics():
    """Get current system metrics."""
    try:
        # Ensure we have current resource data
        await health_monitor._check_system_resources()

        # Get latest metrics
        metrics_history = health_monitor.get_metrics_history(hours=1)
        latest_metrics = metrics_history[-1] if metrics_history else None

        if not latest_metrics:
            raise HTTPException(status_code=503, detail="No metrics data available")

        return {
            "timestamp": latest_metrics["timestamp"],
            "system": {
                "cpu_percent": latest_metrics["cpu_percent"],
                "memory_percent": latest_metrics["memory_percent"],
                "memory_available_gb": latest_metrics["memory_available_gb"],
                "memory_total_gb": latest_metrics["memory_total_gb"],
                "disk_percent": latest_metrics["disk_percent"],
                "disk_free_gb": latest_metrics["disk_free_gb"],
                "disk_total_gb": latest_metrics["disk_total_gb"],
                "active_connections": latest_metrics["active_connections"],
                "process_count": latest_metrics["process_count"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Metrics collection failed: {str(e)}"
        )


@router.get("/metrics/history", summary="Historical metrics")
async def get_metrics_history(
    hours: int = Query(
        default=1, ge=1, le=24, description="Hours of history to retrieve"
    )
):
    """Get historical system metrics."""
    try:
        metrics_history = health_monitor.get_metrics_history(hours=hours)

        return {
            "period_hours": hours,
            "data_points": len(metrics_history),
            "metrics": metrics_history,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Historical metrics retrieval failed: {str(e)}"
        )


@router.get("/alerts", summary="Current system alerts")
async def get_current_alerts():
    """Get current system alerts."""
    try:
        # Ensure we have current health data
        await health_monitor.check_all_components()

        alerts = health_monitor.get_alerts()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_count": len(alerts),
            "alerts": alerts,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")


@router.post("/alerts/thresholds", summary="Update alert thresholds")
async def update_alert_thresholds(thresholds: Dict[str, float]):
    """Update system alert thresholds."""
    try:
        # Validate threshold values
        valid_keys = [
            "cpu_percent",
            "memory_percent",
            "disk_percent",
            "response_time_ms",
        ]
        invalid_keys = [key for key in thresholds.keys() if key not in valid_keys]

        if invalid_keys:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid threshold keys: {invalid_keys}. Valid keys: {valid_keys}",
            )

        # Validate threshold ranges
        for key, value in thresholds.items():
            if key.endswith("_percent") and not (0 <= value <= 100):
                raise HTTPException(
                    status_code=400,
                    detail=f"Percentage threshold {key} must be between 0 and 100",
                )
            elif key == "response_time_ms" and value <= 0:
                raise HTTPException(
                    status_code=400, detail="Response time threshold must be positive"
                )

        # Update thresholds
        health_monitor.update_alert_thresholds(thresholds)

        return {
            "message": "Alert thresholds updated successfully",
            "updated_thresholds": thresholds,
            "current_thresholds": health_monitor.alert_thresholds,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Threshold update failed: {str(e)}"
        )


@router.get("/status", summary="Overall system status")
async def get_system_status():
    """Get overall system status summary."""
    try:
        # Run health checks
        await health_monitor.check_all_components()

        overall_status = health_monitor.get_overall_status()
        alerts = health_monitor.get_alerts()

        # Get component summary
        component_summary = {}
        for name, check in health_monitor.health_checks.items():
            component_summary[name] = check.status

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": component_summary,
            "alert_count": len(alerts),
            "critical_alerts": len([a for a in alerts if a["severity"] == "critical"]),
            "warning_alerts": len([a for a in alerts if a["severity"] == "warning"]),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
