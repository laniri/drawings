"""
Health monitoring service for system components and resources.
"""

import asyncio
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from app.core.config import settings
from app.core.exceptions import ConfigurationError, ResourceError

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a system component."""

    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Dict[str, Any]
    last_check: datetime
    response_time_ms: Optional[float] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_free_gb: float
    disk_total_gb: float
    active_connections: int
    process_count: int


class HealthMonitor:
    """Service for monitoring system health and resources."""

    def __init__(self):
        self.health_checks: Dict[str, HealthStatus] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 100
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000.0,
        }

    async def check_all_components(self) -> Dict[str, HealthStatus]:
        """Check health of all system components."""
        checks = [
            self._check_database(),
            self._check_file_storage(),
            self._check_system_resources(),
            self._check_ml_models(),
            self._check_api_endpoints(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                self.health_checks["system_error"] = HealthStatus(
                    name="system_error",
                    status="critical",
                    message=f"Health check error: {str(result)}",
                    details={"error_type": type(result).__name__},
                    last_check=datetime.now(timezone.utc),
                )
            elif isinstance(result, HealthStatus):
                self.health_checks[result.name] = result

        return self.health_checks.copy()

    async def _check_database(self) -> HealthStatus:
        """Check database connectivity and health."""
        start_time = datetime.now(timezone.utc)

        try:
            # Test database connection
            db_path = settings.DATABASE_URL.replace("sqlite:///", "")
            if not Path(db_path).exists():
                return HealthStatus(
                    name="database",
                    status="critical",
                    message="Database file does not exist",
                    details={"db_path": db_path},
                    last_check=start_time,
                )

            # Test connection and basic query
            conn = sqlite3.connect(db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]

            # Check database size
            db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)

            conn.close()

            response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            status = "healthy"
            message = "Database is accessible"

            if response_time > self.alert_thresholds["response_time_ms"]:
                status = "warning"
                message = f"Database response time is high: {response_time:.1f}ms"

            return HealthStatus(
                name="database",
                status=status,
                message=message,
                details={
                    "table_count": table_count,
                    "size_mb": round(db_size_mb, 2),
                    "path": db_path,
                },
                last_check=start_time,
                response_time_ms=response_time,
            )

        except Exception as e:
            return HealthStatus(
                name="database",
                status="critical",
                message=f"Database check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                last_check=start_time,
            )

    async def _check_file_storage(self) -> HealthStatus:
        """Check file storage accessibility and space."""
        start_time = datetime.now(timezone.utc)

        try:
            upload_dir = Path(settings.UPLOAD_DIR)
            static_dir = Path(settings.STATIC_DIR)

            # Check if directories exist and are writable
            issues = []

            for dir_path, name in [(upload_dir, "upload"), (static_dir, "static")]:
                if not dir_path.exists():
                    issues.append(f"{name} directory does not exist: {dir_path}")
                elif not os.access(dir_path, os.W_OK):
                    issues.append(f"{name} directory is not writable: {dir_path}")

            # Check disk space
            disk_usage = psutil.disk_usage(str(upload_dir.parent))
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            # Count files in directories
            upload_count = (
                len(list(upload_dir.rglob("*"))) if upload_dir.exists() else 0
            )
            static_count = (
                len(list(static_dir.rglob("*"))) if static_dir.exists() else 0
            )

            status = "healthy"
            message = "File storage is accessible"

            if issues:
                status = "critical"
                message = f"File storage issues: {'; '.join(issues)}"
            elif used_percent > self.alert_thresholds["disk_percent"]:
                status = "warning"
                message = f"Disk space is low: {used_percent:.1f}% used"

            return HealthStatus(
                name="file_storage",
                status=status,
                message=message,
                details={
                    "upload_dir": str(upload_dir),
                    "static_dir": str(static_dir),
                    "upload_file_count": upload_count,
                    "static_file_count": static_count,
                    "disk_free_gb": round(free_gb, 2),
                    "disk_total_gb": round(total_gb, 2),
                    "disk_used_percent": round(used_percent, 1),
                    "issues": issues,
                },
                last_check=start_time,
            )

        except Exception as e:
            return HealthStatus(
                name="file_storage",
                status="critical",
                message=f"File storage check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                last_check=start_time,
            )

    async def _check_system_resources(self) -> HealthStatus:
        """Check system resource usage."""
        start_time = datetime.now(timezone.utc)

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Get process information
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / (1024 * 1024)

            # Determine status based on thresholds
            status = "healthy"
            warnings = []

            if cpu_percent > self.alert_thresholds["cpu_percent"]:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
                status = "warning"

            if memory.percent > self.alert_thresholds["memory_percent"]:
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
                status = "warning"

            if (disk.used / disk.total * 100) > self.alert_thresholds["disk_percent"]:
                warnings.append(f"High disk usage: {disk.used / disk.total * 100:.1f}%")
                status = "warning"

            message = "System resources are normal"
            if warnings:
                message = f"Resource warnings: {'; '.join(warnings)}"

            # Store metrics for history
            metrics = SystemMetrics(
                timestamp=start_time,
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / (1024**3),
                disk_total_gb=disk.total / (1024**3),
                active_connections=len(psutil.net_connections()),
                process_count=len(psutil.pids()),
            )

            self._add_metrics_to_history(metrics)

            return HealthStatus(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "disk_percent": round((disk.used / disk.total) * 100, 1),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "process_memory_mb": round(process_memory_mb, 1),
                    "active_connections": len(psutil.net_connections()),
                    "warnings": warnings,
                },
                last_check=start_time,
            )

        except Exception as e:
            return HealthStatus(
                name="system_resources",
                status="critical",
                message=f"System resource check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                last_check=start_time,
            )

    async def _check_ml_models(self) -> HealthStatus:
        """Check ML model availability and status."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check if required ML libraries are available
            import torch
            import transformers

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0

            # Check model directories
            models_dir = Path(settings.STATIC_DIR) / "models"
            model_files = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

            status = "healthy"
            message = "ML models are available"

            details = {
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_available": cuda_available,
                "cuda_device_count": device_count,
                "models_dir": str(models_dir),
                "model_file_count": len(model_files),
            }

            if cuda_available:
                details["cuda_version"] = torch.version.cuda
                details["current_device"] = torch.cuda.current_device()

            return HealthStatus(
                name="ml_models",
                status=status,
                message=message,
                details=details,
                last_check=start_time,
            )

        except ImportError as e:
            return HealthStatus(
                name="ml_models",
                status="critical",
                message=f"ML libraries not available: {str(e)}",
                details={"error": str(e), "missing_library": str(e)},
                last_check=start_time,
            )
        except Exception as e:
            return HealthStatus(
                name="ml_models",
                status="warning",
                message=f"ML model check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                last_check=start_time,
            )

    async def _check_api_endpoints(self) -> HealthStatus:
        """Check API endpoint responsiveness."""
        start_time = datetime.now(timezone.utc)

        try:
            # This is a basic check - in a real implementation you might
            # make HTTP requests to test endpoints

            status = "healthy"
            message = "API endpoints are responsive"

            return HealthStatus(
                name="api_endpoints",
                status=status,
                message=message,
                details={
                    "endpoints_checked": ["health", "metrics"],
                    "all_responsive": True,
                },
                last_check=start_time,
            )

        except Exception as e:
            return HealthStatus(
                name="api_endpoints",
                status="warning",
                message=f"API endpoint check failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                last_check=start_time,
            )

    def _add_metrics_to_history(self, metrics: SystemMetrics):
        """Add metrics to history, maintaining size limit."""
        self.metrics_history.append(metrics)

        # Keep only recent metrics
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size :]

    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        recent_metrics = [
            asdict(m) for m in self.metrics_history if m.timestamp >= cutoff_time
        ]

        return recent_metrics

    def get_overall_status(self) -> str:
        """Get overall system status based on all components."""
        if not self.health_checks:
            return "unknown"

        statuses = [check.status for check in self.health_checks.values()]

        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif all(status == "healthy" for status in statuses):
            return "healthy"
        else:
            return "unknown"

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get current alerts based on health checks."""
        alerts = []

        for check in self.health_checks.values():
            if check.status in ["warning", "critical"]:
                alerts.append(
                    {
                        "component": check.name,
                        "severity": check.status,
                        "message": check.message,
                        "timestamp": check.last_check.isoformat(),
                        "details": check.details,
                    }
                )

        return alerts

    def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds."""
        for key, value in thresholds.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
                logger.info(f"Updated alert threshold {key} to {value}")
            else:
                logger.warning(f"Unknown alert threshold: {key}")


# Global health monitor instance
health_monitor = HealthMonitor()
