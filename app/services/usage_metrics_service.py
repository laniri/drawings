"""
Usage metrics and monitoring service for AWS production deployment.

This service provides application-level metrics collection, CloudWatch integration,
and real-time usage statistics for the dashboard.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Optional AWS dependencies
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    HAS_AWS = True
except ImportError:
    HAS_AWS = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

from app.core.config import settings
from app.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class AnalysisMetric:
    """Data class for analysis metrics."""

    timestamp: datetime
    processing_time: float
    age_group: str
    anomaly_detected: bool
    error_occurred: bool = False
    user_session_id: Optional[str] = None


@dataclass
class SessionMetric:
    """Data class for user session metrics."""

    session_id: str
    start_time: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    page_views: int = 0
    geographic_info: Optional[Dict[str, str]] = None


@dataclass
class SystemHealthMetric:
    """Data class for system health metrics."""

    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    error_count: int
    response_time: float
    uptime_seconds: int


class UsageMetricsService:
    """
    Usage metrics service with CloudWatch integration.

    Provides application-level metrics collection, real-time statistics,
    and CloudWatch custom metrics for monitoring.
    """

    def __init__(self):
        self.namespace = "DrawingAnalysis/Application"
        self.max_metrics_history = 10000  # Keep last 10k metrics in memory

        # In-memory storage for real-time metrics
        self._analysis_metrics: deque = deque(maxlen=self.max_metrics_history)
        self._session_metrics: Dict[str, SessionMetric] = {}
        self._system_health_metrics: deque = deque(maxlen=1000)

        # Aggregated statistics
        self._daily_stats = defaultdict(
            lambda: {
                "analysis_count": 0,
                "total_processing_time": 0.0,
                "error_count": 0,
                "unique_sessions": set(),
            }
        )

        # Thread safety
        self._lock = threading.RLock()

        # CloudWatch client
        self._cloudwatch_client = None
        self._cloudwatch_enabled = False
        self._start_time = datetime.now(timezone.utc)

        # Initialize service
        self._initialize()

    def _initialize(self):
        """Initialize the usage metrics service."""
        try:
            if settings.is_production:
                # In production, use CloudWatch
                self._cloudwatch_client = boto3.client(
                    "cloudwatch", region_name=settings.aws_region or "eu-west-1"
                )
                self._cloudwatch_enabled = True
                logger.info("Initialized CloudWatch client for metrics")
            else:
                self._cloudwatch_enabled = False
                logger.info("Running in local mode - CloudWatch metrics disabled")

        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Failed to initialize CloudWatch client: {e}")
            logger.info("Continuing without CloudWatch integration")
            self._cloudwatch_enabled = False

    def record_analysis(
        self,
        processing_time: float,
        age_group: str,
        anomaly_detected: bool,
        error_occurred: bool = False,
        user_session_id: Optional[str] = None,
    ):
        """
        Record an analysis operation metric.

        Args:
            processing_time: Time taken to process the analysis in seconds
            age_group: Age group of the analyzed drawing
            anomaly_detected: Whether an anomaly was detected
            error_occurred: Whether an error occurred during analysis
            user_session_id: Optional session ID for user tracking
        """
        with self._lock:
            timestamp = datetime.now(timezone.utc)

            # Create analysis metric
            metric = AnalysisMetric(
                timestamp=timestamp,
                processing_time=processing_time,
                age_group=age_group,
                anomaly_detected=anomaly_detected,
                error_occurred=error_occurred,
                user_session_id=user_session_id,
            )

            # Store in memory
            self._analysis_metrics.append(metric)

            # Update daily stats
            date_key = timestamp.date().isoformat()
            self._daily_stats[date_key]["analysis_count"] += 1
            self._daily_stats[date_key]["total_processing_time"] += processing_time

            if error_occurred:
                self._daily_stats[date_key]["error_count"] += 1

            if user_session_id:
                self._daily_stats[date_key]["unique_sessions"].add(user_session_id)

            logger.debug(
                f"Recorded analysis metric: {age_group}, {processing_time:.2f}s"
            )

            # Send to CloudWatch if available
            self._send_cloudwatch_metrics(
                [
                    {
                        "MetricName": "AnalysisCount",
                        "Value": 1,
                        "Unit": "Count",
                        "Dimensions": [
                            {"Name": "AgeGroup", "Value": age_group},
                            {"Name": "AnomalyDetected", "Value": str(anomaly_detected)},
                        ],
                    },
                    {
                        "MetricName": "ProcessingTime",
                        "Value": processing_time,
                        "Unit": "Seconds",
                        "Dimensions": [{"Name": "AgeGroup", "Value": age_group}],
                    },
                ]
            )

    def start_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str,
        geographic_info: Optional[Dict[str, str]] = None,
    ):
        """
        Start tracking a user session.

        Args:
            session_id: Unique session identifier
            ip_address: User's IP address
            user_agent: User's browser user agent
            geographic_info: Optional geographic information
        """
        with self._lock:
            session = SessionMetric(
                session_id=session_id,
                start_time=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                ip_address=ip_address,
                user_agent=user_agent,
                geographic_info=geographic_info,
            )

            self._session_metrics[session_id] = session
            logger.debug(f"Started session tracking: {session_id}")

    def update_session_activity(self, session_id: str):
        """
        Update session activity timestamp.

        Args:
            session_id: Session identifier to update
        """
        with self._lock:
            if session_id in self._session_metrics:
                session = self._session_metrics[session_id]
                session.last_activity = datetime.now(timezone.utc)
                session.page_views += 1
                logger.debug(f"Updated session activity: {session_id}")

    def end_session(self, session_id: str):
        """
        End session tracking.

        Args:
            session_id: Session identifier to end
        """
        with self._lock:
            if session_id in self._session_metrics:
                session = self._session_metrics[session_id]
                duration = (
                    datetime.now(timezone.utc) - session.start_time
                ).total_seconds()

                # Send session metrics to CloudWatch
                self._send_cloudwatch_metrics(
                    [
                        {
                            "MetricName": "SessionDuration",
                            "Value": duration,
                            "Unit": "Seconds",
                        },
                        {
                            "MetricName": "PageViews",
                            "Value": session.page_views,
                            "Unit": "Count",
                        },
                    ]
                )

                # Remove from active sessions
                del self._session_metrics[session_id]
                logger.debug(f"Ended session: {session_id}, duration: {duration:.1f}s")

    def record_response_time(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
    ):
        """
        Record API response time.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            duration: Response time in seconds
            status_code: HTTP status code
        """
        with self._lock:
            # Store in a simple list for now (could be optimized with deque)
            if not hasattr(self, "_response_times"):
                self._response_times = deque(maxlen=1000)

            self._response_times.append(
                {
                    "timestamp": datetime.now(timezone.utc),
                    "endpoint": endpoint,
                    "method": method,
                    "duration": duration,
                    "status_code": status_code,
                }
            )

            # Send to CloudWatch
            self._send_cloudwatch_metrics(
                [
                    {
                        "MetricName": "APIResponseTime",
                        "Value": duration,
                        "Unit": "Seconds",
                        "Dimensions": [
                            {"Name": "Endpoint", "Value": endpoint},
                            {"Name": "Method", "Value": method},
                            {"Name": "StatusCode", "Value": str(status_code)},
                        ],
                    }
                ]
            )

    def record_system_health(
        self,
        cpu_usage: float,
        memory_usage: float,
        error_count: int,
        response_time: float,
    ):
        """
        Record system health metrics.

        Args:
            cpu_usage: CPU usage percentage (0-100)
            memory_usage: Memory usage percentage (0-100)
            error_count: Number of errors in the last period
            response_time: Average response time in seconds
        """
        with self._lock:
            uptime_seconds = int(
                (datetime.now(timezone.utc) - self._start_time).total_seconds()
            )

            metric = SystemHealthMetric(
                timestamp=datetime.now(timezone.utc),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                error_count=error_count,
                response_time=response_time,
                uptime_seconds=uptime_seconds,
            )

            self._system_health_metrics.append(metric)

            # Send to CloudWatch
            self._send_cloudwatch_metrics(
                [
                    {"MetricName": "CPUUsage", "Value": cpu_usage, "Unit": "Percent"},
                    {
                        "MetricName": "MemoryUsage",
                        "Value": memory_usage,
                        "Unit": "Percent",
                    },
                    {"MetricName": "ErrorCount", "Value": error_count, "Unit": "Count"},
                    {
                        "MetricName": "ResponseTime",
                        "Value": response_time,
                        "Unit": "Seconds",
                    },
                    {
                        "MetricName": "UptimeSeconds",
                        "Value": uptime_seconds,
                        "Unit": "Seconds",
                    },
                ]
            )

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        Get real-time statistics for the dashboard.

        Returns:
            Dictionary containing dashboard statistics with nested structure
        """
        from sqlalchemy import func

        from app.core.database import get_db
        from app.models.database import AnomalyAnalysis, Drawing

        with self._lock:
            now = datetime.now(timezone.utc)
            db = next(get_db())

            try:
                # Get database statistics (real data)
                total_drawings = db.query(Drawing).count()
                total_analyses = db.query(AnomalyAnalysis).count()

                # Get threshold manager for dynamic anomaly classification
                from app.services.threshold_manager import get_threshold_manager

                threshold_manager = get_threshold_manager()

                # Count anomalies dynamically based on current thresholds
                analyses_with_drawings = (
                    db.query(AnomalyAnalysis)
                    .join(Drawing)
                    .filter(AnomalyAnalysis.anomaly_score.isnot(None))
                    .all()
                )

                anomaly_count = 0
                normal_count = 0
                for analysis in analyses_with_drawings:
                    is_anomaly, _, _ = threshold_manager.is_anomaly(
                        analysis.anomaly_score, analysis.drawing.age_years, db
                    )
                    if is_anomaly:
                        anomaly_count += 1
                    else:
                        normal_count += 1

                # Get recent analyses (last 24 hours)
                cutoff_24h = now - timedelta(hours=24)
                recent_analyses = (
                    db.query(AnomalyAnalysis)
                    .filter(AnomalyAnalysis.analysis_timestamp >= cutoff_24h)
                    .count()
                )

                # Get time-based counts
                cutoff_7d = now - timedelta(days=7)
                cutoff_30d = now - timedelta(days=30)

                weekly_analyses = (
                    db.query(AnomalyAnalysis)
                    .filter(AnomalyAnalysis.analysis_timestamp >= cutoff_7d)
                    .count()
                )

                monthly_analyses = (
                    db.query(AnomalyAnalysis)
                    .filter(AnomalyAnalysis.analysis_timestamp >= cutoff_30d)
                    .count()
                )

                # Get age groups count
                age_groups_count = (
                    db.query(func.count(func.distinct(Drawing.age_years))).scalar() or 0
                )

                # Calculate average processing time from recent analyses
                recent_analyses_with_time = (
                    db.query(AnomalyAnalysis.processing_time)
                    .filter(AnomalyAnalysis.analysis_timestamp >= cutoff_24h)
                    .filter(AnomalyAnalysis.processing_time.isnot(None))
                    .all()
                )

                if recent_analyses_with_time:
                    avg_processing_time = sum(
                        t[0] for t in recent_analyses_with_time
                    ) / len(recent_analyses_with_time)
                else:
                    avg_processing_time = 0.0

                # Active sessions (from in-memory tracking)
                active_sessions = len(self._session_metrics)
                total_page_views = sum(
                    session.page_views for session in self._session_metrics.values()
                )

                # Geographic distribution (from in-memory tracking)
                geographic_distribution = self._get_geographic_distribution()

                # Response time metrics (from in-memory tracking)
                if hasattr(self, "_response_times") and self._response_times:
                    recent_response_times = [
                        rt["duration"]
                        for rt in self._response_times
                        if rt["timestamp"] >= cutoff_24h
                    ]
                    if recent_response_times:
                        avg_response_time = sum(recent_response_times) / len(
                            recent_response_times
                        )
                    else:
                        avg_response_time = 0.0

                    # Count successful vs failed requests
                    total_requests = len(self._response_times)
                    successful_requests = sum(
                        1 for rt in self._response_times if rt["status_code"] < 400
                    )
                    failed_requests = total_requests - successful_requests
                    error_rate = (
                        (failed_requests / total_requests * 100)
                        if total_requests > 0
                        else 0.0
                    )
                else:
                    avg_response_time = 0.0
                    total_requests = 0
                    successful_requests = 0
                    failed_requests = 0
                    error_rate = 0.0

                # System metrics
                import psutil

                memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                cpu_usage_percent = psutil.cpu_percent(interval=0.1)

                # Uptime
                uptime_seconds = (now - self._start_time).total_seconds()
                uptime_percentage = 99.9  # Simplified

                # Return nested structure expected by frontend
                return {
                    "timestamp": now.isoformat(),
                    "database": {
                        "total_drawings": total_drawings,
                        "total_analyses": total_analyses,
                        "anomaly_count": anomaly_count,
                        "normal_count": normal_count,
                        "recent_analyses_count": recent_analyses,
                        "age_groups_count": age_groups_count,
                    },
                    "time_based": {
                        "daily_analyses": recent_analyses,
                        "weekly_analyses": weekly_analyses,
                        "monthly_analyses": monthly_analyses,
                    },
                    "sessions": {
                        "active_sessions": active_sessions,
                        "total_page_views": total_page_views,
                        "total_session_analyses": total_analyses,
                    },
                    "system_health": {
                        "uptime_seconds": int(uptime_seconds),
                        "uptime_percentage": round(uptime_percentage, 1),
                        "total_requests": total_requests,
                        "successful_requests": successful_requests,
                        "failed_requests": failed_requests,
                        "error_rate": round(error_rate, 2),
                        "average_response_time": round(avg_response_time, 3),
                        "memory_usage_mb": round(memory_usage_mb, 2),
                        "cpu_usage_percent": round(cpu_usage_percent, 2),
                        "average_processing_time": round(avg_processing_time, 2),
                    },
                    "geographic": geographic_distribution,
                    "uptime_seconds": int(uptime_seconds),
                    "last_updated": now.isoformat(),
                }

            finally:
                db.close()

    def get_time_series_data(
        self,
        metric_name: str,
        start_date: datetime,
        end_date: datetime,
        period: str = "daily",
    ) -> List[Dict[str, Any]]:
        """
        Get time series data for a specific metric.

        Args:
            metric_name: Name of the metric to retrieve
            start_date: Start date for the time series
            end_date: End date for the time series
            period: Aggregation period ("daily", "weekly", "monthly")

        Returns:
            List of time series data points
        """
        with self._lock:
            # Filter metrics by date range
            filtered_metrics = [
                m
                for m in self._analysis_metrics
                if start_date <= m.timestamp <= end_date
            ]

            # Aggregate by period
            if period == "daily":
                aggregated = self._aggregate_daily(filtered_metrics)
            elif period == "weekly":
                aggregated = self._aggregate_weekly(filtered_metrics)
            elif period == "monthly":
                aggregated = self._aggregate_monthly(filtered_metrics)
            else:
                raise ValueError(f"Invalid period: {period}")

            return aggregated

    def _get_geographic_distribution(self) -> Dict[str, int]:
        """Get geographic distribution of active sessions."""
        distribution = defaultdict(int)

        for session in self._session_metrics.values():
            if session.geographic_info and "country" in session.geographic_info:
                country = session.geographic_info["country"]
                distribution[country] += 1
            else:
                distribution["Unknown"] += 1

        return dict(distribution)

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics."""
        import psutil

        uptime_seconds = int(
            (datetime.now(timezone.utc) - self._start_time).total_seconds()
        )

        # Get recent system health metrics
        recent_health = (
            list(self._system_health_metrics)[-10:]
            if self._system_health_metrics
            else []
        )

        if recent_health:
            avg_cpu = sum(m.cpu_usage for m in recent_health) / len(recent_health)
            avg_memory = sum(m.memory_usage for m in recent_health) / len(recent_health)
            total_errors = sum(m.error_count for m in recent_health)
        else:
            # Get current metrics
            avg_cpu = psutil.cpu_percent(interval=0.1)
            avg_memory = psutil.virtual_memory().percent
            total_errors = 0

        return {
            "uptime_seconds": uptime_seconds,
            "cpu_usage_percent": round(avg_cpu, 2),
            "memory_usage_percent": round(avg_memory, 2),
            "error_count": total_errors,
            "status": "healthy" if avg_cpu < 80 and avg_memory < 80 else "degraded",
        }

    def _get_session_metrics(self) -> Dict[str, Any]:
        """Get current session metrics."""
        with self._lock:
            total_sessions = len(self._session_metrics)
            total_page_views = sum(
                session.page_views for session in self._session_metrics.values()
            )

            # Calculate average session duration for active sessions
            now = datetime.now(timezone.utc)
            if self._session_metrics:
                avg_duration = sum(
                    (now - session.start_time).total_seconds()
                    for session in self._session_metrics.values()
                ) / len(self._session_metrics)
            else:
                avg_duration = 0.0

            return {
                "active_sessions": total_sessions,
                "total_page_views": total_page_views,
                "average_session_duration": round(avg_duration, 2),
            }

    def _aggregate_daily(self, metrics: List[AnalysisMetric]) -> List[Dict[str, Any]]:
        """Aggregate metrics by day."""
        daily_data = defaultdict(
            lambda: {"count": 0, "total_processing_time": 0.0, "errors": 0}
        )

        for metric in metrics:
            date_key = metric.timestamp.date().isoformat()
            daily_data[date_key]["count"] += 1
            daily_data[date_key]["total_processing_time"] += metric.processing_time
            if metric.error_occurred:
                daily_data[date_key]["errors"] += 1

        result = []
        for date_str, data in sorted(daily_data.items()):
            avg_time = (
                data["total_processing_time"] / data["count"]
                if data["count"] > 0
                else 0
            )
            result.append(
                {
                    "date": date_str,
                    "analysis_count": data["count"],
                    "average_processing_time": round(avg_time, 2),
                    "error_count": data["errors"],
                }
            )

        return result

    def _aggregate_weekly(self, metrics: List[AnalysisMetric]) -> List[Dict[str, Any]]:
        """Aggregate metrics by week."""
        # Simplified weekly aggregation
        weekly_data = defaultdict(
            lambda: {"count": 0, "total_processing_time": 0.0, "errors": 0}
        )

        for metric in metrics:
            # Get week start (Monday)
            week_start = metric.timestamp.date() - timedelta(
                days=metric.timestamp.weekday()
            )
            week_key = week_start.isoformat()

            weekly_data[week_key]["count"] += 1
            weekly_data[week_key]["total_processing_time"] += metric.processing_time
            if metric.error_occurred:
                weekly_data[week_key]["errors"] += 1

        result = []
        for week_str, data in sorted(weekly_data.items()):
            avg_time = (
                data["total_processing_time"] / data["count"]
                if data["count"] > 0
                else 0
            )
            result.append(
                {
                    "week_start": week_str,
                    "analysis_count": data["count"],
                    "average_processing_time": round(avg_time, 2),
                    "error_count": data["errors"],
                }
            )

        return result

    def _aggregate_monthly(self, metrics: List[AnalysisMetric]) -> List[Dict[str, Any]]:
        """Aggregate metrics by month."""
        monthly_data = defaultdict(
            lambda: {"count": 0, "total_processing_time": 0.0, "errors": 0}
        )

        for metric in metrics:
            month_key = metric.timestamp.strftime("%Y-%m")
            monthly_data[month_key]["count"] += 1
            monthly_data[month_key]["total_processing_time"] += metric.processing_time
            if metric.error_occurred:
                monthly_data[month_key]["errors"] += 1

        result = []
        for month_str, data in sorted(monthly_data.items()):
            avg_time = (
                data["total_processing_time"] / data["count"]
                if data["count"] > 0
                else 0
            )
            result.append(
                {
                    "month": month_str,
                    "analysis_count": data["count"],
                    "average_processing_time": round(avg_time, 2),
                    "error_count": data["errors"],
                }
            )

        return result

    def _send_cloudwatch_metrics(self, metrics: List[Dict[str, Any]]):
        """
        Send metrics to CloudWatch.

        Args:
            metrics: List of metric data dictionaries
        """
        if not self._cloudwatch_client or not settings.is_production:
            return

        try:
            # Add timestamp to all metrics
            for metric in metrics:
                if "Timestamp" not in metric:
                    metric["Timestamp"] = datetime.now(timezone.utc)

            # Send metrics to CloudWatch
            response = self._cloudwatch_client.put_metric_data(
                Namespace=self.namespace, MetricData=metrics
            )

            if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
                logger.debug(f"Sent {len(metrics)} metrics to CloudWatch")
            else:
                logger.warning(f"CloudWatch metrics response: {response}")

        except ClientError as e:
            logger.error(f"Failed to send metrics to CloudWatch: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending metrics: {e}")

    def cleanup_old_metrics(self, days_to_keep: int = 30):
        """
        Clean up old metrics from memory.

        Args:
            days_to_keep: Number of days of metrics to keep in memory
        """
        with self._lock:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Clean analysis metrics
            original_count = len(self._analysis_metrics)
            self._analysis_metrics = deque(
                [m for m in self._analysis_metrics if m.timestamp > cutoff_date],
                maxlen=self.max_metrics_history,
            )

            # Clean daily stats
            old_dates = [
                date_str
                for date_str in self._daily_stats.keys()
                if datetime.fromisoformat(date_str) < cutoff_date.date()
            ]
            for date_str in old_dates:
                del self._daily_stats[date_str]

            # Clean system health metrics
            self._system_health_metrics = deque(
                [m for m in self._system_health_metrics if m.timestamp > cutoff_date],
                maxlen=1000,
            )

            cleaned_count = original_count - len(self._analysis_metrics)
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old metrics")

    def get_service_stats(self) -> Dict[str, Any]:
        """
        Get usage metrics service statistics.

        Returns:
            Service statistics dictionary
        """
        with self._lock:
            return {
                "analysis_metrics_count": len(self._analysis_metrics),
                "active_sessions_count": len(self._session_metrics),
                "system_health_metrics_count": len(self._system_health_metrics),
                "daily_stats_count": len(self._daily_stats),
                "cloudwatch_enabled": self._cloudwatch_client is not None,
                "service_uptime_seconds": int(
                    (datetime.now(timezone.utc) - self._start_time).total_seconds()
                ),
                "namespace": self.namespace,
            }


# Global usage metrics service instance
_metrics_service: Optional[UsageMetricsService] = None


def get_metrics_service() -> UsageMetricsService:
    """
    Get the global usage metrics service instance.

    Returns:
        UsageMetricsService instance
    """
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = UsageMetricsService()
    return _metrics_service
