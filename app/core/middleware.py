"""
FastAPI middleware for error handling, logging, and monitoring.
"""

import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict

import psutil
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.exceptions import (
    ConfigurationError,
    DrawingAnalysisException,
    ImageProcessingError,
    ModelError,
    ResourceError,
    StorageError,
    ValidationError,
    configuration_error_to_http,
    image_processing_error_to_http,
    model_error_to_http,
    resource_error_to_http,
    storage_error_to_http,
    validation_error_to_http,
)
from app.services.monitoring_service import AlertLevel, get_monitoring_service

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling and logging."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_counts: Dict[str, int] = {}
        self.monitoring_service = get_monitoring_service()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error handling."""
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request start with structured logging
        start_time = time.time()
        self.monitoring_service.log_structured(
            level="INFO",
            message=f"Request started: {request.method} {request.url}",
            correlation_id=request_id,
            component="error_middleware",
            operation="request_start",
            details={
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown"),
            },
        )

        try:
            # Process the request
            response = await call_next(request)

            # Log successful completion
            duration = time.time() - start_time
            self.monitoring_service.log_structured(
                level="INFO",
                message=f"Request completed successfully",
                correlation_id=request_id,
                component="error_middleware",
                operation="request_complete",
                details={
                    "status_code": response.status_code,
                    "duration_seconds": duration,
                    "response_size": response.headers.get("content-length", "unknown"),
                },
            )

            # Record performance metrics
            self.monitoring_service.record_performance_metrics(
                {
                    "response_time": duration,
                    "request_count": 1,
                    "success_count": 1 if response.status_code < 400 else 0,
                },
                correlation_id=request_id,
            )

            return response

        except DrawingAnalysisException as e:
            # Handle custom application exceptions
            duration = time.time() - start_time
            error_type = type(e).__name__

            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Log the error with structured logging
            self.monitoring_service.log_error(
                message=f"Application error: {e.message}",
                error_type=error_type,
                correlation_id=request_id,
                details={
                    "error_details": e.details,
                    "duration_seconds": duration,
                    "request_method": request.method,
                    "request_url": str(request.url),
                },
            )

            # Send alert if appropriate
            if self.monitoring_service.should_send_alert(error_type, request_id):
                self.monitoring_service.send_alert(
                    level=AlertLevel.ERROR,
                    message=f"Application error: {error_type} - {e.message}",
                    correlation_id=request_id,
                    details={
                        "error_type": error_type,
                        "error_message": e.message,
                        "error_details": e.details,
                        "request_info": {
                            "method": request.method,
                            "url": str(request.url),
                            "duration": duration,
                        },
                    },
                )

            # Record error metrics
            self.monitoring_service.record_performance_metrics(
                {"error_count": 1, "response_time": duration}, correlation_id=request_id
            )

            # Convert to appropriate HTTP exception
            http_exception = self._convert_to_http_exception(e)
            return await self._create_error_response(request_id, http_exception)

        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            duration = time.time() - start_time

            self.monitoring_service.log_structured(
                level="WARNING",
                message=f"HTTP exception: {e.detail}",
                correlation_id=request_id,
                component="error_middleware",
                operation="http_exception",
                details={
                    "status_code": e.status_code,
                    "detail": e.detail,
                    "duration_seconds": duration,
                },
            )

            return await self._create_error_response(request_id, e)

        except Exception as e:
            # Handle unexpected exceptions
            duration = time.time() - start_time
            error_type = type(e).__name__

            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

            # Log the error with full traceback
            self.monitoring_service.log_error(
                message=f"Unexpected error: {str(e)}",
                error_type=error_type,
                correlation_id=request_id,
                details={
                    "traceback": traceback.format_exc(),
                    "duration_seconds": duration,
                    "request_method": request.method,
                    "request_url": str(request.url),
                },
            )

            # Send critical alert for unexpected errors
            self.monitoring_service.send_alert(
                level=AlertLevel.CRITICAL,
                message=f"Unexpected system error: {error_type}",
                correlation_id=request_id,
                details={
                    "error_type": error_type,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "request_info": {
                        "method": request.method,
                        "url": str(request.url),
                        "duration": duration,
                    },
                },
            )

            # Record critical error metrics
            self.monitoring_service.record_performance_metrics(
                {
                    "critical_error_count": 1,
                    "error_count": 1,
                    "response_time": duration,
                },
                correlation_id=request_id,
            )

            # Return generic internal server error
            http_exception = HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Internal server error occurred",
                    "request_id": request_id,
                    "type": "internal_error",
                },
            )

            return await self._create_error_response(request_id, http_exception)

    def _convert_to_http_exception(
        self, error: DrawingAnalysisException
    ) -> HTTPException:
        """Convert custom exceptions to HTTP exceptions."""
        if isinstance(error, ValidationError):
            return validation_error_to_http(error)
        elif isinstance(error, ImageProcessingError):
            return image_processing_error_to_http(error)
        elif isinstance(error, ModelError):
            return model_error_to_http(error)
        elif isinstance(error, StorageError):
            return storage_error_to_http(error)
        elif isinstance(error, ConfigurationError):
            return configuration_error_to_http(error)
        elif isinstance(error, ResourceError):
            return resource_error_to_http(error)
        else:
            # Generic application error
            return HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": error.message,
                    "details": error.details,
                    "type": "application_error",
                },
            )

    async def _create_error_response(
        self, request_id: str, exception: HTTPException
    ) -> JSONResponse:
        """Create consistent error response format."""
        # Ensure detail is a dictionary
        if isinstance(exception.detail, str):
            detail = {
                "error": exception.detail,
                "request_id": request_id,
                "type": "error",
            }
        elif isinstance(exception.detail, dict):
            detail = exception.detail.copy()
            detail["request_id"] = request_id
        else:
            detail = {
                "error": str(exception.detail),
                "request_id": request_id,
                "type": "error",
            }

        return JSONResponse(status_code=exception.status_code, content=detail)

    def get_error_stats(self) -> Dict[str, int]:
        """Get current error statistics."""
        return self.error_counts.copy()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        # Skip logging for health checks and static files
        if request.url.path in ["/health", "/"] or request.url.path.startswith(
            "/static"
        ):
            return await call_next(request)

        # Log request details
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request body size for uploads
        content_length = request.headers.get("content-length")
        if content_length:
            logger.info(
                f"Request {request_id} - Content-Length: {content_length} bytes"
            )

        # Process request
        response = await call_next(request)

        # Log response details
        logger.info(
            f"Response {request_id} - Status: {response.status_code}, "
            f"Content-Type: {response.headers.get('content-type', 'unknown')}"
        )

        return response


class ResourceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring system resources and request queuing."""

    def __init__(self, app: ASGIApp, max_concurrent_requests: int = 10):
        super().__init__(app)
        self.max_concurrent_requests = max_concurrent_requests
        self.active_requests = 0
        self.queued_requests = 0
        self.monitoring_service = get_monitoring_service()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor resources and queue requests if necessary."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

        # Check if we're at capacity
        if self.active_requests >= self.max_concurrent_requests:
            self.queued_requests += 1

            # Log resource limit hit
            self.monitoring_service.log_structured(
                level="WARNING",
                message="Request queued due to resource limits",
                correlation_id=request_id,
                component="resource_middleware",
                operation="request_queue",
                details={
                    "active_requests": self.active_requests,
                    "queued_requests": self.queued_requests,
                    "max_concurrent": self.max_concurrent_requests,
                },
            )

            # Send alert if queue is getting large
            if self.queued_requests > 5:
                self.monitoring_service.send_alert(
                    level=AlertLevel.WARNING,
                    message=f"High request queue: {self.queued_requests} requests queued",
                    correlation_id=request_id,
                    details={
                        "active_requests": self.active_requests,
                        "queued_requests": self.queued_requests,
                        "max_concurrent": self.max_concurrent_requests,
                    },
                )

            # Return 503 Service Unavailable with retry information
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service temporarily unavailable due to high load",
                    "active_requests": self.active_requests,
                    "queued_requests": self.queued_requests,
                    "retry_after": 30,  # Suggest retry after 30 seconds
                    "type": "resource_limit",
                },
                headers={"Retry-After": "30"},
            )

        # Process the request
        self.active_requests += 1

        # Collect system metrics before processing
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        try:
            response = await call_next(request)

            # Record system performance metrics
            self.monitoring_service.record_performance_metrics(
                {
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory_percent,
                    "active_requests": self.active_requests,
                    "queued_requests": self.queued_requests,
                },
                correlation_id=request_id,
            )

            # Check for performance threshold violations
            self._check_performance_thresholds(cpu_percent, memory_percent, request_id)

            return response
        finally:
            self.active_requests -= 1
            if self.queued_requests > 0:
                self.queued_requests -= 1

    def _check_performance_thresholds(
        self, cpu_percent: float, memory_percent: float, correlation_id: str
    ):
        """Check if performance metrics exceed thresholds and send alerts."""
        thresholds = self.monitoring_service.performance_thresholds

        if cpu_percent > thresholds.get("cpu_usage", 80.0):
            self.monitoring_service.send_performance_alert(
                metric_name="cpu_usage",
                current_value=cpu_percent,
                threshold=thresholds["cpu_usage"],
                correlation_id=correlation_id,
            )

        if memory_percent > thresholds.get("memory_usage", 85.0):
            self.monitoring_service.send_performance_alert(
                metric_name="memory_usage",
                current_value=memory_percent,
                threshold=thresholds["memory_usage"],
                correlation_id=correlation_id,
            )

    def get_resource_stats(self) -> Dict[str, int]:
        """Get current resource usage statistics."""
        return {
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "max_concurrent": self.max_concurrent_requests,
        }


@asynccontextmanager
async def error_context(operation_name: str):
    """Context manager for consistent error handling in services."""
    try:
        logger.info(f"Starting operation: {operation_name}")
        yield
        logger.info(f"Completed operation: {operation_name}")
    except DrawingAnalysisException:
        logger.error(f"Application error in operation: {operation_name}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in operation {operation_name}: {str(e)}")
        raise DrawingAnalysisException(
            message=f"Operation '{operation_name}' failed unexpectedly",
            details={"original_error": str(e), "error_type": type(e).__name__},
        )


def setup_error_monitoring():
    """Set up error monitoring and alerting."""
    # Get monitoring service instance
    monitoring_service = get_monitoring_service()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler for errors
    file_handler = logging.FileHandler("app_errors.log")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Initialize monitoring service components
    monitoring_service.create_cloudwatch_dashboard()
    monitoring_service.setup_cost_monitoring()

    # Log initialization
    monitoring_service.log_structured(
        level="INFO",
        message="Error monitoring and alerting system initialized",
        component="middleware",
        operation="setup_monitoring",
        details={
            "cloudwatch_enabled": monitoring_service._cloudwatch_client is not None,
            "sns_enabled": monitoring_service._sns_client is not None,
            "cost_threshold": monitoring_service.cost_threshold,
        },
    )

    logger.info("Error monitoring initialized with CloudWatch and SNS integration")
