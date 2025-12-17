"""
FastAPI middleware for error handling, logging, and monitoring.
"""

import logging
import time
import traceback
import uuid
from typing import Callable, Dict, Any
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.exceptions import (
    DrawingAnalysisException,
    ValidationError,
    ImageProcessingError,
    ModelError,
    StorageError,
    ConfigurationError,
    ResourceError,
    validation_error_to_http,
    image_processing_error_to_http,
    model_error_to_http,
    storage_error_to_http,
    configuration_error_to_http,
    resource_error_to_http,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling and logging."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_counts: Dict[str, int] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with comprehensive error handling."""
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started - ID: {request_id}, Method: {request.method}, "
            f"URL: {request.url}, Client: {request.client.host if request.client else 'unknown'}"
        )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Log successful completion
            duration = time.time() - start_time
            logger.info(
                f"Request completed - ID: {request_id}, Status: {response.status_code}, "
                f"Duration: {duration:.3f}s"
            )
            
            return response
            
        except DrawingAnalysisException as e:
            # Handle custom application exceptions
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Log the error
            logger.error(
                f"Application error - ID: {request_id}, Type: {error_type}, "
                f"Message: {e.message}, Duration: {duration:.3f}s, Details: {e.details}"
            )
            
            # Convert to appropriate HTTP exception
            http_exception = self._convert_to_http_exception(e)
            return await self._create_error_response(request_id, http_exception)
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            duration = time.time() - start_time
            logger.warning(
                f"HTTP exception - ID: {request_id}, Status: {e.status_code}, "
                f"Detail: {e.detail}, Duration: {duration:.3f}s"
            )
            
            return await self._create_error_response(request_id, e)
            
        except Exception as e:
            # Handle unexpected exceptions
            duration = time.time() - start_time
            error_type = type(e).__name__
            
            # Track error counts
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Log the full traceback for debugging
            logger.error(
                f"Unexpected error - ID: {request_id}, Type: {error_type}, "
                f"Message: {str(e)}, Duration: {duration:.3f}s\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Return generic internal server error
            http_exception = HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Internal server error occurred",
                    "request_id": request_id,
                    "type": "internal_error"
                }
            )
            
            return await self._create_error_response(request_id, http_exception)
    
    def _convert_to_http_exception(self, error: DrawingAnalysisException) -> HTTPException:
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
                    "type": "application_error"
                }
            )
    
    async def _create_error_response(self, request_id: str, exception: HTTPException) -> JSONResponse:
        """Create consistent error response format."""
        # Ensure detail is a dictionary
        if isinstance(exception.detail, str):
            detail = {
                "error": exception.detail,
                "request_id": request_id,
                "type": "error"
            }
        elif isinstance(exception.detail, dict):
            detail = exception.detail.copy()
            detail["request_id"] = request_id
        else:
            detail = {
                "error": str(exception.detail),
                "request_id": request_id,
                "type": "error"
            }
        
        return JSONResponse(
            status_code=exception.status_code,
            content=detail
        )
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get current error statistics."""
        return self.error_counts.copy()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for detailed request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        # Skip logging for health checks and static files
        if request.url.path in ["/health", "/"] or request.url.path.startswith("/static"):
            return await call_next(request)
        
        # Log request details
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request body size for uploads
        content_length = request.headers.get('content-length')
        if content_length:
            logger.info(f"Request {request_id} - Content-Length: {content_length} bytes")
        
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
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor resources and queue requests if necessary."""
        # Check if we're at capacity
        if self.active_requests >= self.max_concurrent_requests:
            self.queued_requests += 1
            logger.warning(
                f"Request queued - Active: {self.active_requests}, "
                f"Queued: {self.queued_requests}, Max: {self.max_concurrent_requests}"
            )
            
            # Return 503 Service Unavailable with retry information
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": "Service temporarily unavailable due to high load",
                    "active_requests": self.active_requests,
                    "queued_requests": self.queued_requests,
                    "retry_after": 30,  # Suggest retry after 30 seconds
                    "type": "resource_limit"
                },
                headers={"Retry-After": "30"}
            )
        
        # Process the request
        self.active_requests += 1
        try:
            response = await call_next(request)
            return response
        finally:
            self.active_requests -= 1
            if self.queued_requests > 0:
                self.queued_requests -= 1
    
    def get_resource_stats(self) -> Dict[str, int]:
        """Get current resource usage statistics."""
        return {
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "max_concurrent": self.max_concurrent_requests
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
            details={"original_error": str(e), "error_type": type(e).__name__}
        )


def setup_error_monitoring():
    """Set up error monitoring and alerting."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler for errors
    file_handler = logging.FileHandler('app_errors.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logger.info("Error monitoring initialized")