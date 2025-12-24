"""
Metrics collection middleware for application-level monitoring.

This middleware automatically collects and reports application metrics
to CloudWatch for monitoring and alerting purposes.
"""

import time
import uuid
from typing import Callable, Dict, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.services.monitoring_service import get_monitoring_service


class MetricsCollectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting application-level metrics.
    
    Automatically tracks:
    - Request counts and response times
    - Analysis operations and processing times
    - Error rates and types
    - User session metrics
    - Geographic distribution (if available)
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.monitoring_service = get_monitoring_service()
        
        # Metrics tracking
        self.request_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.analysis_counts = defaultdict(int)
        self.session_tracking = {}
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.total_requests = 0
        self.total_errors = 0
        self.total_analyses = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Collect metrics for each request."""
        # Generate correlation ID if not present
        correlation_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Track request start
        start_time = time.time()
        self.total_requests += 1
        
        # Extract session information
        session_id = self._extract_session_id(request)
        client_ip = self._get_client_ip(request)
        
        # Track session
        if session_id:
            self.session_tracking[session_id] = {
                'last_seen': datetime.utcnow(),
                'client_ip': client_ip,
                'request_count': self.session_tracking.get(session_id, {}).get('request_count', 0) + 1
            }
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Track request by endpoint
            endpoint = self._get_endpoint_name(request)
            self.request_counts[endpoint] += 1
            
            # Track analysis operations
            if self._is_analysis_request(request):
                self.total_analyses += 1
                self.analysis_counts[endpoint] += 1
                
                # Record analysis metrics
                self.monitoring_service.record_performance_metrics({
                    "analysis_count": 1,
                    "analysis_processing_time": response_time
                }, correlation_id=correlation_id)
            
            # Record general request metrics
            self.monitoring_service.record_performance_metrics({
                "request_count": 1,
                "response_time": response_time,
                "status_code": response.status_code
            }, correlation_id=correlation_id)
            
            # Log successful request
            self.monitoring_service.log_structured(
                level="DEBUG",
                message=f"Request metrics collected: {endpoint}",
                correlation_id=correlation_id,
                component="metrics_middleware",
                operation="collect_metrics",
                details={
                    "endpoint": endpoint,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "session_id": session_id,
                    "client_ip": client_ip
                }
            )
            
            return response
            
        except Exception as e:
            # Track error
            self.total_errors += 1
            error_type = type(e).__name__
            self.error_counts[error_type] += 1
            
            # Record error metrics
            response_time = time.time() - start_time
            self.monitoring_service.record_performance_metrics({
                "error_count": 1,
                "response_time": response_time
            }, correlation_id=correlation_id)
            
            # Re-raise the exception
            raise
    
    def _extract_session_id(self, request: Request) -> str:
        """Extract session ID from request."""
        # Try to get session ID from various sources
        session_id = None
        
        # Check cookies
        if 'session_id' in request.cookies:
            session_id = request.cookies['session_id']
        
        # Check headers
        elif 'X-Session-ID' in request.headers:
            session_id = request.headers['X-Session-ID']
        
        # Generate new session ID if none found
        if not session_id:
            session_id = str(uuid.uuid4())
        
        return session_id
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (for load balancers)
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_endpoint_name(self, request: Request) -> str:
        """Get normalized endpoint name for metrics."""
        path = request.url.path
        method = request.method
        
        # Normalize paths with IDs
        normalized_path = path
        
        # Replace common ID patterns
        import re
        normalized_path = re.sub(r'/\d+', '/{id}', normalized_path)
        normalized_path = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', normalized_path)
        
        return f"{method} {normalized_path}"
    
    def _is_analysis_request(self, request: Request) -> bool:
        """Check if request is an analysis operation."""
        analysis_endpoints = [
            '/api/v1/analysis',
            '/api/v1/drawings/analyze',
            '/api/v1/batch/analyze'
        ]
        
        return any(request.url.path.startswith(endpoint) for endpoint in analysis_endpoints)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        uptime = datetime.utcnow() - self.start_time
        
        # Calculate averages
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        error_rate = (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0
        
        # Active sessions (last 30 minutes)
        cutoff_time = datetime.utcnow() - timedelta(minutes=30)
        active_sessions = sum(
            1 for session_data in self.session_tracking.values()
            if session_data['last_seen'] > cutoff_time
        )
        
        # Geographic distribution (simplified by IP)
        geographic_distribution = defaultdict(int)
        for session_data in self.session_tracking.values():
            # Simple geographic grouping by IP prefix
            ip = session_data['client_ip']
            if ip != "unknown":
                # Group by first two octets for privacy
                ip_prefix = '.'.join(ip.split('.')[:2]) + '.x.x'
                geographic_distribution[ip_prefix] += 1
        
        return {
            "uptime_seconds": int(uptime.total_seconds()),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_analyses": self.total_analyses,
            "error_rate_percent": error_rate,
            "average_response_time": avg_response_time,
            "active_sessions": active_sessions,
            "total_sessions": len(self.session_tracking),
            "request_counts_by_endpoint": dict(self.request_counts),
            "error_counts_by_type": dict(self.error_counts),
            "analysis_counts_by_endpoint": dict(self.analysis_counts),
            "geographic_distribution": dict(geographic_distribution),
            "recent_response_times": list(self.response_times)[-10:]  # Last 10 response times
        }
    
    def cleanup_old_sessions(self, hours_to_keep: int = 24):
        """Clean up old session data."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_to_keep)
        
        old_sessions = [
            session_id for session_id, session_data in self.session_tracking.items()
            if session_data['last_seen'] < cutoff_time
        ]
        
        for session_id in old_sessions:
            del self.session_tracking[session_id]
        
        if old_sessions:
            self.monitoring_service.log_structured(
                level="INFO",
                message=f"Cleaned up {len(old_sessions)} old sessions",
                component="metrics_middleware",
                operation="cleanup_sessions",
                details={"cleaned_sessions": len(old_sessions)}
            )