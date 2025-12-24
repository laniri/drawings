"""
Authentication middleware for protecting admin routes.

This middleware implements session-based authentication with route-based
access control, distinguishing between public and protected routes.
"""

import logging
from typing import Callable, List, Set
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.services.auth_service import get_auth_service

logger = logging.getLogger(__name__)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for session-based authentication and access control.
    
    Protects admin routes while allowing public access to demo, upload,
    and documentation endpoints.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.auth_service = get_auth_service()
        
        # Define protected route patterns
        self.protected_patterns = {
            "/api/v1/config",
            "/api/v1/analysis/stats",
            "/api/v1/analysis/batch",
            "/api/v1/models",
            "/api/v1/training",
            "/api/v1/backup",
            "/api/v1/database",
        }
        
        # Define explicitly public routes
        self.public_routes = {
            "/",
            "/health",
            "/health/detailed",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        }
        
        # Define public route patterns
        self.public_patterns = {
            "/static",
            "/api/v1/drawings",
            "/api/v1/analysis/analyze",
            "/api/v1/analysis/embeddings",
            "/api/v1/analysis/drawing",
            "/api/v1/interpretability",
            "/api/v1/documentation",
        }
        
        # Authentication routes
        self.auth_routes = {
            "/auth/login",
            "/auth/logout",
            "/auth/status",
        }
    
    def _is_protected_route(self, path: str) -> bool:
        """
        Check if a route requires authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if route is protected, False otherwise
        """
        # Remove query parameters
        path = path.split('?')[0]
        
        # Check exact matches for public routes
        if path in self.public_routes:
            return False
        
        # Check exact matches for auth routes
        if path in self.auth_routes:
            return False
        
        # Check public patterns
        for pattern in self.public_patterns:
            if path.startswith(pattern):
                return False
        
        # Check protected patterns
        for pattern in self.protected_patterns:
            if path.startswith(pattern):
                return True
        
        # Default to public for unmatched routes
        return False
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Client IP address
        """
        # Check for forwarded IP (from load balancer/proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_session_token(self, request: Request) -> str:
        """
        Extract session token from request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Session token or empty string if not found
        """
        # Check for session cookie
        session_token = request.cookies.get("session_token")
        if session_token:
            return session_token
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        return ""
    
    def _create_login_redirect(self, request: Request) -> Response:
        """
        Create redirect response to login page.
        
        Args:
            request: FastAPI request object
            
        Returns:
            Redirect response
        """
        # For API requests, return JSON error
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "Authentication required",
                    "message": "This endpoint requires admin authentication",
                    "login_url": "/auth/login",
                    "type": "authentication_required"
                }
            )
        
        # For web requests, redirect to login page
        login_url = f"/auth/login?redirect={request.url.path}"
        return RedirectResponse(url=login_url, status_code=302)
    
    def _create_access_denied_response(self, request: Request, reason: str) -> Response:
        """
        Create access denied response.
        
        Args:
            request: FastAPI request object
            reason: Reason for access denial
            
        Returns:
            Access denied response
        """
        # For API requests, return JSON error
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Access denied",
                    "message": reason,
                    "type": "access_denied"
                }
            )
        
        # For web requests, return HTML error page
        return Response(
            content=f"<html><body><h1>Access Denied</h1><p>{reason}</p></body></html>",
            status_code=status.HTTP_403_FORBIDDEN,
            media_type="text/html"
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with authentication checks.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/handler
            
        Returns:
            Response object
        """
        path = request.url.path
        client_ip = self._get_client_ip(request)
        
        # Skip authentication for public routes
        if not self._is_protected_route(path):
            return await call_next(request)
        
        # Handle authentication routes
        if path in self.auth_routes:
            return await call_next(request)
        
        # Get session token
        session_token = self._get_session_token(request)
        
        if not session_token:
            logger.info(f"Authentication required for {path} from {client_ip}")
            return self._create_login_redirect(request)
        
        # Verify session
        if not self.auth_service.verify_session(session_token):
            logger.warning(f"Invalid session token for {path} from {client_ip}")
            return self._create_login_redirect(request)
        
        # Add session info to request state
        session_info = self.auth_service.get_session_info(session_token)
        if session_info:
            request.state.session = session_info
            request.state.is_authenticated = True
            request.state.is_admin = session_info.get("is_admin", False)
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Log successful access
            logger.debug(f"Authenticated access to {path} from {client_ip}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing authenticated request to {path}: {e}")
            raise
    
    def get_protected_routes(self) -> List[str]:
        """
        Get list of protected route patterns.
        
        Returns:
            List of protected route patterns
        """
        return list(self.protected_patterns)
    
    def get_public_routes(self) -> List[str]:
        """
        Get list of public route patterns.
        
        Returns:
            List of public route patterns
        """
        return list(self.public_routes) + list(self.public_patterns)