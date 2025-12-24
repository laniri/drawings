"""
Property-based test for public access availability.

**Feature: aws-production-deployment, Property 7: Public Access Availability**
**Validates: Requirements 12.1**

Property: For any user request to public resources (demo, upload, documentation), 
access should be granted without authentication requirements.
"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

from app.main import app
from app.core.config import settings


# Test client
client = TestClient(app)

# Define public routes that should be accessible without authentication
PUBLIC_ROUTES = [
    # Demo routes
    "/",
    "/health",
    "/health/detailed",
    "/metrics",
    
    # Upload routes (public access)
    "/api/v1/drawings/upload",
    "/api/v1/drawings/",
    "/api/v1/drawings/stats",
    
    # Documentation routes (public access)
    "/api/v1/documentation/",
    
    # Analysis routes (public access for demo)
    "/api/v1/analysis/analyze/1",  # Example analysis endpoint
    
    # Static file routes (public access)
    "/static/",
]

# Define routes that should explicitly NOT be public (protected routes)
PROTECTED_ROUTES = [
    "/api/v1/config/",
    "/api/v1/config/threshold",
    "/api/v1/config/age-grouping",
    "/api/v1/config/stats",
    "/api/v1/config/subjects",
    "/api/v1/config/subjects/statistics",
    "/api/v1/config/models/subject-aware",
    "/api/v1/config/reset",
    "/api/v1/analysis/stats",  # Dashboard stats should be protected
    "/api/v1/analysis/batch",  # Batch operations should be protected
]

@st.composite
def public_route_data(draw):
    """Generate data for public route testing."""
    route = draw(st.sampled_from(PUBLIC_ROUTES))
    method = draw(st.sampled_from(["GET", "POST"]))  # Most public routes are GET or POST
    return {"route": route, "method": method}

@st.composite
def request_headers(draw):
    """Generate various request headers to test public access."""
    headers = {}
    
    # Add common headers that shouldn't affect public access
    if draw(st.booleans()):
        headers["User-Agent"] = draw(st.text(min_size=1, max_size=100))
    
    if draw(st.booleans()):
        headers["Accept"] = draw(st.sampled_from([
            "text/html", "application/json", "application/xml", "*/*"
        ]))
    
    if draw(st.booleans()):
        headers["Accept-Language"] = draw(st.sampled_from([
            "en-US", "en-GB", "es-ES", "fr-FR", "de-DE"
        ]))
    
    return headers

@st.composite
def client_info(draw):
    """Generate client information for testing public access."""
    return {
        "ip": f"{draw(st.integers(1, 255))}.{draw(st.integers(1, 255))}.{draw(st.integers(1, 255))}.{draw(st.integers(1, 255))}",
        "user_agent": draw(st.text(min_size=1, max_size=200)),
        "referer": draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))
    }


class TestPublicAccessAvailability:
    """Test public access availability properties."""
    
    @given(
        route_data=public_route_data(),
        headers=request_headers()
    )
    @hypothesis_settings(max_examples=50, deadline=5000)
    def test_public_routes_accessible_without_authentication(self, route_data, headers):
        """
        Property: Public routes should be accessible without any authentication.
        
        For any public route and any request headers, access should be granted
        without requiring authentication credentials.
        """
        route = route_data["route"]
        method = route_data["method"]
        
        try:
            if method == "GET":
                response = client.get(route, headers=headers)
            elif method == "POST":
                # For POST routes, provide minimal valid data
                if "upload" in route:
                    # Skip file upload tests in this property test
                    # File uploads require specific multipart data
                    return
                else:
                    response = client.post(route, json={}, headers=headers)
            
            # Public routes should not return authentication errors
            assert response.status_code not in [401, 403], (
                f"Public route {route} should not require authentication, "
                f"got status {response.status_code}"
            )
            
            # Valid responses for public routes include:
            # 200 (OK), 201 (Created), 404 (Not Found), 400 (Bad Request), 
            # 422 (Validation Error), 500 (Server Error)
            # But NOT 401 (Unauthorized) or 403 (Forbidden)
            
        except Exception as e:
            # Ensure the exception is not authentication-related
            error_str = str(e).lower()
            assert "401" not in error_str and "403" not in error_str and "unauthorized" not in error_str, (
                f"Public route {route} failed due to authentication: {e}"
            )
    
    @given(
        client_data=client_info(),
        route=st.sampled_from(PUBLIC_ROUTES[:5])  # Test subset for performance
    )
    @hypothesis_settings(max_examples=20, deadline=5000)
    def test_public_routes_accessible_from_any_client(self, client_data, route):
        """
        Property: Public routes should be accessible from any client.
        
        For any client IP, user agent, or referer, public routes should
        remain accessible without authentication.
        """
        headers = {
            "User-Agent": client_data["user_agent"],
            "X-Forwarded-For": client_data["ip"]
        }
        
        if client_data["referer"]:
            headers["Referer"] = client_data["referer"]
        
        try:
            response = client.get(route, headers=headers)
            
            # Should not be blocked due to authentication
            assert response.status_code not in [401, 403], (
                f"Public route {route} should be accessible from any client, "
                f"got status {response.status_code} for client {client_data['ip']}"
            )
            
        except Exception as e:
            error_str = str(e).lower()
            assert "401" not in error_str and "403" not in error_str, (
                f"Public route {route} failed for client {client_data['ip']}: {e}"
            )
    
    @given(st.sampled_from(PUBLIC_ROUTES))
    @hypothesis_settings(max_examples=20, deadline=5000)
    def test_public_routes_no_session_required(self, route):
        """
        Property: Public routes should not require session management.
        
        For any public route, access should work without session cookies
        or session state.
        """
        # Test without any session cookies
        response = client.get(route)
        
        # Should not fail due to missing session
        assert response.status_code not in [401, 403], (
            f"Public route {route} should not require session, "
            f"got status {response.status_code}"
        )
    
    @given(
        route=st.sampled_from(PUBLIC_ROUTES[:3]),  # Test subset
        concurrent_requests=st.integers(min_value=1, max_value=5)
    )
    @hypothesis_settings(max_examples=10, deadline=5000)
    def test_public_routes_handle_concurrent_access(self, route, concurrent_requests):
        """
        Property: Public routes should handle concurrent unauthenticated access.
        
        For any public route, multiple concurrent requests should be handled
        without authentication requirements affecting performance.
        """
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get(route)
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create and start threads
        threads = []
        for _ in range(concurrent_requests):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=5)
        
        # Check that none failed due to authentication
        for result in results:
            if isinstance(result, int):
                assert result not in [401, 403], (
                    f"Concurrent access to public route {route} failed with auth error: {result}"
                )
            else:
                error_str = str(result).lower()
                assert "401" not in error_str and "403" not in error_str, (
                    f"Concurrent access to public route {route} failed with auth error: {result}"
                )
    
    def test_demo_section_specifically_public(self):
        """
        Property: Demo section should be explicitly public.
        
        The demo section (root path and health endpoints) should be
        accessible without any authentication as per requirements.
        """
        demo_routes = ["/", "/health", "/health/detailed"]
        
        for route in demo_routes:
            response = client.get(route)
            
            # Demo routes must be public
            assert response.status_code not in [401, 403], (
                f"Demo route {route} must be public, got status {response.status_code}"
            )
            
            # Should return some content (not just auth redirect)
            if response.status_code == 200:
                assert len(response.content) > 0, (
                    f"Demo route {route} should return content"
                )
    
    def test_upload_functionality_public(self):
        """
        Property: Upload functionality should be publicly accessible.
        
        The drawing upload endpoints should be accessible without
        authentication to allow public demo usage.
        """
        upload_routes = ["/api/v1/drawings/", "/api/v1/drawings/stats"]
        
        for route in upload_routes:
            response = client.get(route)
            
            # Upload routes should be public
            assert response.status_code not in [401, 403], (
                f"Upload route {route} should be public, got status {response.status_code}"
            )
    
    def test_documentation_routes_public(self):
        """
        Property: Documentation routes should be publicly accessible.
        
        Documentation endpoints should be accessible without authentication
        to provide public information about the system.
        """
        # Test documentation route if it exists
        try:
            response = client.get("/api/v1/documentation/")
            
            # Documentation should be public
            assert response.status_code not in [401, 403], (
                "Documentation routes should be public"
            )
            
        except Exception as e:
            # If route doesn't exist, that's fine, but shouldn't be auth error
            error_str = str(e).lower()
            assert "401" not in error_str and "403" not in error_str, (
                f"Documentation route failed with auth error: {e}"
            )
    
    @given(st.sampled_from(PROTECTED_ROUTES))
    @hypothesis_settings(max_examples=10, deadline=5000)
    def test_protected_routes_are_not_public(self, route):
        """
        Property: Protected routes should NOT be publicly accessible.
        
        This validates that the distinction between public and protected
        routes is properly maintained.
        """
        response = client.get(route)
        
        # Protected routes should require authentication
        assert response.status_code in [401, 403, 302], (
            f"Protected route {route} should not be publicly accessible, "
            f"got status {response.status_code}"
        )
    
    def test_static_files_public_access(self):
        """
        Property: Static files should be publicly accessible.
        
        Static file serving should not require authentication to support
        public demo functionality.
        """
        # Test that static file route doesn't require auth
        # Note: Actual static files may not exist, but the route should be public
        try:
            response = client.get("/static/test.png")
            
            # Should not fail due to authentication (may fail due to file not found)
            assert response.status_code not in [401, 403], (
                "Static file routes should be public"
            )
            
        except Exception as e:
            error_str = str(e).lower()
            assert "401" not in error_str and "403" not in error_str, (
                f"Static file access failed with auth error: {e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])