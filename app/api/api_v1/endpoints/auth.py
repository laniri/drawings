"""
Authentication API endpoints.

Provides login, logout, and session management endpoints for admin authentication.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.core.config import settings
from app.services.auth_service import get_auth_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize authentication service
auth_service = get_auth_service()


class LoginRequest(BaseModel):
    """Login request model."""

    password: str
    redirect_url: Optional[str] = None


class LoginResponse(BaseModel):
    """Login response model."""

    success: bool
    message: str
    session_token: Optional[str] = None
    redirect_url: Optional[str] = None


class SessionStatus(BaseModel):
    """Session status model."""

    authenticated: bool
    session_info: Optional[dict] = None
    expires_in: Optional[int] = None


def _get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    if request.client:
        return request.client.host

    return "unknown"


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, redirect: Optional[str] = None):
    """
    Display login page.

    Args:
        request: FastAPI request object
        redirect: URL to redirect to after successful login

    Returns:
        HTML login page
    """
    # Check if already authenticated
    session_token = request.cookies.get("session_token")
    if session_token and auth_service.verify_session(session_token):
        redirect_url = redirect or "/api/v1/config/"
        return RedirectResponse(url=redirect_url, status_code=302)

    # Generate login form HTML
    redirect_input = (
        f'<input type="hidden" name="redirect_url" value="{redirect}">'
        if redirect
        else ""
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Admin Login - Children's Drawing Analysis</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .login-container {{
                background: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                width: 100%;
                max-width: 400px;
            }}
            .login-header {{
                text-align: center;
                margin-bottom: 2rem;
            }}
            .login-header h1 {{
                color: #333;
                margin: 0 0 0.5rem 0;
                font-size: 1.8rem;
            }}
            .login-header p {{
                color: #666;
                margin: 0;
                font-size: 0.9rem;
            }}
            .form-group {{
                margin-bottom: 1.5rem;
            }}
            label {{
                display: block;
                margin-bottom: 0.5rem;
                color: #333;
                font-weight: 500;
            }}
            input[type="password"] {{
                width: 100%;
                padding: 0.75rem;
                border: 2px solid #e1e5e9;
                border-radius: 5px;
                font-size: 1rem;
                transition: border-color 0.3s;
                box-sizing: border-box;
            }}
            input[type="password"]:focus {{
                outline: none;
                border-color: #667eea;
            }}
            .login-button {{
                width: 100%;
                padding: 0.75rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
                transition: transform 0.2s;
            }}
            .login-button:hover {{
                transform: translateY(-1px);
            }}
            .login-button:active {{
                transform: translateY(0);
            }}
            .access-info {{
                margin-top: 2rem;
                padding: 1rem;
                background: #f8f9fa;
                border-radius: 5px;
                font-size: 0.85rem;
                color: #666;
            }}
            .access-info h3 {{
                margin: 0 0 0.5rem 0;
                color: #333;
                font-size: 0.9rem;
            }}
            .access-info ul {{
                margin: 0.5rem 0 0 0;
                padding-left: 1.2rem;
            }}
            .access-info li {{
                margin-bottom: 0.3rem;
            }}
            .public-links {{
                text-align: center;
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid #e1e5e9;
            }}
            .public-links a {{
                color: #667eea;
                text-decoration: none;
                margin: 0 0.5rem;
                font-size: 0.9rem;
            }}
            .public-links a:hover {{
                text-decoration: underline;
            }}
            .error {{
                color: #dc3545;
                background: #f8d7da;
                border: 1px solid #f5c6cb;
                padding: 0.75rem;
                border-radius: 5px;
                margin-bottom: 1rem;
                font-size: 0.9rem;
            }}
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="login-header">
                <h1>Admin Login</h1>
                <p>Children's Drawing Anomaly Detection System</p>
            </div>
            
            <form method="post" action="/auth/login">
                {redirect_input}
                <div class="form-group">
                    <label for="password">Admin Password:</label>
                    <input type="password" id="password" name="password" required 
                           placeholder="Enter admin password" autocomplete="current-password">
                </div>
                <button type="submit" class="login-button">Login</button>
            </form>
            
            <div class="access-info">
                <h3>Access Levels:</h3>
                <ul>
                    <li><strong>Public:</strong> Demo, Upload, Documentation</li>
                    <li><strong>Admin:</strong> Dashboard, Configuration, Analysis History</li>
                </ul>
            </div>
            
            <div class="public-links">
                <a href="/">← Back to Demo</a>
                <a href="/api/v1/drawings/upload">Upload Drawing</a>
                <a href="/docs">API Docs</a>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.post("/login")
async def login(
    request: Request,
    response: Response,
    password: str = Form(...),
    redirect_url: Optional[str] = Form(None),
):
    """
    Process login form submission.

    Args:
        request: FastAPI request object
        response: FastAPI response object
        password: Admin password
        redirect_url: URL to redirect to after successful login

    Returns:
        Redirect response or error page
    """
    client_ip = _get_client_ip(request)

    # Authenticate user
    session_token = auth_service.authenticate(password, client_ip)

    if session_token:
        # Set session cookie
        response = RedirectResponse(
            url=redirect_url or "/api/v1/config/", status_code=302
        )

        # Set secure cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=3600,  # 1 hour
            httponly=True,
            secure=settings.is_production,  # HTTPS only in production
            samesite="lax",
        )

        logger.info(f"Successful login from {client_ip}")
        return response
    else:
        # Login failed - show error page
        error_message = "Invalid password. Please try again."

        # Check if rate limited
        auth_service_instance = get_auth_service()
        if auth_service_instance._is_rate_limited(client_ip):
            error_message = "Too many failed attempts. Please try again later."

        redirect_input = (
            f'<input type="hidden" name="redirect_url" value="{redirect_url}">'
            if redirect_url
            else ""
        )

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login Failed - Children's Drawing Analysis</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 0;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                .login-container {{
                    background: white;
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    width: 100%;
                    max-width: 400px;
                }}
                .error {{
                    color: #dc3545;
                    background: #f8d7da;
                    border: 1px solid #f5c6cb;
                    padding: 0.75rem;
                    border-radius: 5px;
                    margin-bottom: 1rem;
                    font-size: 0.9rem;
                }}
                /* ... rest of styles same as login page ... */
            </style>
        </head>
        <body>
            <div class="login-container">
                <div class="login-header">
                    <h1>Admin Login</h1>
                    <p>Children's Drawing Anomaly Detection System</p>
                </div>
                
                <div class="error">
                    {error_message}
                </div>
                
                <form method="post" action="/auth/login">
                    {redirect_input}
                    <div class="form-group">
                        <label for="password">Admin Password:</label>
                        <input type="password" id="password" name="password" required 
                               placeholder="Enter admin password" autocomplete="current-password">
                    </div>
                    <button type="submit" class="login-button">Login</button>
                </form>
                
                <div class="public-links">
                    <a href="/">← Back to Demo</a>
                    <a href="/api/v1/drawings/upload">Upload Drawing</a>
                    <a href="/docs">API Docs</a>
                </div>
            </div>
        </body>
        </html>
        """

        return HTMLResponse(content=html_content, status_code=401)


@router.post("/api/login", response_model=LoginResponse)
async def api_login(request: Request, login_data: LoginRequest):
    """
    API endpoint for programmatic login.

    Args:
        request: FastAPI request object
        login_data: Login request data

    Returns:
        Login response with session token
    """
    client_ip = _get_client_ip(request)

    # Authenticate user
    session_token = auth_service.authenticate(login_data.password, client_ip)

    if session_token:
        logger.info(f"Successful API login from {client_ip}")
        return LoginResponse(
            success=True,
            message="Login successful",
            session_token=session_token,
            redirect_url=login_data.redirect_url,
        )
    else:
        # Check if rate limited
        auth_service_instance = get_auth_service()
        if auth_service_instance._is_rate_limited(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please try again later.",
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password"
        )


@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    Logout user and invalidate session.

    Args:
        request: FastAPI request object
        response: FastAPI response object

    Returns:
        Redirect to home page
    """
    session_token = request.cookies.get("session_token")

    if session_token:
        auth_service.logout(session_token)
        logger.info("User logged out successfully")

    # Clear session cookie and redirect
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(key="session_token")

    return response


@router.get("/status", response_model=SessionStatus)
async def session_status(request: Request):
    """
    Get current session status.

    Args:
        request: FastAPI request object

    Returns:
        Session status information
    """
    session_token = request.cookies.get("session_token")

    if not session_token:
        return SessionStatus(authenticated=False)

    if auth_service.verify_session(session_token):
        session_info = auth_service.get_session_info(session_token)
        return SessionStatus(
            authenticated=True, session_info=session_info, expires_in=3600  # 1 hour
        )
    else:
        return SessionStatus(authenticated=False)


@router.get("/stats")
async def auth_stats(request: Request):
    """
    Get authentication service statistics (admin only).

    Args:
        request: FastAPI request object

    Returns:
        Authentication statistics
    """
    # This endpoint will be protected by the authentication middleware
    # Only authenticated admin users can access it

    stats = auth_service.get_stats()
    return {
        "authentication_stats": stats,
        "protected_routes": [
            "/api/v1/config/*",
            "/api/v1/analysis/stats",
            "/api/v1/analysis/batch",
            "/api/v1/models/*",
            "/api/v1/training/*",
            "/api/v1/backup/*",
        ],
        "public_routes": [
            "/",
            "/health",
            "/api/v1/drawings/*",
            "/api/v1/analysis/analyze/*",
            "/api/v1/documentation/*",
            "/static/*",
        ],
    }
