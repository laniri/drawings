"""
Authentication service for AWS production deployment.

This service provides session-based authentication using AWS Secrets Manager
for secure password storage and session management with timeouts.
"""

import hashlib
import json
import logging
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

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


class AuthenticationService:
    """
    Authentication service with AWS Secrets Manager integration.

    Provides session-based authentication with secure password storage,
    session management, and timeout enforcement.
    """

    def __init__(self):
        self.secret_name = "drawing-analysis-admin-password"
        self.session_timeout = 3600  # 1 hour default
        self.max_login_attempts = 5
        self.rate_limit_window = 300  # 5 minutes

        # In-memory session storage (in production, use Redis or database)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._login_attempts: Dict[str, list] = {}

        # Initialize AWS client
        self._secrets_client = None
        self._admin_password_hash = None

        # Initialize service
        self._initialize()

    def _initialize(self):
        """Initialize the authentication service."""
        try:
            if settings.is_production:
                # In production, use AWS Secrets Manager
                self._secrets_client = boto3.client(
                    "secretsmanager", region_name=settings.aws_region or "us-east-1"
                )
                logger.info("Initialized AWS Secrets Manager client for authentication")
            else:
                # In local development, use environment variable or default
                logger.info("Using local authentication configuration")

        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Failed to initialize AWS Secrets Manager: {e}")
            logger.info("Falling back to local authentication")

    def _get_admin_password(self) -> str:
        """
        Retrieve admin password from AWS Secrets Manager or local config.

        Returns:
            The admin password string

        Raises:
            ConfigurationError: If password cannot be retrieved
        """
        if settings.is_production and self._secrets_client:
            try:
                response = self._secrets_client.get_secret_value(
                    SecretId=self.secret_name
                )

                secret_data = json.loads(response["SecretString"])
                password = secret_data.get("admin_password")

                if not password:
                    raise ConfigurationError(
                        "Admin password not found in AWS Secrets Manager",
                        {"secret_name": self.secret_name},
                    )

                logger.info("Retrieved admin password from AWS Secrets Manager")
                return password

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "ResourceNotFoundException":
                    logger.error(
                        f"Secret {self.secret_name} not found in AWS Secrets Manager"
                    )
                    raise ConfigurationError(
                        f"Admin password secret not found: {self.secret_name}",
                        {"error_code": error_code},
                    )
                else:
                    logger.error(f"Failed to retrieve admin password: {e}")
                    raise ConfigurationError(
                        "Failed to retrieve admin password from AWS Secrets Manager",
                        {"error": str(e)},
                    )
        else:
            # Local development - use environment variable or default
            password = getattr(settings, "ADMIN_PASSWORD", None)
            if not password:
                # Default password for local development
                password = "admin123"
                logger.warning("Using default admin password for local development")
            else:
                logger.info("Using admin password from environment variable")

            return password

    def _hash_password(self, password: str) -> str:
        """
        Hash password using SHA-256 with salt.

        Args:
            password: Plain text password

        Returns:
            Hashed password string
        """
        # Use a fixed salt for simplicity (in production, use per-user salts)
        salt = "drawing_analysis_salt_2024"
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def _is_rate_limited(self, client_ip: str) -> bool:
        """
        Check if client IP is rate limited for login attempts.

        Args:
            client_ip: Client IP address

        Returns:
            True if rate limited, False otherwise
        """
        now = time.time()

        # Clean old attempts
        if client_ip in self._login_attempts:
            self._login_attempts[client_ip] = [
                attempt_time
                for attempt_time in self._login_attempts[client_ip]
                if now - attempt_time < self.rate_limit_window
            ]

        # Check if rate limited
        attempts = self._login_attempts.get(client_ip, [])
        return len(attempts) >= self.max_login_attempts

    def _record_login_attempt(self, client_ip: str):
        """
        Record a failed login attempt for rate limiting.

        Args:
            client_ip: Client IP address
        """
        now = time.time()
        if client_ip not in self._login_attempts:
            self._login_attempts[client_ip] = []

        self._login_attempts[client_ip].append(now)

    def create_session(self, client_ip: str = "unknown") -> str:
        """
        Create a new session.

        Args:
            client_ip: Client IP address

        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        session_data = {
            "created_at": datetime.now(timezone.utc),
            "last_accessed": datetime.now(timezone.utc),
            "client_ip": client_ip,
            "is_admin": True,
        }

        self._sessions[session_token] = session_data
        logger.info(f"Created new session from {client_ip}")
        return session_token

    def authenticate(self, password: str, client_ip: str = "unknown") -> Optional[str]:
        """
        Authenticate user with password.

        Args:
            password: Password to verify
            client_ip: Client IP address for rate limiting

        Returns:
            Session token if authentication successful, None otherwise
        """
        # Check rate limiting
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limited login attempt from {client_ip}")
            return None

        try:
            # Get admin password
            admin_password = self._get_admin_password()

            # Verify password
            if password == admin_password:
                # Create session
                session_token = self.create_session(client_ip)
                logger.info(f"Successful authentication from {client_ip}")
                return session_token
            else:
                # Record failed attempt
                self._record_login_attempt(client_ip)
                logger.warning(f"Failed authentication attempt from {client_ip}")
                return None

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None

    def verify_session(self, session_token: str) -> bool:
        """
        Verify if session token is valid and not expired.

        Args:
            session_token: Session token to verify

        Returns:
            True if session is valid, False otherwise
        """
        if not session_token or session_token not in self._sessions:
            return False

        session_data = self._sessions[session_token]
        now = datetime.now(timezone.utc)

        # Check if session is expired
        last_accessed = session_data["last_accessed"]
        if (now - last_accessed).total_seconds() > self.session_timeout:
            # Remove expired session
            del self._sessions[session_token]
            logger.info("Removed expired session")
            return False

        # Update last accessed time
        session_data["last_accessed"] = now
        return True

    def logout(self, session_token: str) -> bool:
        """
        Logout user by invalidating session.

        Args:
            session_token: Session token to invalidate

        Returns:
            True if logout successful, False otherwise
        """
        if session_token in self._sessions:
            del self._sessions[session_token]
            logger.info("User logged out successfully")
            return True

        return False

    def get_session_info(self, session_token: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.

        Args:
            session_token: Session token

        Returns:
            Session information dict or None if invalid
        """
        if self.verify_session(session_token):
            session_data = self._sessions[session_token].copy()
            # Convert datetime to string for JSON serialization
            session_data["created_at"] = session_data["created_at"].isoformat()
            session_data["last_accessed"] = session_data["last_accessed"].isoformat()
            return session_data

        return None

    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        now = datetime.now(timezone.utc)
        expired_tokens = []

        for token, session_data in self._sessions.items():
            last_accessed = session_data["last_accessed"]
            if (now - last_accessed).total_seconds() > self.session_timeout:
                expired_tokens.append(token)

        for token in expired_tokens:
            del self._sessions[token]

        if expired_tokens:
            logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get authentication service statistics.

        Returns:
            Statistics dictionary
        """
        self.cleanup_expired_sessions()

        return {
            "active_sessions": len(self._sessions),
            "session_timeout": self.session_timeout,
            "max_login_attempts": self.max_login_attempts,
            "rate_limit_window": self.rate_limit_window,
            "rate_limited_ips": len(
                [
                    ip
                    for ip, attempts in self._login_attempts.items()
                    if len(attempts) >= self.max_login_attempts
                ]
            ),
        }


# Global authentication service instance
_auth_service: Optional[AuthenticationService] = None


def get_auth_service() -> AuthenticationService:
    """
    Get the global authentication service instance.

    Returns:
        AuthenticationService instance
    """
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service
