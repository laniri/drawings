"""
File serving endpoints for S3-stored files.

Provides endpoints to serve files from S3 storage with proper caching headers
to avoid presigned URL expiration issues.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, Response

from app.core.config import settings
from app.services.environment_storage import get_storage_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _generate_etag(file_path: str) -> str:
    """Generate ETag for file caching."""
    try:
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f'"{file_hash}"'
    except Exception:
        return '"unknown"'


def _get_content_type(file_path: str) -> str:
    """Determine content type from file extension."""
    extension = Path(file_path).suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".json": "application/json",
        ".csv": "text/csv",
        ".html": "text/html",
        ".txt": "text/plain",
    }
    return content_types.get(extension, "application/octet-stream")


@router.get("/s3/{file_path:path}")
async def serve_s3_file(file_path: str):
    """
    Serve a file from S3 storage.

    This endpoint downloads files from S3 and serves them with proper caching headers.
    This avoids presigned URL expiration issues and allows CloudFront to cache responses.

    Args:
        file_path: S3 key path (e.g., "drawings/20240108_123456_abc123.png")

    Returns:
        File response with caching headers
    """
    try:
        logger.info(f"[S3 File Serve] Requested path: {file_path}")
        storage_service = get_storage_service()

        # Construct S3 URL
        s3_url = f"s3://{settings.s3_bucket_name}/{file_path}"
        logger.info(f"[S3 File Serve] Constructed S3 URL: {s3_url}")

        # Download file from S3 to local temp storage
        local_path = storage_service.backend.download_to_local(s3_url)
        logger.info(f"[S3 File Serve] Downloaded to local: {local_path}")

        if not local_path or not Path(local_path).exists():
            logger.warning(f"[S3 File Serve] File not found in S3: {s3_url}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}",
            )

        # Determine content type
        content_type = _get_content_type(local_path)
        logger.info(f"[S3 File Serve] Content type: {content_type}")

        # Generate ETag for caching
        etag = _generate_etag(local_path)

        # Serve file with caching headers
        return FileResponse(
            local_path,
            media_type=content_type,
            headers={
                "Cache-Control": "public, max-age=86400, immutable",  # 24 hours
                "ETag": etag,
                "X-Content-Type-Options": "nosniff",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[S3 File Serve] Error serving file {file_path}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve file: {str(e)}",
        )


@router.head("/s3/{file_path:path}")
async def check_s3_file(file_path: str):
    """
    Check if a file exists in S3 storage.

    Args:
        file_path: S3 key path

    Returns:
        Empty response with appropriate status code
    """
    try:
        storage_service = get_storage_service()
        s3_url = f"s3://{settings.s3_bucket_name}/{file_path}"

        # Check if file exists
        file_info = storage_service.backend.get_file_info(s3_url)

        if not file_info or not file_info.get("exists"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="File not found"
            )

        # Return empty response with headers
        return Response(
            status_code=status.HTTP_200_OK,
            headers={
                "Content-Type": _get_content_type(file_path),
                "Cache-Control": "public, max-age=86400",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking S3 file {file_path}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check file",
        )
