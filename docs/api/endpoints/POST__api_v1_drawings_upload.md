# POST /api/v1/drawings/upload

## Summary
Upload Drawing

## Description
Upload drawing with metadata.

This endpoint accepts multipart form data with an image file and metadata.
The image is validated, preprocessed, and stored using the environment-aware storage service
that automatically handles both local filesystem and AWS S3 storage based on deployment configuration.

## Storage Behavior
- **Local Development**: Files stored directly to local filesystem
- **Production**: Files stored to AWS S3 with secure presigned URL access
- **Environment Detection**: Automatic based on APP_ENVIRONMENT and AWS_REGION configuration

## Parameters
### Form Data Parameters
- **file** (required): Drawing image file (PNG, JPEG, BMP, max 10MB)
- **age_years** (required): Child's age in years (2.0-18.0)
- **subject** (optional): Drawing subject category (64 predefined categories supported)
- **expert_label** (optional): Expert assessment label
- **drawing_tool** (optional): Drawing tool used
- **prompt** (optional): Drawing prompt given to child

## Request Body
Multipart form data required

## Responses
- **201**: Successful Response - Returns DrawingResponse with metadata and file information
- **400**: Bad Request - Invalid image format, metadata, or file validation failure
- **413**: Request Entity Too Large - File exceeds maximum size limit
- **422**: Validation Error - Invalid form data or missing required fields
- **500**: Internal Server Error - Storage or database operation failure

## Example
```http
POST /api/v1/drawings/upload
Content-Type: multipart/form-data

--boundary
Content-Disposition: form-data; name="file"; filename="child_drawing.png"
Content-Type: image/png

[binary image data]
--boundary
Content-Disposition: form-data; name="age_years"

5.5
--boundary
Content-Disposition: form-data; name="subject"

person
--boundary--
```

## Response Example
```json
{
  "id": 123,
  "filename": "child_drawing_20241216_123456.png",
  "file_path": "drawings/child_drawing_20241216_123456.png",
  "age_years": 5.5,
  "subject": "person",
  "expert_label": null,
  "drawing_tool": null,
  "prompt": null,
  "upload_timestamp": "2024-12-16T12:34:56.789Z"
}
```
