# POST /api/v1/drawings/upload

## Summary
Upload Drawing

## Description
Upload drawing with metadata.

This endpoint accepts multipart form data with an image file and metadata.
The image is validated, preprocessed, and stored along with the metadata.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **201**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/drawings/upload
```
