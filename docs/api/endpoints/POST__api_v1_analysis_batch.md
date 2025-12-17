# POST /api/v1/analysis/batch

## Summary
Batch Analyze

## Description
Batch analyze multiple drawings.

This endpoint accepts a list of drawing IDs and processes them
in the background, returning a batch ID for progress tracking.

## Parameters
No parameters

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/analysis/batch
```
