# GET /api/v1/training/jobs

## Summary
List Training Jobs

## Description
List training jobs with optional filtering.

This endpoint returns a list of training jobs, optionally filtered
by environment (local/sagemaker) and status.

## Parameters
- **environment** (query): No description
- **status** (query): No description
- **limit** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/training/jobs
```
