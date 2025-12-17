# POST /api/v1/training/models/export

## Summary
Export Model From Training Job

## Description
Export trained model from training job in production-compatible format.

This endpoint exports a trained model from a completed training job,
creating a production-ready model file with metadata and validation.

## Parameters
- **training_job_id** (query): No description
- **age_group_min** (query): No description
- **age_group_max** (query): No description
- **export_format** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/training/models/export
```
