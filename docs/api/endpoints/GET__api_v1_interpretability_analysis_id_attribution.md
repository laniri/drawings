# GET /api/v1/interpretability/{analysis_id}/attribution

## Summary
Get Anomaly Attribution

## Description
Get detailed anomaly attribution breakdown (age vs subject vs visual).

This endpoint provides detailed information about what contributed
to the anomaly detection: age-related factors, subject-specific factors,
or visual characteristics.

## Parameters
- **analysis_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/{analysis_id}/attribution
```
