# GET /api/v1/analysis/{analysis_id}

## Summary
Get Analysis Result

## Description
Get analysis results by analysis ID.

This endpoint retrieves a complete analysis result including
the drawing information, anomaly analysis, and interpretability
results if available.

## Parameters
- **analysis_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/analysis/{analysis_id}
```
