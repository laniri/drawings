# POST /api/v1/interpretability/{analysis_id}/export

## Summary
Export Interpretability Results

## Description
Export interpretability results in multiple formats (PDF, PNG, CSV, JSON, HTML).

This endpoint allows users to export comprehensive interpretability reports
with customizable options for different use cases.

## Parameters
- **analysis_id** (path): No description

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/interpretability/{analysis_id}/export
```
