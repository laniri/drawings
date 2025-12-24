# GET /api/v1/interpretability/{analysis_id}/confidence

## Summary
Get Confidence Metrics

## Description
Get confidence metrics and reliability scores for interpretability results.

This endpoint provides detailed confidence information to help users
assess the trustworthiness of the analysis and interpretations.

## Parameters
- **analysis_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/{analysis_id}/confidence
```
