# GET /api/v1/interpretability/{analysis_id}/simplified

## Summary
Get Simplified Explanation

## Description
Get simplified, non-technical explanations suitable for educators and parents.

This endpoint provides explanations adapted for different user roles
with accessible language and clear recommendations.

## Parameters
- **analysis_id** (path): No description
- **user_role** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/{analysis_id}/simplified
```
