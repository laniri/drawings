# GET /api/v1/analysis/drawing/{drawing_id}

## Summary
Get Drawing Analyses

## Description
Get all analyses for a specific drawing.

This endpoint returns the analysis history for a drawing,
ordered by most recent first.

## Parameters
- **drawing_id** (path): No description
- **limit** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/analysis/drawing/{drawing_id}
```
