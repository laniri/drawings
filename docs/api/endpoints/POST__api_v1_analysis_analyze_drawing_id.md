# POST /api/v1/analysis/analyze/{drawing_id}

## Summary
Analyze Drawing

## Description
Analyze specific drawing for anomalies.

This endpoint performs anomaly detection on a single drawing,
generating embeddings, computing anomaly scores, and providing
interpretability results for all drawings (both normal and anomalous).

## Parameters
- **drawing_id** (path): No description

## Request Body
Request body required

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/analysis/analyze/{drawing_id}
```
