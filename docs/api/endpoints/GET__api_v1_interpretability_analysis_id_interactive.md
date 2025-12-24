# GET /api/v1/interpretability/{analysis_id}/interactive

## Summary
Get Interactive Interpretability

## Description
Get interactive saliency data with hoverable regions and click explanations.

This endpoint provides enhanced interpretability data that supports
interactive user interfaces with hover explanations and click-to-zoom functionality.

## Parameters
- **analysis_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/{analysis_id}/interactive
```
