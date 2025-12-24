# GET /api/v1/interpretability/examples/{age_group}

## Summary
Get Comparison Examples

## Description
Get comparison examples for educational purposes from a specific age group.

This endpoint provides examples of normal and anomalous drawings
to help users understand typical patterns and variations. Now supports
filtering by subject category for more targeted comparisons.

## Parameters
- **age_group** (path): No description
- **example_type** (query): No description
- **subject** (query): No description
- **limit** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/examples/{age_group}
```
