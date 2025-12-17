# GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}

## Summary
Analyze Specific Age Group

## Description
Analyze data sufficiency for a specific age group.

This endpoint provides detailed analysis of data availability,
quality, and distribution for a single age group.

## Parameters
- **age_min** (path): No description
- **age_max** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/models/data-sufficiency/age-group/{age_min}/{age_max}
```
