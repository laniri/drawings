# GET /api/v1/models/age-groups

## Summary
List Age Group Models

## Description
List available age group models.

This endpoint returns all age group models with their status,
sample counts, and threshold information.

## Parameters
- **active_only** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/models/age-groups
```
