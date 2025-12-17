# GET /api/v1/health/component/{component_name}

## Summary
Component-specific health check

## Description
Get health status for a specific component.

## Parameters
- **component_name** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/health/component/{component_name}
```
