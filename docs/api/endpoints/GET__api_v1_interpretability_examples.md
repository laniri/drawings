# GET /api/v1/interpretability/examples

## Summary
Get Example Patterns

## Description
Get example interpretation patterns for educational purposes.

This endpoint provides a gallery of common interpretation patterns
with explanations suitable for different user roles.

## Parameters
- **age_group** (query): No description
- **user_role** (query): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/interpretability/examples
```
