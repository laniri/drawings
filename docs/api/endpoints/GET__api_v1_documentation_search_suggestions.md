# GET /api/v1/documentation/search/suggestions

## Summary
Get Search Suggestions

## Description
Get search suggestions for autocomplete.

Provides intelligent search suggestions based on indexed content
and common search patterns.

## Parameters
- **query** (query): Partial query for suggestions
- **limit** (query): Maximum number of suggestions

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
GET /api/v1/documentation/search/suggestions
```
