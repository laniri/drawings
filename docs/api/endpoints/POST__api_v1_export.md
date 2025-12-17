# POST /api/v1/export

## Summary
Export system data

## Description
Export system data in specified format.

## Parameters
- **format** (query): Export format
- **include_embeddings** (query): Include embedding vectors

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/export
```
