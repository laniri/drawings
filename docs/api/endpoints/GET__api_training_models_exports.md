# GET /api/training/models/exports

## Summary
List Exported Models

## Description
List all exported models with their metadata.

This endpoint returns a list of all models that have been exported,
including their metadata, export timestamps, and file information.

## Tags
training

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```


## Complete Request Example

```http
GET /api/training/models/exports
Content-Type: application/json
Accept: application/json
```

