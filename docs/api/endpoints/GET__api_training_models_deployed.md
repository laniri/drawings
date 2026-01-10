# GET /api/training/models/deployed

## Summary
List Deployed Models

## Description
List all deployed models in production.

This endpoint returns information about all models currently
deployed and active in the production system.

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
GET /api/training/models/deployed
Content-Type: application/json
Accept: application/json
```

