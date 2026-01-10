# GET /api/v1/training/environments/status

## Summary
Get Training Environments Status

## Description
Get status of available training environments.

This endpoint returns information about local and SageMaker
training environments, including availability and configuration.

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
GET /api/v1/training/environments/status
Content-Type: application/json
Accept: application/json
```

