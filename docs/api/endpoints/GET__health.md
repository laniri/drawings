# GET /health

## Summary
Lightweight Health Check

## Description
Lightweight health check endpoint for load balancer monitoring. Returns basic system status with environment information.

## Parameters
No parameters

## Responses
- **200**: Successful Response

### Response Schema
```json
{
  "status": "healthy",
  "service": "drawing-anomaly-detection", 
  "timestamp": "2024-12-24T10:30:45.123456",
  "environment": "production",
  "storage": "s3"
}
```

## Example
```http
GET /health
```

### Example Response
```json
{
  "status": "healthy",
  "service": "drawing-anomaly-detection",
  "timestamp": "2024-12-24T10:30:45.123456", 
  "environment": "production",
  "storage": "s3"
}
```
