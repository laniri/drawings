# GET /health/simple

## Summary
Ultra-Lightweight Health Check

## Description
Ultra-lightweight health check for ALB (Application Load Balancer) with no dependencies. Returns minimal response for fast health verification.

## Parameters
No parameters

## Responses
- **200**: Successful Response

### Response Schema
```json
{
  "status": "ok"
}
```

## Example
```http
GET /health/simple
```

### Example Response
```json
{
  "status": "ok"
}
```

## Usage Notes
- Designed for load balancer health checks where minimal response time is critical
- No database queries or external dependencies
- Fastest possible health check endpoint
- Recommended for ALB target group health checks in AWS deployments