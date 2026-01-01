# GET /api

## Summary
API Root Information Endpoint

## Description
Returns basic information about the Children's Drawing Anomaly Detection System API, including version, documentation URLs, and available endpoints. This endpoint always returns JSON API information, unlike the root endpoint (`/`) which may serve the React frontend.

## Parameters
No parameters

## Responses
- **200**: JSON object with API information

## Response Schema
```json
{
  "message": "string",
  "version": "string", 
  "docs_url": "string",
  "api_url": "string",
  "demo_url": "string"
}
```

## Example
```http
GET /api
```

**Response**:
```json
{
  "message": "Children's Drawing Anomaly Detection System API",
  "version": "0.1.0",
  "docs_url": "/docs",
  "api_url": "/api/v1",
  "demo_url": "/demo"
}
```

## Architecture Notes
- **Guaranteed JSON**: Always returns JSON, regardless of frontend availability
- **API Discovery**: Use this endpoint for programmatic API discovery
- **Distinction**: Unlike `/` which may serve HTML frontend, this always serves API info
- **Integration**: Ideal for API clients and automated tools

## Related Endpoints
- `/` - Root endpoint (may serve frontend or API info)
- `/api/v1/health` - Backend health check
- `/docs` - API documentation
- `/api/v1/` - Main API endpoints