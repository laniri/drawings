# GET /

## Summary
Root Endpoint - Frontend or API Information

## Description
**Behavior Changed**: This endpoint now serves the React frontend if available, or falls back to API information.

- **With Frontend**: Serves the React application (`index.html`) when `frontend_build` directory exists
- **Without Frontend**: Returns basic API information as JSON (development/testing fallback)

## Parameters
No parameters

## Responses
- **200**: HTML content (React frontend) or JSON object with API information

## Response Schema (JSON Fallback)
```json
{
  "message": "string",
  "version": "string", 
  "docs_url": "string",
  "api_url": "string",
  "demo_url": "string"
}
```

## Examples

### With Frontend Available
```http
GET /
```

**Response**: HTML content (React application)
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Children's Drawing Anomaly Detection</title>
    <!-- React app content -->
  </head>
  <body>
    <div id="root"></div>
    <!-- React app scripts -->
  </body>
</html>
```

### Without Frontend (Fallback)
```http
GET /
```

**Response**: JSON API information
```json
{
  "message": "Children's Drawing Anomaly Detection System",
  "version": "0.1.0",
  "docs_url": "/docs",
  "api_url": "/api/v1",
  "demo_url": "/demo"
}
```

## Architecture Notes
- **Production**: Serves React frontend from `frontend_build` directory
- **Development**: Falls back to JSON API information when frontend not built
- **API Access**: Use `/api` endpoint for guaranteed JSON API information
- **Frontend Routing**: React Router handles client-side navigation

## Related Endpoints
- `/api` - **NEW**: Dedicated API information endpoint (always JSON)
- `/api/v1/health` - Backend health check
- `/docs` - API documentation
