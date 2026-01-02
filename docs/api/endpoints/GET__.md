# GET /

## Summary
Root Path - StaticFiles Mount for React Frontend with JSON Fallback

## Description
**Architecture Updated**: This path uses a hybrid approach combining StaticFiles mount with a fallback endpoint.

- **With Frontend**: Serves the React application (`index.html`) when `frontend_build` directory exists
- **Without Frontend**: Returns JSON response with system information (fallback for testing/development)
- **For API Information**: Use the dedicated `/api` endpoint for consistent JSON responses

## Technical Implementation
Uses conditional StaticFiles mount with fallback endpoint:
```python
# In app/main.py
if os.path.exists("frontend_build"):
    app.mount("/", StaticFiles(directory="frontend_build", html=True), name="frontend")
else:
    # Fallback root endpoint when frontend build doesn't exist (e.g., during testing)
    @app.get("/")
    async def root_fallback():
        """Fallback root endpoint when React frontend build is not available."""
        return {
            "message": "Children's Drawing Anomaly Detection System",
            "version": settings.VERSION,
            "docs_url": "/docs",
            "api_url": f"{settings.API_V1_STR}",
            "demo_url": "/demo",
            "status": "Frontend build not available - API only mode"
        }
```

## Parameters
No parameters

## Responses
- **200**: HTML content (React frontend) when `frontend_build` exists
- **200**: JSON response with system information when `frontend_build` doesn't exist

## Examples

### With Frontend Available (Production)
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

### Without Frontend (Development/Testing)
```http
GET /
```

**Response**: JSON system information
```json
{
  "message": "Children's Drawing Anomaly Detection System",
  "version": "2.0.0",
  "docs_url": "/docs",
  "api_url": "/api/v1",
  "demo_url": "/demo",
  "status": "Frontend build not available - API only mode"
}
```

## Architecture Notes
- **Hybrid Design**: Combines StaticFiles mount with fallback endpoint for robust behavior
- **Production**: Always serves React frontend from `frontend_build` directory
- **Development/Testing**: Returns JSON system information when frontend not built
- **Frontend Routing**: React Router handles client-side navigation with `html=True` option
- **Performance**: Direct static file serving without FastAPI route processing overhead
- **Graceful Degradation**: Provides useful information even when frontend is unavailable

## Related Endpoints
- `/api` - Dedicated API information endpoint (always JSON)
- `/api/v1/health` - Backend health check
- `/docs` - API documentation

## Migration Notes
**Improvement**: The root path now provides graceful fallback behavior instead of 404 errors when frontend build is unavailable.

**Benefits of Change**:
- Better development experience with informative fallback response
- Useful for testing scenarios where frontend build may not exist
- Maintains production behavior while improving development workflow
- Provides clear indication of system status and available endpoints
- Part of simplified container architecture for better reliability
