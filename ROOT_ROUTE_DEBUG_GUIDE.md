# Root Route Issue - Debug Guide

## Problem Summary

When accessing `https://d2e6rjfv7d2rgs.cloudfront.net/`, you get:
```json
{"message":"Children's Drawing Anomaly Detection System","version":"0.1.0","docs_url":"/docs"}
```

Instead of the expected React frontend demo page.

## Root Cause Analysis

The issue is in the **request routing configuration**:

1. **CloudFront** forwards all requests (including `/`) to the **Application Load Balancer**
2. **ALB** forwards requests to the **ECS container** on port 80
3. **Container** should serve React app via nginx, but FastAPI is responding instead
4. **FastAPI** doesn't have a root route handler, so it returns the default response

## Solution Applied

### Quick Fix: Added Root Route Handler

Modified `app/main.py` to add a root route handler that:
1. **Checks if React build exists** at `/var/www/html/index.html`
2. **Serves the React app** if available
3. **Redirects to demo page** as fallback

```python
@app.get("/")
async def serve_react_app():
    react_index_path = "/var/www/html/index.html"
    if os.path.exists(react_index_path):
        return FileResponse(react_index_path, media_type="text/html")
    else:
        return RedirectResponse(url="/demo/", status_code=302)
```

## Deployment Steps

### Step 1: Deploy the Fix
```bash
./fix_root_route_deployment.sh
```

This script will:
- Build new Docker image with the fix
- Push to ECR
- Force ECS service update
- Wait for deployment completion

### Step 2: Clear CloudFront Cache
```bash
./invalidate_cloudfront_cache.sh
```

This will immediately clear the CloudFront cache so changes are visible.

## Testing the Fix

### Expected Behavior After Fix

1. **Root URL** (`https://d2e6rjfv7d2rgs.cloudfront.net/`):
   - Should serve the React app `index.html`
   - Or redirect to `/demo/` if React build is missing

2. **Demo URL** (`https://d2e6rjfv7d2rgs.cloudfront.net/demo/`):
   - Should continue working as before

3. **API URLs** (`https://d2e6rjfv7d2rgs.cloudfront.net/api/v1/*`):
   - Should continue working as before

### Test Commands

```bash
# Test root endpoint (should return HTML or redirect)
curl -L https://d2e6rjfv7d2rgs.cloudfront.net/

# Test demo endpoint (should return HTML)
curl https://d2e6rjfv7d2rgs.cloudfront.net/demo/

# Test API endpoint (should return JSON)
curl https://d2e6rjfv7d2rgs.cloudfront.net/health
```

## Long-term Solution (Optional)

For a more robust architecture, consider:

### Option 1: Separate Frontend and Backend
- **S3 + CloudFront** for React app
- **ALB + ECS** for API only
- **CloudFront behaviors** to route `/api/*` to ALB

### Option 2: Fix nginx Configuration
- Debug why nginx isn't serving the React app
- Ensure nginx is properly configured in the container
- Check if both nginx and uvicorn are running correctly

## Troubleshooting

### If the fix doesn't work:

1. **Check ECS service status:**
```bash
aws ecs describe-services --cluster children-drawing-prod-cluster --services children-drawing-prod-service --region eu-west-1
```

2. **Check container logs:**
```bash
aws logs get-log-events --log-group-name "/ecs/children-drawing-prod" --region eu-west-1 --log-stream-name [STREAM_NAME]
```

3. **Check if React build exists in container:**
```bash
# This would require exec into the container
ls -la /var/www/html/
```

4. **Test direct ALB endpoint:**
```bash
curl http://children-drawing-prod-alb-1755835064.eu-west-1.elb.amazonaws.com/
```

### Common Issues

1. **Rate Limiting**: Wait 60 seconds between requests
2. **CloudFront Cache**: Wait 2-3 minutes after deployment
3. **Container Startup**: Check if both nginx and uvicorn are running
4. **React Build Missing**: Frontend build might not be included in container

## Monitoring

After deployment, monitor:
- **ECS service health**
- **ALB target health**
- **CloudFront cache hit ratio**
- **Application response times**

## Rollback Plan

If the fix causes issues:

1. **Revert the code changes:**
```bash
git checkout HEAD~1 app/main.py
```

2. **Redeploy previous version:**
```bash
./fix_root_route_deployment.sh
```

3. **Or use previous task definition:**
```bash
aws ecs update-service --cluster children-drawing-prod-cluster --service children-drawing-prod-service --task-definition children-drawing-prod-task:11 --region eu-west-1
```