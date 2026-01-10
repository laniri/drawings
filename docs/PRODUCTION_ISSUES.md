# Production Issues and Resolutions

This document tracks production issues encountered and their resolutions for the Children's Drawing Anomaly Detection System.

## January 2026

### Issue #1: Usage Metrics Dashboard White Screen

**Date**: January 9, 2026  
**Severity**: High  
**Status**: ✅ Resolved

#### Symptoms
- Dashboard "Usage Metrics" tab displayed blank white screen
- No error messages in browser console
- Backend endpoint `/api/v1/metrics/usage` returned 200 OK
- Response size: 2517 bytes
- Request duration: 23-25 seconds (slow but successful)

#### Root Cause
The `/api/v1/metrics/usage` endpoint returned a flat dictionary structure:
```json
{
  "total_analyses": 123,
  "daily_analyses": 5,
  "active_sessions": 2,
  ...
}
```

But the frontend `UsageMetricsPanel` component expected a nested structure:
```json
{
  "timestamp": "2026-01-09T...",
  "database": { ... },
  "time_based": { ... },
  "sessions": { ... },
  "system_health": { ... },
  "geographic": { ... }
}
```

#### Resolution
1. Modified `UsageMetricsService.get_dashboard_stats()` to return properly nested data structure
2. Added missing helper methods:
   - `_get_health_metrics()` - Returns system health metrics
   - `_get_session_metrics()` - Returns current session metrics
3. Added `_cloudwatch_enabled` property to track CloudWatch integration status
4. Ensured all expected fields are populated with appropriate values

#### Files Modified
- `app/services/usage_metrics_service.py`

#### Verification
```bash
# Test usage metrics endpoint
curl -f https://your-domain.com/api/v1/metrics/usage

# Expected response structure with nested data
```

#### Commit
- Hash: `8d3b6a9`
- Message: "Fix dashboard issues: usage metrics white screen, drawing file 404, and remove markdown viewer tab"

---

### Issue #2: Existing Analysis Drawings Return 404

**Date**: January 9, 2026  
**Severity**: High  
**Status**: ✅ Resolved

#### Symptoms
- When opening existing analysis (not just uploaded), drawing images failed to load
- API returned 404 for `/api/v1/drawings/{id}/file`
- Example: `/api/v1/drawings/16600/file` returned 404 after 0.2 seconds
- Interpretability endpoints worked correctly (200 OK)

#### Root Cause
The S3 storage backend's `get_file_info()` method had two issues:

1. **Path Format Handling**: Only handled S3 URLs (starting with `s3://`) and returned `None` for relative paths stored in the database (like `uploads/file.png`)

2. **Path Mismatch**: Database stores paths as `uploads/file.png` but S3 bucket has them as `drawings/file.png`

When `get_file_info()` returned `None`, the drawing file endpoint raised a 404 error.

#### Resolution
1. Enhanced `S3StorageBackend.get_file_info()` to handle three path formats:
   - S3 URLs: `s3://bucket/drawings/file.png`
   - Relative uploads: `uploads/file.png` → Maps to `drawings/file.png` in S3
   - Other relative paths: Used as-is

2. Enhanced `S3StorageBackend.download_to_local()` with same path handling logic

3. Added debug logging for S3 file path resolution

#### Path Mapping Logic
```
Database Path: uploads/file.png
S3 Key: drawings/file.png
Local Synced: /app/uploads/file.png
```

#### Files Modified
- `app/services/environment_storage.py`

#### Verification
```bash
# Test drawing file endpoint
curl -f https://your-domain.com/api/v1/drawings/16600/file

# Check logs for path resolution
docker logs <container> | grep "S3 URL Generation"
docker logs <container> | grep "S3 get_file_info"
```

#### Commit
- Hash: `8d3b6a9`
- Message: "Fix dashboard issues: usage metrics white screen, drawing file 404, and remove markdown viewer tab"

---

### Issue #3: Markdown Viewer in User Navigation

**Date**: January 9, 2026  
**Severity**: Low  
**Status**: ✅ Resolved

#### Symptoms
- Markdown viewer appeared as a navigation tab in the sidebar
- Feature was intended only for internal documentation viewing
- Cluttered user interface with non-user-facing feature

#### Root Cause
The markdown viewer was added to the `menuItems` array in the Layout component, making it visible in the user-facing navigation.

#### Resolution
1. Removed "Markdown Viewer" entry from `menuItems` array
2. Removed unused `Article` icon import
3. Route and component remain functional at `/markdown-viewer` for programmatic/internal use

#### Files Modified
- `frontend/src/components/Layout/Layout.tsx`

#### Note
The markdown viewer page is still accessible at `/markdown-viewer` for internal documentation purposes, but is no longer exposed in the user-facing navigation.

#### Commit
- Hash: `8d3b6a9`
- Message: "Fix dashboard issues: usage metrics white screen, drawing file 404, and remove markdown viewer tab"

---

## Debugging Process

### Log Analysis
Production logs from `/ecs/children-drawing-prod` (21:17-21:22 UTC) were analyzed:
- Identified 404 errors for drawing file endpoint
- Confirmed `/api/v1/analysis/stats` succeeding but slow (23-25 seconds)
- No errors for usage metrics endpoint (not being called due to frontend error)

### Time Zone Considerations
- Local time: PST/PDT (UTC-8/-7)
- AWS CloudWatch: UTC
- GitHub Actions: UTC
- Production logs: UTC

When analyzing logs, timestamps were converted to understand the sequence of events.

### Testing Approach
1. Checked production logs first (as per steering guidelines)
2. Read last 2 commits of suspicious files
3. Analyzed frontend component expectations
4. Traced backend data structure
5. Identified mismatches
6. Implemented fixes
7. Verified with local tests

---

## Prevention Measures

### For Future Development

1. **API Contract Testing**: Add tests to verify API response structure matches frontend expectations
2. **Path Handling**: Document path format conventions for database storage vs S3 keys
3. **Navigation Review**: Review navigation items before adding to ensure they're user-facing features
4. **Integration Tests**: Add tests for S3 storage backend path handling
5. **Log Analysis**: Always check production logs before making assumptions about root cause

### Monitoring Improvements

1. Add frontend error tracking for white screen scenarios
2. Add metrics for 404 errors on file endpoints
3. Monitor API response times (23-25 seconds is too slow for stats endpoint)
4. Add alerts for missing data structure fields

---

## Related Documentation

- [Deployment Guide](../DEPLOYMENT.md) - Production troubleshooting section updated
- [Tech Steering](../.kiro/steering/tech.md) - Troubleshooting section updated
- [Structure Steering](../.kiro/steering/structure.md) - Service descriptions updated
- [Dashboard Fixes Summary](../tmp_files/dashboard_fixes_summary.md) - Detailed technical summary
