# S3 Storage Integration Fix Specification

## Problem Statement

The deployed application is missing demo images (both original drawings and saliency maps) that should be retrieved from S3 in the production environment. While the database is working correctly with 37,835+ drawings, the images are not loading because the S3 storage integration is not functioning properly.

## Current State Analysis

### Environment Detection
- Production environment is correctly detected (AWS_REGION present)
- Environment configuration shows `storage_backend: "s3"` in production
- S3 bucket name and AWS region are configured

### Storage Architecture
- **FileStorageService**: Legacy service, deprecated in favor of environment-aware storage
- **EnvironmentAwareStorageService**: ✅ **INTEGRATED** - Now used by all endpoints, supports both local and S3 storage
- **S3StorageBackend**: Implemented with presigned URL generation for secure access

### Current Status
✅ **RESOLVED**: The application endpoints now use `EnvironmentAwareStorageService` (environment-aware with S3 support) instead of the legacy `FileStorageService`.

## User Stories

### Story 1: Demo Images Display
**As a** user visiting the deployed demo page  
**I want** to see the original drawings and saliency maps  
**So that** I can understand how the AI system works through visual examples

**Acceptance Criteria:**
- Demo page loads with all sample images visible
- Original drawings display correctly from S3
- Saliency maps display correctly from S3
- Images load with reasonable performance (< 3 seconds)

### Story 2: File Serving Integration
**As a** developer  
**I want** the file serving endpoints to use environment-aware storage  
**So that** files are served from the correct storage backend (local vs S3) based on environment

**Acceptance Criteria:**
- `/api/v1/drawings/{id}/file` endpoint uses environment-aware storage
- Static file serving uses environment-aware storage
- S3 presigned URLs are generated for production file access
- Local file serving continues to work in development

### Story 3: Storage Service Integration
**As a** system administrator  
**I want** all file operations to use the unified storage service  
**So that** storage backend switching is transparent and consistent

**Acceptance Criteria:**
- All file upload operations use `EnvironmentAwareStorageService`
- All file retrieval operations use `EnvironmentAwareStorageService`
- File URL generation works correctly for both local and S3 backends
- Storage statistics reflect the correct backend usage

## Technical Requirements

### 1. Service Integration ✅ **COMPLETED**
- ✅ Replace `FileStorageService` usage with `EnvironmentAwareStorageService` in:
  - ✅ `app/api/api_v1/endpoints/drawings.py` - **COMPLETED**
  - ✅ `app/services/demo_service.py` - **COMPLETED**
  - Any other endpoints that serve files

### 2. URL Generation
- Ensure S3 presigned URLs are generated for production file access
- Maintain local file URLs for development environment
- Handle both absolute and relative file paths correctly

### 3. Error Handling
- Graceful fallback when S3 is unavailable
- Clear error messages for storage configuration issues
- Proper logging for storage operations

### 4. Performance Optimization
- Implement appropriate presigned URL expiration (1 hour)
- Cache file URLs where appropriate
- Minimize S3 API calls

## Implementation Tasks

### Phase 1: Core Integration
1. **Update drawings endpoint** to use `EnvironmentAwareStorageService`
2. **Update demo service** to use environment-aware storage for image URLs
3. **Test file serving** in both local and production environments

### Phase 2: Validation
1. **Verify S3 configuration** in production environment
2. **Test presigned URL generation** for existing files
3. **Validate demo page** image loading in production

### Phase 3: Monitoring
1. **Add storage operation logging** for debugging
2. **Monitor S3 API usage** and costs
3. **Set up alerts** for storage failures

## Testing Strategy

### Unit Tests
- Test environment-aware storage service initialization
- Test S3 backend URL generation
- Test local backend URL generation
- Test storage backend switching

### Integration Tests
- Test file upload and retrieval flow
- Test demo page image loading
- Test error handling for missing files

### Production Validation
- Verify demo images load correctly
- Check S3 presigned URL generation
- Monitor application logs for storage errors

## Success Criteria

1. **Demo images visible**: All demo page images load correctly in production
2. **Performance acceptable**: Images load within 3 seconds
3. **No errors**: No storage-related errors in application logs
4. **Environment consistency**: Storage backend switches correctly between environments
5. **Cost efficient**: S3 usage is optimized with appropriate caching

## Risk Assessment

### High Risk
- **S3 permissions**: Incorrect IAM permissions could prevent file access
- **Presigned URL expiration**: URLs expiring too quickly could cause broken images

### Medium Risk
- **Performance impact**: S3 latency could slow down image loading
- **Cost implications**: Excessive S3 API calls could increase costs

### Low Risk
- **Local development**: Changes should not affect local development workflow
- **Backward compatibility**: Existing file paths should continue to work

## Dependencies

### External
- AWS S3 bucket with proper permissions
- boto3 library for S3 integration
- IAM role with S3 access permissions

### Internal
- Environment configuration system
- File storage service architecture
- Demo service implementation

## Rollback Plan

If issues arise:
1. **Revert to FileStorageService** in critical endpoints
2. **Add feature flag** to control storage backend usage
3. **Implement gradual rollout** for storage service migration

## Monitoring and Alerting

### Metrics to Track
- File serving response times
- S3 API call frequency
- Storage error rates
- Demo page load success rate

### Alerts to Configure
- Storage service initialization failures
- S3 access permission errors
- High S3 API usage costs
- Demo page image loading failures