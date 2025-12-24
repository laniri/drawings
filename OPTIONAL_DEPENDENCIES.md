# Optional Dependencies Guide

## Overview

The Children's Drawing Anomaly Detection System has been updated to make several dependencies optional, improving deployment flexibility and reducing installation complexity. The system now provides graceful fallbacks for OpenCV functionality using PIL and NumPy, and AWS services are optional for local development.

## Changes Made

### 1. OpenCV Made Optional

**Files Modified:**
- `app/services/data_pipeline.py` - Added optional OpenCV import with `HAS_OPENCV` flag
- `app/services/interpretability_engine.py` - Added optional OpenCV import and fallback functions
- `requirements.txt` - Moved OpenCV to optional section with comments
- `requirements-enhanced.txt` - New file for enhanced functionality
- `pyproject.toml` - Added `enhanced` optional dependency group
- `.github/workflows/deploy-production.yml` - Removed OpenCV system dependencies from CI
- `Dockerfile.prod` - Simplified system dependencies

### 2. AWS Dependencies Made Optional

**Files Modified:**
- `app/services/cost_optimization_service.py` - Added optional AWS import with `HAS_AWS` flag
- `app/services/database_migration_service.py` - Added optional AWS import with `HAS_AWS` flag
- `app/services/monitoring_service.py` - Added optional AWS import handling
- `app/services/security_service.py` - Added optional AWS import with graceful fallback behavior
- `requirements.txt` - AWS dependencies remain required but gracefully handled when missing

**New Import Pattern:**
```python
# Optional AWS dependencies
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_AWS = True
except ImportError:
    HAS_AWS = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception
```

### 3. Fallback Implementations Added

**New Functions in `interpretability_engine.py`:**
- `_resize_with_pil()` - PIL-based image resizing fallback
- `_rgb_to_grayscale()` - NumPy-based RGB to grayscale conversion
- `_simple_edge_detection()` - Gradient-based edge detection fallback
- `_safe_resize()` - Safe resizing with OpenCV/PIL detection
- `_safe_rgb_to_gray()` - Safe grayscale conversion with fallback
- `_safe_edge_detection()` - Safe edge detection with fallback

**AWS Service Fallbacks:**
- Cost optimization service works without AWS clients for local development
- Database migration service works without AWS S3 backup
- Monitoring service operates without CloudWatch integration
- Security service provides validation without AWS clients
- Local development doesn't require AWS credentials
- Production features gracefully degrade when AWS is unavailable

### 4. Documentation Updates

### 4. Documentation Updates

**Files Updated:**
- `README.md` - Added Optional Dependencies section, updated installation instructions
- `OPTIONAL_DEPENDENCIES.md` - Added AWS dependencies section
- `.kiro/steering/tech.md` - Updated dependency descriptions
- `.kiro/steering/interpretability.md` - Updated dependency information

## Installation Options

### Minimal Installation (Core Functionality)
```bash
pip install -r requirements.txt
```

**Provides:**
- All core drawing analysis functionality
- PIL-based image processing
- Basic saliency map generation
- Web interface and API
- Model training and inference

### Enhanced Installation (Full Features)
```bash
pip install -r requirements-enhanced.txt
# OR
pip install -e .[enhanced]
```

**Additional Features:**
- Advanced contour detection
- High-quality image resizing (cubic interpolation)
- Canny edge detection for complexity analysis
- Enhanced saliency map overlays
- PDF report generation (ReportLab)

## Functionality Comparison

| Feature | Without OpenCV | With OpenCV | Without AWS | With AWS |
|---------|---------------|-------------|-------------|----------|
| Image Resizing | PIL Lanczos | OpenCV Cubic | ✓ | ✓ |
| Grayscale Conversion | NumPy weights | OpenCV optimized | ✓ | ✓ |
| Edge Detection | Simple gradients | Canny algorithm | ✓ | ✓ |
| Contour Detection | Basic regions | Precise contours | ✓ | ✓ |
| Saliency Maps | PIL-based | OpenCV overlays | ✓ | ✓ |
| Database Backups | Local only | Local + S3 | ✓ | ✓ |
| Cost Optimization | Local only | AWS integration | ✓ | ✓ |
| Monitoring | Local logs | CloudWatch | ✓ | ✓ |
| Alerting | Local only | SNS alerts | ✓ | ✓ |
| Performance | Good | Optimized | Good | Enhanced |

## Deployment Benefits

### CI/CD Improvements
- Faster CI builds (no OpenCV compilation)
- Smaller Docker images
- Reduced system dependencies
- Better compatibility across platforms

### Production Flexibility
- Deploy with minimal dependencies for basic functionality
- Add enhanced features only when needed
- Easier troubleshooting of dependency issues
- Better resource utilization

## Migration Guide

### For Existing Deployments
1. **No action required** - OpenCV and AWS services will continue to work if already installed
2. **To optimize**: Remove OpenCV and test functionality
3. **To enhance**: Install enhanced requirements for full features
4. **Local development**: AWS services are now optional for local development and testing

### For New Deployments
1. Start with minimal installation for local development
2. Test core functionality without AWS dependencies
3. Add AWS services for production deployment
4. Add enhanced features (OpenCV, ReportLab) if needed

## Testing

The system includes comprehensive tests that work with or without OpenCV:
- Property-based tests validate functionality in both modes
- CI runs with minimal dependencies
- Local development can use enhanced features

## Troubleshooting

### AWS Import Errors
```bash
# Error: No module named 'boto3'
# Solution: This is expected for local development and handled gracefully
# The system will work without AWS services for local development
# For production deployment, install AWS dependencies:
pip install boto3 botocore
```

### OpenCV Import Errors
```bash
# Error: No module named 'cv2'
# Solution: This is expected and handled gracefully
# The system will use PIL fallbacks automatically
```

### Performance Differences
- PIL fallbacks may be slightly slower for large images
- Edge detection may be less precise without Canny algorithm
- Contour detection uses simplified region analysis

### Feature Limitations
- Without OpenCV: Basic saliency overlays
- Without ReportLab: No PDF exports (other formats available)
- Without AWS: No S3 backups, CloudWatch monitoring, or SNS alerts
- All core ML functionality remains unchanged regardless of optional dependencies

## Future Considerations

### Planned Enhancements
- Additional PIL-based optimizations
- More sophisticated fallback algorithms
- Performance benchmarking tools
- Automatic dependency detection

### Compatibility
- Python 3.11+ support maintained
- Cross-platform compatibility improved
- Docker image size optimization
- Cloud deployment flexibility

## Summary

This update significantly improves the system's deployment flexibility while maintaining full functionality. Users can choose between minimal installation for basic needs or enhanced installation for advanced features, making the system more accessible and easier to deploy in various environments.