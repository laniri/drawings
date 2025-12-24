# Children's Drawing Anomaly Detection System - Documentation

**Generated**: 2025-12-18 15:30:00  
**Version**: 2.0.0  
**Status**: Subject-Aware System - Fully Functional

## Quick Navigation

### 🚀 Subject-Aware System (v2.0)
- [Subject-Aware Upgrade Guide](./SUBJECT_AWARE_UPGRADE.md) - **NEW**: Complete upgrade documentation
- [Subject-Aware Architecture](./architecture/08-subject-aware-system.md) - **NEW**: System architecture with hybrid embeddings

### 🏗️ Architecture
- [System Overview](./architecture/01-system-context.md)
- [Component Architecture](./architecture/06-backend-components.md)
- [Database Schema](./database/schema.md) - **UPDATED**: Subject-aware schema

### 📡 API Reference
- [API Overview](./api/README.md)
- [OpenAPI Schema](./api/openapi.json)
- [Endpoint Documentation](./api/endpoints/)

### 🧮 Algorithms
- [Hybrid Embedding System](./algorithms/01-hybrid-embeddings.md)
- [Subject-Aware Anomaly Detection](./algorithms/02-subject-aware-detection.md)
- [Score Normalization](./algorithms/07-score-normalization.md)
- [Algorithm Implementations](./algorithms/implementations/)

### 🔄 Workflows
- [Subject-Aware Analysis](./workflows/business/02-subject-aware-analysis.md) - **NEW**: Enhanced workflow with subject support
- [Upload and Analysis](./workflows/business/01-drawing-upload-analysis.md)
- [System Workflows](./workflows/README.md) - **UPDATED**: Includes subject-aware workflows

### 🔌 Interfaces
- [Service Interfaces](./interfaces/services/)
- [API Contracts](./interfaces/api/)

### 🚀 Deployment
- [Docker Deployment](./deployment/docker.md)
- [Environment Setup](./deployment/environment-setup.md)

### 🧪 Testing
- [Testing Documentation](./testing.md) - **NEW**: Comprehensive testing guide with robust fixtures and property-based testing

## Documentation Standards

This documentation follows industry standards:
- **C4 Model** for architecture
- **OpenAPI 3.0** for API documentation
- **BPMN 2.0** for process flows
- **UML 2.5** for interfaces
- **IEEE 830** for requirements

## Maintenance

Documentation is automatically generated from code using:
```bash
python scripts/generate_docs.py
```

For manual updates, see [Documentation Guidelines](./CONTRIBUTING.md).
