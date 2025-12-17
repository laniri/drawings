# Interface Documentation

This section documents all system interfaces using UML diagrams and formal specifications.

## Interface Categories

### 1. User Interfaces
- [Web Application Interface](./ui/web-application.md)
- [Upload Interface Specification](./ui/upload-interface.md)
- [Analysis Results Interface](./ui/analysis-interface.md)
- [Configuration Interface](./ui/configuration-interface.md)

### 2. API Interfaces
- [REST API Interface Contracts](./api/rest-contracts.md)
- [Request/Response Schemas](./api/schemas.md)
- [Error Handling Interfaces](./api/error-handling.md)
- [Authentication Interfaces](./api/authentication.md)

### 3. Service Interfaces
- [Internal Service Contracts](./services/internal-contracts.md)
- [Database Interface Specifications](./services/database-interfaces.md)
- [File Storage Interfaces](./services/storage-interfaces.md)
- [ML Model Interfaces](./services/ml-interfaces.md)

### 4. External Interfaces
- [AWS Integration Interfaces](./external/aws-interfaces.md)
- [Cloud Storage Interfaces](./external/storage-interfaces.md)
- [Third-party Service Interfaces](./external/third-party.md)

## UML Diagrams

### Sequence Diagrams
- [Analysis Request Sequence](./uml/sequences/analysis-request.md)
- [Upload Process Sequence](./uml/sequences/upload-process.md)
- [Configuration Update Sequence](./uml/sequences/config-update.md)

### Class Diagrams
- [Core Domain Classes](./uml/classes/domain-classes.md)
- [Service Layer Classes](./uml/classes/service-classes.md)
- [Data Model Classes](./uml/classes/data-models.md)

### Component Diagrams
- [System Component Overview](./uml/components/system-overview.md)
- [Backend Component Structure](./uml/components/backend-structure.md)
- [Frontend Component Structure](./uml/components/frontend-structure.md)

## Interface Specifications

### Data Transfer Objects (DTOs)
- [Analysis Request DTO](./dtos/analysis-request.md)
- [Analysis Response DTO](./dtos/analysis-response.md)
- [Configuration DTO](./dtos/configuration.md)
- [Error Response DTO](./dtos/error-response.md)

### Service Contracts
- [Analysis Service Contract](./contracts/analysis-service.md)
- [Model Management Contract](./contracts/model-management.md)
- [Configuration Service Contract](./contracts/configuration-service.md)

## Integration Patterns

### Communication Patterns
- [Request-Response Pattern](./patterns/request-response.md)
- [Event-Driven Communication](./patterns/event-driven.md)
- [Batch Processing Pattern](./patterns/batch-processing.md)

### Data Exchange Formats
- [JSON Schema Specifications](./formats/json-schemas.md)
- [File Format Specifications](./formats/file-formats.md)
- [Binary Protocol Specifications](./formats/binary-protocols.md)

## Interface Testing

### Contract Testing
- [API Contract Tests](./testing/api-contracts.md)
- [Service Contract Tests](./testing/service-contracts.md)
- [Integration Contract Tests](./testing/integration-contracts.md)

### Interface Validation
- [Schema Validation Rules](./validation/schema-validation.md)
- [Data Type Validation](./validation/data-types.md)
- [Business Rule Validation](./validation/business-rules.md)