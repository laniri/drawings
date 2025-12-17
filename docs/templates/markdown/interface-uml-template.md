# {{INTERFACE_NAME}} Interface Documentation

## Overview

{{INTERFACE_DESCRIPTION}}

**Interface Type**: {{INTERFACE_TYPE}}  
**UML Version**: 2.5  
**Specification Level**: {{SPECIFICATION_LEVEL}}

## Interface Contract

### Service Interface

```python
{{INTERFACE_DEFINITION}}
```

### Contract Specification

{{#CONTRACT_METHODS}}
#### {{METHOD_NAME}}

**Signature**: `{{METHOD_SIGNATURE}}`

**Description**: {{METHOD_DESCRIPTION}}

**Parameters**:
{{#PARAMETERS}}
- **{{PARAM_NAME}}** ({{PARAM_TYPE}}, {{REQUIRED_STATUS}}): {{PARAM_DESCRIPTION}}
  {{#PARAM_CONSTRAINTS}}
  - **Constraints**: {{CONSTRAINTS}}
  {{/PARAM_CONSTRAINTS}}
{{/PARAMETERS}}

**Returns**: {{RETURN_TYPE}} - {{RETURN_DESCRIPTION}}

**Exceptions**:
{{#EXCEPTIONS}}
- **{{EXCEPTION_TYPE}}**: {{EXCEPTION_DESCRIPTION}}
{{/EXCEPTIONS}}

**Preconditions**:
{{#PRECONDITIONS}}
- {{PRECONDITION}}
{{/PRECONDITIONS}}

**Postconditions**:
{{#POSTCONDITIONS}}
- {{POSTCONDITION}}
{{/POSTCONDITIONS}}

**Example Usage**:
```python
{{USAGE_EXAMPLE}}
```
{{/CONTRACT_METHODS}}

## UML Diagrams

### Class Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
classDiagram
    {{CLASS_DIAGRAM_CONTENT}}
```

### Sequence Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
sequenceDiagram
    {{SEQUENCE_DIAGRAM_CONTENT}}
```

### Component Diagram

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#ff6b6b', 'primaryTextColor':'#fff', 'primaryBorderColor':'#ff4757', 'lineColor':'#5f27cd', 'secondaryColor':'#00d2d3', 'tertiaryColor':'#ff9ff3'}}}%%
graph TB
    {{COMPONENT_DIAGRAM_CONTENT}}
```

## Data Transfer Objects (DTOs)

{{#DTOS}}
### {{DTO_NAME}}

**Purpose**: {{DTO_PURPOSE}}

**Schema**:
```python
{{DTO_SCHEMA}}
```

**JSON Representation**:
```json
{{DTO_JSON_EXAMPLE}}
```

**Validation Rules**:
{{#VALIDATION_RULES}}
- **{{FIELD_NAME}}**: {{VALIDATION_RULE}}
{{/VALIDATION_RULES}}

**Serialization**:
- **Format**: {{SERIALIZATION_FORMAT}}
- **Encoding**: {{SERIALIZATION_ENCODING}}
- **Compression**: {{SERIALIZATION_COMPRESSION}}
{{/DTOS}}

## Interaction Patterns

### Request-Response Pattern

{{#REQUEST_RESPONSE_PATTERNS}}
#### {{PATTERN_NAME}}

**Description**: {{PATTERN_DESCRIPTION}}

**Sequence**:
{{#SEQUENCE_STEPS}}
{{STEP_NUMBER}}. **{{ACTOR}}** → **{{TARGET}}**: {{MESSAGE}}
   - {{STEP_DESCRIPTION}}
{{/SEQUENCE_STEPS}}

**Error Handling**:
{{#ERROR_SCENARIOS}}
- **{{ERROR_TYPE}}**: {{ERROR_HANDLING}}
{{/ERROR_SCENARIOS}}
{{/REQUEST_RESPONSE_PATTERNS}}

### Event-Driven Patterns

{{#EVENT_PATTERNS}}
#### {{EVENT_NAME}}

**Trigger**: {{EVENT_TRIGGER}}

**Publishers**:
{{#PUBLISHERS}}
- **{{PUBLISHER_NAME}}**: {{PUBLISHER_DESCRIPTION}}
{{/PUBLISHERS}}

**Subscribers**:
{{#SUBSCRIBERS}}
- **{{SUBSCRIBER_NAME}}**: {{SUBSCRIBER_DESCRIPTION}}
{{/SUBSCRIBERS}}

**Event Schema**:
```json
{{EVENT_SCHEMA}}
```
{{/EVENT_PATTERNS}}

## Service Dependencies

### Required Services

{{#REQUIRED_SERVICES}}
#### {{SERVICE_NAME}}

- **Purpose**: {{SERVICE_PURPOSE}}
- **Interface**: {{SERVICE_INTERFACE}}
- **Version**: {{SERVICE_VERSION}}
- **Criticality**: {{SERVICE_CRITICALITY}}

**Dependency Type**: {{DEPENDENCY_TYPE}}

**Fallback Strategy**: {{FALLBACK_STRATEGY}}
{{/REQUIRED_SERVICES}}

### Provided Services

{{#PROVIDED_SERVICES}}
#### {{SERVICE_NAME}}

- **Consumers**: {{SERVICE_CONSUMERS}}
- **SLA**: {{SERVICE_SLA}}
- **Capacity**: {{SERVICE_CAPACITY}}
- **Monitoring**: {{SERVICE_MONITORING}}
{{/PROVIDED_SERVICES}}

## Quality of Service (QoS)

### Performance Requirements

{{#PERFORMANCE_REQUIREMENTS}}
#### {{REQUIREMENT_NAME}}

- **Metric**: {{PERFORMANCE_METRIC}}
- **Target**: {{PERFORMANCE_TARGET}}
- **Measurement**: {{PERFORMANCE_MEASUREMENT}}
- **Monitoring**: {{PERFORMANCE_MONITORING}}
{{/PERFORMANCE_REQUIREMENTS}}

### Reliability Requirements

{{#RELIABILITY_REQUIREMENTS}}
- **{{RELIABILITY_ASPECT}}**: {{RELIABILITY_REQUIREMENT}}
  - **Measurement**: {{RELIABILITY_MEASUREMENT}}
  - **Target**: {{RELIABILITY_TARGET}}
{{/RELIABILITY_REQUIREMENTS}}

### Security Requirements

{{#SECURITY_REQUIREMENTS}}
#### {{SECURITY_ASPECT}}

**Requirement**: {{SECURITY_REQUIREMENT}}

**Implementation**: {{SECURITY_IMPLEMENTATION}}

**Validation**: {{SECURITY_VALIDATION}}
{{/SECURITY_REQUIREMENTS}}

## Interface Versioning

### Version Strategy

- **Current Version**: {{CURRENT_VERSION}}
- **Versioning Scheme**: {{VERSIONING_SCHEME}}
- **Backward Compatibility**: {{BACKWARD_COMPATIBILITY}}

### Version History

{{#VERSION_HISTORY}}
#### Version {{VERSION_NUMBER}} ({{RELEASE_DATE}})

**Changes**:
{{#VERSION_CHANGES}}
- **{{CHANGE_TYPE}}**: {{CHANGE_DESCRIPTION}}
{{/VERSION_CHANGES}}

**Migration Guide**: {{MIGRATION_GUIDE}}

**Deprecation Notice**: {{DEPRECATION_NOTICE}}
{{/VERSION_HISTORY}}

## Testing and Validation

### Contract Testing

{{#CONTRACT_TESTS}}
#### Test: {{TEST_NAME}}

**Purpose**: {{TEST_PURPOSE}}

**Test Implementation**:
```python
{{TEST_IMPLEMENTATION}}
```

**Validation Criteria**: {{VALIDATION_CRITERIA}}
{{/CONTRACT_TESTS}}

### Integration Testing

{{#INTEGRATION_TESTS}}
#### Integration Test: {{TEST_NAME}}

**Scope**: {{TEST_SCOPE}}

**Test Scenario**: {{TEST_SCENARIO}}

**Expected Behavior**: {{EXPECTED_BEHAVIOR}}

**Test Data**:
```json
{{TEST_DATA}}
```
{{/INTEGRATION_TESTS}}

### Property-Based Testing

{{#PROPERTY_TESTS}}
#### Property: {{PROPERTY_NAME}}

**Property Statement**: {{PROPERTY_STATEMENT}}

**Test Implementation**:
```python
{{PROPERTY_TEST_CODE}}
```

**Validates**: Requirements {{REQUIREMENT_REFERENCES}}
{{/PROPERTY_TESTS}}

## Error Handling

### Error Categories

{{#ERROR_CATEGORIES}}
#### {{ERROR_CATEGORY}}

**Description**: {{ERROR_DESCRIPTION}}

**Error Codes**:
{{#ERROR_CODES}}
- **{{ERROR_CODE}}**: {{ERROR_MESSAGE}}
  - **Cause**: {{ERROR_CAUSE}}
  - **Resolution**: {{ERROR_RESOLUTION}}
{{/ERROR_CODES}}
{{/ERROR_CATEGORIES}}

### Exception Hierarchy

```python
{{EXCEPTION_HIERARCHY}}
```

### Error Response Format

```json
{{ERROR_RESPONSE_FORMAT}}
```

## Monitoring and Observability

### Metrics

{{#METRICS}}
#### {{METRIC_NAME}}

- **Type**: {{METRIC_TYPE}}
- **Description**: {{METRIC_DESCRIPTION}}
- **Collection**: {{METRIC_COLLECTION}}
- **Alerting**: {{METRIC_ALERTING}}
{{/METRICS}}

### Logging

{{#LOGGING_REQUIREMENTS}}
- **{{LOG_LEVEL}}**: {{LOG_DESCRIPTION}}
  - **Format**: {{LOG_FORMAT}}
  - **Destination**: {{LOG_DESTINATION}}
{{/LOGGING_REQUIREMENTS}}

### Tracing

- **Trace Context**: {{TRACE_CONTEXT}}
- **Span Information**: {{SPAN_INFORMATION}}
- **Correlation IDs**: {{CORRELATION_IDS}}

## Implementation Guidelines

### Best Practices

{{#BEST_PRACTICES}}
- **{{PRACTICE_CATEGORY}}**: {{PRACTICE_DESCRIPTION}}
{{/BEST_PRACTICES}}

### Anti-Patterns

{{#ANTI_PATTERNS}}
- **{{ANTI_PATTERN_NAME}}**: {{ANTI_PATTERN_DESCRIPTION}}
  - **Why to Avoid**: {{AVOIDANCE_REASON}}
  - **Alternative**: {{ALTERNATIVE_APPROACH}}
{{/ANTI_PATTERNS}}

### Code Examples

{{#CODE_EXAMPLES}}
#### {{EXAMPLE_NAME}}

**Scenario**: {{EXAMPLE_SCENARIO}}

**Implementation**:
```python
{{EXAMPLE_CODE}}
```

**Explanation**: {{EXAMPLE_EXPLANATION}}
{{/CODE_EXAMPLES}}

---

**Generated**: {{GENERATION_DATE}}  
**Interface Version**: {{INTERFACE_VERSION}}  
**UML Specification**: {{UML_SPECIFICATION}}  
**Last Updated**: {{LAST_UPDATED}}  
**Validated**: {{VALIDATION_STATUS}}