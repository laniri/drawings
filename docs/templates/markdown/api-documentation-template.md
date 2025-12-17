# {{API_NAME}} API Documentation

## Overview

{{API_DESCRIPTION}}

**Base URL**: `{{BASE_URL}}`  
**Version**: {{API_VERSION}}  
**OpenAPI Specification**: {{OPENAPI_VERSION}}

## Authentication

{{#AUTHENTICATION_REQUIRED}}
### {{AUTH_TYPE}}

{{AUTH_DESCRIPTION}}

**Headers Required**:
```http
{{AUTH_HEADERS}}
```

**Example**:
```bash
curl -H "{{AUTH_HEADER_EXAMPLE}}" {{BASE_URL}}/{{EXAMPLE_ENDPOINT}}
```
{{/AUTHENTICATION_REQUIRED}}

## Endpoints

{{#ENDPOINTS}}
### {{HTTP_METHOD}} {{ENDPOINT_PATH}}

{{ENDPOINT_DESCRIPTION}}

**Parameters**:

{{#PATH_PARAMETERS}}
- **{{PARAM_NAME}}** ({{PARAM_TYPE}}, required): {{PARAM_DESCRIPTION}}
{{/PATH_PARAMETERS}}

{{#QUERY_PARAMETERS}}
- **{{PARAM_NAME}}** ({{PARAM_TYPE}}, {{REQUIRED_STATUS}}): {{PARAM_DESCRIPTION}}
{{/QUERY_PARAMETERS}}

**Request Body**:
{{#REQUEST_BODY}}
```json
{{REQUEST_SCHEMA}}
```

{{REQUEST_DESCRIPTION}}
{{/REQUEST_BODY}}

**Response**:

{{#RESPONSES}}
#### {{STATUS_CODE}} {{STATUS_MESSAGE}}

{{RESPONSE_DESCRIPTION}}

```json
{{RESPONSE_SCHEMA}}
```

{{#RESPONSE_EXAMPLES}}
**Example**:
```json
{{EXAMPLE_RESPONSE}}
```
{{/RESPONSE_EXAMPLES}}
{{/RESPONSES}}

**Example Request**:
```bash
curl -X {{HTTP_METHOD}} \
  "{{BASE_URL}}{{ENDPOINT_PATH}}" \
  {{#REQUEST_HEADERS}}
  -H "{{HEADER_NAME}}: {{HEADER_VALUE}}" \
  {{/REQUEST_HEADERS}}
  {{#REQUEST_BODY}}
  -d '{{EXAMPLE_REQUEST_BODY}}'
  {{/REQUEST_BODY}}
```

{{#ERROR_CODES}}
**Error Codes**:
- **{{ERROR_CODE}}**: {{ERROR_DESCRIPTION}}
{{/ERROR_CODES}}

---
{{/ENDPOINTS}}

## Data Models

{{#MODELS}}
### {{MODEL_NAME}}

{{MODEL_DESCRIPTION}}

```json
{{MODEL_SCHEMA}}
```

**Properties**:
{{#MODEL_PROPERTIES}}
- **{{PROPERTY_NAME}}** ({{PROPERTY_TYPE}}, {{REQUIRED_STATUS}}): {{PROPERTY_DESCRIPTION}}
{{/MODEL_PROPERTIES}}

**Validation Rules**:
{{#VALIDATION_RULES}}
- {{RULE_DESCRIPTION}}
{{/VALIDATION_RULES}}

---
{{/MODELS}}

## Error Handling

### Standard Error Response

All API errors follow this standard format:

```json
{
  "error": {
    "code": "{{ERROR_CODE}}",
    "message": "{{ERROR_MESSAGE}}",
    "details": "{{ERROR_DETAILS}}",
    "timestamp": "{{TIMESTAMP}}",
    "request_id": "{{REQUEST_ID}}"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
{{#COMMON_ERRORS}}
| {{ERROR_CODE}} | {{HTTP_STATUS}} | {{ERROR_DESCRIPTION}} |
{{/COMMON_ERRORS}}

## Rate Limiting

{{#RATE_LIMITING}}
- **Limit**: {{RATE_LIMIT}} requests per {{TIME_WINDOW}}
- **Headers**: 
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Requests remaining in current window
  - `X-RateLimit-Reset`: Time when rate limit resets

**Rate Limit Exceeded Response**:
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again later.",
    "retry_after": {{RETRY_AFTER_SECONDS}}
  }
}
```
{{/RATE_LIMITING}}

## SDK and Code Examples

{{#SDK_EXAMPLES}}
### {{LANGUAGE}}

```{{LANGUAGE_CODE}}
{{CODE_EXAMPLE}}
```
{{/SDK_EXAMPLES}}

## Testing

### Interactive API Explorer

Access the interactive Swagger UI documentation at:
**{{SWAGGER_UI_URL}}**

### Postman Collection

Download the Postman collection: [{{API_NAME}} Collection]({{POSTMAN_COLLECTION_URL}})

### Test Environment

- **Base URL**: {{TEST_BASE_URL}}
- **Test Credentials**: {{TEST_CREDENTIALS}}

## Changelog

{{#CHANGELOG}}
### Version {{VERSION}} ({{RELEASE_DATE}})

{{#CHANGES}}
- **{{CHANGE_TYPE}}**: {{CHANGE_DESCRIPTION}}
{{/CHANGES}}
{{/CHANGELOG}}

## Support

- **Documentation**: {{DOCUMENTATION_URL}}
- **Issues**: {{ISSUES_URL}}
- **Contact**: {{CONTACT_EMAIL}}

---

**Generated**: {{GENERATION_DATE}}  
**OpenAPI Spec**: [{{OPENAPI_SPEC_URL}}]({{OPENAPI_SPEC_URL}})  
**Last Updated**: {{LAST_UPDATED}}  
**Validated**: {{VALIDATION_STATUS}}