# Error Handling

## Common Error Responses

The following error responses are used across multiple endpoints:

### 422 - Unprocessable Entity - Validation error

**Used by**: 71 endpoint(s)

## Error Response Schemas

### HTTPValidationError

**Properties**:

- `detail` (array): No description

### ValidationError

**Properties**:

- `loc` (array): No description
- `msg` (string): No description
- `type` (string): No description

## Endpoint-Specific Errors

Some endpoints may return additional error responses:

### POST /api/v1/drawings/upload

- **422**: Validation Error

### GET /api/v1/drawings/upload/progress/{upload_id}

- **422**: Validation Error

### GET /api/v1/drawings/{drawing_id}

- **422**: Validation Error

### DELETE /api/v1/drawings/{drawing_id}

- **422**: Validation Error

### GET /api/v1/drawings/{drawing_id}/file

- **422**: Validation Error

