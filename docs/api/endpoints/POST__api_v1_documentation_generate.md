# POST /api/v1/documentation/generate

## Summary
Generate Documentation

## Description
Trigger documentation generation.

Starts documentation generation process in the background.
Use the status endpoint to monitor progress.

## Tags
documentation

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "categories": {},
  "force": true,
  "validate_after": true
}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "success": true,
  "duration": 3.14,
  "generated_files": [
    "example_string",
    "example_string"
  ],
  "errors": [
    "example_string",
    "example_string"
  ],
  "warnings": [
    "example_string",
    "example_string"
  ],
  "validation_result": {}
}
```

### 422 - Validation Error

**application/json**:
```json
{
  "detail": [
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    },
    {
      "loc": [
        {},
        {}
      ],
      "msg": "example_string",
      "type": "example_string"
    }
  ]
}
```


## Complete Request Example

```http
POST /api/v1/documentation/generate
Content-Type: application/json
Accept: application/json
```

