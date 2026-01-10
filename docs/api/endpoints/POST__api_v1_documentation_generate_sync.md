# POST /api/v1/documentation/generate/sync

## Summary
Generate Documentation Sync

## Description
Generate documentation synchronously.

Runs documentation generation and waits for completion.
Use this for smaller generation tasks or when immediate results are needed.

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
POST /api/v1/documentation/generate/sync
Content-Type: application/json
Accept: application/json
```

