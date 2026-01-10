# POST /api/documentation/batch/generate

## Summary
Batch Generate Documentation

## Description
Batch generate multiple documentation categories with scheduling.

Allows generating multiple categories in sequence with different
configurations for each category.

## Tags
documentation

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{}
```


## Responses

### 200 - Successful Response

**application/json**:
```json
{}
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
POST /api/documentation/batch/generate
Content-Type: application/json
Accept: application/json
```

