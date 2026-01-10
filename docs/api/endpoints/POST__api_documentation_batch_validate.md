# POST /api/documentation/batch/validate

## Summary
Batch Validate Documentation

## Description
Batch validate multiple documentation categories.

Runs validation on multiple categories in parallel for faster processing.

## Tags
documentation

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
[
  "example_string",
  "example_string"
]
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
POST /api/documentation/batch/validate
Content-Type: application/json
Accept: application/json
```

