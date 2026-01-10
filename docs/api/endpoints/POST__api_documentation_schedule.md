# POST /api/documentation/schedule

## Summary
Schedule Generation

## Description
Schedule documentation generation for later execution.

Allows scheduling generation tasks for specific times or intervals.

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
POST /api/documentation/schedule
Content-Type: application/json
Accept: application/json
```

