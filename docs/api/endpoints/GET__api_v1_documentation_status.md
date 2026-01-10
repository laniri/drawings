# GET /api/v1/documentation/status

## Summary
Get Documentation Status

## Description
Get current documentation generation status.

Returns real-time status of documentation generation including progress,
current task, and any errors or warnings.

## Tags
documentation

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "is_generating": true,
  "current_task": {},
  "progress": 42,
  "start_time": {},
  "last_update": {},
  "errors": [
    "example_string",
    "example_string"
  ],
  "warnings": [
    "example_string",
    "example_string"
  ]
}
```


## Complete Request Example

```http
GET /api/v1/documentation/status
Content-Type: application/json
Accept: application/json
```

