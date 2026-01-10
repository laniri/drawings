# GET /api/models/training/{job_id}/status

## Summary
Get Training Status

## Description
Get training job status.

## Tags
models

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| job_id | path | string | Yes | No description |

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
GET /api/models/training/{job_id}/status
Content-Type: application/json
Accept: application/json
```

