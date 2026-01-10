# GET /api/v1/training/jobs/{job_id}

## Summary
Get Training Job Status

## Description
Get detailed status of a specific training job.

This endpoint returns comprehensive information about a training job,
including progress, metrics, and environment-specific details.

## Tags
training

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| job_id | path | integer | Yes | No description |

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
GET /api/v1/training/jobs/{job_id}
Content-Type: application/json
Accept: application/json
```

