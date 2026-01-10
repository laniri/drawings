# POST /api/v1/cost-optimization/apply-s3-lifecycle/{bucket_name}

## Summary
Apply S3 Lifecycle Optimization

## Description
Apply S3 lifecycle optimization to a specific bucket.

Args:
    bucket_name: Name of the S3 bucket to optimize

Returns:
    Success status of the lifecycle policy application

## Tags
cost-optimization

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| bucket_name | path | string | Yes | No description |

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
POST /api/v1/cost-optimization/apply-s3-lifecycle/{bucket_name}
Content-Type: application/json
Accept: application/json
```

