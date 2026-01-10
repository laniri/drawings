# GET /api/v1/cost-optimization/optimization

## Summary
Get Cost Optimization

## Description
Get cost optimization configurations and recommendations.

Returns optimized configurations for ECS Fargate, S3, and CloudFront.

## Tags
cost-optimization

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "ecs_fargate_config": {},
  "s3_lifecycle_policy": {},
  "cloudfront_cache_config": {},
  "recommendations": [
    "example_string",
    "example_string"
  ]
}
```


## Complete Request Example

```http
GET /api/v1/cost-optimization/optimization
Content-Type: application/json
Accept: application/json
```

