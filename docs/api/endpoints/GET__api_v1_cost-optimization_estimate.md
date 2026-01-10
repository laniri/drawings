# GET /api/v1/cost-optimization/estimate

## Summary
Get Cost Estimate

## Description
Get estimated monthly costs for optimized AWS resources.

Returns cost breakdown and compliance status for the production deployment.

## Tags
cost-optimization

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "total_monthly_cost": 3.14,
  "is_within_budget": true,
  "cost_breakdown": [
    {
      "service_name": "example_string",
      "monthly_cost_usd": 3.14,
      "resource_type": "example_string",
      "configuration": {},
      "optimization_applied": true
    },
    {
      "service_name": "example_string",
      "monthly_cost_usd": 3.14,
      "resource_type": "example_string",
      "configuration": {},
      "optimization_applied": true
    }
  ],
  "target_range": {}
}
```


## Complete Request Example

```http
GET /api/v1/cost-optimization/estimate
Content-Type: application/json
Accept: application/json
```

