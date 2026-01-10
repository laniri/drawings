# GET /api/cost-optimization/compliance

## Summary
Validate Cost Compliance

## Description
Validate cost compliance against budget requirements.

Returns compliance status and detailed cost analysis.

## Tags
cost-optimization

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "is_compliant": true,
  "total_estimated_cost": 3.14,
  "budget_limit": 3.14,
  "target_range": {},
  "cost_breakdown": [
    {},
    {}
  ],
  "recommendations": [
    "example_string",
    "example_string"
  ]
}
```


## Complete Request Example

```http
GET /api/cost-optimization/compliance
Content-Type: application/json
Accept: application/json
```

