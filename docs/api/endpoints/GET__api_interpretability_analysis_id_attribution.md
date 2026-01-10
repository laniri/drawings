# GET /api/interpretability/{analysis_id}/attribution

## Summary
Get Anomaly Attribution

## Description
Get detailed anomaly attribution breakdown (age vs subject vs visual).

This endpoint provides detailed information about what contributed
to the anomaly detection: age-related factors, subject-specific factors,
or visual characteristics.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| analysis_id | path | integer | Yes | No description |

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
GET /api/interpretability/{analysis_id}/attribution
Content-Type: application/json
Accept: application/json
```

