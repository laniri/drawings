# GET /api/demo/samples/{sample_id}

## Summary
Get Demo Sample

## Description
Get a specific demo sample by ID.

Args:
    sample_id: ID of the demo sample

Returns:
    Demo sample with complete analysis data

## Tags
demo

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| sample_id | path | integer | Yes | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "success": true,
  "message": "example_string",
  "data": {}
}
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
GET /api/demo/samples/{sample_id}
Content-Type: application/json
Accept: application/json
```

