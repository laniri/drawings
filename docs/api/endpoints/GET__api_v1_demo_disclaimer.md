# GET /api/v1/demo/disclaimer

## Summary
Get Medical Disclaimer

## Description
Get medical disclaimer and warnings for demo content.

Returns:
    Medical disclaimer with all required warnings

## Tags
demo

## Parameters
No parameters required.

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


## Complete Request Example

```http
GET /api/v1/demo/disclaimer
Content-Type: application/json
Accept: application/json
```

