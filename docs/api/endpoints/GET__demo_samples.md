# GET /demo/samples

## Summary
Get Demo Samples

## Description
Get all demo samples with analysis results.

Returns:
    List of demo samples with complete analysis data

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
GET /demo/samples
Content-Type: application/json
Accept: application/json
```

