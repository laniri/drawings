# GET /demo/statistics

## Summary
Get Demo Statistics

## Description
Get demo-specific statistics and metrics.

Returns:
    Demo statistics including sample counts and distributions

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
GET /demo/statistics
Content-Type: application/json
Accept: application/json
```

