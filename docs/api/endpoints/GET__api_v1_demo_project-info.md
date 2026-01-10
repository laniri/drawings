# GET /api/v1/demo/project-info

## Summary
Get Project Info

## Description
Get comprehensive project information for demo page.

Returns:
    Project description with technical details and features

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
GET /api/v1/demo/project-info
Content-Type: application/json
Accept: application/json
```

