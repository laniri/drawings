# GET /api/config/subjects

## Summary
Get Supported Subject Categories

## Description
Get list of supported subject categories.

This endpoint returns all supported subject categories that can be used
when uploading drawings, along with usage statistics.

## Tags
configuration

## Parameters
No parameters required.

## Responses

### 200 - Successful Response

**application/json**:
```json
{}
```


## Complete Request Example

```http
GET /api/config/subjects
Content-Type: application/json
Accept: application/json
```

