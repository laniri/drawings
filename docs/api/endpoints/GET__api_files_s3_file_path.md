# GET /api/files/s3/{file_path}

## Summary
Serve S3 File

## Description
Serve a file from S3 storage.

This endpoint downloads files from S3 and serves them with proper caching headers.
This avoids presigned URL expiration issues and allows CloudFront to cache responses.

Args:
    file_path: S3 key path (e.g., "drawings/20240108_123456_abc123.png")

Returns:
    File response with caching headers

## Tags
files

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| file_path | path | string | Yes | No description |

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
GET /api/files/s3/{file_path}
Content-Type: application/json
Accept: application/json
```

