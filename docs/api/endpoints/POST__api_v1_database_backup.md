# POST /api/v1/database/backup

## Summary
Create Database Backup

## Description
Create a database backup with optional S3 upload.

- **upload_to_s3**: Whether to upload to S3 (defaults to environment setting)
- **include_files**: Whether to include uploaded files and static content

## Tags
database

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "upload_to_s3": {},
  "include_files": true
}
```


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
POST /api/v1/database/backup
Content-Type: application/json
Accept: application/json
```

