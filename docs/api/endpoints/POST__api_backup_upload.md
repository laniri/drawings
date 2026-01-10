# POST /api/backup/upload

## Summary
Upload backup file

## Description
Upload a backup file for restoration.

## Tags
backup

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**multipart/form-data**:
```json
{
  "file": "example_string"
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
POST /api/backup/upload
Content-Type: application/json
Accept: application/json
```

