# POST /api/database/migrate

## Summary
Run Database Migration

## Description
Run database migrations to the specified revision.

- **target_revision**: Target migration revision (defaults to "head")

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
  "target_revision": "example_string"
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
POST /api/database/migrate
Content-Type: application/json
Accept: application/json
```

