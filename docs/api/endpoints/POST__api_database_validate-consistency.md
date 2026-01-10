# POST /api/database/validate-consistency

## Summary
Validate Cross Environment Consistency

## Description
Validate database schema consistency across environments.

- **other_db_url**: Database URL of the other environment to compare

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
  "other_db_url": "example_string"
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
POST /api/database/validate-consistency
Content-Type: application/json
Accept: application/json
```

