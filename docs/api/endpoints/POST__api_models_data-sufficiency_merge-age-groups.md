# POST /api/models/data-sufficiency/merge-age-groups

## Summary
Merge Age Groups

## Description
Merge age groups to improve data sufficiency.

This endpoint deactivates the original age group models and creates
a new merged age group model with combined data.

## Tags
models

## Parameters
No parameters required.

## Request Body
Request body required

### Request Body Examples

**application/json**:
```json
{
  "original_groups": [
    [
      3.14,
      3.14
    ],
    [
      3.14,
      3.14
    ]
  ],
  "merged_group": [
    3.14,
    3.14
  ]
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
POST /api/models/data-sufficiency/merge-age-groups
Content-Type: application/json
Accept: application/json
```

