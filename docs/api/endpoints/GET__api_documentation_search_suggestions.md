# GET /api/documentation/search/suggestions

## Summary
Get Search Suggestions

## Description
Get search suggestions for autocomplete.

Provides intelligent search suggestions based on indexed content
and common search patterns.

## Tags
documentation

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| query | query | string | Yes | Partial query for suggestions |
| limit | query | integer | No | Maximum number of suggestions |

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
GET /api/documentation/search/suggestions
Content-Type: application/json
Accept: application/json
```

