# POST /api/v1/export

## Summary
Export system data

## Description
Export system data in specified format.

## Tags
backup

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| format | query | string | No | Export format |
| include_embeddings | query | boolean | No | Include embedding vectors |

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
POST /api/v1/export
Content-Type: application/json
Accept: application/json
```

