# POST /api/analysis/embeddings/{drawing_id}

## Summary
Generate Embedding

## Description
Generate embedding for a drawing without requiring a trained model.

This endpoint is used during the training phase to generate embeddings
for all drawings before training the autoencoder models.

## Tags
analysis

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| drawing_id | path | integer | Yes | No description |

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
POST /api/analysis/embeddings/{drawing_id}
Content-Type: application/json
Accept: application/json
```

