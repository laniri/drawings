# GET /api/interpretability/{analysis_id}/interactive

## Summary
Get Interactive Interpretability

## Description
Get interactive saliency data with hoverable regions and click explanations.

This endpoint provides enhanced interpretability data that supports
interactive user interfaces with hover explanations and click-to-zoom functionality.

## Tags
interpretability

## Parameters
| Name | Location | Type | Required | Description |
|------|----------|------|----------|-------------|
| analysis_id | path | integer | Yes | No description |

## Responses

### 200 - Successful Response

**application/json**:
```json
{
  "saliency_regions": [
    {
      "region_id": "example_string",
      "bounding_box": [
        42,
        42
      ],
      "importance_score": 3.14,
      "spatial_location": "example_string",
      "hover_explanation": "example_string",
      "click_explanation": "example_string"
    },
    {
      "region_id": "example_string",
      "bounding_box": [
        42,
        42
      ],
      "importance_score": 3.14,
      "spatial_location": "example_string",
      "hover_explanation": "example_string",
      "click_explanation": "example_string"
    }
  ],
  "attention_patches": [
    {
      "patch_id": "example_string",
      "coordinates": [
        42,
        42
      ],
      "attention_weight": 3.14,
      "layer_index": 42,
      "head_index": 42
    },
    {
      "patch_id": "example_string",
      "coordinates": [
        42,
        42
      ],
      "attention_weight": 3.14,
      "layer_index": 42,
      "head_index": 42
    }
  ],
  "region_explanations": {},
  "confidence_scores": {},
  "interaction_metadata": {}
}
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
GET /api/interpretability/{analysis_id}/interactive
Content-Type: application/json
Accept: application/json
```

