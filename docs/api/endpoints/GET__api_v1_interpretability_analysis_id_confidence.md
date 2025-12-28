# GET /api/v1/interpretability/{analysis_id}/confidence

## Summary
Get Comprehensive Confidence Metrics

## Description
Get detailed confidence metrics and reliability assessment for interpretability results.

This endpoint provides multi-dimensional confidence information including model certainty,
explanation reliability, data sufficiency, and technical breakdown to help users
assess the trustworthiness of the analysis and interpretations.

## Parameters
- **analysis_id** (path, integer, required): The ID of the analysis to get confidence metrics for

## Response Schema
```json
{
  "overall_confidence": 0.85,
  "explanation_reliability": 0.82,
  "model_certainty": 0.88,
  "data_sufficiency": "sufficient",
  "warnings": [
    "Limited training data for this specific subject category"
  ],
  "technical_details": {
    "base_model_confidence": 0.87,
    "training_data_quality": 0.90,
    "score_extremity": 0.75,
    "age_group_sample_count": 1250,
    "analysis_method": "subject-aware autoencoder",
    "vision_model": "Vision Transformer (ViT-Base)"
  }
}
```

## Response Fields

### Core Metrics
- **overall_confidence** (float): Composite confidence score (0.0-1.0)
- **explanation_reliability** (float): Trustworthiness of visual explanations (0.0-1.0)
- **model_certainty** (float): AI model's confidence in its prediction (0.0-1.0)
- **data_sufficiency** (string): Quality assessment ("sufficient", "limited", "insufficient")
- **warnings** (array): Important considerations and limitations

### Technical Details
- **base_model_confidence** (float): Core model confidence before adjustments
- **training_data_quality** (float): Quality score of training data for this age group
- **score_extremity** (float): How extreme the anomaly score is relative to training data
- **age_group_sample_count** (integer): Number of training samples for this age group
- **analysis_method** (string): Method used for anomaly detection
- **vision_model** (string): Vision model used for feature extraction

## Confidence Level Interpretation

### High Confidence (0.8+)
- Strong evidence supports the analysis
- Results are likely reliable for decision-making
- Model has high certainty based on training data
- Explanations are trustworthy and actionable

### Medium Confidence (0.6-0.79)
- Moderate evidence supports the analysis
- Consider additional context or assessment
- Model shows reasonable certainty
- Use results as one factor among others

### Low Confidence (<0.6)
- Limited evidence available
- Use results cautiously
- Seek additional professional input
- Consider as preliminary screening only

## Responses
- **200**: Successful Response - Returns confidence metrics object
- **404**: Analysis not found
- **422**: Validation Error - Invalid analysis_id format

## Example Request
```http
GET /api/v1/interpretability/522/confidence
```

## Example Response
```json
{
  "overall_confidence": 0.85,
  "explanation_reliability": 0.82,
  "model_certainty": 0.88,
  "data_sufficiency": "sufficient",
  "warnings": [],
  "technical_details": {
    "base_model_confidence": 0.87,
    "training_data_quality": 0.90,
    "score_extremity": 0.75,
    "age_group_sample_count": 1250,
    "analysis_method": "subject-aware autoencoder",
    "vision_model": "Vision Transformer (ViT-Base)"
  }
}
```
