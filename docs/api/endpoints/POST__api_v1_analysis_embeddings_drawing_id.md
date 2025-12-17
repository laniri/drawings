# POST /api/v1/analysis/embeddings/{drawing_id}

## Summary
Generate Embedding

## Description
Generate embedding for a drawing without requiring a trained model.

This endpoint is used during the training phase to generate embeddings
for all drawings before training the autoencoder models.

## Parameters
- **drawing_id** (path): No description

## Responses
- **200**: Successful Response
- **422**: Validation Error

## Example
```http
POST /api/v1/analysis/embeddings/{drawing_id}
```
