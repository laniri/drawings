# Analysis Results Contradictions - Fix Documentation

## Issues Identified

### 1. Contradictory Attribution for Normal Drawings
**Problem**: Analysis ID 38385 shows:
- `is_anomaly: false` (marked as normal)
- `anomaly_attribution: "age"` (suggests different age group)
- Explanation: "The drawing appears more typical for a different age group"

**Root Cause**: The `determine_attribution()` function is called for ALL drawings, including normal ones, and returns "age" as a default/fallback value even when there's no anomaly.

**Fix Location**: `app/api/api_v1/endpoints/analysis.py` - `perform_single_analysis()` function

**Solution**:
```python
# Only determine attribution if the drawing is actually anomalous
if is_anomaly:
    anomaly_attribution = model_manager.determine_attribution(
        embedding_data, age_group_model.id, db
    )
else:
    anomaly_attribution = None  # No attribution needed for normal drawings
```

### 2. Misleading Explanation Text
**Problem**: The interpretability explanation says:
- "The low anomaly score of 0.001 (normalized: 65.3/100)"
- But 65.3/100 is NOT a low score - it's medium-high

**Root Cause**: The explanation template in `_create_simple_saliency_map()` uses "low anomaly score" for all non-anomalous drawings, regardless of the actual normalized score value.

**Fix Location**: `app/api/api_v1/endpoints/analysis.py` - explanation generation in `perform_single_analysis()`

**Solution**:
```python
# Generate explanation based on normalized score ranges
if is_anomaly:
    if normalized_score >= 80:
        score_description = "high"
    elif normalized_score >= 60:
        score_description = "moderately high"
    else:
        score_description = "moderate"
    
    explanation_text = f"Analysis reveals patterns that deviate from typical developmental expectations for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates significant differences..."
else:
    if normalized_score < 40:
        score_description = "low"
    elif normalized_score < 60:
        score_description = "moderate"
    else:
        score_description = "moderately elevated but still within normal range"
    
    explanation_text = f"This drawing demonstrates age-appropriate developmental patterns for a {drawing.age_years}-year-old child. The {score_description} anomaly score of {anomaly_score:.3f} (normalized: {normalized_score:.1f}/100) indicates the drawing aligns with expected developmental milestones..."
```

### 3. Confusing Attribution Display
**Problem**: The frontend shows "Attribution Explanation: The drawing appears more typical for a different age group" even for normal drawings.

**Fix Location**: `frontend/src/pages/AnalysisPage.tsx` - Attribution explanation section

**Solution**:
```typescript
{/* Attribution Explanation - Only show for anomalies */}
{analysis.anomaly_attribution && analysis.is_anomaly && (
  <Alert severity="info" sx={{ mt: 2 }}>
    <Typography variant="body2">
      <strong>Attribution Explanation:</strong>{' '}
      {analysis.anomaly_attribution === 'visual' && 
        'The anomaly is primarily in the visual features of the drawing (shapes, lines, spatial relationships).'}
      {analysis.anomaly_attribution === 'subject' && 
        'The anomaly is primarily related to the subject category representation.'}
      {analysis.anomaly_attribution === 'both' && 
        'The anomaly involves both visual features and subject representation.'}
      {analysis.anomaly_attribution === 'age' && 
        'The drawing appears more typical for a different age group.'}
    </Typography>
  </Alert>
)}
```

### 4. Confidence Display Inconsistency
**Problem**: The page shows both "High Confidence" (green) and "Medium Confidence" (orange) simultaneously.

**Root Cause**: The main analysis page shows confidence from `analysis.confidence` (99.75%), while the ConfidenceIndicator component fetches separate confidence metrics from the `/api/interpretability/{analysisId}/confidence` endpoint which calculates different confidence values.

**Fix Location**: Frontend should clarify which confidence metric is being displayed

**Solution**: Update the display to show:
- "Analysis Confidence: 99.8%" (from analysis.confidence)
- "Explanation Reliability: Medium" (from confidence metrics endpoint)

## Implementation Priority

1. **HIGH**: Fix attribution logic to not assign attribution to normal drawings
2. **HIGH**: Fix explanation text to accurately describe normalized scores
3. **MEDIUM**: Update frontend to only show attribution for anomalies
4. **MEDIUM**: Clarify confidence metric labels in the UI

## Testing Checklist

After fixes:
- [ ] Normal drawings (is_anomaly: false) should have anomaly_attribution: null
- [ ] Explanation text should accurately describe the normalized score level
- [ ] Attribution explanation should only appear for anomalous drawings
- [ ] Confidence labels should clearly indicate which metric they represent
- [ ] Re-analyze drawing ID 1 (analysis 38385) and verify results are consistent
