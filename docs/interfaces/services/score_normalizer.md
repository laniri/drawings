# Score Normalizer Service

Score Normalization Service for cross-age-group reconstruction loss normalization.

This service handles normalization of anomaly scores to a 0-100 scale where:
- 0 represents no anomaly (lowest scores in age group)
- 100 represents maximal anomaly (highest scores in age group)

Uses percentile ranking within age groups for intuitive score interpretation.

## Class: ScoreNormalizationError

Base exception for score normalization errors.

## Class: NormalizationConfig

Configuration for score normalization.

## Class: ScoreNormalizer

Service for normalizing anomaly scores across age groups.

### normalize_score

Normalize a raw anomaly score to a 0-100 scale using percentile ranking.

0 = no anomaly (lowest scores in age group)
100 = maximal anomaly (highest scores in age group)

Args:
    raw_score: Raw anomaly score to normalize
    age_group_model_id: ID of the age group model used
    db: Database session
    
Returns:
    Normalized score on 0-100 scale

**Signature**: `normalize_score(raw_score, age_group_model_id, db)`

### calculate_confidence

Calculate confidence level for an anomaly decision.

Args:
    raw_score: Raw anomaly score
    normalized_score: Normalized anomaly score
    age_group_model_id: ID of the age group model used (optional for test compatibility)
    db: Database session (optional for test compatibility)
    threshold: Threshold value (optional for test compatibility)
    age_group_stats: Pre-computed statistics (optional for test compatibility)
    
Returns:
    Confidence value between 0 and 1

**Signature**: `calculate_confidence(raw_score, normalized_score, age_group_model_id, db, threshold, age_group_stats)`

### compare_scores

Compare and rank multiple scores across potentially different age groups.

Args:
    scores_and_ages: List of (score, age) tuples
    db: Database session
    
Returns:
    List of dictionaries with normalized scores and rankings

**Signature**: `compare_scores(scores_and_ages, db)`

### get_normalization_summary

Get a summary of normalization statistics across all age groups.

Args:
    db: Database session
    
Returns:
    Dictionary containing normalization summary

**Signature**: `get_normalization_summary(db)`

### update_normalization_config

Update normalization configuration and clear cache.

Args:
    normalization_method: New normalization method
    confidence_method: New confidence calculation method

**Signature**: `update_normalization_config(normalization_method, confidence_method)`

### clear_cache

Clear the statistics cache.

**Signature**: `clear_cache()`

### detect_outliers

Detect outliers in a list of scores.

Args:
    scores: List of scores to analyze
    method: Outlier detection method ("z_score", "iqr")
    
Returns:
    List of boolean values indicating outliers

**Signature**: `detect_outliers(scores, method)`

