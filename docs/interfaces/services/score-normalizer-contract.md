# Score Normalizer Contract

## Overview
Service contract for Score Normalizer (service)

**Source File**: `app/services/score_normalizer.py`

## Interface Specification

### Classes

#### ScoreNormalizationError

Base exception for score normalization errors.

**Inherits from**: Exception

#### NormalizationConfig

Configuration for score normalization.

**Attributes**:

- `normalization_method: str`
- `confidence_method: str`
- `min_samples_for_stats: int`
- `outlier_threshold: float`

#### ScoreNormalizer

Service for normalizing anomaly scores across age groups.

## Methods

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

**Signature**: `normalize_score(raw_score: float, age_group_model_id: int, db: Session) -> float`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `raw_score` | `float` | Parameter description |
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `float`

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

**Signature**: `calculate_confidence(raw_score: float, normalized_score: float, age_group_model_id: int, db: Session, threshold: float, age_group_stats: Dict) -> float`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `raw_score` | `float` | Parameter description |
| `normalized_score` | `float` | Parameter description |
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |
| `threshold` | `float` | Parameter description |
| `age_group_stats` | `Dict` | Parameter description |

**Returns**: `float`

### compare_scores

Compare and rank multiple scores across potentially different age groups.

Args:
    scores_and_ages: List of (score, age) tuples
    db: Database session
    
Returns:
    List of dictionaries with normalized scores and rankings

**Signature**: `compare_scores(scores_and_ages: <ast.Subscript object at 0x11047e250>, db: Session) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `scores_and_ages` | `<ast.Subscript object at 0x11047e250>` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `List[Dict]`

### get_normalization_summary

Get a summary of normalization statistics across all age groups.

Args:
    db: Database session
    
Returns:
    Dictionary containing normalization summary

**Signature**: `get_normalization_summary(db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### update_normalization_config

Update normalization configuration and clear cache.

Args:
    normalization_method: New normalization method
    confidence_method: New confidence calculation method

**Signature**: `update_normalization_config(normalization_method: Optional[str], confidence_method: Optional[str]) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `normalization_method` | `Optional[str]` | Parameter description |
| `confidence_method` | `Optional[str]` | Parameter description |

**Returns**: `None`

### clear_cache

Clear the statistics cache.

**Signature**: `clear_cache() -> None`

**Returns**: `None`

### detect_outliers

Detect outliers in a list of scores.

Args:
    scores: List of scores to analyze
    method: Outlier detection method ("z_score", "iqr")
    
Returns:
    List of boolean values indicating outliers

**Signature**: `detect_outliers(scores: List[float], method: str) -> List[bool]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `scores` | `List[float]` | Parameter description |
| `method` | `str` | Parameter description |

**Returns**: `List[bool]`

## Dependencies

- `app.models.database.AgeGroupModel`
- `app.models.database.AnomalyAnalysis`
- `app.models.database.Drawing`
- `app.services.model_manager.get_model_manager`
- `app.services.age_group_manager.get_age_group_manager`

## Defined Interfaces

### ScoreNormalizerInterface

**Type**: Protocol
**Implemented by**: ScoreNormalizer

**Methods**:

- `normalize_score(raw_score: float, age_group_model_id: int, db: Session) -> float`
- `calculate_confidence(raw_score: float, normalized_score: float, age_group_model_id: int, db: Session, threshold: float, age_group_stats: Dict) -> float`
- `compare_scores(scores_and_ages: <ast.Subscript object at 0x11047e250>, db: Session) -> List[Dict]`
- `get_normalization_summary(db: Session) -> Dict`
- `update_normalization_config(normalization_method: Optional[str], confidence_method: Optional[str]) -> None`
- `clear_cache() -> None`
- `detect_outliers(scores: List[float], method: str) -> List[bool]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/score_normalizer.py`
- Last validated: 2025-12-16 15:47:04

