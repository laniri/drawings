# Threshold Manager Contract

## Overview
Service contract for Threshold Manager (service)

**Source File**: `app/services/threshold_manager.py`

## Interface Specification

### Classes

#### ThresholdManagerError

Base exception for threshold manager errors.

**Inherits from**: Exception

#### ThresholdCalculationError

Raised when threshold calculation fails.

**Inherits from**: ThresholdManagerError

#### ThresholdConfig

Configuration for threshold management.

**Attributes**:

- `default_percentile: float`
- `min_samples_for_calculation: int`
- `confidence_levels: List[float]`

#### ThresholdManager

Manager for threshold calculation and dynamic updates.

## Methods

### calculate_percentile_threshold

Calculate threshold based on percentile of scores.

Args:
    scores: Array of anomaly scores
    percentile: Percentile value (0-100)
    
Returns:
    Threshold value at the specified percentile

**Signature**: `calculate_percentile_threshold(scores: np.ndarray, percentile: float) -> float`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `scores` | `np.ndarray` | Parameter description |
| `percentile` | `float` | Parameter description |

**Returns**: `float`

### calculate_model_threshold

Calculate threshold for a specific age group model using existing analysis results.

This optimized version uses existing anomaly scores from the database instead
of recalculating reconstruction losses, making it much faster.

Args:
    age_group_model_id: ID of the age group model
    db: Database session
    percentile: Percentile to use (defaults to config default)
    
Returns:
    Dictionary containing threshold information

**Signature**: `calculate_model_threshold(age_group_model_id: int, db: Session, percentile: Optional[float]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_model_id` | `int` | Parameter description |
| `db` | `Session` | Parameter description |
| `percentile` | `Optional[float]` | Parameter description |

**Returns**: `Dict`

### update_model_threshold

Update the threshold for a specific age group model.

Args:
    age_group_model_id: ID of the age group model
    new_threshold: New threshold value
    db: Database session
    
Returns:
    True if update was successful

**Signature**: `update_model_threshold(age_group_model_id: int, new_threshold: float, db: Session) -> bool`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_model_id` | `int` | Parameter description |
| `new_threshold` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `bool`

### recalculate_all_thresholds

Recalculate thresholds for all active age group models.

Args:
    db: Database session
    percentile: Percentile to use (defaults to config default)
    
Returns:
    Dictionary containing results for all models

**Signature**: `recalculate_all_thresholds(db: Session, percentile: Optional[float]) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |
| `percentile` | `Optional[float]` | Parameter description |

**Returns**: `Dict`

### get_threshold_for_age

Get the appropriate threshold for a given age.

Args:
    age: Age to get threshold for
    db: Database session
    
Returns:
    Threshold value or None if no appropriate model found

**Signature**: `get_threshold_for_age(age: float, db: Session) -> Optional[float]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Optional[float]`

### is_anomaly

Determine if a score represents an anomaly for a given age.

Args:
    anomaly_score: The computed anomaly score
    age: Age of the subject
    db: Database session
    
Returns:
    Tuple of (is_anomaly, threshold_used, model_info)

**Signature**: `is_anomaly(anomaly_score: float, age: float, db: Session) -> Tuple[bool, float, <ast.Subscript object at 0x110474dd0>]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `anomaly_score` | `float` | Parameter description |
| `age` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Tuple[bool, float, <ast.Subscript object at 0x110474dd0>]`

### get_current_percentile

Get the current threshold percentile.

**Signature**: `get_current_percentile() -> float`

**Returns**: `float`

### set_current_percentile

Set the current threshold percentile.

**Signature**: `set_current_percentile(percentile: float) -> None`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `percentile` | `float` | Parameter description |

**Returns**: `None`

### get_threshold_statistics

Get statistics about all thresholds in the system.

Args:
    db: Database session
    
Returns:
    Dictionary containing threshold statistics

**Signature**: `get_threshold_statistics(db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### clear_threshold_cache

Clear the threshold calculation cache.

**Signature**: `clear_threshold_cache() -> None`

**Returns**: `None`

## Dependencies

- `app.models.database.AgeGroupModel`
- `app.models.database.AnomalyAnalysis`
- `app.models.database.Drawing`
- `app.models.database.DrawingEmbedding`
- `app.services.model_manager.get_model_manager`
- `app.services.age_group_manager.get_age_group_manager`

## Defined Interfaces

### ThresholdManagerInterface

**Type**: Protocol
**Implemented by**: ThresholdManager

**Methods**:

- `calculate_percentile_threshold(scores: np.ndarray, percentile: float) -> float`
- `calculate_model_threshold(age_group_model_id: int, db: Session, percentile: Optional[float]) -> Dict`
- `update_model_threshold(age_group_model_id: int, new_threshold: float, db: Session) -> bool`
- `recalculate_all_thresholds(db: Session, percentile: Optional[float]) -> Dict`
- `get_threshold_for_age(age: float, db: Session) -> Optional[float]`
- `is_anomaly(anomaly_score: float, age: float, db: Session) -> Tuple[bool, float, <ast.Subscript object at 0x110474dd0>]`
- `get_current_percentile() -> float`
- `set_current_percentile(percentile: float) -> None`
- `get_threshold_statistics(db: Session) -> Dict`
- `clear_threshold_cache() -> None`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/threshold_manager.py`
- Last validated: 2025-12-16 15:47:05

