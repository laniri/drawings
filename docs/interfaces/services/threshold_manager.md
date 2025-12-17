# Threshold Manager Service

Threshold Management Service for configurable anomaly detection thresholds.

This service handles threshold calculation, management, and dynamic updates
for anomaly detection in children's drawings analysis.

## Class: ThresholdManagerError

Base exception for threshold manager errors.

## Class: ThresholdCalculationError

Raised when threshold calculation fails.

## Class: ThresholdConfig

Configuration for threshold management.

## Class: ThresholdManager

Manager for threshold calculation and dynamic updates.

### calculate_percentile_threshold

Calculate threshold based on percentile of scores.

Args:
    scores: Array of anomaly scores
    percentile: Percentile value (0-100)
    
Returns:
    Threshold value at the specified percentile

**Signature**: `calculate_percentile_threshold(scores, percentile)`

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

**Signature**: `calculate_model_threshold(age_group_model_id, db, percentile)`

### update_model_threshold

Update the threshold for a specific age group model.

Args:
    age_group_model_id: ID of the age group model
    new_threshold: New threshold value
    db: Database session
    
Returns:
    True if update was successful

**Signature**: `update_model_threshold(age_group_model_id, new_threshold, db)`

### recalculate_all_thresholds

Recalculate thresholds for all active age group models.

Args:
    db: Database session
    percentile: Percentile to use (defaults to config default)
    
Returns:
    Dictionary containing results for all models

**Signature**: `recalculate_all_thresholds(db, percentile)`

### get_threshold_for_age

Get the appropriate threshold for a given age.

Args:
    age: Age to get threshold for
    db: Database session
    
Returns:
    Threshold value or None if no appropriate model found

**Signature**: `get_threshold_for_age(age, db)`

### is_anomaly

Determine if a score represents an anomaly for a given age.

Args:
    anomaly_score: The computed anomaly score
    age: Age of the subject
    db: Database session
    
Returns:
    Tuple of (is_anomaly, threshold_used, model_info)

**Signature**: `is_anomaly(anomaly_score, age, db)`

### get_current_percentile

Get the current threshold percentile.

**Signature**: `get_current_percentile()`

### set_current_percentile

Set the current threshold percentile.

**Signature**: `set_current_percentile(percentile)`

### get_threshold_statistics

Get statistics about all thresholds in the system.

Args:
    db: Database session
    
Returns:
    Dictionary containing threshold statistics

**Signature**: `get_threshold_statistics(db)`

### clear_threshold_cache

Clear the threshold calculation cache.

**Signature**: `clear_threshold_cache()`

