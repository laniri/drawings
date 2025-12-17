# Percentile-Based Score Normalization Algorithm

## Overview

The score normalization algorithm transforms raw reconstruction loss values into interpretable scores on a 0-100 scale, where 0 represents no anomaly (most normal) and 100 represents maximal anomaly within the age group.

## Mathematical Formulation

### Percentile Ranking Method

Given a set of historical anomaly scores for an age group and a new score to normalize:

```
Let S = {s₁, s₂, ..., sₙ} be the set of historical anomaly scores
Let x be the new score to normalize

Percentile Rank (PR) = (Number of scores ≤ x) / Total number of scores × 100

Normalized Score = PR
```

### Implementation Algorithm

```python
def normalize_score(raw_score: float, age_group_model_id: int, db: Session) -> float:
    """
    Normalize anomaly score using percentile ranking within age group.
    
    Args:
        raw_score: Raw reconstruction loss value
        age_group_model_id: Age group for comparison
        db: Database session for historical data
        
    Returns:
        Normalized score on 0-100 scale
    """
    
    # Step 1: Retrieve historical scores for age group
    historical_scores = get_historical_scores(age_group_model_id, db)
    
    if not historical_scores:
        return 50.0  # Default middle value if no history
    
    # Step 2: Calculate percentile rank
    scores_array = np.array(historical_scores)
    
    # Method 1: Using scipy (preferred)
    if scipy_available:
        percentile_rank = scipy.stats.percentileofscore(
            scores_array, raw_score, kind='rank'
        )
    else:
        # Method 2: Manual calculation
        percentile_rank = (np.sum(scores_array <= raw_score) / len(scores_array)) * 100
    
    # Step 3: Ensure bounds and return
    return np.clip(percentile_rank, 0.0, 100.0)
```

## Algorithm Properties

### Monotonicity
- **Property**: If score A > score B, then normalized(A) ≥ normalized(B)
- **Guarantee**: Higher raw scores always result in higher or equal normalized scores
- **Importance**: Preserves relative ordering of anomaly severity

### Bounded Output
- **Range**: [0.0, 100.0]
- **Lower Bound**: 0.0 represents the lowest score in the historical dataset
- **Upper Bound**: 100.0 represents the highest score in the historical dataset

### Age Group Specificity
- **Isolation**: Normalization is performed independently for each age group
- **Rationale**: Developmental expectations vary significantly by age
- **Benefit**: Enables fair comparison within developmental cohorts

## Edge Cases and Handling

### Insufficient Historical Data
```python
if len(historical_scores) < min_samples_threshold:
    logger.warning(f"Limited historical data: {len(historical_scores)} samples")
    # Continue with available data but flag low confidence
```

### Identical Scores
```python
if np.std(historical_scores) == 0:
    # All historical scores are identical
    if raw_score == historical_scores[0]:
        return 50.0  # Same as historical average
    elif raw_score > historical_scores[0]:
        return 100.0  # Higher than all historical
    else:
        return 0.0   # Lower than all historical
```

### Extreme Outliers
```python
# Scores beyond historical range are clamped to bounds
if raw_score <= np.min(historical_scores):
    return 0.0
elif raw_score >= np.max(historical_scores):
    return 100.0
```

## Performance Characteristics

### Time Complexity
- **Best Case**: O(1) - cached statistics available
- **Average Case**: O(n) - where n is number of historical scores
- **Worst Case**: O(n log n) - if sorting is required

### Space Complexity
- **Memory Usage**: O(n) for storing historical scores array
- **Optimization**: Caching of statistics reduces repeated database queries

### Scalability
- **Linear Growth**: Performance scales linearly with historical dataset size
- **Optimization Strategies**:
  - Statistical caching for frequently accessed age groups
  - Incremental updates when new scores are added
  - Periodic cleanup of very old historical data

## Validation and Testing

### Property-Based Tests

#### Property 1: Bounded Output
```python
@given(st.floats(min_value=0.0, max_value=10.0))
def test_normalized_score_bounds(raw_score):
    normalized = normalize_score(raw_score, age_group_id, db)
    assert 0.0 <= normalized <= 100.0
```

#### Property 2: Monotonicity
```python
@given(st.floats(min_value=0.0, max_value=5.0), 
       st.floats(min_value=5.0, max_value=10.0))
def test_monotonicity(lower_score, higher_score):
    norm_lower = normalize_score(lower_score, age_group_id, db)
    norm_higher = normalize_score(higher_score, age_group_id, db)
    assert norm_lower <= norm_higher
```

#### Property 3: Consistency
```python
def test_normalization_consistency():
    # Same score should always produce same normalized value
    score = 0.5
    norm1 = normalize_score(score, age_group_id, db)
    norm2 = normalize_score(score, age_group_id, db)
    assert norm1 == norm2
```

### Unit Tests

#### Edge Case Testing
```python
def test_empty_historical_data():
    # Test behavior with no historical data
    result = normalize_score(0.5, empty_age_group_id, db)
    assert result == 50.0

def test_single_historical_score():
    # Test behavior with only one historical score
    setup_single_score_history(0.6, age_group_id, db)
    
    assert normalize_score(0.6, age_group_id, db) == 50.0  # Same as history
    assert normalize_score(0.7, age_group_id, db) == 100.0  # Higher than history
    assert normalize_score(0.5, age_group_id, db) == 0.0   # Lower than history
```

## Integration with System Components

### Score Normalizer Service
- **Location**: `app/services/score_normalizer.py`
- **Interface**: `ScoreNormalizer.normalize_score()`
- **Caching**: Implements statistical caching for performance

### Database Integration
- **Historical Data**: Retrieved from `AnomalyAnalysis` table
- **Age Group Filtering**: Uses `age_group_model_id` for isolation
- **Query Optimization**: Indexed queries for performance

### API Integration
- **Real-time Normalization**: Called during analysis pipeline
- **Batch Processing**: Supports bulk normalization operations
- **Recalculation**: Handles legacy data migration

## Configuration Parameters

### Normalization Config
```python
@dataclass
class NormalizationConfig:
    min_samples_for_stats: int = 30      # Minimum historical samples
    cache_ttl: int = 3600               # Cache time-to-live (seconds)
    outlier_threshold: float = 3.0       # Z-score threshold for outliers
```

### Tuning Guidelines
- **min_samples_for_stats**: Increase for more stable statistics, decrease for faster cold-start
- **cache_ttl**: Balance between performance and data freshness
- **outlier_threshold**: Adjust based on expected data distribution characteristics