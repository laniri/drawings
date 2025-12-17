# Comparison Service Service

Comparison service for finding similar normal examples.

This service provides functionality to find similar normal drawings
from the same age group for comparison purposes when displaying
analysis results.

## Class: ComparisonService

Service for finding similar normal examples for comparison.

### find_similar_normal_examples

Find similar normal examples from the same age group.

Args:
    target_drawing_id: ID of the drawing to find comparisons for
    age_group_min: Minimum age for the age group
    age_group_max: Maximum age for the age group
    db: Database session
    max_examples: Maximum number of examples to return
    similarity_threshold: Minimum similarity score (0-1)
    
Returns:
    List of similar normal examples with metadata

**Signature**: `find_similar_normal_examples(target_drawing_id, age_group_min, age_group_max, db, max_examples, similarity_threshold)`

### get_comparison_statistics

Get statistics about available comparison examples in an age group.

**Signature**: `get_comparison_statistics(age_group_min, age_group_max, db)`

