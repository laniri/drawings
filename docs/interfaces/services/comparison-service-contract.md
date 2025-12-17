# Comparison Service Contract

## Overview
Service contract for Comparison Service (service)

**Source File**: `app/services/comparison_service.py`

## Interface Specification

### Classes

#### ComparisonService

Service for finding similar normal examples for comparison.

## Methods

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

**Signature**: `find_similar_normal_examples(target_drawing_id: int, age_group_min: float, age_group_max: float, db: Session, max_examples: int, similarity_threshold: float) -> <ast.Subscript object at 0x1104823d0>`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `target_drawing_id` | `int` | Parameter description |
| `age_group_min` | `float` | Parameter description |
| `age_group_max` | `float` | Parameter description |
| `db` | `Session` | Parameter description |
| `max_examples` | `int` | Parameter description |
| `similarity_threshold` | `float` | Parameter description |

**Returns**: `<ast.Subscript object at 0x1104823d0>`

### get_comparison_statistics

Get statistics about available comparison examples in an age group.

**Signature**: `get_comparison_statistics(age_group_min: float, age_group_max: float, db: Session) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_min` | `float` | Parameter description |
| `age_group_max` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Dict[str, Any]`

## Dependencies

- `app.models.database.Drawing`
- `app.models.database.DrawingEmbedding`
- `app.models.database.AnomalyAnalysis`

## Defined Interfaces

### ComparisonServiceInterface

**Type**: Protocol
**Implemented by**: ComparisonService

**Methods**:

- `find_similar_normal_examples(target_drawing_id: int, age_group_min: float, age_group_max: float, db: Session, max_examples: int, similarity_threshold: float) -> <ast.Subscript object at 0x1104823d0>`
- `get_comparison_statistics(age_group_min: float, age_group_max: float, db: Session) -> Dict[str, Any]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/comparison_service.py`
- Last validated: 2025-12-16 15:47:04

