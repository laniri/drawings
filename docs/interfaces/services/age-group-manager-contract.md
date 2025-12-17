# Age Group Manager Contract

## Overview
Service contract for Age Group Manager (service)

**Source File**: `app/services/age_group_manager.py`

## Interface Specification

### Classes

#### AgeGroupManagerError

Base exception for age group manager errors.

**Inherits from**: Exception

#### InsufficientDataError

Raised when there's insufficient data for age group modeling.

**Inherits from**: AgeGroupManagerError

#### AgeGroupConfig

Configuration for age group management.

**Attributes**:

- `min_samples_per_group: int`
- `default_age_span: float`
- `max_age_span: float`
- `merge_threshold: int`

#### AgeGroupManager

Manager for automatic age group creation and merging.

## Methods

### analyze_age_distribution

Analyze the distribution of ages in the drawing dataset.

Args:
    db: Database session
    
Returns:
    Dictionary containing age distribution statistics

**Signature**: `analyze_age_distribution(db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### suggest_age_groups

Suggest optimal age groups based on data distribution.

Args:
    db: Database session
    
Returns:
    List of (age_min, age_max) tuples for suggested age groups

**Signature**: `suggest_age_groups(db: Session) -> <ast.Subscript object at 0x11046f010>`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `<ast.Subscript object at 0x11046f010>`

### create_age_groups

Create age group models based on data distribution.

Args:
    db: Database session
    force_recreate: Whether to recreate existing models
    
Returns:
    List of created age group model information

**Signature**: `create_age_groups(db: Session, force_recreate: bool) -> List[Dict]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |
| `force_recreate` | `bool` | Parameter description |

**Returns**: `List[Dict]`

### find_appropriate_model

Find the appropriate age group model for a given age.

Args:
    age: Age to find model for
    db: Database session
    
Returns:
    AgeGroupModel instance or None if no appropriate model found

**Signature**: `find_appropriate_model(age: float, db: Session) -> Optional[AgeGroupModel]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `Optional[AgeGroupModel]`

### get_age_group_coverage

Get information about age group coverage.

Args:
    db: Database session
    
Returns:
    Dictionary containing coverage information

**Signature**: `get_age_group_coverage(db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

### validate_age_group_data

Validate that age groups have sufficient data and are properly configured.

Args:
    db: Database session
    
Returns:
    Dictionary containing validation results

**Signature**: `validate_age_group_data(db: Session) -> Dict`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `db` | `Session` | Parameter description |

**Returns**: `Dict`

## Dependencies

- `app.models.database.Drawing`
- `app.models.database.AgeGroupModel`
- `app.services.model_manager.ModelManager`
- `app.services.model_manager.TrainingConfig`
- `app.services.model_manager.get_model_manager`

## Defined Interfaces

### AgeGroupManagerInterface

**Type**: Protocol
**Implemented by**: AgeGroupManager

**Methods**:

- `analyze_age_distribution(db: Session) -> Dict`
- `suggest_age_groups(db: Session) -> <ast.Subscript object at 0x11046f010>`
- `create_age_groups(db: Session, force_recreate: bool) -> List[Dict]`
- `find_appropriate_model(age: float, db: Session) -> Optional[AgeGroupModel]`
- `get_age_group_coverage(db: Session) -> Dict`
- `validate_age_group_data(db: Session) -> Dict`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/age_group_manager.py`
- Last validated: 2025-12-16 15:47:04

