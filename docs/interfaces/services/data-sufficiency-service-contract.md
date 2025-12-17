# Data Sufficiency Service Contract

## Overview
Service contract for Data Sufficiency Service (service)

**Source File**: `app/services/data_sufficiency_service.py`

## Interface Specification

### Classes

#### DataSufficiencyError

Base exception for data sufficiency errors.

**Inherits from**: Exception

#### InsufficientDataError

Raised when there is insufficient data for training.

**Inherits from**: DataSufficiencyError

#### AgeGroupDataInfo

Information about data availability for an age group.

**Attributes**:

- `age_min: float`
- `age_max: float`
- `sample_count: int`
- `is_sufficient: bool`
- `recommended_min_samples: int`
- `data_quality_score: float`
- `subjects_distribution: Dict[str, int]`
- `age_distribution: List[float]`

#### DataSufficiencyWarning

Warning about data sufficiency issues.

**Attributes**:

- `warning_type: str`
- `severity: str`
- `age_group_min: float`
- `age_group_max: float`
- `current_samples: int`
- `recommended_samples: int`
- `message: str`
- `suggestions: List[str]`

#### AgeGroupMergingSuggestion

Suggestion for merging age groups to improve data sufficiency.

**Attributes**:

- `original_groups: <ast.Subscript object at 0x1105aebd0>`
- `merged_group: Tuple[float, float]`
- `combined_sample_count: int`
- `improvement_score: float`
- `rationale: str`

#### DataSufficiencyAnalyzer

Analyzer for data sufficiency and quality assessment.

#### DataAugmentationSuggester

Service for suggesting data augmentation strategies.

## Methods

### to_dict

Convert to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### to_dict

Convert to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### to_dict

Convert to dictionary.

**Signature**: `to_dict() -> Dict[str, Any]`

**Returns**: `Dict[str, Any]`

### analyze_age_group_data

Analyze data sufficiency for a specific age group.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    db: Database session
    
Returns:
    Data information for the age group

**Signature**: `analyze_age_group_data(age_min: float, age_max: float, db: Session) -> AgeGroupDataInfo`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_min` | `float` | Parameter description |
| `age_max` | `float` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `AgeGroupDataInfo`

### generate_data_warnings

Generate warnings for data sufficiency issues across age groups.

Args:
    age_groups: List of (age_min, age_max) tuples
    db: Database session
    
Returns:
    List of data sufficiency warnings

**Signature**: `generate_data_warnings(age_groups: <ast.Subscript object at 0x1104e40d0>, db: Session) -> List[DataSufficiencyWarning]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_groups` | `<ast.Subscript object at 0x1104e40d0>` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `List[DataSufficiencyWarning]`

### suggest_age_group_merging

Suggest age group merging strategies to improve data sufficiency.

Args:
    age_groups: List of (age_min, age_max) tuples
    db: Database session
    
Returns:
    List of merging suggestions

**Signature**: `suggest_age_group_merging(age_groups: <ast.Subscript object at 0x110477990>, db: Session) -> List[AgeGroupMergingSuggestion]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_groups` | `<ast.Subscript object at 0x110477990>` | Parameter description |
| `db` | `Session` | Parameter description |

**Returns**: `List[AgeGroupMergingSuggestion]`

### suggest_augmentation_strategies

Suggest data augmentation strategies based on data analysis.

Args:
    age_group_data: Data information for the age group
    
Returns:
    Dictionary with augmentation suggestions

**Signature**: `suggest_augmentation_strategies(age_group_data: AgeGroupDataInfo) -> Dict[str, Any]`

**Parameters**:

| Name | Type | Description |
|------|------|-------------|
| `age_group_data` | `AgeGroupDataInfo` | Parameter description |

**Returns**: `Dict[str, Any]`

## Dependencies

- `app.models.database.Drawing`
- `app.models.database.AgeGroupModel`

## Defined Interfaces

### AgeGroupDataInfoInterface

**Type**: Protocol
**Implemented by**: AgeGroupDataInfo

**Methods**:

- `to_dict() -> Dict[str, Any]`

### DataSufficiencyWarningInterface

**Type**: Protocol
**Implemented by**: DataSufficiencyWarning

**Methods**:

- `to_dict() -> Dict[str, Any]`

### AgeGroupMergingSuggestionInterface

**Type**: Protocol
**Implemented by**: AgeGroupMergingSuggestion

**Methods**:

- `to_dict() -> Dict[str, Any]`

### DataSufficiencyAnalyzerInterface

**Type**: Protocol
**Implemented by**: DataSufficiencyAnalyzer

**Methods**:

- `analyze_age_group_data(age_min: float, age_max: float, db: Session) -> AgeGroupDataInfo`
- `generate_data_warnings(age_groups: <ast.Subscript object at 0x1104e40d0>, db: Session) -> List[DataSufficiencyWarning]`
- `suggest_age_group_merging(age_groups: <ast.Subscript object at 0x110477990>, db: Session) -> List[AgeGroupMergingSuggestion]`

### DataAugmentationSuggesterInterface

**Type**: Protocol
**Implemented by**: DataAugmentationSuggester

**Methods**:

- `suggest_augmentation_strategies(age_group_data: AgeGroupDataInfo) -> Dict[str, Any]`

## Usage Examples

```python
# Example usage of the service contract
# TODO: Add specific usage examples
```

## Validation

This contract is automatically validated against the implementation in:
- Source file: `app/services/data_sufficiency_service.py`
- Last validated: 2025-12-16 15:47:05

