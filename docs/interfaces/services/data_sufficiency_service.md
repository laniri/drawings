# Data Sufficiency Service Service

Data Sufficiency Service for Children's Drawing Anomaly Detection System

This module provides data count validation for age groups, warning generation for insufficient
training data, and suggestions for data augmentation and age group merging.

## Class: DataSufficiencyError

Base exception for data sufficiency errors.

## Class: InsufficientDataError

Raised when there is insufficient data for training.

## Class: AgeGroupDataInfo

Information about data availability for an age group.

### to_dict

Convert to dictionary.

**Signature**: `to_dict()`

## Class: DataSufficiencyWarning

Warning about data sufficiency issues.

### to_dict

Convert to dictionary.

**Signature**: `to_dict()`

## Class: AgeGroupMergingSuggestion

Suggestion for merging age groups to improve data sufficiency.

### to_dict

Convert to dictionary.

**Signature**: `to_dict()`

## Class: DataSufficiencyAnalyzer

Analyzer for data sufficiency and quality assessment.

### analyze_age_group_data

Analyze data sufficiency for a specific age group.

Args:
    age_min: Minimum age for the group
    age_max: Maximum age for the group
    db: Database session
    
Returns:
    Data information for the age group

**Error Handling**: 
- Invalid age ranges (age_min >= age_max) are handled gracefully
- Returns AgeGroupDataInfo with sample_count=0 and is_sufficient=False for invalid ranges
- Does not raise exceptions for malformed age parameters

**Signature**: `analyze_age_group_data(age_min, age_max, db)`

### generate_data_warnings

Generate warnings for data sufficiency issues across age groups.

Args:
    age_groups: List of (age_min, age_max) tuples
    db: Database session
    
Returns:
    List of data sufficiency warnings

**Signature**: `generate_data_warnings(age_groups, db)`

### suggest_age_group_merging

Suggest age group merging strategies to improve data sufficiency.

Args:
    age_groups: List of (age_min, age_max) tuples
    db: Database session
    
Returns:
    List of merging suggestions

**Signature**: `suggest_age_group_merging(age_groups, db)`

## Class: DataAugmentationSuggester

Service for suggesting data augmentation strategies.

### suggest_augmentation_strategies

Suggest data augmentation strategies based on data analysis.

Args:
    age_group_data: Data information for the age group
    
Returns:
    Dictionary with augmentation suggestions

**Signature**: `suggest_augmentation_strategies(age_group_data)`

