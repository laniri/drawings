# Age Group Manager Service

Age Group Management Service for automatic age group creation and merging.

This service handles age group creation, merging when insufficient data is available,
and manages the lifecycle of age-specific autoencoder models.

## Class: AgeGroupManagerError

Base exception for age group manager errors.

## Class: InsufficientDataError

Raised when there's insufficient data for age group modeling.

## Class: AgeGroupConfig

Configuration for age group management.

## Class: AgeGroupManager

Manager for automatic age group creation and merging.

### analyze_age_distribution

Analyze the distribution of ages in the drawing dataset.

Args:
    db: Database session
    
Returns:
    Dictionary containing age distribution statistics

**Signature**: `analyze_age_distribution(db)`

### suggest_age_groups

Suggest optimal age groups based on data distribution.

Args:
    db: Database session
    
Returns:
    List of (age_min, age_max) tuples for suggested age groups

**Signature**: `suggest_age_groups(db)`

### create_age_groups

Create age group models based on data distribution.

Args:
    db: Database session
    force_recreate: Whether to recreate existing models
    
Returns:
    List of created age group model information

**Signature**: `create_age_groups(db, force_recreate)`

### find_appropriate_model

Find the appropriate age group model for a given age.

Args:
    age: Age to find model for
    db: Database session
    
Returns:
    AgeGroupModel instance or None if no appropriate model found

**Signature**: `find_appropriate_model(age, db)`

### get_age_group_coverage

Get information about age group coverage.

Args:
    db: Database session
    
Returns:
    Dictionary containing coverage information

**Signature**: `get_age_group_coverage(db)`

### validate_age_group_data

Validate that age groups have sufficient data and are properly configured.

Args:
    db: Database session
    
Returns:
    Dictionary containing validation results

**Signature**: `validate_age_group_data(db)`

