"""
Data Sufficiency Service for Children's Drawing Anomaly Detection System

This module provides data count validation for age groups, warning generation for insufficient
training data, and suggestions for data augmentation and age group merging.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.core.exceptions import ValidationError
from app.models.database import AgeGroupModel, Drawing

logger = logging.getLogger(__name__)


class DataSufficiencyError(Exception):
    """Base exception for data sufficiency errors."""

    pass


class InsufficientDataError(DataSufficiencyError):
    """Raised when there is insufficient data for training."""

    pass


@dataclass
class AgeGroupDataInfo:
    """Information about data availability for an age group."""

    age_min: float
    age_max: float
    sample_count: int
    is_sufficient: bool
    recommended_min_samples: int
    data_quality_score: float
    subjects_distribution: Dict[str, int]
    age_distribution: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DataSufficiencyWarning:
    """Warning about data sufficiency issues."""

    warning_type: (
        str  # "insufficient_data", "unbalanced_distribution", "narrow_age_range"
    )
    severity: str  # "low", "medium", "high", "critical"
    age_group_min: float
    age_group_max: float
    current_samples: int
    recommended_samples: int
    message: str
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AgeGroupMergingSuggestion:
    """Suggestion for merging age groups to improve data sufficiency."""

    original_groups: List[Tuple[float, float]]
    merged_group: Tuple[float, float]
    combined_sample_count: int
    improvement_score: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataSufficiencyAnalyzer:
    """Analyzer for data sufficiency and quality assessment."""

    def __init__(self):
        # Configuration for data sufficiency thresholds
        self.min_samples_per_group = 50
        self.recommended_samples_per_group = 100
        self.critical_threshold = 20
        self.max_age_group_span = 2.0  # Maximum years for a single age group
        self.min_age_group_span = 0.5  # Minimum years for a single age group

        logger.info("Data Sufficiency Analyzer initialized")

    def analyze_age_group_data(
        self, age_min: float, age_max: float, db: Session
    ) -> AgeGroupDataInfo:
        """
        Analyze data sufficiency for a specific age group.

        Args:
            age_min: Minimum age for the group
            age_max: Maximum age for the group
            db: Database session

        Returns:
            Data information for the age group
        """
        try:
            logger.info(f"Analyzing data sufficiency for age group {age_min}-{age_max}")

            # Query drawings in the age range
            drawings = (
                db.query(Drawing)
                .filter(Drawing.age_years >= age_min, Drawing.age_years < age_max)
                .all()
            )

            sample_count = len(drawings)

            # Calculate data quality metrics
            data_quality_score = self._calculate_data_quality_score(drawings)

            # Analyze subject distribution
            subjects_distribution = self._analyze_subject_distribution(drawings)

            # Analyze age distribution
            age_distribution = [d.age_years for d in drawings]

            # Determine if data is sufficient
            is_sufficient = sample_count >= self.min_samples_per_group

            return AgeGroupDataInfo(
                age_min=age_min,
                age_max=age_max,
                sample_count=sample_count,
                is_sufficient=is_sufficient,
                recommended_min_samples=self.recommended_samples_per_group,
                data_quality_score=data_quality_score,
                subjects_distribution=subjects_distribution,
                age_distribution=age_distribution,
            )

        except Exception as e:
            logger.error(f"Failed to analyze age group data: {str(e)}")
            raise DataSufficiencyError(f"Data analysis failed: {str(e)}")

    def _calculate_data_quality_score(self, drawings: List[Any]) -> float:
        """Calculate a quality score for the data (0.0 to 1.0)."""
        if not drawings:
            return 0.0

        score = 0.0

        # Factor 1: Sample size (0.4 weight)
        sample_size_score = min(len(drawings) / self.recommended_samples_per_group, 1.0)
        score += 0.4 * sample_size_score

        # Factor 2: Age distribution (0.3 weight)
        ages = [d.age_years for d in drawings]
        age_range = max(ages) - min(ages) if len(ages) > 1 else 0
        age_distribution_score = min(
            age_range / 1.0, 1.0
        )  # Good if spans at least 1 year
        score += 0.3 * age_distribution_score

        # Factor 3: Subject diversity (0.2 weight)
        subjects = [d.subject for d in drawings if d.subject]
        unique_subjects = len(set(subjects)) if subjects else 0
        subject_diversity_score = min(
            unique_subjects / 3.0, 1.0
        )  # Good if at least 3 subjects
        score += 0.2 * subject_diversity_score

        # Factor 4: Metadata completeness (0.1 weight)
        complete_metadata_count = sum(1 for d in drawings if d.subject and d.age_years)
        metadata_completeness_score = (
            complete_metadata_count / len(drawings) if drawings else 0
        )
        score += 0.1 * metadata_completeness_score

        return min(score, 1.0)

    def _analyze_subject_distribution(self, drawings: List[Any]) -> Dict[str, int]:
        """Analyze the distribution of drawing subjects."""
        subjects = {}
        for drawing in drawings:
            subject = drawing.subject or "unknown"
            subjects[subject] = subjects.get(subject, 0) + 1
        return subjects

    def generate_data_warnings(
        self, age_groups: List[Tuple[float, float]], db: Session
    ) -> List[DataSufficiencyWarning]:
        """
        Generate warnings for data sufficiency issues across age groups.

        Args:
            age_groups: List of (age_min, age_max) tuples
            db: Database session

        Returns:
            List of data sufficiency warnings
        """
        try:
            warnings = []

            for age_min, age_max in age_groups:
                data_info = self.analyze_age_group_data(age_min, age_max, db)

                # Check for insufficient data
                if data_info.sample_count <= self.critical_threshold:
                    warnings.append(
                        DataSufficiencyWarning(
                            warning_type="insufficient_data",
                            severity="critical",
                            age_group_min=age_min,
                            age_group_max=age_max,
                            current_samples=data_info.sample_count,
                            recommended_samples=self.min_samples_per_group,
                            message=f"Critical: Only {data_info.sample_count} samples for age group {age_min}-{age_max}",
                            suggestions=[
                                "Consider merging with adjacent age groups",
                                "Collect more data for this age range",
                                "Use data augmentation techniques",
                                "Skip training for this age group until more data is available",
                            ],
                        )
                    )
                elif data_info.sample_count < self.min_samples_per_group:
                    warnings.append(
                        DataSufficiencyWarning(
                            warning_type="insufficient_data",
                            severity="high",
                            age_group_min=age_min,
                            age_group_max=age_max,
                            current_samples=data_info.sample_count,
                            recommended_samples=self.min_samples_per_group,
                            message=f"Insufficient data: {data_info.sample_count} samples for age group {age_min}-{age_max}",
                            suggestions=[
                                "Consider merging with adjacent age groups",
                                "Collect additional data for this age range",
                                "Use data augmentation if appropriate",
                            ],
                        )
                    )
                elif data_info.sample_count < self.recommended_samples_per_group:
                    warnings.append(
                        DataSufficiencyWarning(
                            warning_type="insufficient_data",
                            severity="medium",
                            age_group_min=age_min,
                            age_group_max=age_max,
                            current_samples=data_info.sample_count,
                            recommended_samples=self.recommended_samples_per_group,
                            message=f"Below recommended: {data_info.sample_count} samples for age group {age_min}-{age_max}",
                            suggestions=[
                                "Consider collecting more data for better model performance",
                                "Monitor model performance closely",
                            ],
                        )
                    )

                # Check for unbalanced subject distribution
                if data_info.subjects_distribution:
                    max_subject_count = max(data_info.subjects_distribution.values())
                    min_subject_count = min(data_info.subjects_distribution.values())

                    if max_subject_count > min_subject_count * 3:  # Highly unbalanced
                        warnings.append(
                            DataSufficiencyWarning(
                                warning_type="unbalanced_distribution",
                                severity="medium",
                                age_group_min=age_min,
                                age_group_max=age_max,
                                current_samples=data_info.sample_count,
                                recommended_samples=self.recommended_samples_per_group,
                                message=f"Unbalanced subject distribution in age group {age_min}-{age_max}",
                                suggestions=[
                                    "Collect more drawings of underrepresented subjects",
                                    "Consider subject-specific data augmentation",
                                    "Monitor for subject-specific biases in model performance",
                                ],
                            )
                        )

                # Check for narrow age range
                if data_info.age_distribution:
                    age_span = max(data_info.age_distribution) - min(
                        data_info.age_distribution
                    )
                    if age_span < self.min_age_group_span:
                        warnings.append(
                            DataSufficiencyWarning(
                                warning_type="narrow_age_range",
                                severity="low",
                                age_group_min=age_min,
                                age_group_max=age_max,
                                current_samples=data_info.sample_count,
                                recommended_samples=self.recommended_samples_per_group,
                                message=f"Narrow age range ({age_span:.1f} years) in group {age_min}-{age_max}",
                                suggestions=[
                                    "Collect data from a wider age range within the group",
                                    "Consider adjusting age group boundaries",
                                ],
                            )
                        )

            logger.info(f"Generated {len(warnings)} data sufficiency warnings")
            return warnings

        except Exception as e:
            logger.error(f"Failed to generate data warnings: {str(e)}")
            raise DataSufficiencyError(f"Warning generation failed: {str(e)}")

    def suggest_age_group_merging(
        self, age_groups: List[Tuple[float, float]], db: Session
    ) -> List[AgeGroupMergingSuggestion]:
        """
        Suggest age group merging strategies to improve data sufficiency.

        Args:
            age_groups: List of (age_min, age_max) tuples
            db: Database session

        Returns:
            List of merging suggestions
        """
        try:
            suggestions = []

            # Analyze each age group
            group_data = []
            for age_min, age_max in age_groups:
                data_info = self.analyze_age_group_data(age_min, age_max, db)
                group_data.append((age_min, age_max, data_info))

            # Find groups with insufficient data
            insufficient_groups = [
                (age_min, age_max, info)
                for age_min, age_max, info in group_data
                if info.sample_count < self.min_samples_per_group
            ]

            # Generate merging suggestions for insufficient groups
            for i, (age_min1, age_max1, info1) in enumerate(insufficient_groups):
                # Look for adjacent groups to merge with
                for age_min2, age_max2, info2 in group_data:
                    if (age_min1, age_max1) == (age_min2, age_max2):
                        continue

                    # Check if groups are adjacent or overlapping
                    if self._are_groups_mergeable(
                        age_min1, age_max1, age_min2, age_max2
                    ):
                        merged_min = min(age_min1, age_min2)
                        merged_max = max(age_max1, age_max2)
                        merged_span = merged_max - merged_min

                        # Don't suggest merging if the result would be too large
                        if merged_span <= self.max_age_group_span:
                            combined_samples = info1.sample_count + info2.sample_count

                            # Calculate improvement score
                            improvement_score = self._calculate_improvement_score(
                                info1.sample_count, info2.sample_count, combined_samples
                            )

                            # Generate rationale
                            rationale = self._generate_merging_rationale(
                                (age_min1, age_max1, info1.sample_count),
                                (age_min2, age_max2, info2.sample_count),
                                combined_samples,
                            )

                            suggestions.append(
                                AgeGroupMergingSuggestion(
                                    original_groups=[
                                        (age_min1, age_max1),
                                        (age_min2, age_max2),
                                    ],
                                    merged_group=(merged_min, merged_max),
                                    combined_sample_count=combined_samples,
                                    improvement_score=improvement_score,
                                    rationale=rationale,
                                )
                            )

            # Sort suggestions by improvement score (highest first)
            suggestions.sort(key=lambda x: x.improvement_score, reverse=True)

            # Remove duplicate suggestions (same merged group)
            unique_suggestions = []
            seen_merged_groups = set()

            for suggestion in suggestions:
                merged_group_key = suggestion.merged_group
                if merged_group_key not in seen_merged_groups:
                    unique_suggestions.append(suggestion)
                    seen_merged_groups.add(merged_group_key)

            logger.info(
                f"Generated {len(unique_suggestions)} age group merging suggestions"
            )
            return unique_suggestions

        except Exception as e:
            logger.error(f"Failed to generate merging suggestions: {str(e)}")
            raise DataSufficiencyError(f"Merging suggestion failed: {str(e)}")

    def _are_groups_mergeable(
        self, age_min1: float, age_max1: float, age_min2: float, age_max2: float
    ) -> bool:
        """Check if two age groups can be merged."""
        # Groups are mergeable if they are adjacent or overlapping
        return (
            (age_max1 >= age_min2 and age_min1 <= age_max2)
            or (abs(age_max1 - age_min2) <= 0.5)
            or (abs(age_max2 - age_min1) <= 0.5)
        )

    def _calculate_improvement_score(
        self, samples1: int, samples2: int, combined_samples: int
    ) -> float:
        """Calculate improvement score for merging (0.0 to 1.0)."""
        # Base score: how much the combined group improves sufficiency
        sufficiency_before = min(samples1 / self.min_samples_per_group, 1.0) + min(
            samples2 / self.min_samples_per_group, 1.0
        )
        sufficiency_after = min(combined_samples / self.min_samples_per_group, 1.0)

        # Improvement is the difference, normalized
        improvement = max(0, sufficiency_after - sufficiency_before / 2)

        # Bonus for reaching recommended threshold
        if combined_samples >= self.recommended_samples_per_group:
            improvement += 0.2

        # Penalty for very small individual groups (they might be outliers)
        if samples1 < 10 or samples2 < 10:
            improvement *= 0.8

        return min(improvement, 1.0)

    def _generate_merging_rationale(
        self,
        group1: Tuple[float, float, int],
        group2: Tuple[float, float, int],
        combined_samples: int,
    ) -> str:
        """Generate human-readable rationale for merging suggestion."""
        age_min1, age_max1, samples1 = group1
        age_min2, age_max2, samples2 = group2

        rationale = (
            f"Merging age groups {age_min1}-{age_max1} ({samples1} samples) and "
            f"{age_min2}-{age_max2} ({samples2} samples) would create a group with "
            f"{combined_samples} samples."
        )

        if combined_samples >= self.recommended_samples_per_group:
            rationale += " This meets the recommended sample size for robust training."
        elif combined_samples >= self.min_samples_per_group:
            rationale += " This meets the minimum sample size for training."
        else:
            rationale += (
                " This improves data sufficiency but may still require additional data."
            )

        return rationale


class DataAugmentationSuggester:
    """Service for suggesting data augmentation strategies."""

    def __init__(self):
        self.augmentation_techniques = {
            "geometric": [
                "Rotation (±15 degrees)",
                "Translation (±10% of image size)",
                "Scaling (0.9-1.1x)",
                "Horizontal flipping (for symmetric subjects)",
            ],
            "appearance": [
                "Brightness adjustment (±20%)",
                "Contrast adjustment (±15%)",
                "Gaussian noise addition (low intensity)",
                "Slight blur/sharpening",
            ],
            "drawing_specific": [
                "Line thickness variation",
                "Stroke style variation",
                "Color palette shifts (if applicable)",
                "Background texture variation",
            ],
        }

        logger.info("Data Augmentation Suggester initialized")

    def suggest_augmentation_strategies(
        self, age_group_data: AgeGroupDataInfo
    ) -> Dict[str, Any]:
        """
        Suggest data augmentation strategies based on data analysis.

        Args:
            age_group_data: Data information for the age group

        Returns:
            Dictionary with augmentation suggestions
        """
        suggestions = {
            "recommended_techniques": [],
            "multiplier_target": 1.0,
            "priority_subjects": [],
            "cautions": [],
        }

        # Calculate how much augmentation is needed
        if age_group_data.sample_count < 20:
            suggestions["multiplier_target"] = 5.0  # Aggressive augmentation
            suggestions["recommended_techniques"].extend(
                self.augmentation_techniques["geometric"]
            )
            suggestions["recommended_techniques"].extend(
                self.augmentation_techniques["appearance"]
            )
        elif age_group_data.sample_count < 50:
            suggestions["multiplier_target"] = 2.0  # Moderate augmentation
            suggestions["recommended_techniques"].extend(
                self.augmentation_techniques["geometric"]
            )
        elif age_group_data.sample_count < 100:
            suggestions["multiplier_target"] = 1.5  # Light augmentation
            suggestions["recommended_techniques"].extend(
                ["Rotation (±10 degrees)", "Translation (±5%)"]
            )

        # Identify subjects that need more data
        if age_group_data.subjects_distribution:
            total_samples = sum(age_group_data.subjects_distribution.values())
            for subject, count in age_group_data.subjects_distribution.items():
                if count < total_samples * 0.2:  # Less than 20% of total
                    suggestions["priority_subjects"].append(subject)

        # Add cautions for children's drawings
        suggestions["cautions"] = [
            "Preserve developmental characteristics appropriate for the age group",
            "Avoid over-augmentation that might create unrealistic drawings",
            "Test augmented samples with domain experts",
            "Monitor for bias introduction through augmentation",
        ]

        return suggestions


# Global service instances
_data_sufficiency_analyzer = None
_data_augmentation_suggester = None


def get_data_sufficiency_analyzer() -> DataSufficiencyAnalyzer:
    """Get the global data sufficiency analyzer instance."""
    global _data_sufficiency_analyzer
    if _data_sufficiency_analyzer is None:
        _data_sufficiency_analyzer = DataSufficiencyAnalyzer()
    return _data_sufficiency_analyzer


def get_data_augmentation_suggester() -> DataAugmentationSuggester:
    """Get the global data augmentation suggester instance."""
    global _data_augmentation_suggester
    if _data_augmentation_suggester is None:
        _data_augmentation_suggester = DataAugmentationSuggester()
    return _data_augmentation_suggester
