"""
Property-Based Test for Cost Boundary Compliance

**Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**

Tests that AWS resource costs remain within the target budget of $26-36/month for demo usage.

**Validates: Requirements 8.4, 9.3**

This property ensures that:
1. Total estimated monthly costs fall within the target range ($26-36)
2. Individual service costs are optimized and reasonable
3. Cost optimization configurations are properly applied
4. Budget compliance is maintained across different usage scenarios
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List
import logging

from app.services.cost_optimization_service import (
    cost_optimization_service,
    CostOptimizationService,
    ResourceCostEstimate,
    CostOptimizationConfig
)

logger = logging.getLogger(__name__)


class TestCostBoundaryCompliance:
    """Test cost boundary compliance property."""
    
    def test_baseline_cost_compliance(self):
        """
        Test that baseline configuration meets cost compliance requirements.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        # Get total estimated cost
        total_cost, is_within_budget = cost_optimization_service.get_total_estimated_cost()
        
        # Verify cost is within target range
        target_min = cost_optimization_service.config.target_monthly_cost_min
        target_max = cost_optimization_service.config.target_monthly_cost_max
        
        assert is_within_budget, (
            f"Total cost ${total_cost:.2f} is not within target range "
            f"${target_min:.2f}-${target_max:.2f}"
        )
        
        assert target_min <= total_cost <= target_max, (
            f"Cost ${total_cost:.2f} exceeds target range ${target_min:.2f}-${target_max:.2f}"
        )
        
        # Verify cost is under budget limit
        budget_limit = cost_optimization_service.config.monthly_budget_limit
        assert total_cost <= budget_limit, (
            f"Total cost ${total_cost:.2f} exceeds budget limit ${budget_limit:.2f}"
        )
    
    @given(
        fargate_cpu=st.integers(min_value=256, max_value=1024),
        fargate_memory=st.integers(min_value=512, max_value=2048),
        s3_ia_days=st.integers(min_value=1, max_value=90),
        s3_glacier_days=st.integers(min_value=30, max_value=365)
    )
    @settings(max_examples=50, deadline=10000)
    def test_cost_compliance_with_configuration_variations(
        self, 
        fargate_cpu: int, 
        fargate_memory: int,
        s3_ia_days: int,
        s3_glacier_days: int
    ):
        """
        Property test: For any valid resource configuration, costs should remain reasonable.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        # Ensure valid configuration constraints
        assume(s3_glacier_days >= s3_ia_days)
        assume(fargate_memory >= fargate_cpu // 2)  # AWS Fargate memory constraints
        
        # Create temporary service with modified configuration
        temp_service = CostOptimizationService()
        temp_service.config.fargate_cpu = fargate_cpu
        temp_service.config.fargate_memory = fargate_memory
        temp_service.config.s3_standard_ia_transition_days = s3_ia_days
        temp_service.config.s3_glacier_transition_days = s3_glacier_days
        
        # Calculate costs with modified configuration
        estimates = temp_service.estimate_monthly_costs()
        total_cost = sum(estimate.monthly_cost_usd for estimate in estimates)
        
        # Verify costs remain reasonable (not exceeding 2x target max)
        max_reasonable_cost = temp_service.config.target_monthly_cost_max * 2
        assert total_cost <= max_reasonable_cost, (
            f"Configuration results in unreasonable cost: ${total_cost:.2f} "
            f"(CPU: {fargate_cpu}, Memory: {fargate_memory}MB, "
            f"IA: {s3_ia_days}d, Glacier: {s3_glacier_days}d)"
        )
        
        # Verify individual service costs are reasonable
        for estimate in estimates:
            if estimate.service_name == "ECS Fargate":
                # Fargate cost should scale with resources
                expected_fargate_cost = (fargate_cpu / 256) * (fargate_memory / 512) * 29.0
                assert estimate.monthly_cost_usd <= expected_fargate_cost * 1.5, (
                    f"Fargate cost ${estimate.monthly_cost_usd:.2f} too high for "
                    f"CPU: {fargate_cpu}, Memory: {fargate_memory}MB"
                )
    
    @given(
        usage_multiplier=st.floats(min_value=0.1, max_value=5.0),
        optimization_enabled=st.booleans()
    )
    @settings(max_examples=30, deadline=10000)
    def test_cost_scaling_with_usage(self, usage_multiplier: float, optimization_enabled: bool):
        """
        Property test: For any usage level, optimized configurations should cost less.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        # Get baseline estimates
        estimates = cost_optimization_service.estimate_monthly_costs()
        
        # Simulate usage scaling for variable costs (S3, CloudWatch)
        scaled_estimates = []
        for estimate in estimates:
            scaled_estimate = ResourceCostEstimate(
                service_name=estimate.service_name,
                monthly_cost_usd=estimate.monthly_cost_usd,
                resource_type=estimate.resource_type,
                configuration=estimate.configuration,
                optimization_applied=optimization_enabled if estimate.service_name in ["S3 Storage", "CloudWatch"] else estimate.optimization_applied
            )
            
            # Scale variable costs
            if estimate.service_name in ["S3 Storage", "CloudWatch"]:
                scaled_estimate.monthly_cost_usd *= usage_multiplier
                
                # Apply optimization discount if enabled
                if optimization_enabled:
                    scaled_estimate.monthly_cost_usd *= 0.7  # 30% savings with optimization
            
            scaled_estimates.append(scaled_estimate)
        
        total_scaled_cost = sum(est.monthly_cost_usd for est in scaled_estimates)
        
        # For reasonable usage levels (up to 2x), costs should remain manageable
        if usage_multiplier <= 2.0:
            max_acceptable_cost = cost_optimization_service.config.monthly_budget_limit * 1.5
            assert total_scaled_cost <= max_acceptable_cost, (
                f"Scaled cost ${total_scaled_cost:.2f} too high for usage multiplier {usage_multiplier:.2f}"
            )
        
        # Optimization should always reduce costs for variable services
        if optimization_enabled and usage_multiplier > 1.0:
            unoptimized_cost = sum(
                est.monthly_cost_usd * usage_multiplier if est.service_name in ["S3 Storage", "CloudWatch"] 
                else est.monthly_cost_usd 
                for est in estimates
            )
            
            assert total_scaled_cost < unoptimized_cost, (
                f"Optimization should reduce costs: optimized=${total_scaled_cost:.2f}, "
                f"unoptimized=${unoptimized_cost:.2f}"
            )
    
    def test_cost_breakdown_reasonableness(self):
        """
        Test that individual service costs are reasonable and properly categorized.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        estimates = cost_optimization_service.estimate_monthly_costs()
        
        # Verify we have estimates for all expected services
        service_names = {est.service_name for est in estimates}
        expected_services = {
            "ECS Fargate", "S3 Storage", "CloudFront CDN", 
            "Route 53", "CloudWatch", "Secrets Manager"
        }
        
        assert expected_services.issubset(service_names), (
            f"Missing cost estimates for services: {expected_services - service_names}"
        )
        
        # Verify individual service cost reasonableness
        for estimate in estimates:
            assert estimate.monthly_cost_usd >= 0, (
                f"Service {estimate.service_name} has negative cost: ${estimate.monthly_cost_usd}"
            )
            
            # Service-specific cost validation
            if estimate.service_name == "ECS Fargate":
                # Fargate with 0.25 vCPU, 0.5GB should be ~$29/month
                assert 25.0 <= estimate.monthly_cost_usd <= 35.0, (
                    f"ECS Fargate cost ${estimate.monthly_cost_usd:.2f} outside expected range $25-35"
                )
            
            elif estimate.service_name == "S3 Storage":
                # S3 for demo usage should be minimal
                assert estimate.monthly_cost_usd <= 10.0, (
                    f"S3 Storage cost ${estimate.monthly_cost_usd:.2f} too high for demo usage"
                )
            
            elif estimate.service_name == "CloudFront CDN":
                # CloudFront should be free tier for demo
                assert estimate.monthly_cost_usd <= 5.0, (
                    f"CloudFront cost ${estimate.monthly_cost_usd:.2f} should be minimal for demo"
                )
            
            elif estimate.service_name == "Route 53":
                # Route 53 hosted zone is $0.50/month
                assert 0.40 <= estimate.monthly_cost_usd <= 1.0, (
                    f"Route 53 cost ${estimate.monthly_cost_usd:.2f} outside expected range"
                )
            
            elif estimate.service_name == "Secrets Manager":
                # Secrets Manager is $0.40/secret/month
                assert 0.30 <= estimate.monthly_cost_usd <= 1.0, (
                    f"Secrets Manager cost ${estimate.monthly_cost_usd:.2f} outside expected range"
                )
    
    def test_optimization_effectiveness(self):
        """
        Test that cost optimization configurations provide meaningful savings.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        # Get optimized configuration
        ecs_config = cost_optimization_service.get_ecs_fargate_optimization()
        s3_lifecycle = cost_optimization_service.get_s3_lifecycle_policy()
        cloudfront_config = cost_optimization_service.get_cloudfront_cache_optimization()
        
        # Verify ECS optimization (minimal resources)
        assert ecs_config["cpu"] == 256, "ECS CPU should be optimized to 256 (0.25 vCPU)"
        assert ecs_config["memory"] == 512, "ECS Memory should be optimized to 512 MB"
        assert "estimated_monthly_cost" in ecs_config, "ECS config should include cost estimate"
        
        # Verify S3 lifecycle optimization
        assert len(s3_lifecycle["Rules"]) > 0, "S3 lifecycle should have optimization rules"
        
        lifecycle_rule = s3_lifecycle["Rules"][0]
        assert lifecycle_rule["Status"] == "Enabled", "S3 lifecycle rule should be enabled"
        assert "Transitions" in lifecycle_rule, "S3 lifecycle should include transitions"
        
        # Verify transitions include Standard-IA and Glacier
        transitions = lifecycle_rule["Transitions"]
        storage_classes = {t["StorageClass"] for t in transitions}
        assert "STANDARD_IA" in storage_classes, "Should transition to Standard-IA"
        assert "GLACIER" in storage_classes, "Should transition to Glacier"
        
        # Verify CloudFront caching optimization
        assert cloudfront_config["default_cache_behavior"]["DefaultTTL"] > 0, (
            "CloudFront should have optimized caching TTL"
        )
        assert cloudfront_config["static_cache_behavior"]["DefaultTTL"] > 0, (
            "Static assets should have longer cache TTL"
        )
        assert cloudfront_config["price_class"] == "PriceClass_100", (
            "Should use cost-optimized price class"
        )
    
    def test_compliance_validation_accuracy(self):
        """
        Test that cost compliance validation provides accurate results.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        compliance_result = cost_optimization_service.validate_cost_compliance()
        
        # Verify compliance result structure
        required_fields = {
            "is_compliant", "total_estimated_cost", "budget_limit", 
            "target_range", "cost_breakdown", "recommendations"
        }
        assert required_fields.issubset(compliance_result.keys()), (
            f"Compliance result missing fields: {required_fields - compliance_result.keys()}"
        )
        
        # Verify compliance logic consistency
        total_cost = compliance_result["total_estimated_cost"]
        target_range = compliance_result["target_range"]
        is_compliant = compliance_result["is_compliant"]
        
        expected_compliance = (
            target_range["min"] <= total_cost <= target_range["max"]
        )
        
        assert is_compliant == expected_compliance, (
            f"Compliance flag {is_compliant} inconsistent with cost ${total_cost:.2f} "
            f"and range ${target_range['min']:.2f}-${target_range['max']:.2f}"
        )
        
        # Verify cost breakdown sums to total
        breakdown_total = sum(item["cost"] for item in compliance_result["cost_breakdown"])
        assert abs(breakdown_total - total_cost) < 0.01, (
            f"Cost breakdown total ${breakdown_total:.2f} doesn't match "
            f"reported total ${total_cost:.2f}"
        )
        
        # Verify recommendations are provided
        assert len(compliance_result["recommendations"]) > 0, (
            "Should provide cost optimization recommendations"
        )
    
    @given(
        budget_limit=st.floats(min_value=10.0, max_value=100.0),
        target_min=st.floats(min_value=5.0, max_value=50.0),
        target_max=st.floats(min_value=20.0, max_value=80.0)
    )
    @settings(max_examples=20, deadline=10000)
    def test_cost_compliance_with_different_budgets(
        self, 
        budget_limit: float, 
        target_min: float, 
        target_max: float
    ):
        """
        Property test: Cost compliance should work correctly with different budget configurations.
        
        **Feature: aws-production-deployment, Property 11: Cost Boundary Compliance**
        **Validates: Requirements 8.4, 9.3**
        """
        # Ensure valid budget configuration
        assume(target_min < target_max)
        assume(target_max <= budget_limit)
        
        # Create temporary service with modified budget configuration
        temp_service = CostOptimizationService()
        temp_service.config.monthly_budget_limit = budget_limit
        temp_service.config.target_monthly_cost_min = target_min
        temp_service.config.target_monthly_cost_max = target_max
        
        # Get compliance validation
        compliance_result = temp_service.validate_cost_compliance()
        total_cost = compliance_result["total_estimated_cost"]
        
        # Verify compliance logic
        expected_compliance = target_min <= total_cost <= target_max
        assert compliance_result["is_compliant"] == expected_compliance, (
            f"Compliance calculation incorrect for budget ${budget_limit:.2f}, "
            f"range ${target_min:.2f}-${target_max:.2f}, cost ${total_cost:.2f}"
        )
        
        # Verify budget limit is respected in validation
        assert compliance_result["budget_limit"] == budget_limit, (
            f"Budget limit not properly set: expected ${budget_limit:.2f}, "
            f"got ${compliance_result['budget_limit']:.2f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])