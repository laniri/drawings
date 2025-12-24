"""
Cost Optimization API endpoints for AWS production deployment.

Provides endpoints for:
- Cost estimation and monitoring
- Resource optimization recommendations
- Cost compliance validation
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.services.cost_optimization_service import (
    cost_optimization_service,
    ResourceCostEstimate,
    CostOptimizationConfig
)

router = APIRouter()


class CostEstimateResponse(BaseModel):
    """Response model for cost estimates."""
    
    total_monthly_cost: float
    is_within_budget: bool
    cost_breakdown: List[ResourceCostEstimate]
    target_range: Dict[str, float]


class CostOptimizationResponse(BaseModel):
    """Response model for cost optimization recommendations."""
    
    ecs_fargate_config: Dict[str, int]
    s3_lifecycle_policy: Dict
    cloudfront_cache_config: Dict
    recommendations: List[str]


class CostComplianceResponse(BaseModel):
    """Response model for cost compliance validation."""
    
    is_compliant: bool
    total_estimated_cost: float
    budget_limit: float
    target_range: Dict[str, float]
    cost_breakdown: List[Dict[str, Any]]
    recommendations: List[str]


@router.get("/estimate", response_model=CostEstimateResponse)
async def get_cost_estimate():
    """
    Get estimated monthly costs for optimized AWS resources.
    
    Returns cost breakdown and compliance status for the production deployment.
    """
    try:
        estimates = cost_optimization_service.estimate_monthly_costs()
        total_cost, is_within_budget = cost_optimization_service.get_total_estimated_cost()
        
        return CostEstimateResponse(
            total_monthly_cost=total_cost,
            is_within_budget=is_within_budget,
            cost_breakdown=estimates,
            target_range={
                "min": cost_optimization_service.config.target_monthly_cost_min,
                "max": cost_optimization_service.config.target_monthly_cost_max
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cost estimate: {str(e)}")


@router.get("/optimization", response_model=CostOptimizationResponse)
async def get_cost_optimization():
    """
    Get cost optimization configurations and recommendations.
    
    Returns optimized configurations for ECS Fargate, S3, and CloudFront.
    """
    try:
        return CostOptimizationResponse(
            ecs_fargate_config=cost_optimization_service.get_ecs_fargate_optimization(),
            s3_lifecycle_policy=cost_optimization_service.get_s3_lifecycle_policy(),
            cloudfront_cache_config=cost_optimization_service.get_cloudfront_cache_optimization(),
            recommendations=cost_optimization_service.get_cost_optimization_recommendations()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get optimization config: {str(e)}")


@router.get("/compliance", response_model=CostComplianceResponse)
async def validate_cost_compliance():
    """
    Validate cost compliance against budget requirements.
    
    Returns compliance status and detailed cost analysis.
    """
    try:
        compliance_result = cost_optimization_service.validate_cost_compliance()
        
        return CostComplianceResponse(**compliance_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate compliance: {str(e)}")


@router.post("/apply-s3-lifecycle/{bucket_name}")
async def apply_s3_lifecycle_optimization(bucket_name: str):
    """
    Apply S3 lifecycle optimization to a specific bucket.
    
    Args:
        bucket_name: Name of the S3 bucket to optimize
        
    Returns:
        Success status of the lifecycle policy application
    """
    try:
        success = cost_optimization_service.apply_s3_lifecycle_optimization(bucket_name)
        
        if success:
            return {"message": f"Lifecycle optimization applied to bucket: {bucket_name}"}
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to apply lifecycle optimization to bucket: {bucket_name}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error applying S3 optimization: {str(e)}")


@router.post("/setup-monitoring")
async def setup_cost_monitoring():
    """
    Set up cost monitoring and budget alerts.
    
    Returns:
        Success status of the cost monitoring setup
    """
    try:
        success = cost_optimization_service.setup_cost_monitoring()
        
        if success:
            return {"message": "Cost monitoring and budget alerts configured successfully"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to set up cost monitoring"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error setting up cost monitoring: {str(e)}")


@router.get("/config")
async def get_cost_optimization_config():
    """
    Get current cost optimization configuration.
    
    Returns:
        Current cost optimization settings
    """
    try:
        config = cost_optimization_service.config
        return {
            "fargate_cpu": config.fargate_cpu,
            "fargate_memory": config.fargate_memory,
            "s3_standard_ia_transition_days": config.s3_standard_ia_transition_days,
            "s3_glacier_transition_days": config.s3_glacier_transition_days,
            "cloudfront_default_ttl": config.cloudfront_default_ttl,
            "cloudfront_static_ttl": config.cloudfront_static_ttl,
            "monthly_budget_limit": config.monthly_budget_limit,
            "cost_alert_threshold": config.cost_alert_threshold,
            "target_monthly_cost_min": config.target_monthly_cost_min,
            "target_monthly_cost_max": config.target_monthly_cost_max
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")