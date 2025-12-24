#!/usr/bin/env python3
"""
Cost Optimization Validation Script

This script validates that cost optimization configurations are properly applied
and that the system meets cost compliance requirements.
"""

import sys
import json
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.cost_optimization_service import cost_optimization_service


def main():
    """Main validation function."""
    print("🔍 Validating Cost Optimization Configuration...")
    print("=" * 60)
    
    # 1. Validate cost compliance
    print("\n1. Cost Compliance Validation")
    print("-" * 30)
    
    compliance_result = cost_optimization_service.validate_cost_compliance()
    
    if compliance_result["is_compliant"]:
        print(f"✅ Cost compliance: PASSED")
        print(f"   Total estimated cost: ${compliance_result['total_estimated_cost']:.2f}")
        print(f"   Target range: ${compliance_result['target_range']['min']:.2f}-${compliance_result['target_range']['max']:.2f}")
    else:
        print(f"❌ Cost compliance: FAILED")
        print(f"   Total estimated cost: ${compliance_result['total_estimated_cost']:.2f}")
        print(f"   Target range: ${compliance_result['target_range']['min']:.2f}-${compliance_result['target_range']['max']:.2f}")
    
    # 2. Validate ECS Fargate optimization
    print("\n2. ECS Fargate Optimization")
    print("-" * 30)
    
    ecs_config = cost_optimization_service.get_ecs_fargate_optimization()
    
    if ecs_config["cpu"] == 256 and ecs_config["memory"] == 512:
        print("✅ ECS Fargate: Optimally configured")
        print(f"   CPU: {ecs_config['cpu']} (0.25 vCPU)")
        print(f"   Memory: {ecs_config['memory']} MB (0.5 GB)")
        print(f"   Estimated cost: ${ecs_config['estimated_monthly_cost']:.2f}/month")
    else:
        print("❌ ECS Fargate: Not optimally configured")
        print(f"   Current CPU: {ecs_config['cpu']} (should be 256)")
        print(f"   Current Memory: {ecs_config['memory']} (should be 512)")
    
    # 3. Validate S3 lifecycle optimization
    print("\n3. S3 Lifecycle Optimization")
    print("-" * 30)
    
    s3_lifecycle = cost_optimization_service.get_s3_lifecycle_policy()
    
    if s3_lifecycle["Rules"]:
        rule = s3_lifecycle["Rules"][0]
        transitions = rule.get("Transitions", [])
        storage_classes = {t["StorageClass"] for t in transitions}
        
        if "STANDARD_IA" in storage_classes and "GLACIER" in storage_classes:
            print("✅ S3 Lifecycle: Properly configured")
            print("   Transitions: Standard → Standard-IA → Glacier")
        else:
            print("❌ S3 Lifecycle: Missing required transitions")
            print(f"   Found storage classes: {storage_classes}")
    else:
        print("❌ S3 Lifecycle: No rules configured")
    
    # 4. Validate CloudFront caching
    print("\n4. CloudFront Caching Optimization")
    print("-" * 30)
    
    cf_config = cost_optimization_service.get_cloudfront_cache_optimization()
    
    default_ttl = cf_config["default_cache_behavior"]["DefaultTTL"]
    static_ttl = cf_config["static_cache_behavior"]["DefaultTTL"]
    price_class = cf_config["price_class"]
    
    if default_ttl > 0 and static_ttl > 0 and price_class == "PriceClass_100":
        print("✅ CloudFront: Optimally configured")
        print(f"   Default TTL: {default_ttl} seconds")
        print(f"   Static TTL: {static_ttl} seconds")
        print(f"   Price Class: {price_class}")
    else:
        print("❌ CloudFront: Not optimally configured")
        print(f"   Default TTL: {default_ttl} (should be > 0)")
        print(f"   Static TTL: {static_ttl} (should be > 0)")
        print(f"   Price Class: {price_class} (should be PriceClass_100)")
    
    # 5. Cost breakdown analysis
    print("\n5. Cost Breakdown Analysis")
    print("-" * 30)
    
    estimates = cost_optimization_service.estimate_monthly_costs()
    
    print("Service-level cost estimates:")
    total_cost = 0
    for estimate in estimates:
        optimization_status = "✅" if estimate.optimization_applied else "⚠️"
        print(f"   {optimization_status} {estimate.service_name}: ${estimate.monthly_cost_usd:.2f}")
        total_cost += estimate.monthly_cost_usd
    
    print(f"\nTotal estimated cost: ${total_cost:.2f}/month")
    
    # 6. Optimization recommendations
    print("\n6. Optimization Recommendations")
    print("-" * 30)
    
    recommendations = cost_optimization_service.get_cost_optimization_recommendations()
    
    for i, recommendation in enumerate(recommendations, 1):
        print(f"   {i}. {recommendation}")
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("📊 COST OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    if compliance_result["is_compliant"]:
        print("🎉 Overall Status: COMPLIANT")
        print(f"💰 Estimated Monthly Cost: ${compliance_result['total_estimated_cost']:.2f}")
        print(f"🎯 Target Range: ${compliance_result['target_range']['min']:.2f}-${compliance_result['target_range']['max']:.2f}")
        print("✅ All cost optimization strategies are properly configured")
        return 0
    else:
        print("⚠️  Overall Status: NON-COMPLIANT")
        print(f"💰 Estimated Monthly Cost: ${compliance_result['total_estimated_cost']:.2f}")
        print(f"🎯 Target Range: ${compliance_result['target_range']['min']:.2f}-${compliance_result['target_range']['max']:.2f}")
        print("❌ Cost optimization requires attention")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)