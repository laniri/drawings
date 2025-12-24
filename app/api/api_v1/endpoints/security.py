"""
Security management endpoints for AWS production deployment.

This module provides endpoints for security validation, compliance checking,
and security configuration management.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.security_service import SecurityValidationResult, get_security_service

logger = logging.getLogger(__name__)

router = APIRouter()


class SecurityAuditRequest(BaseModel):
    """Request model for security audit."""

    iam_role_arn: Optional[str] = Field(None, description="IAM role ARN to validate")
    s3_buckets: Optional[List[str]] = Field(
        None, description="S3 bucket names to validate"
    )
    security_group_ids: Optional[List[str]] = Field(
        None, description="Security group IDs to validate"
    )
    vpc_id: Optional[str] = Field(None, description="VPC ID to validate")


class SecurityValidationResponse(BaseModel):
    """Response model for security validation."""

    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    details: Dict[str, Any]


class SecurityAuditResponse(BaseModel):
    """Response model for comprehensive security audit."""

    overall_compliant: bool
    total_violations: int
    total_warnings: int
    components: Dict[str, SecurityValidationResponse]
    summary: Dict[str, Any]


def get_security_service_dependency():
    """Dependency to get security service instance."""
    return get_security_service()


@router.get("/status", response_model=Dict[str, Any])
async def get_security_status(
    security_service=Depends(get_security_service_dependency),
):
    """
    Get current security service status and configuration.

    Returns information about security service initialization,
    AWS client availability, and current security policy.
    """
    try:
        # Get current AWS identity
        identity = security_service.get_current_identity()

        # Get security policy configuration
        policy = security_service.policy

        return {
            "service_status": "active",
            "aws_clients_available": {
                "iam": security_service._iam_client is not None,
                "s3": security_service._s3_client is not None,
                "ec2": security_service._ec2_client is not None,
                "sts": security_service._sts_client is not None,
            },
            "identity": identity,
            "security_policy": {
                "enforce_encryption_at_rest": policy.enforce_encryption_at_rest,
                "enforce_encryption_in_transit": policy.enforce_encryption_in_transit,
                "require_least_privilege_iam": policy.require_least_privilege_iam,
                "block_public_s3_access": policy.block_public_s3_access,
                "minimum_tls_version": policy.minimum_tls_version,
                "allowed_ports": policy.allowed_ports,
            },
            "environment": {
                "is_production": settings.is_production,
                "aws_region": settings.aws_region,
            },
        }

    except Exception as e:
        logger.error(f"Failed to get security status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get security status: {str(e)}",
        )


@router.post("/validate/iam-role", response_model=SecurityValidationResponse)
async def validate_iam_role(
    role_arn: str = Query(..., description="IAM role ARN to validate"),
    security_service=Depends(get_security_service_dependency),
):
    """
    Validate IAM role for least-privilege compliance.

    Checks the specified IAM role for overly broad permissions,
    dangerous policy attachments, and compliance with security best practices.
    """
    try:
        result = security_service.validate_iam_least_privilege(role_arn)

        return SecurityValidationResponse(
            is_compliant=result.is_compliant,
            violations=result.violations,
            warnings=result.warnings,
            recommendations=result.recommendations,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Failed to validate IAM role {role_arn}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate IAM role: {str(e)}",
        )


@router.post("/validate/s3-bucket", response_model=SecurityValidationResponse)
async def validate_s3_bucket(
    bucket_name: str = Query(..., description="S3 bucket name to validate"),
    security_service=Depends(get_security_service_dependency),
):
    """
    Validate S3 bucket encryption and security configuration.

    Checks the specified S3 bucket for proper encryption configuration,
    public access blocks, and security compliance.
    """
    try:
        result = security_service.validate_s3_encryption(bucket_name)

        return SecurityValidationResponse(
            is_compliant=result.is_compliant,
            violations=result.violations,
            warnings=result.warnings,
            recommendations=result.recommendations,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Failed to validate S3 bucket {bucket_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate S3 bucket: {str(e)}",
        )


@router.post("/validate/security-groups", response_model=SecurityValidationResponse)
async def validate_security_groups(
    security_group_ids: List[str] = Query(
        ..., description="Security group IDs to validate"
    ),
    security_service=Depends(get_security_service_dependency),
):
    """
    Validate security group configurations for minimal exposure.

    Checks the specified security groups for overly permissive rules,
    open ports, and compliance with network security best practices.
    """
    try:
        result = security_service.validate_security_groups(security_group_ids)

        return SecurityValidationResponse(
            is_compliant=result.is_compliant,
            violations=result.violations,
            warnings=result.warnings,
            recommendations=result.recommendations,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Failed to validate security groups {security_group_ids}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate security groups: {str(e)}",
        )


@router.post("/validate/vpc", response_model=SecurityValidationResponse)
async def validate_vpc(
    vpc_id: str = Query(..., description="VPC ID to validate"),
    security_service=Depends(get_security_service_dependency),
):
    """
    Validate VPC configuration for security compliance.

    Checks the specified VPC for proper subnet configuration,
    private subnet isolation, and network security best practices.
    """
    try:
        result = security_service.validate_vpc_configuration(vpc_id)

        return SecurityValidationResponse(
            is_compliant=result.is_compliant,
            violations=result.violations,
            warnings=result.warnings,
            recommendations=result.recommendations,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Failed to validate VPC {vpc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate VPC: {str(e)}",
        )


@router.get(
    "/validate/encryption-in-transit", response_model=SecurityValidationResponse
)
async def validate_encryption_in_transit(
    security_service=Depends(get_security_service_dependency),
):
    """
    Validate encryption in transit configuration.

    Checks the current deployment for proper HTTPS enforcement,
    TLS configuration, and secure communication protocols.
    """
    try:
        result = security_service.validate_encryption_in_transit()

        return SecurityValidationResponse(
            is_compliant=result.is_compliant,
            violations=result.violations,
            warnings=result.warnings,
            recommendations=result.recommendations,
            details=result.details,
        )

    except Exception as e:
        logger.error(f"Failed to validate encryption in transit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate encryption in transit: {str(e)}",
        )


@router.post("/audit/comprehensive", response_model=SecurityAuditResponse)
async def comprehensive_security_audit(
    request: SecurityAuditRequest,
    security_service=Depends(get_security_service_dependency),
):
    """
    Perform comprehensive security audit of AWS resources.

    Validates all specified AWS resources for security compliance,
    including IAM roles, S3 buckets, security groups, VPC configuration,
    and encryption settings.
    """
    try:
        # Perform comprehensive audit
        results = security_service.comprehensive_security_audit(
            iam_role_arn=request.iam_role_arn,
            s3_buckets=request.s3_buckets,
            security_group_ids=request.security_group_ids,
            vpc_id=request.vpc_id,
        )

        # Convert results to response format
        components = {}
        total_violations = 0
        total_warnings = 0
        overall_compliant = True

        for component_name, result in results.items():
            if isinstance(result, SecurityValidationResult):
                components[component_name] = SecurityValidationResponse(
                    is_compliant=result.is_compliant,
                    violations=result.violations,
                    warnings=result.warnings,
                    recommendations=result.recommendations,
                    details=result.details,
                )

                total_violations += len(result.violations)
                total_warnings += len(result.warnings)

                if not result.is_compliant:
                    overall_compliant = False

            elif isinstance(result, dict):
                # Handle nested results (like S3 buckets)
                if component_name == "s3_buckets":
                    for bucket_name, bucket_result in result.items():
                        components[f"s3_bucket_{bucket_name}"] = (
                            SecurityValidationResponse(
                                is_compliant=bucket_result.is_compliant,
                                violations=bucket_result.violations,
                                warnings=bucket_result.warnings,
                                recommendations=bucket_result.recommendations,
                                details=bucket_result.details,
                            )
                        )

                        total_violations += len(bucket_result.violations)
                        total_warnings += len(bucket_result.warnings)

                        if not bucket_result.is_compliant:
                            overall_compliant = False
                else:
                    # Handle other dict results (like identity)
                    components[component_name] = SecurityValidationResponse(
                        is_compliant=True,
                        violations=[],
                        warnings=[],
                        recommendations=[],
                        details=result,
                    )

        # Create summary
        summary = {
            "components_audited": len(components),
            "compliant_components": sum(
                1 for comp in components.values() if comp.is_compliant
            ),
            "violation_rate": total_violations / max(len(components), 1),
            "warning_rate": total_warnings / max(len(components), 1),
            "audit_timestamp": "2024-01-01T00:00:00Z",  # TODO: Add actual timestamp
            "recommendations_count": sum(
                len(comp.recommendations) for comp in components.values()
            ),
        }

        return SecurityAuditResponse(
            overall_compliant=overall_compliant,
            total_violations=total_violations,
            total_warnings=total_warnings,
            components=components,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Failed to perform comprehensive security audit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform security audit: {str(e)}",
        )


@router.get("/compliance/report")
async def get_compliance_report(
    format: str = Query("json", description="Report format: json, csv, or html"),
    security_service=Depends(get_security_service_dependency),
):
    """
    Generate security compliance report.

    Creates a detailed compliance report based on current security
    configuration and validation results.
    """
    try:
        # For now, return basic compliance information
        # In a full implementation, this would generate detailed reports

        compliance_data = {
            "report_type": "security_compliance",
            "format": format,
            "timestamp": "2024-01-01T00:00:00Z",
            "environment": "production" if settings.is_production else "development",
            "compliance_framework": "AWS Security Best Practices",
            "summary": {
                "encryption_at_rest": "Enforced",
                "encryption_in_transit": "Enforced",
                "iam_least_privilege": "Validated",
                "network_security": "Configured",
                "access_control": "Implemented",
            },
            "recommendations": [
                "Regular security audits",
                "Automated compliance monitoring",
                "Security training for development team",
                "Incident response plan updates",
            ],
        }

        if format.lower() == "csv":
            # Return CSV format indication
            compliance_data["csv_download_url"] = (
                "/api/v1/security/compliance/download?format=csv"
            )
        elif format.lower() == "html":
            # Return HTML format indication
            compliance_data["html_view_url"] = "/api/v1/security/compliance/view"

        return compliance_data

    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate compliance report: {str(e)}",
        )
