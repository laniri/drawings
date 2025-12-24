"""
Security service for AWS production deployment.

This service provides security controls and compliance validation including
IAM role management, encryption enforcement, and security group validation.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.core.config import settings
from app.core.exceptions import ConfigurationError, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class SecurityPolicy:
    """Security policy configuration."""

    enforce_encryption_at_rest: bool = True
    enforce_encryption_in_transit: bool = True
    require_least_privilege_iam: bool = True
    block_public_s3_access: bool = True
    require_vpc_endpoints: bool = True
    enable_cloudtrail_logging: bool = True
    minimum_tls_version: str = "1.2"
    allowed_ports: List[int] = None

    def __post_init__(self):
        if self.allowed_ports is None:
            self.allowed_ports = [80, 443, 8000]  # HTTP, HTTPS, App port


@dataclass
class SecurityValidationResult:
    """Result of security validation."""

    is_compliant: bool
    violations: List[str]
    warnings: List[str]
    recommendations: List[str]
    details: Dict[str, Any]


class SecurityService:
    """
    Security service for AWS production deployment.

    Provides security controls validation, IAM policy enforcement,
    and encryption compliance checking.
    """

    def __init__(self):
        self.policy = SecurityPolicy()

        # AWS clients
        self._iam_client = None
        self._s3_client = None
        self._ec2_client = None
        self._sts_client = None

        # Initialize service
        self._initialize()

    def _initialize(self):
        """Initialize the security service."""
        try:
            if settings.is_production:
                # In production, use AWS clients
                region = settings.aws_region or "us-east-1"

                self._iam_client = boto3.client("iam", region_name=region)
                self._s3_client = boto3.client("s3", region_name=region)
                self._ec2_client = boto3.client("ec2", region_name=region)
                self._sts_client = boto3.client("sts", region_name=region)

                logger.info("Initialized AWS security clients")
            else:
                logger.info("Security service initialized for local development")

        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Failed to initialize AWS clients: {e}")
            logger.info("Security service running in limited mode")

    def validate_iam_least_privilege(self, role_arn: str) -> SecurityValidationResult:
        """
        Validate that IAM role follows least-privilege principle.

        Args:
            role_arn: IAM role ARN to validate

        Returns:
            SecurityValidationResult with validation details
        """
        violations = []
        warnings = []
        recommendations = []
        details = {}

        if not self._iam_client:
            warnings.append("IAM client not available - skipping IAM validation")
            return SecurityValidationResult(
                is_compliant=True,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                details=details,
            )

        try:
            # Extract role name from ARN
            role_name = role_arn.split("/")[-1]

            # Get role details
            role_response = self._iam_client.get_role(RoleName=role_name)
            role = role_response["Role"]

            # Get attached policies
            attached_policies = self._iam_client.list_attached_role_policies(
                RoleName=role_name
            )

            # Get inline policies
            inline_policies = self._iam_client.list_role_policies(RoleName=role_name)

            details["role_name"] = role_name
            details["attached_policies"] = [
                p["PolicyName"] for p in attached_policies["AttachedPolicies"]
            ]
            details["inline_policies"] = inline_policies["PolicyNames"]

            # Check for overly broad policies
            dangerous_policies = [
                "arn:aws:iam::aws:policy/AdministratorAccess",
                "arn:aws:iam::aws:policy/PowerUserAccess",
                "arn:aws:iam::aws:policy/IAMFullAccess",
            ]

            for policy in attached_policies["AttachedPolicies"]:
                if policy["PolicyArn"] in dangerous_policies:
                    violations.append(
                        f"Role has overly broad policy: {policy['PolicyName']}"
                    )

            # Check for wildcard permissions in inline policies
            for policy_name in inline_policies["PolicyNames"]:
                policy_doc = self._iam_client.get_role_policy(
                    RoleName=role_name, PolicyName=policy_name
                )

                policy_document = policy_doc["PolicyDocument"]
                if self._has_wildcard_permissions(policy_document):
                    warnings.append(
                        f"Inline policy '{policy_name}' may have overly broad permissions"
                    )

            # Recommendations for improvement
            if len(attached_policies["AttachedPolicies"]) > 5:
                recommendations.append(
                    "Consider consolidating policies to reduce complexity"
                )

            if not inline_policies["PolicyNames"]:
                recommendations.append(
                    "Consider using inline policies for role-specific permissions"
                )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchEntity":
                violations.append(f"IAM role not found: {role_name}")
            else:
                violations.append(f"Failed to validate IAM role: {str(e)}")

        is_compliant = len(violations) == 0

        return SecurityValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    def _has_wildcard_permissions(self, policy_document: Dict) -> bool:
        """Check if policy document has wildcard permissions."""
        statements = policy_document.get("Statement", [])
        if not isinstance(statements, list):
            statements = [statements]

        for statement in statements:
            if statement.get("Effect") == "Allow":
                actions = statement.get("Action", [])
                resources = statement.get("Resource", [])

                if not isinstance(actions, list):
                    actions = [actions]
                if not isinstance(resources, list):
                    resources = [resources]

                # Check for wildcard actions or resources
                if "*" in actions or "*" in resources:
                    return True

        return False

    def validate_s3_encryption(self, bucket_name: str) -> SecurityValidationResult:
        """
        Validate S3 bucket encryption configuration.

        Args:
            bucket_name: S3 bucket name to validate

        Returns:
            SecurityValidationResult with validation details
        """
        violations = []
        warnings = []
        recommendations = []
        details = {}

        if not self._s3_client:
            warnings.append("S3 client not available - skipping S3 validation")
            return SecurityValidationResult(
                is_compliant=True,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                details=details,
            )

        try:
            # Check bucket encryption
            try:
                encryption_response = self._s3_client.get_bucket_encryption(
                    Bucket=bucket_name
                )

                encryption_config = encryption_response[
                    "ServerSideEncryptionConfiguration"
                ]
                rules = encryption_config["Rules"]

                details["encryption_enabled"] = True
                details["encryption_rules"] = len(rules)

                # Validate encryption algorithm
                for rule in rules:
                    sse_config = rule["ApplyServerSideEncryptionByDefault"]
                    algorithm = sse_config["SSEAlgorithm"]

                    if algorithm not in ["AES256", "aws:kms"]:
                        violations.append(
                            f"Bucket uses unsupported encryption algorithm: {algorithm}"
                        )

                    details["encryption_algorithm"] = algorithm

            except ClientError as e:
                if (
                    e.response["Error"]["Code"]
                    == "ServerSideEncryptionConfigurationNotFoundError"
                ):
                    violations.append(
                        f"Bucket encryption not configured: {bucket_name}"
                    )
                    details["encryption_enabled"] = False
                else:
                    raise

            # Check public access block
            try:
                public_access_response = self._s3_client.get_public_access_block(
                    Bucket=bucket_name
                )

                public_access_config = public_access_response[
                    "PublicAccessBlockConfiguration"
                ]
                details["public_access_block"] = public_access_config

                required_blocks = [
                    "BlockPublicAcls",
                    "IgnorePublicAcls",
                    "BlockPublicPolicy",
                    "RestrictPublicBuckets",
                ]

                for block_setting in required_blocks:
                    if not public_access_config.get(block_setting, False):
                        violations.append(
                            f"Public access block setting not enabled: {block_setting}"
                        )

            except ClientError as e:
                if (
                    e.response["Error"]["Code"]
                    == "NoSuchPublicAccessBlockConfiguration"
                ):
                    violations.append(
                        f"Public access block not configured: {bucket_name}"
                    )
                else:
                    raise

            # Check bucket versioning
            versioning_response = self._s3_client.get_bucket_versioning(
                Bucket=bucket_name
            )

            versioning_status = versioning_response.get("Status", "Disabled")
            details["versioning_enabled"] = versioning_status == "Enabled"

            if versioning_status != "Enabled":
                recommendations.append("Enable bucket versioning for data protection")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchBucket":
                violations.append(f"S3 bucket not found: {bucket_name}")
            else:
                violations.append(f"Failed to validate S3 bucket: {str(e)}")

        is_compliant = len(violations) == 0

        return SecurityValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    def validate_security_groups(
        self, security_group_ids: List[str]
    ) -> SecurityValidationResult:
        """
        Validate security group configurations.

        Args:
            security_group_ids: List of security group IDs to validate

        Returns:
            SecurityValidationResult with validation details
        """
        violations = []
        warnings = []
        recommendations = []
        details = {}

        if not self._ec2_client:
            warnings.append(
                "EC2 client not available - skipping security group validation"
            )
            return SecurityValidationResult(
                is_compliant=True,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                details=details,
            )

        try:
            # Get security group details
            response = self._ec2_client.describe_security_groups(
                GroupIds=security_group_ids
            )

            security_groups = response["SecurityGroups"]
            details["security_groups"] = []

            for sg in security_groups:
                sg_details = {
                    "group_id": sg["GroupId"],
                    "group_name": sg["GroupName"],
                    "description": sg["Description"],
                    "ingress_rules": len(sg["IpPermissions"]),
                    "egress_rules": len(sg["IpPermissionsEgress"]),
                }

                # Check ingress rules
                for rule in sg["IpPermissions"]:
                    from_port = rule.get("FromPort", 0)
                    to_port = rule.get("ToPort", 65535)
                    protocol = rule.get("IpProtocol", "all")

                    # Check for overly permissive rules
                    for ip_range in rule.get("IpRanges", []):
                        cidr = ip_range.get("CidrIp", "")

                        if cidr == "0.0.0.0/0":
                            if from_port == 0 and to_port == 65535:
                                violations.append(
                                    f"Security group {sg['GroupId']} allows all traffic from anywhere"
                                )
                            elif from_port not in self.policy.allowed_ports:
                                warnings.append(
                                    f"Security group {sg['GroupId']} allows port {from_port} from anywhere"
                                )

                    # Check for SSH access
                    if from_port == 22 and to_port == 22:
                        for ip_range in rule.get("IpRanges", []):
                            if ip_range.get("CidrIp") == "0.0.0.0/0":
                                violations.append(
                                    f"Security group {sg['GroupId']} allows SSH from anywhere"
                                )

                details["security_groups"].append(sg_details)

            # Recommendations
            if len(security_groups) > 3:
                recommendations.append(
                    "Consider consolidating security groups to reduce complexity"
                )

        except ClientError as e:
            violations.append(f"Failed to validate security groups: {str(e)}")

        is_compliant = len(violations) == 0

        return SecurityValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    def validate_vpc_configuration(self, vpc_id: str) -> SecurityValidationResult:
        """
        Validate VPC security configuration.

        Args:
            vpc_id: VPC ID to validate

        Returns:
            SecurityValidationResult with validation details
        """
        violations = []
        warnings = []
        recommendations = []
        details = {}

        if not self._ec2_client:
            warnings.append("EC2 client not available - skipping VPC validation")
            return SecurityValidationResult(
                is_compliant=True,
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                details=details,
            )

        try:
            # Get VPC details
            vpc_response = self._ec2_client.describe_vpcs(VpcIds=[vpc_id])
            vpc = vpc_response["Vpcs"][0]

            details["vpc_id"] = vpc_id
            details["cidr_block"] = vpc["CidrBlock"]
            details["dns_hostnames"] = vpc.get("EnableDnsHostnames", False)
            details["dns_support"] = vpc.get("EnableDnsSupport", False)

            # Check DNS configuration
            if not vpc.get("EnableDnsHostnames", False):
                warnings.append("VPC DNS hostnames not enabled")

            if not vpc.get("EnableDnsSupport", False):
                violations.append("VPC DNS support not enabled")

            # Get subnets
            subnets_response = self._ec2_client.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )

            subnets = subnets_response["Subnets"]
            public_subnets = []
            private_subnets = []

            for subnet in subnets:
                if subnet.get("MapPublicIpOnLaunch", False):
                    public_subnets.append(subnet)
                else:
                    private_subnets.append(subnet)

            details["public_subnets"] = len(public_subnets)
            details["private_subnets"] = len(private_subnets)

            # Validate subnet configuration
            if len(private_subnets) == 0:
                violations.append("VPC has no private subnets")

            if len(public_subnets) == 0:
                warnings.append("VPC has no public subnets - may limit internet access")

            # Check for multiple AZs
            azs = set(subnet["AvailabilityZone"] for subnet in subnets)
            details["availability_zones"] = len(azs)

            if len(azs) < 2:
                recommendations.append(
                    "Use multiple availability zones for high availability"
                )

            # Check NAT gateways
            nat_gateways_response = self._ec2_client.describe_nat_gateways(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )

            nat_gateways = nat_gateways_response["NatGateways"]
            active_nat_gateways = [
                ng for ng in nat_gateways if ng["State"] == "available"
            ]

            details["nat_gateways"] = len(active_nat_gateways)

            if len(private_subnets) > 0 and len(active_nat_gateways) == 0:
                warnings.append(
                    "Private subnets without NAT gateway may lack internet access"
                )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "InvalidVpcID.NotFound":
                violations.append(f"VPC not found: {vpc_id}")
            else:
                violations.append(f"Failed to validate VPC: {str(e)}")

        is_compliant = len(violations) == 0

        return SecurityValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    def validate_encryption_in_transit(self) -> SecurityValidationResult:
        """
        Validate encryption in transit configuration.

        Returns:
            SecurityValidationResult with validation details
        """
        violations = []
        warnings = []
        recommendations = []
        details = {}

        # Check TLS configuration
        details["minimum_tls_version"] = self.policy.minimum_tls_version
        details["https_enforcement"] = True  # Enforced by CloudFront/ALB

        # In production, HTTPS should be enforced
        if settings.is_production:
            details["production_https_required"] = True
            recommendations.append("Ensure CloudFront and ALB enforce HTTPS redirects")
        else:
            details["production_https_required"] = False
            warnings.append("HTTPS enforcement not required in development")

        # Check for secure communication protocols
        secure_protocols = ["TLSv1.2", "TLSv1.3"]
        if self.policy.minimum_tls_version not in secure_protocols:
            violations.append(
                f"Minimum TLS version {self.policy.minimum_tls_version} is not secure"
            )

        is_compliant = len(violations) == 0

        return SecurityValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            details=details,
        )

    def get_current_identity(self) -> Dict[str, Any]:
        """
        Get current AWS identity information.

        Returns:
            Dictionary with identity information
        """
        if not self._sts_client:
            return {"identity_available": False, "reason": "STS client not available"}

        try:
            identity = self._sts_client.get_caller_identity()
            return {
                "identity_available": True,
                "account_id": identity.get("Account"),
                "user_id": identity.get("UserId"),
                "arn": identity.get("Arn"),
            }
        except ClientError as e:
            return {"identity_available": False, "error": str(e)}

    def comprehensive_security_audit(
        self,
        iam_role_arn: Optional[str] = None,
        s3_buckets: Optional[List[str]] = None,
        security_group_ids: Optional[List[str]] = None,
        vpc_id: Optional[str] = None,
    ) -> Dict[str, SecurityValidationResult]:
        """
        Perform comprehensive security audit.

        Args:
            iam_role_arn: IAM role ARN to validate
            s3_buckets: List of S3 bucket names to validate
            security_group_ids: List of security group IDs to validate
            vpc_id: VPC ID to validate

        Returns:
            Dictionary of validation results by component
        """
        results = {}

        # Validate IAM role
        if iam_role_arn:
            results["iam_role"] = self.validate_iam_least_privilege(iam_role_arn)

        # Validate S3 buckets
        if s3_buckets:
            results["s3_buckets"] = {}
            for bucket in s3_buckets:
                results["s3_buckets"][bucket] = self.validate_s3_encryption(bucket)

        # Validate security groups
        if security_group_ids:
            results["security_groups"] = self.validate_security_groups(
                security_group_ids
            )

        # Validate VPC
        if vpc_id:
            results["vpc"] = self.validate_vpc_configuration(vpc_id)

        # Validate encryption in transit
        results["encryption_in_transit"] = self.validate_encryption_in_transit()

        # Add identity information
        results["identity"] = self.get_current_identity()

        return results


# Global security service instance
_security_service: Optional[SecurityService] = None


def get_security_service() -> SecurityService:
    """
    Get the global security service instance.

    Returns:
        SecurityService instance
    """
    global _security_service
    if _security_service is None:
        _security_service = SecurityService()
    return _security_service
