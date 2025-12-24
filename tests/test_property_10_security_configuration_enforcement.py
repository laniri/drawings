"""
Property-based test for security configuration enforcement.

**Feature: aws-production-deployment, Property 10: Security Configuration Enforcement**
**Validates: Requirements 7.2, 7.3, 7.4, 7.5**

Property: For any AWS resource deployment, security controls should enforce 
least-privilege IAM permissions and encryption for data at rest and in transit.
"""

import pytest
from hypothesis import given, strategies as st, settings as hypothesis_settings, assume
from unittest.mock import patch, MagicMock
import json

# Optional AWS dependencies for testing
try:
    import boto3
    from botocore.exceptions import ClientError

    HAS_AWS = True
except ImportError:
    HAS_AWS = False
    boto3 = None
    ClientError = Exception

from app.services.security_service import (
    get_security_service,
    SecurityService,
    SecurityPolicy,
    SecurityValidationResult,
)
from app.core.config import settings


# Test data generators
@st.composite
def iam_role_arn(draw):
    """Generate valid IAM role ARN."""
    account_id = draw(st.text(min_size=12, max_size=12, alphabet="0123456789"))
    role_name = draw(
        st.text(
            min_size=1,
            max_size=64,
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
        )
    )
    return f"arn:aws:iam::{account_id}:role/{role_name}"


@st.composite
def s3_bucket_name(draw):
    """Generate valid S3 bucket name."""
    # S3 bucket naming rules: 3-63 chars, lowercase, numbers, hyphens
    return draw(
        st.text(
            min_size=3, max_size=63, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-"
        )
    ).strip("-")


@st.composite
def security_group_id(draw):
    """Generate valid security group ID."""
    return f"sg-{draw(st.text(min_size=8, max_size=17, alphabet='0123456789abcdef'))}"


@st.composite
def vpc_id(draw):
    """Generate valid VPC ID."""
    return f"vpc-{draw(st.text(min_size=8, max_size=17, alphabet='0123456789abcdef'))}"


@st.composite
def iam_policy_document(draw):
    """Generate IAM policy document with varying privilege levels."""
    effect = draw(st.sampled_from(["Allow", "Deny"]))

    # Generate actions with different privilege levels
    action_type = draw(st.integers(min_value=1, max_value=4))

    if action_type == 1:
        # Least privilege - specific actions
        actions = [
            draw(
                st.sampled_from(
                    [
                        "s3:GetObject",
                        "s3:PutObject",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "secretsmanager:GetSecretValue",
                    ]
                )
            )
        ]
    elif action_type == 2:
        # Moderate privilege - service-specific wildcards
        actions = [draw(st.sampled_from(["s3:*", "logs:*", "secretsmanager:*"]))]
    elif action_type == 3:
        # High privilege - broad wildcards
        actions = ["*"]
    else:
        # Multiple mixed actions
        actions = draw(
            st.lists(
                st.sampled_from(
                    ["s3:GetObject", "s3:*", "logs:*", "*", "iam:CreateRole", "ec2:*"]
                ),
                min_size=1,
                max_size=5,
            )
        )

    # Generate resources
    resource_type = draw(st.integers(min_value=1, max_value=3))

    if resource_type == 1:
        # Specific resources
        resources = [
            f"arn:aws:s3:::my-bucket/{draw(st.text(min_size=1, max_size=20))}/*"
        ]
    elif resource_type == 2:
        # Wildcard resources
        resources = ["*"]
    else:
        # Mixed resources
        resources = draw(
            st.lists(
                st.sampled_from(
                    ["arn:aws:s3:::my-bucket/*", "*", "arn:aws:logs:*:*:*"]
                ),
                min_size=1,
                max_size=3,
            )
        )

    return {
        "Version": "2012-10-17",
        "Statement": [{"Effect": effect, "Action": actions, "Resource": resources}],
    }


@st.composite
def s3_encryption_config(draw):
    """Generate S3 encryption configuration."""
    algorithm = draw(st.sampled_from(["AES256", "aws:kms", "invalid-algorithm"]))

    config = {
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": algorithm}}]
    }

    if algorithm == "aws:kms":
        config["Rules"][0]["ApplyServerSideEncryptionByDefault"][
            "KMSMasterKeyID"
        ] = f"arn:aws:kms:us-east-1:123456789012:key/{draw(st.text(min_size=36, max_size=36, alphabet='0123456789abcdef-'))}"

    return config


@st.composite
def security_group_rules(draw):
    """Generate security group ingress rules."""
    num_rules = draw(st.integers(min_value=0, max_value=5))
    rules = []

    for _ in range(num_rules):
        from_port = draw(st.integers(min_value=0, max_value=65535))
        to_port = draw(st.integers(min_value=from_port, max_value=65535))
        protocol = draw(st.sampled_from(["tcp", "udp", "icmp", "-1"]))

        # Generate CIDR blocks with varying security levels
        cidr_type = draw(st.integers(min_value=1, max_value=3))

        if cidr_type == 1:
            # Secure - specific IP ranges
            cidr_ip = "10.0.0.0/16"
        elif cidr_type == 2:
            # Less secure - broader ranges
            cidr_ip = "0.0.0.0/8"
        else:
            # Insecure - open to world
            cidr_ip = "0.0.0.0/0"

        rules.append(
            {
                "IpProtocol": protocol,
                "FromPort": from_port,
                "ToPort": to_port,
                "IpRanges": [{"CidrIp": cidr_ip}],
            }
        )

    return rules


class TestSecurityConfigurationEnforcement:
    """Test security configuration enforcement properties."""

    def setup_method(self):
        """Set up test environment."""
        # Mock AWS clients
        self.iam_patcher = patch("app.services.security_service.boto3.client")
        self.mock_boto3_client = self.iam_patcher.start()

        # Configure mock clients
        self.mock_iam_client = MagicMock()
        self.mock_s3_client = MagicMock()
        self.mock_ec2_client = MagicMock()
        self.mock_sts_client = MagicMock()

        def client_factory(service_name, **kwargs):
            if service_name == "iam":
                return self.mock_iam_client
            elif service_name == "s3":
                return self.mock_s3_client
            elif service_name == "ec2":
                return self.mock_ec2_client
            elif service_name == "sts":
                return self.mock_sts_client
            else:
                return MagicMock()

        self.mock_boto3_client.side_effect = client_factory

        # Create security service with mocked clients
        self.security_service = SecurityService()
        # Force initialization with mocked clients
        self.security_service._iam_client = self.mock_iam_client
        self.security_service._s3_client = self.mock_s3_client
        self.security_service._ec2_client = self.mock_ec2_client
        self.security_service._sts_client = self.mock_sts_client

    def teardown_method(self):
        """Clean up test environment."""
        self.iam_patcher.stop()

    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    @given(role_arn=iam_role_arn(), policy_doc=iam_policy_document())
    @hypothesis_settings(max_examples=50, deadline=5000)
    def test_iam_least_privilege_enforcement(self, role_arn, policy_doc):
        """
        Property: IAM roles should enforce least-privilege permissions.

        For any IAM role, overly broad permissions should be detected and flagged.
        """
        # Extract role name from ARN
        role_name = role_arn.split("/")[-1]

        # Mock IAM responses
        self.mock_iam_client.get_role.return_value = {
            "Role": {
                "RoleName": role_name,
                "Arn": role_arn,
                "AssumeRolePolicyDocument": {},
            }
        }

        # Determine if policy has dangerous permissions
        has_wildcard_actions = False
        has_wildcard_resources = False
        has_dangerous_policies = False

        statements = policy_doc.get("Statement", [])
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

                if "*" in actions:
                    has_wildcard_actions = True
                if "*" in resources:
                    has_wildcard_resources = True

        # Mock attached policies (some dangerous, some safe)
        dangerous_policies = [
            "arn:aws:iam::aws:policy/AdministratorAccess",
            "arn:aws:iam::aws:policy/PowerUserAccess",
        ]

        attached_policies = []
        if has_wildcard_actions or has_wildcard_resources:
            # Add a dangerous policy
            attached_policies.append(
                {
                    "PolicyName": "AdministratorAccess",
                    "PolicyArn": "arn:aws:iam::aws:policy/AdministratorAccess",
                }
            )
            has_dangerous_policies = True
        else:
            # Add safe policies
            attached_policies.append(
                {
                    "PolicyName": "S3ReadOnlyAccess",
                    "PolicyArn": "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
                }
            )

        self.mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": attached_policies
        }

        # Mock inline policies
        inline_policy_names = ["CustomPolicy"] if has_wildcard_actions else []
        self.mock_iam_client.list_role_policies.return_value = {
            "PolicyNames": inline_policy_names
        }

        if inline_policy_names:
            self.mock_iam_client.get_role_policy.return_value = {
                "PolicyDocument": policy_doc
            }

        # Test the validation
        result = self.security_service.validate_iam_least_privilege(role_arn)

        # Verify property: dangerous permissions should be flagged
        if has_dangerous_policies:
            assert (
                not result.is_compliant
            ), f"Role with dangerous policies should not be compliant: {role_arn}"
            assert (
                len(result.violations) > 0
            ), "Dangerous policies should generate violations"

        if has_wildcard_actions and has_wildcard_resources:
            # Should have warnings about overly broad permissions
            assert (
                len(result.warnings) > 0 or len(result.violations) > 0
            ), "Wildcard permissions should generate warnings or violations"

        # Verify result structure
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.details, dict)

        # Verify details contain expected information
        assert "role_name" in result.details
        assert "attached_policies" in result.details
        assert result.details["role_name"] == role_name

    @given(bucket_name=s3_bucket_name(), encryption_config=s3_encryption_config())
    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    @hypothesis_settings(max_examples=50, deadline=5000)
    def test_s3_encryption_enforcement(self, bucket_name, encryption_config):
        """
        Property: S3 buckets should enforce encryption at rest.

        For any S3 bucket, encryption should be properly configured and
        public access should be blocked.
        """
        assume(len(bucket_name) >= 3)  # Valid bucket name
        assume(not bucket_name.startswith("-"))  # Valid bucket name
        assume(not bucket_name.endswith("-"))  # Valid bucket name

        # Determine if encryption is properly configured
        rules = encryption_config.get("Rules", [])
        has_valid_encryption = False

        for rule in rules:
            sse_config = rule.get("ApplyServerSideEncryptionByDefault", {})
            algorithm = sse_config.get("SSEAlgorithm", "")

            if algorithm in ["AES256", "aws:kms"]:
                has_valid_encryption = True
                break

        # Mock S3 responses
        # Reset any previous side effects first
        self.mock_s3_client.reset_mock()

        if has_valid_encryption:
            self.mock_s3_client.get_bucket_encryption.return_value = {
                "ServerSideEncryptionConfiguration": encryption_config
            }
            self.mock_s3_client.get_bucket_encryption.side_effect = None
        else:
            # For invalid algorithms, still return the config but let the service detect it's invalid
            # OR simulate no encryption configured
            if any(
                rule.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
                == "invalid-algorithm"
                for rule in rules
            ):
                # Return the config with invalid algorithm - let service validate it
                self.mock_s3_client.get_bucket_encryption.return_value = {
                    "ServerSideEncryptionConfiguration": encryption_config
                }
                self.mock_s3_client.get_bucket_encryption.side_effect = None
            else:
                # Simulate no encryption configured
                self.mock_s3_client.get_bucket_encryption.side_effect = ClientError(
                    {
                        "Error": {
                            "Code": "ServerSideEncryptionConfigurationNotFoundError"
                        }
                    },
                    "GetBucketEncryption",
                )

        # Mock public access block (secure configuration)
        self.mock_s3_client.get_public_access_block.return_value = {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }
        }

        # Mock versioning
        self.mock_s3_client.get_bucket_versioning.return_value = {"Status": "Enabled"}

        # Reset any previous side effects
        self.mock_s3_client.get_bucket_encryption.side_effect = None

        # Test the validation
        result = self.security_service.validate_s3_encryption(bucket_name)

        # Verify property: encryption should be enforced
        if not has_valid_encryption:
            # Check if we have invalid algorithm or no encryption
            has_invalid_algorithm = any(
                rule.get("ApplyServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
                == "invalid-algorithm"
                for rule in rules
            )

            if has_invalid_algorithm:
                # Should have violations about invalid algorithm
                assert (
                    not result.is_compliant
                ), f"Bucket with invalid encryption algorithm should not be compliant: {bucket_name}"
                assert any(
                    "unsupported encryption algorithm" in violation.lower()
                    for violation in result.violations
                ), "Invalid encryption algorithm should generate algorithm-related violations"
            else:
                # Should have violations about missing encryption
                assert (
                    not result.is_compliant
                ), f"Bucket without proper encryption should not be compliant: {bucket_name}"
                assert any(
                    "encryption" in violation.lower() for violation in result.violations
                ), "Missing encryption should generate encryption-related violations"
        else:
            # Should be compliant if encryption is properly configured
            encryption_violations = [
                v for v in result.violations if "encryption" in v.lower()
            ]
            assert (
                len(encryption_violations) == 0
            ), f"Properly encrypted bucket should not have encryption violations: {encryption_violations}"

        # Verify result structure
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.details, dict)

        # Verify details contain encryption information
        assert "encryption_enabled" in result.details
        assert "public_access_block" in result.details

    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    @given(
        sg_ids=st.lists(security_group_id(), min_size=1, max_size=3),
        ingress_rules=security_group_rules(),
    )
    @hypothesis_settings(max_examples=50, deadline=5000)
    def test_security_group_minimal_exposure(self, sg_ids, ingress_rules):
        """
        Property: Security groups should minimize port exposure.

        For any security group, overly permissive rules should be detected,
        especially those allowing access from 0.0.0.0/0.
        """
        # Analyze rules for security violations
        has_open_ssh = False
        has_open_all_ports = False
        has_world_accessible = False

        for rule in ingress_rules:
            from_port = rule.get("FromPort", 0)
            to_port = rule.get("ToPort", 65535)

            for ip_range in rule.get("IpRanges", []):
                cidr = ip_range.get("CidrIp", "")

                if cidr == "0.0.0.0/0":
                    has_world_accessible = True

                    if from_port == 22 and to_port == 22:
                        has_open_ssh = True

                    if from_port == 0 and to_port == 65535:
                        has_open_all_ports = True

        # Mock EC2 responses
        security_groups = []
        for sg_id in sg_ids:
            security_groups.append(
                {
                    "GroupId": sg_id,
                    "GroupName": f"test-sg-{sg_id[-8:]}",
                    "Description": "Test security group",
                    "IpPermissions": ingress_rules,
                    "IpPermissionsEgress": [],
                }
            )

        self.mock_ec2_client.describe_security_groups.return_value = {
            "SecurityGroups": security_groups
        }

        # Test the validation
        result = self.security_service.validate_security_groups(sg_ids)

        # Verify property: overly permissive rules should be flagged
        if has_open_ssh:
            assert (
                not result.is_compliant
            ), "Security group allowing SSH from anywhere should not be compliant"
            assert any(
                "SSH" in violation or "22" in violation
                for violation in result.violations
            ), "Open SSH access should generate SSH-related violations"

        if has_open_all_ports:
            assert (
                not result.is_compliant
            ), "Security group allowing all ports from anywhere should not be compliant"
            assert any(
                "all traffic" in violation.lower() for violation in result.violations
            ), "Open all ports should generate traffic-related violations"

        # World-accessible ports should generate warnings (except for allowed ports)
        if has_world_accessible:
            # Should have violations or warnings about world accessibility
            total_issues = len(result.violations) + len(result.warnings)
            assert (
                total_issues > 0
            ), "World-accessible security groups should generate violations or warnings"

        # Verify result structure
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.violations, list)
        assert isinstance(result.details, dict)

        # Verify details contain security group information
        assert "security_groups" in result.details
        assert len(result.details["security_groups"]) == len(sg_ids)

    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    @given(vpc_id=vpc_id())
    @hypothesis_settings(max_examples=30, deadline=5000)
    def test_vpc_private_subnet_requirement(self, vpc_id):
        """
        Property: VPC should have private subnets for application tier.

        For any VPC deployment, there should be private subnets to isolate
        application resources from direct internet access.
        """
        # Generate subnet configuration
        has_private_subnets = True  # Most deployments should have private subnets
        has_public_subnets = True  # Need public subnets for load balancers

        # Mock VPC response
        self.mock_ec2_client.describe_vpcs.return_value = {
            "Vpcs": [
                {
                    "VpcId": vpc_id,
                    "CidrBlock": "10.0.0.0/16",
                    "EnableDnsHostnames": True,
                    "EnableDnsSupport": True,
                }
            ]
        }

        # Mock subnets
        subnets = []

        if has_public_subnets:
            subnets.extend(
                [
                    {
                        "SubnetId": "subnet-public1",
                        "VpcId": vpc_id,
                        "CidrBlock": "10.0.1.0/24",
                        "AvailabilityZone": "us-east-1a",
                        "MapPublicIpOnLaunch": True,
                    },
                    {
                        "SubnetId": "subnet-public2",
                        "VpcId": vpc_id,
                        "CidrBlock": "10.0.2.0/24",
                        "AvailabilityZone": "us-east-1b",
                        "MapPublicIpOnLaunch": True,
                    },
                ]
            )

        if has_private_subnets:
            subnets.extend(
                [
                    {
                        "SubnetId": "subnet-private1",
                        "VpcId": vpc_id,
                        "CidrBlock": "10.0.11.0/24",
                        "AvailabilityZone": "us-east-1a",
                        "MapPublicIpOnLaunch": False,
                    },
                    {
                        "SubnetId": "subnet-private2",
                        "VpcId": vpc_id,
                        "CidrBlock": "10.0.12.0/24",
                        "AvailabilityZone": "us-east-1b",
                        "MapPublicIpOnLaunch": False,
                    },
                ]
            )

        self.mock_ec2_client.describe_subnets.return_value = {"Subnets": subnets}

        # Mock NAT gateways
        self.mock_ec2_client.describe_nat_gateways.return_value = {
            "NatGateways": [
                {"NatGatewayId": "nat-12345", "VpcId": vpc_id, "State": "available"}
            ]
            if has_private_subnets
            else []
        }

        # Test the validation
        result = self.security_service.validate_vpc_configuration(vpc_id)

        # Verify property: should have private subnets
        if not has_private_subnets:
            assert (
                not result.is_compliant
            ), "VPC without private subnets should not be compliant"
            assert any(
                "private subnet" in violation.lower() for violation in result.violations
            ), "Missing private subnets should generate subnet-related violations"

        # Verify result structure
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.details, dict)

        # Verify details contain VPC information
        assert "vpc_id" in result.details
        assert "private_subnets" in result.details
        assert "public_subnets" in result.details
        assert result.details["vpc_id"] == vpc_id

    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    def test_encryption_in_transit_enforcement(self):
        """
        Property: Encryption in transit should be enforced.

        For any production deployment, HTTPS should be enforced and
        minimum TLS version should be secure.
        """
        # Test encryption in transit validation
        result = self.security_service.validate_encryption_in_transit()

        # Verify property: should enforce secure protocols
        assert isinstance(result.is_compliant, bool)
        assert isinstance(result.details, dict)

        # Verify TLS configuration
        assert "minimum_tls_version" in result.details
        assert "https_enforcement" in result.details

        # In production, HTTPS should be required
        if settings.is_production:
            assert (
                result.details["production_https_required"] == True
            ), "Production environment should require HTTPS"

        # Minimum TLS version should be secure
        min_tls = result.details["minimum_tls_version"]
        secure_versions = ["1.2", "1.3"]

        if min_tls not in secure_versions:
            assert (
                not result.is_compliant
            ), f"Insecure TLS version {min_tls} should not be compliant"

    @pytest.mark.skipif(not HAS_AWS, reason="AWS dependencies not available")
    @given(
        role_arn=iam_role_arn(),
        bucket_names=st.lists(s3_bucket_name(), min_size=1, max_size=3),
        sg_ids=st.lists(security_group_id(), min_size=1, max_size=2),
        vpc_id=vpc_id(),
    )
    @hypothesis_settings(max_examples=20, deadline=10000)
    def test_comprehensive_security_audit(self, role_arn, bucket_names, sg_ids, vpc_id):
        """
        Property: Comprehensive security audit should validate all components.

        For any complete AWS deployment, all security components should be
        validated together and overall compliance determined.
        """
        # Filter valid bucket names
        valid_buckets = [
            bucket
            for bucket in bucket_names
            if len(bucket) >= 3
            and not bucket.startswith("-")
            and not bucket.endswith("-")
        ]

        assume(len(valid_buckets) > 0)

        # Mock all AWS service responses for comprehensive audit
        self._setup_comprehensive_mocks(role_arn, valid_buckets, sg_ids, vpc_id)

        # Perform comprehensive audit
        results = self.security_service.comprehensive_security_audit(
            iam_role_arn=role_arn,
            s3_buckets=valid_buckets,
            security_group_ids=sg_ids,
            vpc_id=vpc_id,
        )

        # Verify property: all components should be validated
        expected_components = [
            "iam_role",
            "s3_buckets",
            "security_groups",
            "vpc",
            "encryption_in_transit",
            "identity",
        ]

        for component in expected_components:
            assert (
                component in results
            ), f"Comprehensive audit should include {component} validation"

        # Verify IAM role validation
        iam_result = results["iam_role"]
        assert isinstance(iam_result, SecurityValidationResult)
        assert isinstance(iam_result.is_compliant, bool)

        # Verify S3 bucket validations
        s3_results = results["s3_buckets"]
        assert isinstance(s3_results, dict)

        for bucket in valid_buckets:
            assert bucket in s3_results, f"S3 validation should include bucket {bucket}"
            bucket_result = s3_results[bucket]
            assert isinstance(bucket_result, SecurityValidationResult)

        # Verify security group validation
        sg_result = results["security_groups"]
        assert isinstance(sg_result, SecurityValidationResult)

        # Verify VPC validation
        vpc_result = results["vpc"]
        assert isinstance(vpc_result, SecurityValidationResult)

        # Verify encryption in transit
        transit_result = results["encryption_in_transit"]
        assert isinstance(transit_result, SecurityValidationResult)

        # Verify identity information
        identity_result = results["identity"]
        assert isinstance(identity_result, dict)
        assert "identity_available" in identity_result

    def _setup_comprehensive_mocks(self, role_arn, bucket_names, sg_ids, vpc_id):
        """Set up mocks for comprehensive security audit."""
        role_name = role_arn.split("/")[-1]

        # Mock IAM
        self.mock_iam_client.get_role.return_value = {
            "Role": {"RoleName": role_name, "Arn": role_arn}
        }
        self.mock_iam_client.list_attached_role_policies.return_value = {
            "AttachedPolicies": [
                {
                    "PolicyName": "S3Access",
                    "PolicyArn": "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess",
                }
            ]
        }
        self.mock_iam_client.list_role_policies.return_value = {"PolicyNames": []}

        # Mock S3
        self.mock_s3_client.get_bucket_encryption.return_value = {
            "ServerSideEncryptionConfiguration": {
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            }
        }
        self.mock_s3_client.get_public_access_block.return_value = {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }
        }
        self.mock_s3_client.get_bucket_versioning.return_value = {"Status": "Enabled"}

        # Mock EC2
        self.mock_ec2_client.describe_security_groups.return_value = {
            "SecurityGroups": [
                {
                    "GroupId": sg_id,
                    "GroupName": f"sg-{sg_id[-8:]}",
                    "Description": "Test SG",
                    "IpPermissions": [
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 443,
                            "ToPort": 443,
                            "IpRanges": [{"CidrIp": "0.0.0.0/0"}],
                        }
                    ],
                    "IpPermissionsEgress": [],
                }
                for sg_id in sg_ids
            ]
        }

        self.mock_ec2_client.describe_vpcs.return_value = {
            "Vpcs": [
                {
                    "VpcId": vpc_id,
                    "CidrBlock": "10.0.0.0/16",
                    "EnableDnsHostnames": True,
                    "EnableDnsSupport": True,
                }
            ]
        }

        self.mock_ec2_client.describe_subnets.return_value = {
            "Subnets": [
                {
                    "SubnetId": "subnet-private1",
                    "VpcId": vpc_id,
                    "CidrBlock": "10.0.11.0/24",
                    "AvailabilityZone": "us-east-1a",
                    "MapPublicIpOnLaunch": False,
                }
            ]
        }

        self.mock_ec2_client.describe_nat_gateways.return_value = {"NatGateways": []}

        # Mock STS
        self.mock_sts_client.get_caller_identity.return_value = {
            "Account": "123456789012",
            "UserId": "AIDACKCEVSQ6C2EXAMPLE",
            "Arn": "arn:aws:iam::123456789012:user/test-user",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
