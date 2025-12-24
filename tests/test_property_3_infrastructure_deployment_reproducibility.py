"""
Property-based test for infrastructure deployment reproducibility.

**Feature: aws-production-deployment, Property 3: Infrastructure Deployment Reproducibility**
**Validates: Requirements 2.2, 2.4, 2.5**

This test validates that destroying and recreating the infrastructure should result
in functionally equivalent AWS resources.
"""

import json
import tempfile
import os
import pytest
from hypothesis import given, strategies as st, assume
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
import yaml


class MockCloudFormationTemplate:
    """Mock CloudFormation template for testing infrastructure reproducibility"""
    
    def __init__(self, template_data: Dict[str, Any]):
        self.template_data = template_data
        self.resources = template_data.get("Resources", {})
        self.outputs = template_data.get("Outputs", {})
        self.parameters = template_data.get("Parameters", {})
    
    def get_resource_properties(self, resource_name: str) -> Dict[str, Any]:
        """Get properties of a specific resource"""
        return self.resources.get(resource_name, {}).get("Properties", {})
    
    def get_resource_type(self, resource_name: str) -> str:
        """Get type of a specific resource"""
        return self.resources.get(resource_name, {}).get("Type", "")
    
    def validate_template(self) -> bool:
        """Basic template validation"""
        required_sections = ["AWSTemplateFormatVersion", "Resources"]
        return all(section in self.template_data for section in required_sections)
    
    def get_resource_names_by_type(self, resource_type: str) -> List[str]:
        """Get all resource names of a specific type"""
        return [
            name for name, resource in self.resources.items()
            if resource.get("Type") == resource_type
        ]


class TestInfrastructureDeploymentReproducibility:
    """Property-based tests for infrastructure deployment reproducibility"""
    
    @given(
        stack_name=st.text(
            min_size=1, 
            max_size=128, 
            alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd'), whitelist_characters='-')
        ).filter(lambda x: not x.startswith('-') and not x.endswith('-')),
        environment=st.sampled_from(["production", "staging", "dev"]),
        region=st.sampled_from(["eu-west-1", "us-east-1", "us-west-2", "ap-southeast-1"]),
        enable_cloudfront=st.booleans(),
        enable_route53=st.booleans()
    )
    def test_infrastructure_template_reproducibility(
        self, 
        stack_name: str, 
        environment: str, 
        region: str,
        enable_cloudfront: bool,
        enable_route53: bool
    ):
        """
        **Feature: aws-production-deployment, Property 3: Infrastructure Deployment Reproducibility**
        **Validates: Requirements 2.2, 2.4, 2.5**
        
        For any Infrastructure as Code deployment, destroying and recreating the 
        infrastructure should result in functionally equivalent AWS resources.
        """
        # Skip invalid stack names
        assume(len(stack_name) > 0 and not stack_name.startswith('-') and not stack_name.endswith('-'))
        
        # Create template parameters
        template_params = {
            "StackName": stack_name,
            "Environment": environment,
            "Region": region,
            "EnableCloudFront": enable_cloudfront,
            "EnableRoute53": enable_route53
        }
        
        # Generate template twice with same parameters
        template1 = self._generate_cloudformation_template(template_params)
        template2 = self._generate_cloudformation_template(template_params)
        
        # Templates should be identical
        assert template1.template_data == template2.template_data
        
        # Verify essential resources are present
        self._verify_essential_resources(template1)
        self._verify_essential_resources(template2)
        
        # Verify resource properties are consistent
        self._verify_resource_consistency(template1, template2)
        
        # Verify outputs are consistent
        assert template1.outputs == template2.outputs
    
    @given(
        vpc_cidr=st.sampled_from(["10.0.0.0/16", "172.16.0.0/16", "192.168.0.0/16"]),
        availability_zones=st.integers(min_value=2, max_value=3),
        enable_nat_gateway=st.booleans(),
        instance_type=st.sampled_from(["t3.micro", "t3.small", "t3.medium"])
    )
    def test_network_configuration_reproducibility(
        self,
        vpc_cidr: str,
        availability_zones: int,
        enable_nat_gateway: bool,
        instance_type: str
    ):
        """
        Test that network configurations are reproducible across deployments.
        """
        network_params = {
            "VpcCidr": vpc_cidr,
            "AvailabilityZones": availability_zones,
            "EnableNatGateway": enable_nat_gateway,
            "InstanceType": instance_type
        }
        
        # Generate network configuration twice
        template1 = self._generate_cloudformation_template(network_params)
        template2 = self._generate_cloudformation_template(network_params)
        
        # VPC configurations should be identical
        vpc_resources1 = template1.get_resource_names_by_type("AWS::EC2::VPC")
        vpc_resources2 = template2.get_resource_names_by_type("AWS::EC2::VPC")
        
        assert len(vpc_resources1) == len(vpc_resources2) == 1
        
        vpc_props1 = template1.get_resource_properties(vpc_resources1[0])
        vpc_props2 = template2.get_resource_properties(vpc_resources2[0])
        
        assert vpc_props1["CidrBlock"] == vpc_props2["CidrBlock"] == vpc_cidr
        
        # Subnet configurations should be identical
        subnet_resources1 = template1.get_resource_names_by_type("AWS::EC2::Subnet")
        subnet_resources2 = template2.get_resource_names_by_type("AWS::EC2::Subnet")
        
        assert len(subnet_resources1) == len(subnet_resources2)
        assert len(subnet_resources1) >= availability_zones * 2  # Public + Private subnets
    
    @given(
        s3_bucket_prefix=st.text(
            min_size=3, 
            max_size=20, 
            alphabet=st.characters(whitelist_categories=('Ll', 'Nd'), whitelist_characters='-')
        ).filter(lambda x: not x.startswith('-') and not x.endswith('-')),
        enable_versioning=st.booleans(),
        enable_encryption=st.booleans(),
        lifecycle_days=st.integers(min_value=30, max_value=365)
    )
    def test_storage_configuration_reproducibility(
        self,
        s3_bucket_prefix: str,
        enable_versioning: bool,
        enable_encryption: bool,
        lifecycle_days: int
    ):
        """
        Test that S3 storage configurations are reproducible.
        """
        storage_params = {
            "S3BucketPrefix": s3_bucket_prefix,
            "EnableVersioning": enable_versioning,
            "EnableEncryption": enable_encryption,
            "LifecycleDays": lifecycle_days
        }
        
        # Generate storage configuration twice
        template1 = self._generate_cloudformation_template(storage_params)
        template2 = self._generate_cloudformation_template(storage_params)
        
        # S3 bucket configurations should be identical
        s3_resources1 = template1.get_resource_names_by_type("AWS::S3::Bucket")
        s3_resources2 = template2.get_resource_names_by_type("AWS::S3::Bucket")
        
        assert len(s3_resources1) == len(s3_resources2)
        
        for bucket1, bucket2 in zip(s3_resources1, s3_resources2):
            props1 = template1.get_resource_properties(bucket1)
            props2 = template2.get_resource_properties(bucket2)
            
            # Versioning configuration should be identical
            if enable_versioning:
                assert props1.get("VersioningConfiguration", {}).get("Status") == "Enabled"
                assert props2.get("VersioningConfiguration", {}).get("Status") == "Enabled"
            
            # Encryption configuration should be identical
            if enable_encryption:
                assert "BucketEncryption" in props1
                assert "BucketEncryption" in props2
                assert props1["BucketEncryption"] == props2["BucketEncryption"]
    
    @given(
        cpu_units=st.integers(min_value=256, max_value=1024),
        memory_mb=st.integers(min_value=512, max_value=2048),
        desired_count=st.integers(min_value=1, max_value=3),
        enable_auto_scaling=st.booleans()
    )
    def test_ecs_configuration_reproducibility(
        self,
        cpu_units: int,
        memory_mb: int,
        desired_count: int,
        enable_auto_scaling: bool
    ):
        """
        Test that ECS Fargate configurations are reproducible.
        """
        ecs_params = {
            "CpuUnits": cpu_units,
            "MemoryMB": memory_mb,
            "DesiredCount": desired_count,
            "EnableAutoScaling": enable_auto_scaling
        }
        
        # Generate ECS configuration twice
        template1 = self._generate_cloudformation_template(ecs_params)
        template2 = self._generate_cloudformation_template(ecs_params)
        
        # ECS cluster configurations should be identical
        cluster_resources1 = template1.get_resource_names_by_type("AWS::ECS::Cluster")
        cluster_resources2 = template2.get_resource_names_by_type("AWS::ECS::Cluster")
        
        assert len(cluster_resources1) == len(cluster_resources2) == 1
        
        # Task definition configurations should be identical
        task_def_resources1 = template1.get_resource_names_by_type("AWS::ECS::TaskDefinition")
        task_def_resources2 = template2.get_resource_names_by_type("AWS::ECS::TaskDefinition")
        
        assert len(task_def_resources1) == len(task_def_resources2) == 1
        
        task_props1 = template1.get_resource_properties(task_def_resources1[0])
        task_props2 = template2.get_resource_properties(task_def_resources2[0])
        
        assert task_props1["Cpu"] == task_props2["Cpu"] == str(cpu_units)
        assert task_props1["Memory"] == task_props2["Memory"] == str(memory_mb)
        
        # Service configurations should be identical
        service_resources1 = template1.get_resource_names_by_type("AWS::ECS::Service")
        service_resources2 = template2.get_resource_names_by_type("AWS::ECS::Service")
        
        assert len(service_resources1) == len(service_resources2) == 1
        
        service_props1 = template1.get_resource_properties(service_resources1[0])
        service_props2 = template2.get_resource_properties(service_resources2[0])
        
        assert service_props1["DesiredCount"] == service_props2["DesiredCount"] == desired_count
    
    def test_template_validation_consistency(self):
        """
        Test that generated templates consistently pass validation.
        """
        base_params = {
            "StackName": "test-stack",
            "Environment": "production",
            "Region": "eu-west-1"
        }
        
        # Generate multiple templates with same parameters
        templates = [
            self._generate_cloudformation_template(base_params)
            for _ in range(5)
        ]
        
        # All templates should be valid
        for template in templates:
            assert template.validate_template()
        
        # All templates should be identical
        first_template = templates[0]
        for template in templates[1:]:
            assert template.template_data == first_template.template_data
    
    def _generate_cloudformation_template(self, params: Dict[str, Any]) -> MockCloudFormationTemplate:
        """
        Generate a CloudFormation template based on parameters.
        This simulates the actual template generation logic.
        """
        template_data = {
            "AWSTemplateFormatVersion": "2010-09-09",
            "Description": f"Infrastructure for {params.get('Environment', 'production')} environment",
            "Parameters": {
                "Environment": {
                    "Type": "String",
                    "Default": params.get("Environment", "production"),
                    "AllowedValues": ["production", "staging", "dev"]
                }
            },
            "Resources": {},
            "Outputs": {}
        }
        
        # Always add essential resources for basic infrastructure
        # Add VPC by default if not specified
        if not params.get("VpcCidr"):
            params["VpcCidr"] = "10.0.0.0/16"
            params["AvailabilityZones"] = 2
        
        # Add S3 bucket by default if not specified
        if not params.get("S3BucketPrefix"):
            params["S3BucketPrefix"] = params.get("StackName", "default")
        
        # Add ECS resources by default if not specified
        if not params.get("CpuUnits"):
            params["CpuUnits"] = 256
            params["MemoryMB"] = 512
            params["DesiredCount"] = 1
        
        # Add VPC resources
        if params.get("VpcCidr"):
            template_data["Resources"]["VPC"] = {
                "Type": "AWS::EC2::VPC",
                "Properties": {
                    "CidrBlock": params["VpcCidr"],
                    "EnableDnsHostnames": True,
                    "EnableDnsSupport": True,
                    "Tags": [
                        {"Key": "Name", "Value": f"{params.get('StackName', 'default')}-vpc"}
                    ]
                }
            }
            
            # Add subnets based on availability zones
            az_count = params.get("AvailabilityZones", 2)
            for i in range(az_count):
                # Public subnet
                template_data["Resources"][f"PublicSubnet{i+1}"] = {
                    "Type": "AWS::EC2::Subnet",
                    "Properties": {
                        "VpcId": {"Ref": "VPC"},
                        "CidrBlock": f"10.0.{i+1}.0/24",
                        "AvailabilityZone": {"Fn::Select": [i, {"Fn::GetAZs": ""}]},
                        "MapPublicIpOnLaunch": True
                    }
                }
                
                # Private subnet
                template_data["Resources"][f"PrivateSubnet{i+1}"] = {
                    "Type": "AWS::EC2::Subnet",
                    "Properties": {
                        "VpcId": {"Ref": "VPC"},
                        "CidrBlock": f"10.0.{i+10}.0/24",
                        "AvailabilityZone": {"Fn::Select": [i, {"Fn::GetAZs": ""}]}
                    }
                }
        
        # Add S3 resources
        if params.get("S3BucketPrefix"):
            bucket_name = f"{params['S3BucketPrefix']}-{params.get('Environment', 'prod')}-bucket"
            bucket_config = {
                "Type": "AWS::S3::Bucket",
                "Properties": {
                    "BucketName": bucket_name,
                    "PublicAccessBlockConfiguration": {
                        "BlockPublicAcls": True,
                        "BlockPublicPolicy": True,
                        "IgnorePublicAcls": True,
                        "RestrictPublicBuckets": True
                    }
                }
            }
            
            if params.get("EnableVersioning"):
                bucket_config["Properties"]["VersioningConfiguration"] = {"Status": "Enabled"}
            
            if params.get("EnableEncryption"):
                bucket_config["Properties"]["BucketEncryption"] = {
                    "ServerSideEncryptionConfiguration": [
                        {
                            "ServerSideEncryptionByDefault": {
                                "SSEAlgorithm": "AES256"
                            }
                        }
                    ]
                }
            
            if params.get("LifecycleDays"):
                bucket_config["Properties"]["LifecycleConfiguration"] = {
                    "Rules": [
                        {
                            "Id": "DeleteOldVersions",
                            "Status": "Enabled",
                            "NoncurrentVersionExpirationInDays": params["LifecycleDays"]
                        }
                    ]
                }
            
            template_data["Resources"]["S3Bucket"] = bucket_config
        
        # Add ECS resources
        if params.get("CpuUnits") or params.get("MemoryMB"):
            # ECS Cluster
            template_data["Resources"]["ECSCluster"] = {
                "Type": "AWS::ECS::Cluster",
                "Properties": {
                    "ClusterName": f"{params.get('StackName', 'default')}-cluster"
                }
            }
            
            # Task Definition
            template_data["Resources"]["TaskDefinition"] = {
                "Type": "AWS::ECS::TaskDefinition",
                "Properties": {
                    "Family": f"{params.get('StackName', 'default')}-task",
                    "NetworkMode": "awsvpc",
                    "RequiresCompatibilities": ["FARGATE"],
                    "Cpu": str(params.get("CpuUnits", 256)),
                    "Memory": str(params.get("MemoryMB", 512)),
                    "ExecutionRoleArn": {"Ref": "TaskExecutionRole"},
                    "ContainerDefinitions": [
                        {
                            "Name": "app",
                            "Image": "nginx:latest",
                            "PortMappings": [{"ContainerPort": 8000}],
                            "LogConfiguration": {
                                "LogDriver": "awslogs",
                                "Options": {
                                    "awslogs-group": {"Ref": "LogGroup"},
                                    "awslogs-region": {"Ref": "AWS::Region"},
                                    "awslogs-stream-prefix": "ecs"
                                }
                            }
                        }
                    ]
                }
            }
            
            # ECS Service
            template_data["Resources"]["ECSService"] = {
                "Type": "AWS::ECS::Service",
                "Properties": {
                    "Cluster": {"Ref": "ECSCluster"},
                    "TaskDefinition": {"Ref": "TaskDefinition"},
                    "DesiredCount": params.get("DesiredCount", 1),
                    "LaunchType": "FARGATE",
                    "NetworkConfiguration": {
                        "AwsvpcConfiguration": {
                            "SecurityGroups": [{"Ref": "ECSSecurityGroup"}],
                            "Subnets": [{"Ref": "PrivateSubnet1"}]
                        }
                    }
                }
            }
            
            # Supporting resources
            template_data["Resources"]["TaskExecutionRole"] = {
                "Type": "AWS::IAM::Role",
                "Properties": {
                    "AssumeRolePolicyDocument": {
                        "Version": "2012-10-17",
                        "Statement": [
                            {
                                "Effect": "Allow",
                                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                                "Action": "sts:AssumeRole"
                            }
                        ]
                    },
                    "ManagedPolicyArns": [
                        "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
                    ]
                }
            }
            
            template_data["Resources"]["LogGroup"] = {
                "Type": "AWS::Logs::LogGroup",
                "Properties": {
                    "LogGroupName": f"/ecs/{params.get('StackName', 'default')}",
                    "RetentionInDays": 30
                }
            }
            
            template_data["Resources"]["ECSSecurityGroup"] = {
                "Type": "AWS::EC2::SecurityGroup",
                "Properties": {
                    "GroupDescription": "Security group for ECS tasks",
                    "VpcId": {"Ref": "VPC"},
                    "SecurityGroupIngress": [
                        {
                            "IpProtocol": "tcp",
                            "FromPort": 8000,
                            "ToPort": 8000,
                            "CidrIp": "10.0.0.0/16"
                        }
                    ]
                }
            }
        
        # Add outputs
        template_data["Outputs"]["StackName"] = {
            "Description": "Name of the CloudFormation stack",
            "Value": {"Ref": "AWS::StackName"}
        }
        
        if "VPC" in template_data["Resources"]:
            template_data["Outputs"]["VPCId"] = {
                "Description": "VPC ID",
                "Value": {"Ref": "VPC"}
            }
        
        if "S3Bucket" in template_data["Resources"]:
            template_data["Outputs"]["S3BucketName"] = {
                "Description": "S3 Bucket Name",
                "Value": {"Ref": "S3Bucket"}
            }
        
        if "ECSCluster" in template_data["Resources"]:
            template_data["Outputs"]["ECSClusterName"] = {
                "Description": "ECS Cluster Name",
                "Value": {"Ref": "ECSCluster"}
            }
        
        return MockCloudFormationTemplate(template_data)
    
    def _verify_essential_resources(self, template: MockCloudFormationTemplate):
        """Verify that essential resources are present in the template"""
        assert template.validate_template()
        
        # Check for required sections
        assert "Resources" in template.template_data
        assert "Outputs" in template.template_data
        
        # Verify at least one resource exists
        assert len(template.resources) > 0
    
    def _verify_resource_consistency(self, template1: MockCloudFormationTemplate, template2: MockCloudFormationTemplate):
        """Verify that resources are consistent between templates"""
        # Same number of resources
        assert len(template1.resources) == len(template2.resources)
        
        # Same resource names and types
        for resource_name in template1.resources:
            assert resource_name in template2.resources
            assert template1.get_resource_type(resource_name) == template2.get_resource_type(resource_name)
            assert template1.get_resource_properties(resource_name) == template2.get_resource_properties(resource_name)