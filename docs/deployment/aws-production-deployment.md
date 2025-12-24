# AWS Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Children's Drawing Anomaly Detection System to AWS using Infrastructure as Code (CloudFormation) and automated CI/CD pipelines.

## Architecture Summary

The production deployment uses:
- **ECS Fargate**: Serverless container hosting (0.25 vCPU, 0.5 GB RAM)
- **Application Load Balancer**: Traffic distribution and health checks
- **CloudFront CDN**: Global content delivery and caching
- **S3 Buckets**: Storage for drawings, models, and backups
- **VPC**: Secure networking with public/private subnets
- **Route 53**: DNS management (optional)
- **CloudWatch**: Monitoring, logging, and alerting
- **Secrets Manager**: Secure credential storage

## Prerequisites

### Required Tools
- AWS CLI v2.0+ configured with appropriate permissions
- Docker 20.10+ for local image building
- Git for source code management
- jq for JSON processing (optional but recommended)

### AWS Permissions Required
Your AWS user/role needs the following permissions:
- CloudFormation: Full access
- ECS: Full access
- EC2: VPC, Security Groups, Load Balancer management
- S3: Bucket creation and management
- IAM: Role and policy creation
- Route 53: DNS management (if using custom domain)
- CloudWatch: Metrics and logging
- Secrets Manager: Secret creation and access

### Domain Setup (Optional)
If using a custom domain:
1. Register domain in Route 53 or external registrar
2. Request SSL certificate in AWS Certificate Manager (ACM)
3. Note the certificate ARN for deployment

## Quick Start Deployment

### 1. Clone Repository and Setup
```bash
git clone <repository-url>
cd children-drawing-anomaly-detection
cd infrastructure
```

### 2. Configure AWS CLI
```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and region (eu-west-1)
```

### 3. Deploy Infrastructure
```bash
# Make deployment script executable
chmod +x deploy.sh

# Run interactive deployment
./deploy.sh
```

The script will prompt for:
- Domain name (optional)
- S3 bucket prefix
- ECR repository URI (optional)
- Certificate ARN (if using custom domain)
- Cost alert email address

### 4. Build and Deploy Application
```bash
# Build Docker image
docker build -t children-drawing-app -f ../Dockerfile.prod ..

# Tag for ECR (replace with your ECR URI)
docker tag children-drawing-app:latest <account-id>.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# Push to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.eu-west-1.amazonaws.com
docker push <account-id>.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# Update ECS service
aws ecs update-service --cluster children-drawing-prod-cluster --service children-drawing-prod-service --force-new-deployment --region eu-west-1
```

## Detailed Deployment Steps

### Step 1: Infrastructure Validation
```bash
# Validate CloudFormation templates
./deploy.sh validate

# Check AWS CLI configuration
aws sts get-caller-identity
```

### Step 2: Parameter Configuration
Create a parameters file for consistent deployments:

```json
# parameters.json
[
  {
    "ParameterKey": "Environment",
    "ParameterValue": "production"
  },
  {
    "ParameterKey": "DomainName",
    "ParameterValue": "your-domain.com"
  },
  {
    "ParameterKey": "S3BucketPrefix",
    "ParameterValue": "children-drawing"
  },
  {
    "ParameterKey": "CostAlertEmail",
    "ParameterValue": "admin@your-domain.com"
  }
]
```

### Step 3: Deploy with Parameters File
```bash
aws cloudformation deploy \
  --template-file main-infrastructure.yaml \
  --stack-name children-drawing-prod \
  --parameter-overrides file://parameters.json \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region eu-west-1 \
  --tags Environment=production Project=ChildrenDrawingAnomalyDetection
```

### Step 4: Verify Deployment
```bash
# Check stack status
aws cloudformation describe-stacks --stack-name children-drawing-prod --region eu-west-1

# Get stack outputs
aws cloudformation describe-stacks \
  --stack-name children-drawing-prod \
  --region eu-west-1 \
  --query 'Stacks[0].Outputs' \
  --output table
```

### Step 5: Application Deployment
```bash
# Create ECR repository if not exists
aws ecr create-repository --repository-name children-drawing-app --region eu-west-1

# Build and push application image
docker build -t children-drawing-app -f ../Dockerfile.prod ..
docker tag children-drawing-app:latest $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# Login to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com

# Push image
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# Update ECS task definition with new image
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --force-new-deployment \
  --region eu-west-1
```

## Environment Configuration

### Environment Variables
The application uses the following environment variables in production:

| Variable | Description | Source |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | CloudFormation |
| `AWS_DEFAULT_REGION` | AWS region | CloudFormation |
| `S3_BUCKET_DRAWINGS` | Drawings storage bucket | CloudFormation |
| `S3_BUCKET_MODELS` | ML models bucket | CloudFormation |
| `S3_BUCKET_BACKUPS` | Database backups bucket | CloudFormation |
| `ENABLE_METRICS` | Enable CloudWatch metrics | CloudFormation |
| `ENABLE_SECRETS_MANAGER` | Use Secrets Manager | CloudFormation |
| `ADMIN_PASSWORD_SECRET_ARN` | Admin password secret ARN | Secrets Manager |

### Database Configuration
- **Type**: SQLite with S3 backup
- **Backup Schedule**: Every 6 hours via ECS scheduled task
- **Retention**: 90 days in S3, then moved to Glacier
- **Migration**: Automatic via Alembic on deployment

### Storage Configuration
- **Drawings**: S3 Standard, lifecycle to Standard-IA after 30 days
- **Models**: S3 Standard with versioning enabled
- **Backups**: S3 Standard, lifecycle to Glacier after 30 days
- **Static Assets**: Served via CloudFront with 7-day cache TTL

## Security Configuration

### Network Security
- **VPC**: Isolated network (10.0.0.0/16)
- **Private Subnets**: Application tier (10.0.11.0/24, 10.0.12.0/24)
- **Public Subnets**: Load balancer tier (10.0.1.0/24, 10.0.2.0/24)
- **NAT Gateway**: Outbound internet access for private subnets
- **Security Groups**: Minimal port exposure (80, 443, 8000)

### IAM Security
- **ECS Execution Role**: ECR and Secrets Manager access
- **ECS Task Role**: S3 and CloudWatch access with least privilege
- **Resource-based Policies**: S3 bucket policies for cross-service access

### Data Security
- **Encryption at Rest**: S3 server-side encryption (AES-256)
- **Encryption in Transit**: HTTPS/TLS 1.2+ enforced
- **Secrets Management**: AWS Secrets Manager for sensitive data
- **Access Logging**: CloudTrail and VPC Flow Logs enabled

### Authentication Security
- **Admin Access**: Password-based authentication via Secrets Manager
- **Session Management**: Secure session cookies with timeout
- **Public Endpoints**: Demo, upload, documentation (no auth required)
- **Protected Endpoints**: Dashboard, configuration (auth required)

## Monitoring and Alerting

### CloudWatch Metrics
- **ECS Metrics**: CPU, memory utilization
- **ALB Metrics**: Request count, response time, error rate
- **Custom Metrics**: Analysis count, processing time
- **S3 Metrics**: Storage usage, request metrics

### Alarms and Alerts
- **High CPU**: >80% for 10 minutes
- **High Memory**: >80% for 10 minutes
- **High Response Time**: >5 seconds for 10 minutes
- **Cost Alerts**: 80% and 100% of $40 monthly budget

### Log Management
- **Application Logs**: CloudWatch Logs with 30-day retention
- **ECS Logs**: Container stdout/stderr
- **ALB Logs**: Access logs to S3 (optional)
- **VPC Flow Logs**: Network traffic analysis (optional)

### Dashboards
CloudWatch dashboard includes:
- ECS service health and performance
- Application Load Balancer metrics
- S3 storage utilization
- Cost and billing metrics
- Custom application metrics

## Cost Optimization

### Resource Sizing
- **ECS Fargate**: 0.25 vCPU, 0.5 GB RAM (minimal viable)
- **Single Task**: No auto-scaling for demo workloads
- **ALB**: Application Load Balancer (required for ECS Fargate)
- **NAT Gateway**: Single AZ deployment

### Storage Optimization
- **S3 Lifecycle Policies**: Automatic transition to cheaper storage classes
- **CloudFront Caching**: Reduce origin requests and data transfer costs
- **Log Retention**: 30-day retention to minimize storage costs

### Cost Monitoring
- **Budget Alerts**: $40 monthly threshold with email notifications
- **Cost Allocation Tags**: Environment and project tagging
- **Resource Optimization**: Regular review of unused resources

### Estimated Monthly Costs (USD)
| Service | Configuration | Estimated Cost |
|---------|---------------|----------------|
| ECS Fargate | 0.25 vCPU, 0.5 GB | $10-15 |
| Application Load Balancer | Standard | $16 |
| NAT Gateway | Single AZ | $32 |
| S3 Storage | Demo usage | $2-5 |
| CloudFront | Free tier | $0-3 |
| Route 53 | Hosted zone | $1 |
| CloudWatch | Basic monitoring | $1-2 |
| **Total** | | **$62-74** |

**Cost Reduction Options:**
- Remove NAT Gateway if outbound internet not needed (-$32/month)
- Use EC2 instead of Fargate (-$5-10/month, +management overhead)
- Disable ALB and use CloudFront only (-$16/month, -health checks)

## Backup and Recovery

### Database Backup
- **Automated Backup**: SQLite database backed up to S3 every 6 hours
- **Backup Retention**: 90 days in Standard, then Glacier
- **Point-in-Time Recovery**: Timestamped backups for specific recovery points

### Application Data Backup
- **S3 Cross-Region Replication**: Optional for critical data
- **Versioning**: Enabled on all S3 buckets
- **Lifecycle Management**: Automatic cleanup of old versions

### Disaster Recovery
- **Infrastructure**: CloudFormation templates for rapid rebuild
- **Application**: Docker images in ECR for quick redeployment
- **Data**: S3 backup restoration procedures
- **DNS**: Route 53 for quick failover (if using custom domain)

### Recovery Procedures
```bash
# Restore database from backup
aws s3 cp s3://children-drawing-production-backups-<account>/backup-YYYY-MM-DD-HH-MM.db ./restored.db

# Redeploy infrastructure
./deploy.sh

# Redeploy application with restored data
# (Upload restored database to new environment)
```

## Compliance and Governance

### Compliance Features
- **Data Encryption**: At rest and in transit
- **Access Controls**: IAM roles and policies
- **Audit Logging**: CloudTrail for API calls
- **Network Isolation**: VPC with private subnets
- **Backup Procedures**: Automated and tested

### Governance Policies
- **Tagging Strategy**: Environment, Project, Owner tags required
- **Resource Naming**: Consistent naming convention
- **Cost Management**: Budget alerts and regular reviews
- **Security Reviews**: Quarterly security assessment

### Documentation Requirements
- **Infrastructure Changes**: All changes via CloudFormation
- **Deployment Procedures**: Documented and version controlled
- **Incident Response**: Runbooks for common issues
- **Access Management**: Regular access reviews

## Next Steps

After successful deployment:

1. **Configure GitHub Actions**: Set up automated CI/CD pipeline
2. **SSL Certificate**: Obtain and configure production SSL certificate
3. **Domain Configuration**: Point DNS to CloudFront distribution
4. **Monitoring Setup**: Configure additional custom metrics
5. **Backup Testing**: Verify backup and restore procedures
6. **Security Hardening**: Implement additional security measures
7. **Performance Optimization**: Monitor and optimize based on usage patterns

## Support and Maintenance

### Regular Maintenance Tasks
- **Weekly**: Review CloudWatch alarms and logs
- **Monthly**: Check cost reports and optimize resources
- **Quarterly**: Security review and updates
- **Annually**: Disaster recovery testing

### Getting Help
- **AWS Support**: Use AWS Support Center for infrastructure issues
- **Application Issues**: Check CloudWatch logs and ECS service events
- **Cost Issues**: Use AWS Cost Explorer and billing alerts
- **Security Issues**: Follow AWS security best practices documentation