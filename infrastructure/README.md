# AWS Infrastructure as Code

This directory contains CloudFormation templates for deploying the Children's Drawing Anomaly Detection System to AWS.

## 🚀 Current Deployment Status: SUCCESSFUL ✅

**Live Application**: https://d2e6rjfv7d2rgs.cloudfront.net/demo/  
**API Documentation**: https://d2e6rjfv7d2rgs.cloudfront.net/docs  
**Last Updated**: December 23, 2025

## Architecture Overview

The infrastructure consists of:
- **ECS Fargate Cluster**: Serverless container hosting (1/1 tasks running)
- **S3 Buckets**: Storage for drawings, models, and static assets
- **CloudFront Distribution**: Content delivery network (E34MC6W2KLQE7H)
- **Application Load Balancer**: Traffic distribution and health checks
- **VPC & Networking**: Secure network configuration
- **IAM Roles**: Least-privilege access control
- **ECR Repository**: Container image storage

## Quick Start (Automated Deployment)

Use the deployment script for guided setup:
```bash
./deploy.sh
```

The script will:
- Validate AWS CLI configuration
- Check CloudFormation templates
- Guide you through parameter selection
- Deploy infrastructure components
- Provide deployment status and URLs

## Manual Deployment

1. **Prerequisites**:
   - AWS CLI configured with appropriate permissions
   - Docker installed for building application images
   - ECR repository access for container images

2. **Deploy Infrastructure**:
   ```bash
   # Deploy main infrastructure stack
   aws cloudformation deploy \
     --template-file main-infrastructure.yaml \
     --stack-name children-drawing-prod \
     --parameter-overrides \
       Environment=production \
       S3BucketPrefix=children-drawing \
     --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
     --region eu-west-1

   # Deploy networking stack (if using separate stack)
   aws cloudformation deploy \
     --template-file networking.yaml \
     --stack-name children-drawing-network \
     --parameter-overrides Environment=production \
     --region eu-west-1
   ```

3. **Deploy Application**:
   ```bash
   # Build and push Docker image
   docker build -f ../Dockerfile.prod -t children-drawing-app:latest ..
   docker tag children-drawing-app:latest 921400262514.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest
   docker push 921400262514.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

   # Update ECS service
   aws ecs update-service \
     --cluster children-drawing-prod-cluster \
     --service children-drawing-prod-service \
     --task-definition children-drawing-prod-task:12 \
     --region eu-west-1
   ```

## Troubleshooting Resolved Issues

### ✅ Container Startup Failures
**Issue**: Missing dependencies causing application crashes  
**Solution**: Ensure all requirements are in `requirements.txt` and rebuild image  
**Prevention**: Test Docker image locally before pushing to ECR

### ✅ Wrong Container Image
**Issue**: ECS running basic nginx instead of application  
**Solution**: Update task definition with correct ECR image URI  
**Prevention**: Verify task definition image URI matches pushed image

### ⚠️ Rate Limiting
**Issue**: Aggressive rate limiting blocking root path access  
**Workaround**: Use `/demo/` endpoint for full application access  
**Status**: Functional but may need adjustment for production use

## Cost Estimation

**Monthly costs (actual deployment)**:
- ECS Fargate (0.5 vCPU, 2 GB): ~$15-20
- Application Load Balancer: ~$16
- S3 Storage: ~$2-5
- CloudFront: ~$1-5 (low traffic)
- CloudWatch: Free tier covers basic monitoring
- **Total: ~$34-46/month**

**Cost optimization tips**:
- Monitor CloudWatch metrics for right-sizing
- Use S3 lifecycle policies for old data
- Consider scheduled scaling for predictable workloads

## Security Features

- Private subnets for application tier
- Security groups with minimal port exposure
- IAM roles with least-privilege permissions
- Encryption for S3 storage and data in transit
- VPC endpoints for AWS service access
- Rate limiting for API protection (configurable)

## Monitoring & Health Checks

- CloudWatch logs collection from ECS containers
- Custom metrics for application monitoring
- SNS alerts for errors and cost thresholds
- CloudWatch dashboards for system visibility
- Application health checks via `/health` endpoint
- ECS service health monitoring with auto-recovery

## Current Infrastructure Status

**Active Resources**:
- ECS Cluster: `children-drawing-prod-cluster`
- ECS Service: `children-drawing-prod-service` (1/1 tasks)
- Task Definition: `children-drawing-prod-task:12`
- Load Balancer: `children-drawing-prod-alb`
- CloudFront: `E34MC6W2KLQE7H`
- ECR Repository: `921400262514.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app`

**Health Status**: ✅ All systems operational  
**Last Verified**: December 23, 2025