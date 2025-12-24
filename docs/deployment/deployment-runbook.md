# AWS Production Deployment Runbook

## Overview

This runbook provides step-by-step procedures for deploying, operating, and maintaining the Children's Drawing Anomaly Detection System in AWS production environment.

## Quick Reference

### Emergency Contacts
- **Primary On-Call**: [Your contact information]
- **AWS Support**: [Your AWS support plan details]
- **Security Team**: [Security team contact]

### Critical URLs
- **Production Application**: https://your-domain.com
- **AWS Console**: https://console.aws.amazon.com
- **CloudWatch Dashboard**: [Dashboard URL]
- **GitHub Repository**: [Repository URL]

### Key AWS Resources
- **Stack Name**: children-drawing-prod
- **ECS Cluster**: children-drawing-prod-cluster
- **ECS Service**: children-drawing-prod-service
- **Load Balancer**: children-drawing-prod-alb
- **S3 Buckets**: children-drawing-production-*
- **CloudFront Distribution**: [Distribution ID]

## Pre-Deployment Checklist

### Prerequisites Verification
```bash
# 1. Verify AWS CLI configuration
aws sts get-caller-identity
aws configure list

# 2. Check required permissions
aws iam simulate-principal-policy \
  --policy-source-arn $(aws sts get-caller-identity --query Arn --output text) \
  --action-names cloudformation:CreateStack \
  --resource-arns "*"

# 3. Validate CloudFormation templates
cd infrastructure
./deploy.sh validate

# 4. Check domain and certificate (if applicable)
aws acm list-certificates --region eu-west-1
aws route53 list-hosted-zones
```

### Environment Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd children-drawing-anomaly-detection

# 2. Set up environment variables
cp .env.example .env.production
# Edit .env.production with production values

# 3. Verify Docker setup
docker --version
docker-compose --version
```

## Deployment Procedures

### Initial Infrastructure Deployment

#### Step 1: Deploy Core Infrastructure
```bash
cd infrastructure

# Interactive deployment
./deploy.sh

# Or automated deployment with parameters
aws cloudformation deploy \
  --template-file main-infrastructure.yaml \
  --stack-name children-drawing-prod \
  --parameter-overrides \
    Environment=production \
    DomainName=your-domain.com \
    S3BucketPrefix=children-drawing \
    CostAlertEmail=admin@your-domain.com \
  --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
  --region eu-west-1 \
  --tags Environment=production Project=ChildrenDrawingAnomalyDetection
```

#### Step 2: Verify Infrastructure Deployment
```bash
# Check stack status
aws cloudformation describe-stacks \
  --stack-name children-drawing-prod \
  --region eu-west-1 \
  --query 'Stacks[0].StackStatus'

# Get stack outputs
aws cloudformation describe-stacks \
  --stack-name children-drawing-prod \
  --region eu-west-1 \
  --query 'Stacks[0].Outputs' \
  --output table

# Verify ECS cluster
aws ecs describe-clusters \
  --clusters children-drawing-prod-cluster \
  --region eu-west-1

# Check S3 buckets
aws s3 ls | grep children-drawing
```

#### Step 3: Application Deployment
```bash
# 1. Create ECR repository (if not exists)
aws ecr create-repository \
  --repository-name children-drawing-app \
  --region eu-west-1

# 2. Build and tag Docker image
docker build -t children-drawing-app -f Dockerfile.prod .
docker tag children-drawing-app:latest \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# 3. Login to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com

# 4. Push image to ECR
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest

# 5. Update ECS task definition
aws ecs register-task-definition \
  --family children-drawing-prod-task \
  --task-role-arn $(aws cloudformation describe-stacks --stack-name children-drawing-prod --query 'Stacks[0].Outputs[?OutputKey==`ECSTaskRoleArn`].OutputValue' --output text --region eu-west-1) \
  --execution-role-arn $(aws cloudformation describe-stacks --stack-name children-drawing-prod --query 'Stacks[0].Outputs[?OutputKey==`ECSTaskExecutionRoleArn`].OutputValue' --output text --region eu-west-1) \
  --network-mode awsvpc \
  --requires-compatibilities FARGATE \
  --cpu 256 \
  --memory 512 \
  --container-definitions '[
    {
      "name": "app",
      "image": "'$(aws sts get-caller-identity --query Account --output text)'.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/children-drawing-prod",
          "awslogs-region": "eu-west-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]' \
  --region eu-west-1

# 6. Update ECS service
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --task-definition children-drawing-prod-task \
  --region eu-west-1
```

#### Step 4: Post-Deployment Verification
```bash
# 1. Check ECS service status
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1 \
  --query 'services[0].[serviceName,status,runningCount,desiredCount]'

# 2. Check ALB target health
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups --names children-drawing-prod-tg --query 'TargetGroups[0].TargetGroupArn' --output text --region eu-west-1) \
  --region eu-west-1

# 3. Test application endpoints
ALB_DNS=$(aws elbv2 describe-load-balancers --names children-drawing-prod-alb --query 'LoadBalancers[0].DNSName' --output text --region eu-west-1)
curl -f http://$ALB_DNS/health

# 4. Test CloudFront distribution
CLOUDFRONT_DNS=$(aws cloudformation describe-stacks --stack-name children-drawing-prod --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionDNS`].OutputValue' --output text --region eu-west-1)
curl -f https://$CLOUDFRONT_DNS/health
```

### Application Updates

#### Rolling Update Procedure
```bash
# 1. Build new image with version tag
VERSION=$(date +%Y%m%d-%H%M%S)
docker build -t children-drawing-app:$VERSION -f Dockerfile.prod .
docker tag children-drawing-app:$VERSION \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:$VERSION

# 2. Push new image
docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app:$VERSION

# 3. Update task definition with new image
# (Use the same register-task-definition command with new image URI)

# 4. Update service (ECS will perform rolling update)
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --task-definition children-drawing-prod-task:LATEST \
  --region eu-west-1

# 5. Monitor deployment
aws ecs wait services-stable \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1

# 6. Verify new deployment
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1 \
  --query 'services[0].deployments'
```

#### Rollback Procedure
```bash
# 1. List previous task definitions
aws ecs list-task-definitions \
  --family-prefix children-drawing-prod-task \
  --status ACTIVE \
  --sort DESC \
  --region eu-west-1

# 2. Rollback to previous version
PREVIOUS_TASK_DEF=$(aws ecs list-task-definitions --family-prefix children-drawing-prod-task --status ACTIVE --sort DESC --region eu-west-1 --query 'taskDefinitionArns[1]' --output text)

aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --task-definition $PREVIOUS_TASK_DEF \
  --region eu-west-1

# 3. Monitor rollback
aws ecs wait services-stable \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1
```

## Operational Procedures

### Daily Operations

#### Morning Health Check
```bash
#!/bin/bash
# daily-health-check.sh

echo "=== Daily Health Check - $(date) ==="

# 1. Check ECS service health
echo "1. ECS Service Status:"
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1 \
  --query 'services[0].[serviceName,status,runningCount,desiredCount,pendingCount]' \
  --output table

# 2. Check ALB target health
echo "2. Load Balancer Target Health:"
aws elbv2 describe-target-health \
  --target-group-arn $(aws elbv2 describe-target-groups --names children-drawing-prod-tg --query 'TargetGroups[0].TargetGroupArn' --output text --region eu-west-1) \
  --region eu-west-1 \
  --query 'TargetHealthDescriptions[*].[Target.Id,TargetHealth.State,TargetHealth.Description]' \
  --output table

# 3. Check CloudWatch alarms
echo "3. Active Alarms:"
aws cloudwatch describe-alarms \
  --state-value ALARM \
  --region eu-west-1 \
  --query 'MetricAlarms[?contains(AlarmName, `children-drawing`)].[AlarmName,StateReason]' \
  --output table

# 4. Check application health
echo "4. Application Health:"
ALB_DNS=$(aws elbv2 describe-load-balancers --names children-drawing-prod-alb --query 'LoadBalancers[0].DNSName' --output text --region eu-west-1)
curl -s -o /dev/null -w "HTTP Status: %{http_code}, Response Time: %{time_total}s\n" http://$ALB_DNS/health

# 5. Check recent errors in logs
echo "5. Recent Errors (last 1 hour):"
aws logs filter-log-events \
  --log-group-name /ecs/children-drawing-prod \
  --start-time $(date -d '1 hour ago' +%s)000 \
  --filter-pattern "ERROR" \
  --region eu-west-1 \
  --query 'events[*].[logStreamName,message]' \
  --output table | head -20

# 6. Check yesterday's costs
echo "6. Yesterday's Costs:"
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)
aws ce get-cost-and-usage \
  --time-period Start=$YESTERDAY,End=$TODAY \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1 \
  --query 'ResultsByTime[0].Total.BlendedCost.Amount' \
  --output text

echo "=== Health Check Complete ==="
```

#### Weekly Maintenance
```bash
#!/bin/bash
# weekly-maintenance.sh

echo "=== Weekly Maintenance - $(date) ==="

# 1. Clean up old Docker images in ECR
echo "1. Cleaning up old ECR images..."
aws ecr list-images \
  --repository-name children-drawing-app \
  --filter tagStatus=UNTAGGED \
  --region eu-west-1 \
  --query 'imageIds[?imageDigest!=null]' | \
aws ecr batch-delete-image \
  --repository-name children-drawing-app \
  --image-ids file:///dev/stdin \
  --region eu-west-1

# 2. Review and clean up old CloudWatch logs
echo "2. Reviewing log retention..."
aws logs describe-log-groups \
  --log-group-name-prefix /ecs/children-drawing \
  --region eu-west-1 \
  --query 'logGroups[*].[logGroupName,retentionInDays,storedBytes]' \
  --output table

# 3. Check S3 storage usage and lifecycle policies
echo "3. S3 Storage Analysis:"
for bucket in $(aws s3 ls | grep children-drawing | awk '{print $3}'); do
  echo "Bucket: $bucket"
  aws s3 ls s3://$bucket --recursive --human-readable --summarize | tail -2
  aws s3api get-bucket-lifecycle-configuration --bucket $bucket 2>/dev/null || echo "  No lifecycle policy"
done

# 4. Review security groups and NACLs
echo "4. Security Review:"
aws ec2 describe-security-groups \
  --filters Name=group-name,Values=*children-drawing* \
  --region eu-west-1 \
  --query 'SecurityGroups[*].[GroupName,GroupId,IpPermissions[?IpProtocol==`-1`]]' \
  --output table

# 5. Check for unused resources
echo "5. Unused Resources Check:"
# Unattached EBS volumes
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --region eu-west-1 \
  --query 'Volumes[*].[VolumeId,Size,CreateTime]' \
  --output table

# Unattached Elastic IPs
aws ec2 describe-addresses \
  --region eu-west-1 \
  --query 'Addresses[?AssociationId==null].[PublicIp,AllocationId]' \
  --output table

echo "=== Weekly Maintenance Complete ==="
```

### Monitoring and Alerting

#### CloudWatch Dashboard Setup
```bash
# Create CloudWatch dashboard
aws cloudwatch put-dashboard \
  --dashboard-name "ChildrenDrawing-Production" \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "x": 0,
        "y": 0,
        "width": 12,
        "height": 6,
        "properties": {
          "metrics": [
            ["AWS/ECS", "CPUUtilization", "ServiceName", "children-drawing-prod-service", "ClusterName", "children-drawing-prod-cluster"],
            [".", "MemoryUtilization", ".", ".", ".", "."]
          ],
          "view": "timeSeries",
          "stacked": false,
          "region": "eu-west-1",
          "title": "ECS Resource Utilization",
          "period": 300
        }
      },
      {
        "type": "metric",
        "x": 12,
        "y": 0,
        "width": 12,
        "height": 6,
        "properties": {
          "metrics": [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", "'$(aws elbv2 describe-load-balancers --names children-drawing-prod-alb --query 'LoadBalancers[0].LoadBalancerFullName' --output text --region eu-west-1)'"],
            [".", "TargetResponseTime", ".", "."],
            [".", "HTTPCode_Target_2XX_Count", ".", "."],
            [".", "HTTPCode_Target_4XX_Count", ".", "."],
            [".", "HTTPCode_Target_5XX_Count", ".", "."]
          ],
          "view": "timeSeries",
          "stacked": false,
          "region": "eu-west-1",
          "title": "Application Load Balancer Metrics",
          "period": 300
        }
      }
    ]
  }' \
  --region eu-west-1
```

#### Alert Configuration
```bash
# Create SNS topic for alerts
aws sns create-topic \
  --name children-drawing-prod-alerts \
  --region eu-west-1

# Subscribe email to topic
aws sns subscribe \
  --topic-arn arn:aws:sns:eu-west-1:$(aws sts get-caller-identity --query Account --output text):children-drawing-prod-alerts \
  --protocol email \
  --notification-endpoint admin@your-domain.com \
  --region eu-west-1

# Create CloudWatch alarms
aws cloudwatch put-metric-alarm \
  --alarm-name "children-drawing-prod-high-cpu" \
  --alarm-description "ECS service high CPU utilization" \
  --metric-name CPUUtilization \
  --namespace AWS/ECS \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:eu-west-1:$(aws sts get-caller-identity --query Account --output text):children-drawing-prod-alerts \
  --dimensions Name=ServiceName,Value=children-drawing-prod-service Name=ClusterName,Value=children-drawing-prod-cluster \
  --region eu-west-1
```

### Backup and Recovery

#### Database Backup
```bash
#!/bin/bash
# backup-database.sh

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BACKUP_BUCKET="children-drawing-production-backups-$(aws sts get-caller-identity --query Account --output text)"

echo "Starting database backup - $TIMESTAMP"

# 1. Create database backup from ECS task
TASK_ARN=$(aws ecs list-tasks --cluster children-drawing-prod-cluster --service-name children-drawing-prod-service --query 'taskArns[0]' --output text --region eu-west-1)

# Execute backup command in running container
aws ecs execute-command \
  --cluster children-drawing-prod-cluster \
  --task $TASK_ARN \
  --container app \
  --command "sqlite3 /app/drawings.db '.backup /tmp/backup-$TIMESTAMP.db'" \
  --interactive \
  --region eu-west-1

# 2. Copy backup to S3
aws ecs execute-command \
  --cluster children-drawing-prod-cluster \
  --task $TASK_ARN \
  --container app \
  --command "aws s3 cp /tmp/backup-$TIMESTAMP.db s3://$BACKUP_BUCKET/database/backup-$TIMESTAMP.db" \
  --interactive \
  --region eu-west-1

# 3. Verify backup
aws s3 ls s3://$BACKUP_BUCKET/database/backup-$TIMESTAMP.db

echo "Database backup completed: backup-$TIMESTAMP.db"
```

#### Disaster Recovery
```bash
#!/bin/bash
# disaster-recovery.sh

echo "=== Disaster Recovery Procedure ==="

# 1. Assess the situation
echo "1. Assessing current infrastructure state..."
aws cloudformation describe-stacks \
  --stack-name children-drawing-prod \
  --region eu-west-1 \
  --query 'Stacks[0].StackStatus'

# 2. If stack is corrupted, redeploy infrastructure
read -p "Redeploy infrastructure? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Redeploying infrastructure..."
  cd infrastructure
  ./deploy.sh
fi

# 3. Restore application data
echo "3. Restoring application data..."
BACKUP_BUCKET="children-drawing-production-backups-$(aws sts get-caller-identity --query Account --output text)"

# List available backups
echo "Available backups:"
aws s3 ls s3://$BACKUP_BUCKET/database/ --recursive

read -p "Enter backup filename to restore: " BACKUP_FILE

# Download and restore backup
aws s3 cp s3://$BACKUP_BUCKET/database/$BACKUP_FILE ./restored.db

# 4. Redeploy application with restored data
echo "4. Redeploying application..."
# (Follow application deployment procedure)

echo "=== Disaster Recovery Complete ==="
```

## Troubleshooting Procedures

### Common Issues and Solutions

#### Issue: ECS Service Not Starting
```bash
# Diagnosis
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1 \
  --query 'services[0].events[0:5]'

# Check task definition
aws ecs describe-task-definition \
  --task-definition children-drawing-prod-task \
  --region eu-west-1

# Check logs
aws logs tail /ecs/children-drawing-prod --follow --region eu-west-1

# Solutions:
# 1. Check image exists in ECR
# 2. Verify IAM roles have correct permissions
# 3. Check security group allows traffic
# 4. Verify resource allocation (CPU/memory)
```

#### Issue: High Response Times
```bash
# Diagnosis
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=$(aws elbv2 describe-load-balancers --names children-drawing-prod-alb --query 'LoadBalancers[0].LoadBalancerFullName' --output text --region eu-west-1) \
  --start-time $(date -d '1 hour ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 300 \
  --statistics Average,Maximum \
  --region eu-west-1

# Solutions:
# 1. Scale up ECS service
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 2 \
  --region eu-west-1

# 2. Increase task resources
# Update task definition with more CPU/memory

# 3. Check application performance
aws logs filter-log-events \
  --log-group-name /ecs/children-drawing-prod \
  --filter-pattern "slow" \
  --region eu-west-1
```

#### Issue: Cost Overrun
```bash
# Diagnosis
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1

# Immediate cost reduction actions:
# 1. Scale down ECS service
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 0 \
  --region eu-west-1

# 2. Delete NAT Gateway (if not needed)
NAT_GATEWAY_ID=$(aws ec2 describe-nat-gateways --filter Name=tag:Name,Values=children-drawing-prod-nat-1 --query 'NatGateways[0].NatGatewayId' --output text --region eu-west-1)
aws ec2 delete-nat-gateway --nat-gateway-id $NAT_GATEWAY_ID --region eu-west-1

# 3. Review S3 storage classes
aws s3api list-objects-v2 \
  --bucket children-drawing-production-drawings-$(aws sts get-caller-identity --query Account --output text) \
  --query 'Contents[?StorageClass!=`STANDARD_IA`].[Key,StorageClass,Size]' \
  --output table
```

## Security Procedures

### Security Incident Response
```bash
#!/bin/bash
# security-incident-response.sh

echo "🚨 SECURITY INCIDENT RESPONSE 🚨"

# 1. Immediate containment
echo "1. Implementing immediate containment..."
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 0 \
  --region eu-west-1

# 2. Preserve evidence
echo "2. Preserving evidence..."
INCIDENT_ID="incident-$(date +%Y%m%d-%H%M%S)"
mkdir -p /tmp/$INCIDENT_ID

# Collect CloudTrail logs
aws logs create-export-task \
  --log-group-name /aws/cloudtrail/children-drawing-prod \
  --from $(date -d '24 hours ago' +%s)000 \
  --to $(date +%s)000 \
  --destination s3://children-drawing-production-backups-$(aws sts get-caller-identity --query Account --output text) \
  --destination-prefix security-incident/$INCIDENT_ID/ \
  --region eu-west-1

# 3. Rotate credentials
echo "3. Rotating credentials..."
aws secretsmanager rotate-secret \
  --secret-id children-drawing-prod-admin-password \
  --region eu-west-1

# 4. Send alert
aws sns publish \
  --topic-arn arn:aws:sns:eu-west-1:$(aws sts get-caller-identity --query Account --output text):children-drawing-prod-alerts \
  --message "Security incident detected. Incident ID: $INCIDENT_ID. Immediate containment measures activated." \
  --subject "SECURITY INCIDENT: $INCIDENT_ID" \
  --region eu-west-1

echo "Security incident response initiated. Incident ID: $INCIDENT_ID"
```

### Regular Security Audit
```bash
#!/bin/bash
# security-audit.sh

echo "=== Security Audit - $(date) ==="

# 1. Check IAM permissions
echo "1. IAM Security Review:"
aws iam list-roles \
  --query 'Roles[?contains(RoleName, `children-drawing`)].[RoleName,CreateDate,MaxSessionDuration]' \
  --output table

# 2. Check S3 bucket security
echo "2. S3 Security Review:"
for bucket in $(aws s3 ls | grep children-drawing | awk '{print $3}'); do
  echo "Bucket: $bucket"
  aws s3api get-bucket-encryption --bucket $bucket 2>/dev/null || echo "  ❌ No encryption"
  aws s3api get-public-access-block --bucket $bucket 2>/dev/null || echo "  ❌ No public access block"
  aws s3api get-bucket-policy --bucket $bucket 2>/dev/null || echo "  ✅ No bucket policy"
done

# 3. Check security groups
echo "3. Security Groups Review:"
aws ec2 describe-security-groups \
  --filters Name=group-name,Values=*children-drawing* \
  --query 'SecurityGroups[*].[GroupName,GroupId,IpPermissions[?IpProtocol==`-1` || (FromPort==`0` && ToPort==`65535`)]]' \
  --output table \
  --region eu-west-1

# 4. Check CloudTrail status
echo "4. CloudTrail Status:"
aws cloudtrail describe-trails \
  --query 'trailList[?contains(Name, `children-drawing`)].[Name,IsMultiRegionTrail,LogFileValidationEnabled,IncludeGlobalServiceEvents]' \
  --output table

# 5. Check for failed login attempts
echo "5. Recent Failed Login Attempts:"
aws logs filter-log-events \
  --log-group-name /ecs/children-drawing-prod \
  --start-time $(date -d '24 hours ago' +%s)000 \
  --filter-pattern "authentication_attempt" \
  --region eu-west-1 \
  --query 'events[?contains(message, `"success": false`)].[logStreamName,message]' \
  --output table | head -10

echo "=== Security Audit Complete ==="
```

## Maintenance Windows

### Planned Maintenance Procedure
```bash
#!/bin/bash
# planned-maintenance.sh

echo "=== Planned Maintenance Window ==="

# 1. Pre-maintenance checks
echo "1. Pre-maintenance health check..."
./daily-health-check.sh

# 2. Create maintenance backup
echo "2. Creating maintenance backup..."
./backup-database.sh

# 3. Scale down service (if needed)
echo "3. Scaling down for maintenance..."
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 0 \
  --region eu-west-1

# Wait for tasks to stop
aws ecs wait services-stable \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1

# 4. Perform maintenance tasks
echo "4. Performing maintenance tasks..."
# - Update task definitions
# - Apply security patches
# - Update configurations
# - Clean up resources

# 5. Scale service back up
echo "5. Scaling service back up..."
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 1 \
  --region eu-west-1

# Wait for service to stabilize
aws ecs wait services-stable \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1

# 6. Post-maintenance verification
echo "6. Post-maintenance verification..."
./daily-health-check.sh

echo "=== Maintenance Window Complete ==="
```

## Performance Optimization

### Performance Monitoring
```bash
#!/bin/bash
# performance-monitoring.sh

echo "=== Performance Monitoring Report ==="

# 1. ECS resource utilization
echo "1. ECS Resource Utilization (last 24 hours):"
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=children-drawing-prod-service Name=ClusterName,Value=children-drawing-prod-cluster \
  --start-time $(date -d '24 hours ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 3600 \
  --statistics Average,Maximum \
  --region eu-west-1 \
  --query 'Datapoints[*].[Timestamp,Average,Maximum]' \
  --output table

# 2. Application response times
echo "2. Application Response Times:"
aws cloudwatch get-metric-statistics \
  --namespace AWS/ApplicationELB \
  --metric-name TargetResponseTime \
  --dimensions Name=LoadBalancer,Value=$(aws elbv2 describe-load-balancers --names children-drawing-prod-alb --query 'LoadBalancers[0].LoadBalancerFullName' --output text --region eu-west-1) \
  --start-time $(date -d '24 hours ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 3600 \
  --statistics Average,Maximum \
  --region eu-west-1 \
  --query 'Datapoints[*].[Timestamp,Average,Maximum]' \
  --output table

# 3. CloudFront cache performance
echo "3. CloudFront Cache Performance:"
DISTRIBUTION_ID=$(aws cloudformation describe-stacks --stack-name children-drawing-prod --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionId`].OutputValue' --output text --region eu-west-1)
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=$DISTRIBUTION_ID \
  --start-time $(date -d '24 hours ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 3600 \
  --statistics Average \
  --region us-east-1 \
  --query 'Datapoints[*].[Timestamp,Average]' \
  --output table

echo "=== Performance Monitoring Complete ==="
```

## Documentation and Knowledge Management

### Runbook Updates
- Update this runbook after any significant changes to infrastructure or procedures
- Version control all runbook changes in Git
- Review and test procedures quarterly
- Maintain change log of all modifications

### Knowledge Transfer
- Ensure all team members are familiar with these procedures
- Conduct regular runbook walkthroughs
- Document lessons learned from incidents
- Maintain up-to-date contact information and access credentials

### Compliance and Audit
- Regular security audits and compliance checks
- Document all changes and maintenance activities
- Maintain audit trails for all operational activities
- Review and update procedures based on audit findings

This runbook provides comprehensive operational procedures for the Children's Drawing Anomaly Detection System. Keep it updated and accessible to all team members responsible for system operations.