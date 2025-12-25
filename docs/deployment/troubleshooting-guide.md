# AWS Production Deployment Troubleshooting Guide

## Overview

This guide provides solutions for common issues encountered during AWS production deployment and operation of the Children's Drawing Anomaly Detection System.

## Pre-Deployment Issues

### AWS CLI Configuration Problems

#### Issue: AWS CLI not configured or invalid credentials
```bash
# Error: Unable to locate credentials
aws sts get-caller-identity
```

**Solution:**
```bash
# Configure AWS CLI
aws configure
# Enter: Access Key ID, Secret Access Key, Region (eu-west-1), Output format (json)

# Verify configuration
aws sts get-caller-identity
aws configure list
```

#### Issue: Insufficient permissions
```bash
# Error: User is not authorized to perform: cloudformation:CreateStack
```

**Solution:**
1. Ensure your AWS user has the required permissions:
   - CloudFormationFullAccess
   - ECSFullAccess
   - EC2FullAccess (for VPC management)
   - S3FullAccess
   - IAMFullAccess
   - Route53FullAccess (if using custom domain)

2. Or create a custom policy with minimal required permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:*",
        "ecs:*",
        "ec2:*",
        "s3:*",
        "iam:*",
        "route53:*",
        "cloudwatch:*",
        "logs:*",
        "secretsmanager:*",
        "elasticloadbalancing:*",
        "cloudfront:*"
      ],
      "Resource": "*"
    }
  ]
}
```

### CloudFormation Template Validation

#### Issue: Template validation errors
```bash
# Error: Template format error: YAML not well-formed
./deploy.sh validate
```

**Solution:**
```bash
# Check YAML syntax
yamllint infrastructure/main-infrastructure.yaml

# Validate specific template
aws cloudformation validate-template \
  --template-body file://main-infrastructure.yaml \
  --region eu-west-1
```

#### Issue: Parameter validation errors
```bash
# Error: Parameter validation failed
```

**Solution:**
1. Check parameter constraints in template
2. Ensure S3 bucket names are globally unique
3. Verify domain name format if provided
4. Check certificate ARN format if provided

## Deployment Issues

### CloudFormation Stack Failures

#### Issue: Stack creation failed - Resource already exists
```bash
# Error: S3 bucket already exists
```

**Solution:**
```bash
# Check existing resources
aws s3 ls | grep children-drawing

# Use different S3 bucket prefix
./deploy.sh
# Enter unique prefix when prompted

# Or delete existing resources if safe to do so
aws s3 rb s3://existing-bucket-name --force
```

#### Issue: Stack rollback due to resource creation failure
```bash
# Error: CREATE_FAILED - The specified VPC does not exist
```

**Solution:**
```bash
# Check stack events for detailed error
aws cloudformation describe-stack-events \
  --stack-name children-drawing-prod \
  --region eu-west-1 \
  --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`]'

# Common fixes:
# 1. Ensure region has enough availability zones
aws ec2 describe-availability-zones --region eu-west-1

# 2. Check service limits
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-F678F1CE \
  --region eu-west-1
```

#### Issue: IAM role creation failed
```bash
# Error: Cannot create role - already exists
```

**Solution:**
```bash
# Check existing roles
aws iam list-roles --query 'Roles[?contains(RoleName, `children-drawing`)]'

# Delete existing role if safe
aws iam delete-role --role-name existing-role-name

# Or use different stack name
aws cloudformation deploy \
  --stack-name children-drawing-prod-v2 \
  --template-file main-infrastructure.yaml
```

### ECS Service Issues

#### Issue: ECS service fails to start tasks
```bash
# Error: Service tasks keep stopping
```

**Solution:**
```bash
# Check ECS service events
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1 \
  --query 'services[0].events'

# Check task definition
aws ecs describe-task-definition \
  --task-definition children-drawing-prod-task \
  --region eu-west-1

# Common issues and fixes:
# 1. Image not found in ECR
aws ecr describe-images \
  --repository-name children-drawing-app \
  --region eu-west-1

# 2. Insufficient memory/CPU
# Update task definition with higher resources

# 3. Health check failures
# Check application logs
aws logs tail /ecs/children-drawing-prod --follow
```

#### Issue: Tasks fail health checks
```bash
# Error: Task failed ELB health checks
```

**Solution:**
```bash
# Check application health endpoint
curl -f http://<alb-dns>/health

# Check ECS task logs
aws logs tail /ecs/children-drawing-prod --follow

# Verify security group allows ALB to reach ECS tasks
aws ec2 describe-security-groups \
  --group-ids sg-xxxxxxxxx \
  --region eu-west-1

# Common fixes:
# 1. Ensure health check endpoint returns 200
# 2. Verify port 8000 is exposed in container
# 3. Check security group rules allow port 8000 from ALB
```

### Load Balancer Issues

#### Issue: ALB returns 503 Service Unavailable
```bash
# Error: No healthy targets
```

**Solution:**
```bash
# Check target group health
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:eu-west-1:account:targetgroup/children-drawing-prod-tg/xxxxx \
  --region eu-west-1

# Check ECS service status
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --region eu-west-1

# Force new deployment
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --force-new-deployment \
  --region eu-west-1
```

#### Issue: SSL certificate issues
```bash
# Error: SSL handshake failed
```

**Solution:**
```bash
# Verify certificate status
aws acm describe-certificate \
  --certificate-arn arn:aws:acm:eu-west-1:account:certificate/xxxxx \
  --region eu-west-1

# Check certificate validation
# Ensure DNS validation records are in place for domain

# Test SSL connection
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

## Runtime Issues

### Application Performance Issues

#### Issue: High response times
```bash
# Symptoms: ALB response time alarms firing
```

**Solution:**
```bash
# Check ECS task resource utilization
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=children-drawing-prod-service Name=ClusterName,Value=children-drawing-prod-cluster \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-01T23:59:59Z \
  --period 300 \
  --statistics Average \
  --region eu-west-1

# Scale up resources if needed
# Update task definition with more CPU/memory
aws ecs register-task-definition \
  --family children-drawing-prod-task \
  --cpu 512 \
  --memory 1024 \
  --region eu-west-1

# Increase desired count for horizontal scaling
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 2 \
  --region eu-west-1
```

#### Issue: Out of memory errors
```bash
# Error: Container killed due to memory limit
```

**Solution:**
```bash
# Check memory utilization
aws logs filter-log-events \
  --log-group-name /ecs/children-drawing-prod \
  --filter-pattern "OutOfMemoryError" \
  --region eu-west-1

# Increase memory allocation
# Update task definition with more memory
# Consider optimizing application memory usage
```

### Storage Issues

#### Issue: S3 access denied errors
```bash
# Error: Access Denied when uploading to S3
```

**Solution:**
```bash
# Check ECS task role permissions
aws iam get-role-policy \
  --role-name children-drawing-prod-ecs-task-role \
  --policy-name S3Access

# Verify S3 bucket policy
aws s3api get-bucket-policy \
  --bucket children-drawing-production-drawings-account

# Test S3 access from ECS task
aws ecs execute-command \
  --cluster children-drawing-prod-cluster \
  --task task-id \
  --container app \
  --command "aws s3 ls s3://bucket-name" \
  --interactive
```

#### Issue: Database backup failures
```bash
# Error: SQLite backup to S3 failed
```

**Solution:**
```bash
# Check backup service logs
aws logs filter-log-events \
  --log-group-name /ecs/children-drawing-prod \
  --filter-pattern "backup" \
  --region eu-west-1

# Verify S3 backup bucket permissions
aws s3api head-bucket \
  --bucket children-drawing-production-backups-account

# Check database URL format in environment variables
# The backup service supports multiple SQLite URL formats:
# - sqlite:///absolute/path/to/database.db (recommended)
# - sqlite://relative/path/to/database.db
# - sqlite://:memory: (in-memory - limited backup support)

# Verify DATABASE_URL environment variable
aws ecs describe-task-definition \
  --task-definition children-drawing-prod-task \
  --query 'taskDefinition.containerDefinitions[0].environment[?name==`DATABASE_URL`]'

# Manual backup test
aws s3 cp /path/to/database.db s3://backup-bucket/test-backup.db
```

### Networking Issues

#### Issue: CloudFront distribution not serving content
```bash
# Error: CloudFront returns errors or stale content
```

**Solution:**
```bash
# Check CloudFront distribution status
aws cloudfront get-distribution \
  --id DISTRIBUTION_ID \
  --region eu-west-1

# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id DISTRIBUTION_ID \
  --paths "/*" \
  --region eu-west-1

# Check origin health
curl -I http://alb-dns-name/health
```

#### Issue: DNS resolution problems
```bash
# Error: Domain not resolving to CloudFront
```

**Solution:**
```bash
# Check Route 53 records
aws route53 list-resource-record-sets \
  --hosted-zone-id ZONE_ID

# Verify DNS propagation
dig your-domain.com
nslookup your-domain.com

# Check CloudFront distribution domain
aws cloudfront get-distribution \
  --id DISTRIBUTION_ID \
  --query 'Distribution.DomainName'
```

## Monitoring and Alerting Issues

### CloudWatch Issues

#### Issue: Metrics not appearing in CloudWatch
```bash
# Error: Custom metrics not visible
```

**Solution:**
```bash
# Check ECS task role permissions for CloudWatch
aws iam get-role-policy \
  --role-name children-drawing-prod-ecs-task-role \
  --policy-name CloudWatchMetrics

# Verify metric namespace
aws cloudwatch list-metrics \
  --namespace "ChildrenDrawing/Application" \
  --region eu-west-1

# Test metric publishing
aws cloudwatch put-metric-data \
  --namespace "ChildrenDrawing/Application" \
  --metric-data MetricName=TestMetric,Value=1 \
  --region eu-west-1
```

#### Issue: Alarms not triggering
```bash
# Error: No notifications received despite threshold breach
```

**Solution:**
```bash
# Check alarm configuration
aws cloudwatch describe-alarms \
  --alarm-names children-drawing-prod-ecs-cpu-high \
  --region eu-west-1

# Verify SNS topic subscription
aws sns list-subscriptions-by-topic \
  --topic-arn arn:aws:sns:eu-west-1:account:children-drawing-prod-cost-alerts

# Test SNS notification
aws sns publish \
  --topic-arn arn:aws:sns:eu-west-1:account:children-drawing-prod-cost-alerts \
  --message "Test notification" \
  --region eu-west-1
```

### Log Management Issues

#### Issue: Logs not appearing in CloudWatch Logs
```bash
# Error: Application logs missing
```

**Solution:**
```bash
# Check log group exists
aws logs describe-log-groups \
  --log-group-name-prefix /ecs/children-drawing-prod \
  --region eu-west-1

# Verify ECS task execution role permissions
aws iam get-role-policy \
  --role-name children-drawing-prod-ecs-execution-role \
  --policy-name CloudWatchLogs

# Check ECS task definition log configuration
aws ecs describe-task-definition \
  --task-definition children-drawing-prod-task \
  --query 'taskDefinition.containerDefinitions[0].logConfiguration'
```

## Cost Management Issues

### Unexpected High Costs

#### Issue: AWS bill higher than expected
```bash
# Symptoms: Cost alerts firing, unexpected charges
```

**Solution:**
```bash
# Check cost breakdown
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --region us-east-1

# Identify expensive resources
aws ce get-dimension-values \
  --dimension SERVICE \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --region us-east-1

# Common cost optimization actions:
# 1. Remove NAT Gateway if not needed
aws ec2 delete-nat-gateway --nat-gateway-id nat-xxxxxxxxx

# 2. Reduce ECS task size
# Update task definition with smaller CPU/memory

# 3. Enable S3 lifecycle policies
aws s3api put-bucket-lifecycle-configuration \
  --bucket bucket-name \
  --lifecycle-configuration file://lifecycle.json
```

#### Issue: Budget alerts not working
```bash
# Error: No cost alert emails received
```

**Solution:**
```bash
# Check budget configuration
aws budgets describe-budgets \
  --account-id $(aws sts get-caller-identity --query Account --output text)

# Verify SNS subscription
aws sns list-subscriptions \
  --region eu-west-1

# Confirm email subscription
# Check spam folder for AWS notification emails
```

## Security Issues

### Access Control Problems

#### Issue: Unable to access admin features
```bash
# Error: Authentication failed
```

**Solution:**
```bash
# Check Secrets Manager secret
aws secretsmanager get-secret-value \
  --secret-id children-drawing-prod-admin-password \
  --region eu-west-1

# Verify ECS task can access secret
aws iam simulate-principal-policy \
  --policy-source-arn arn:aws:iam::account:role/children-drawing-prod-ecs-task-role \
  --action-names secretsmanager:GetSecretValue \
  --resource-arns arn:aws:secretsmanager:eu-west-1:account:secret:children-drawing-prod-admin-password-xxxxx
```

#### Issue: S3 bucket access denied
```bash
# Error: Access denied for S3 operations
```

**Solution:**
```bash
# Check bucket policy
aws s3api get-bucket-policy \
  --bucket bucket-name

# Verify IAM role permissions
aws iam get-role-policy \
  --role-name children-drawing-prod-ecs-task-role \
  --policy-name S3Access

# Test access with specific role
aws sts assume-role \
  --role-arn arn:aws:iam::account:role/children-drawing-prod-ecs-task-role \
  --role-session-name test-session
```

## Recovery Procedures

### Complete Stack Recovery

#### Issue: Entire stack needs to be rebuilt
```bash
# Scenario: Stack corrupted or accidentally deleted
```

**Solution:**
```bash
# 1. Backup any remaining data
aws s3 sync s3://children-drawing-production-drawings-account ./backup-drawings/
aws s3 sync s3://children-drawing-production-backups-account ./backup-db/

# 2. Delete corrupted stack
aws cloudformation delete-stack \
  --stack-name children-drawing-prod \
  --region eu-west-1

# 3. Wait for deletion to complete
aws cloudformation wait stack-delete-complete \
  --stack-name children-drawing-prod \
  --region eu-west-1

# 4. Redeploy infrastructure
./deploy.sh

# 5. Restore data
aws s3 sync ./backup-drawings/ s3://new-drawings-bucket/
aws s3 sync ./backup-db/ s3://new-backups-bucket/

# 6. Redeploy application
docker build -t children-drawing-app -f ../Dockerfile.prod ..
# Push to ECR and update ECS service
```

### Database Recovery

#### Issue: Database corruption or data loss
```bash
# Scenario: Need to restore from backup
```

**Solution:**
```bash
# 1. List available backups
aws s3 ls s3://children-drawing-production-backups-account/

# 2. Download specific backup
aws s3 cp s3://children-drawing-production-backups-account/backup-2024-01-15-12-00.db ./restore.db

# 3. Stop ECS service
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 0 \
  --region eu-west-1

# 4. Replace database file (implementation specific)
# Upload restored database to application

# 5. Restart ECS service
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 1 \
  --region eu-west-1
```

## Emergency Contacts and Escalation

### Immediate Response (P0 - Service Down)
1. Check AWS Service Health Dashboard
2. Review CloudWatch alarms and logs
3. Verify ECS service status
4. Check ALB target health
5. Escalate to AWS Support if infrastructure issue

### Performance Issues (P1 - Degraded Service)
1. Monitor CloudWatch metrics
2. Check ECS task resource utilization
3. Review application logs
4. Consider scaling up resources
5. Implement temporary fixes

### Security Issues (P0 - Security Breach)
1. Immediately rotate all credentials
2. Review CloudTrail logs
3. Check for unauthorized access
4. Implement security patches
5. Contact AWS Security team if needed

### Cost Issues (P2 - Budget Exceeded)
1. Review AWS Cost Explorer
2. Identify expensive resources
3. Implement immediate cost controls
4. Schedule cost optimization review
5. Update budget alerts

## Preventive Measures

### Regular Health Checks
```bash
# Weekly health check script
#!/bin/bash
echo "=== Weekly Health Check ==="

# Check ECS service health
aws ecs describe-services --cluster children-drawing-prod-cluster --services children-drawing-prod-service --region eu-west-1

# Check ALB target health
aws elbv2 describe-target-health --target-group-arn $(aws elbv2 describe-target-groups --names children-drawing-prod-tg --query 'TargetGroups[0].TargetGroupArn' --output text) --region eu-west-1

# Check CloudWatch alarms
aws cloudwatch describe-alarms --state-value ALARM --region eu-west-1

# Check recent costs
aws ce get-cost-and-usage --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) --granularity DAILY --metrics BlendedCost --region us-east-1

echo "=== Health Check Complete ==="
```

### Automated Monitoring
- Set up comprehensive CloudWatch alarms
- Configure SNS notifications for all critical alerts
- Implement automated backup verification
- Schedule regular security scans
- Monitor cost trends and set up budget alerts

### Documentation Maintenance
- Keep runbooks updated with latest procedures
- Document all configuration changes
- Maintain incident response procedures
- Regular review of troubleshooting guides
- Update contact information and escalation procedures