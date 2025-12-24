# AWS Cost Monitoring and Optimization Guide

## Overview

This guide provides comprehensive cost monitoring, estimation, and optimization strategies for the Children's Drawing Anomaly Detection System deployed on AWS.

## Cost Architecture Overview

### Target Cost Structure
- **Target Monthly Budget**: $26-36 for demo usage
- **Alert Threshold**: $40 monthly budget with 80% and 100% alerts
- **Cost Optimization Priority**: Minimize operational costs while maintaining functionality

### Cost Components Breakdown

| Service | Configuration | Monthly Cost (USD) | Optimization Notes |
|---------|---------------|-------------------|-------------------|
| **ECS Fargate** | 0.25 vCPU, 0.5 GB RAM | $10-15 | Cost-optimized minimal sizing |
| **Application Load Balancer** | Standard ALB | $16 | Required for ECS Fargate |
| **NAT Gateway** | Single AZ | $32 | **Removable for cost savings** |
| **S3 Storage** | Standard + lifecycle | $2-5 | Lifecycle policies enabled |
| **CloudFront** | Free tier usage | $0-3 | Optimized for free tier |
| **Route 53** | Hosted zone (optional) | $1 | Only if using custom domain |
| **CloudWatch** | Basic monitoring | $1-2 | Free tier covers most usage |
| **Secrets Manager** | 1 secret | $0.40 | Minimal usage |
| **Data Transfer** | Minimal usage | $1-5 | CloudFront reduces costs |
| **Total** | | **$62-74** | **Can be reduced to $26-36** |

## Cost Optimization Strategies

### Immediate Cost Reductions

#### 1. Remove NAT Gateway (Save $32/month)
**Impact**: Removes outbound internet access from private subnets
**Suitable if**: Application doesn't need to make outbound API calls

```bash
# Check if NAT Gateway is being used
aws ec2 describe-nat-gateways --region eu-west-1

# Remove NAT Gateway from CloudFormation template
# Comment out or remove these resources:
# - NatGateway1
# - NatGateway1EIP
# - DefaultPrivateRoute1

# Update route table to remove NAT Gateway route
aws ec2 delete-route \
  --route-table-id rtb-xxxxxxxxx \
  --destination-cidr-block 0.0.0.0/0 \
  --region eu-west-1
```

#### 2. Use EC2 Instead of Fargate (Save $5-10/month)
**Impact**: Requires more management overhead
**Suitable if**: You can manage EC2 instances

```yaml
# Alternative EC2 configuration in CloudFormation
EC2Instance:
  Type: AWS::EC2::Instance
  Properties:
    InstanceType: t3.micro  # Free tier eligible
    ImageId: ami-0c02fb55956c7d316  # Amazon Linux 2
    SecurityGroupIds:
      - !Ref ECSSecurityGroup
    SubnetId: !Ref PrivateSubnet1
    IamInstanceProfile: !Ref EC2InstanceProfile
    UserData:
      Fn::Base64: !Sub |
        #!/bin/bash
        yum update -y
        yum install -y docker
        service docker start
        usermod -a -G docker ec2-user
```

#### 3. Optimize S3 Storage Classes
**Impact**: Reduces storage costs for infrequently accessed data

```json
{
  "Rules": [
    {
      "ID": "CostOptimizationRule",
      "Status": "Enabled",
      "Filter": {"Prefix": "drawings/"},
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        },
        {
          "Days": 365,
          "StorageClass": "DEEP_ARCHIVE"
        }
      ]
    },
    {
      "ID": "DeleteOldVersions",
      "Status": "Enabled",
      "NoncurrentVersionExpiration": {
        "NoncurrentDays": 30
      }
    }
  ]
}
```

#### 4. CloudFront Optimization
**Impact**: Maximizes free tier usage and reduces data transfer costs

```yaml
# Optimized CloudFront configuration
DefaultCacheBehavior:
  TargetOriginId: ALBOrigin
  ViewerProtocolPolicy: redirect-to-https
  CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad  # CachingOptimized
  OriginRequestPolicyId: 88a5eaf4-2fd4-4709-b370-b4c650ea3fcf  # CORS-S3Origin
  DefaultTTL: 86400  # 24 hours
  MaxTTL: 31536000   # 1 year
  MinTTL: 0
```

### Advanced Cost Optimization

#### 1. Spot Instances for ECS (Save 50-70%)
**Impact**: Potential interruptions but significant cost savings

```yaml
ECSService:
  Type: AWS::ECS::Service
  Properties:
    CapacityProviderStrategy:
      - CapacityProvider: FARGATE_SPOT
        Weight: 100
    # Fargate Spot can save 50-70% on compute costs
```

#### 2. Reserved Instances for Predictable Workloads
**Impact**: 30-60% savings for 1-3 year commitments

```bash
# Check Reserved Instance recommendations
aws ce get-reservation-purchase-recommendation \
  --service EC2-Instance \
  --region eu-west-1

# Purchase Reserved Instance (example)
aws ec2 purchase-reserved-instances-offering \
  --reserved-instances-offering-id xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
  --instance-count 1
```

#### 3. S3 Intelligent Tiering
**Impact**: Automatic cost optimization for varying access patterns

```yaml
S3Bucket:
  Type: AWS::S3::Bucket
  Properties:
    IntelligentTieringConfigurations:
      - Id: EntireBucket
        Status: Enabled
        OptionalFields:
          - BucketKeyStatus
```

## Cost Monitoring Setup

### 1. AWS Budgets Configuration

#### Monthly Budget with Alerts
```bash
# Create budget via CLI
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "ChildrenDrawing-Monthly-Budget",
    "BudgetLimit": {
      "Amount": "40",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
      "TagKey": ["Environment"],
      "TagValue": ["production"]
    }
  }' \
  --notifications-with-subscribers '[
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "admin@example.com"
        }
      ]
    },
    {
      "Notification": {
        "NotificationType": "FORECASTED",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 100,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "admin@example.com"
        }
      ]
    }
  ]'
```

#### Service-Specific Budgets
```bash
# ECS-specific budget
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "ChildrenDrawing-ECS-Budget",
    "BudgetLimit": {
      "Amount": "20",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
      "Service": ["Amazon Elastic Container Service"]
    }
  }'
```

### 2. CloudWatch Cost Metrics

#### Custom Cost Metrics Dashboard
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Billing", "EstimatedCharges", "Currency", "USD"]
        ],
        "period": 86400,
        "stat": "Maximum",
        "region": "us-east-1",
        "title": "Estimated Monthly Charges"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/ECS", "CPUUtilization", "ServiceName", "children-drawing-prod-service"],
          ["AWS/ECS", "MemoryUtilization", "ServiceName", "children-drawing-prod-service"]
        ],
        "period": 300,
        "stat": "Average",
        "region": "eu-west-1",
        "title": "ECS Resource Utilization"
      }
    }
  ]
}
```

#### Cost Anomaly Detection
```bash
# Enable Cost Anomaly Detection
aws ce create-anomaly-detector \
  --anomaly-detector '{
    "DetectorName": "ChildrenDrawing-Cost-Anomaly",
    "MonitorType": "DIMENSIONAL",
    "DimensionKey": "SERVICE",
    "MatchOptions": ["EQUALS"],
    "MonitorSpecification": "{\\"Dimension\\": \\"SERVICE\\", \\"MatchOptions\\": [\\"EQUALS\\"], \\"Values\\": [\\"Amazon Elastic Container Service\\"]}"
  }'
```

### 3. Cost Allocation Tags

#### Tagging Strategy
```yaml
# CloudFormation resource tagging
Tags:
  - Key: Environment
    Value: !Ref Environment
  - Key: Project
    Value: ChildrenDrawingAnomalyDetection
  - Key: CostCenter
    Value: Research
  - Key: Owner
    Value: DataScienceTeam
  - Key: Application
    Value: AnomalyDetection
```

#### Tag-Based Cost Reports
```bash
# Generate cost report by tags
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=TAG \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1
```

## Cost Analysis and Reporting

### 1. Daily Cost Monitoring Script

```bash
#!/bin/bash
# daily-cost-check.sh

# Get yesterday's costs
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

echo "=== Daily Cost Report for $YESTERDAY ==="

# Get total cost for yesterday
TOTAL_COST=$(aws ce get-cost-and-usage \
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
  --output text)

echo "Total Cost: \$$TOTAL_COST"

# Get cost by service
echo "=== Cost by Service ==="
aws ce get-cost-and-usage \
  --time-period Start=$YESTERDAY,End=$TODAY \
  --granularity DAILY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1 \
  --query 'ResultsByTime[0].Groups[?Total.BlendedCost.Amount>`0`].[Keys[0],Total.BlendedCost.Amount]' \
  --output table

# Check if cost exceeds daily budget (monthly budget / 30)
DAILY_BUDGET=$(echo "scale=2; 40 / 30" | bc)
if (( $(echo "$TOTAL_COST > $DAILY_BUDGET" | bc -l) )); then
  echo "⚠️  WARNING: Daily cost ($TOTAL_COST) exceeds daily budget ($DAILY_BUDGET)"
  # Send alert (implement notification logic)
fi

echo "=== End of Report ==="
```

### 2. Monthly Cost Analysis

```bash
#!/bin/bash
# monthly-cost-analysis.sh

CURRENT_MONTH=$(date +%Y-%m-01)
NEXT_MONTH=$(date -d "next month" +%Y-%m-01)

echo "=== Monthly Cost Analysis ==="

# Get month-to-date costs
aws ce get-cost-and-usage \
  --time-period Start=$CURRENT_MONTH,End=$NEXT_MONTH \
  --granularity MONTHLY \
  --metrics BlendedCost,UnblendedCost,UsageQuantity \
  --group-by Type=DIMENSION,Key=SERVICE \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1

# Get cost forecast
echo "=== Cost Forecast ==="
aws ce get-cost-forecast \
  --time-period Start=$CURRENT_MONTH,End=$NEXT_MONTH \
  --metric BLENDED_COST \
  --granularity MONTHLY \
  --filter '{
    "Tags": {
      "Key": "Project",
      "Values": ["ChildrenDrawingAnomalyDetection"]
    }
  }' \
  --region us-east-1

# Get rightsizing recommendations
echo "=== Rightsizing Recommendations ==="
aws ce get-rightsizing-recommendation \
  --service EC2-Instance \
  --region us-east-1
```

### 3. Cost Optimization Recommendations

```bash
#!/bin/bash
# cost-optimization-recommendations.sh

echo "=== Cost Optimization Recommendations ==="

# Check for unused resources
echo "1. Checking for unused EBS volumes..."
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --query 'Volumes[?State==`available`].[VolumeId,Size,VolumeType]' \
  --output table \
  --region eu-west-1

echo "2. Checking for unattached Elastic IPs..."
aws ec2 describe-addresses \
  --query 'Addresses[?AssociationId==null].[PublicIp,AllocationId]' \
  --output table \
  --region eu-west-1

echo "3. Checking ECS service utilization..."
aws ecs describe-services \
  --cluster children-drawing-prod-cluster \
  --services children-drawing-prod-service \
  --query 'services[0].[serviceName,runningCount,desiredCount]' \
  --output table \
  --region eu-west-1

echo "4. Checking S3 storage class distribution..."
aws s3api list-objects-v2 \
  --bucket children-drawing-production-drawings-$(aws sts get-caller-identity --query Account --output text) \
  --query 'Contents[?StorageClass!=`STANDARD`].[Key,StorageClass,Size]' \
  --output table

echo "5. CloudFront cache hit ratio..."
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=$(aws cloudformation describe-stacks --stack-name children-drawing-prod --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionId`].OutputValue' --output text --region eu-west-1) \
  --start-time $(date -d '7 days ago' --iso-8601) \
  --end-time $(date --iso-8601) \
  --period 86400 \
  --statistics Average \
  --region us-east-1
```

## Cost Alerts and Automation

### 1. Lambda Function for Cost Alerts

```python
import json
import boto3
import os
from datetime import datetime, timedelta

def lambda_handler(event, context):
    """
    Lambda function to check daily costs and send alerts
    """
    ce_client = boto3.client('ce', region_name='us-east-1')
    sns_client = boto3.client('sns')
    
    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get cost for yesterday
    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': yesterday,
            'End': today
        },
        Granularity='DAILY',
        Metrics=['BlendedCost'],
        Filter={
            'Tags': {
                'Key': 'Project',
                'Values': ['ChildrenDrawingAnomalyDetection']
            }
        }
    )
    
    if response['ResultsByTime']:
        daily_cost = float(response['ResultsByTime'][0]['Total']['BlendedCost']['Amount'])
        daily_budget = float(os.environ.get('DAILY_BUDGET', '1.33'))  # $40/30 days
        
        if daily_cost > daily_budget:
            message = f"""
            🚨 COST ALERT: Children Drawing Anomaly Detection System
            
            Daily cost for {yesterday}: ${daily_cost:.2f}
            Daily budget: ${daily_budget:.2f}
            Overage: ${daily_cost - daily_budget:.2f}
            
            Please review resource usage and optimize costs.
            """
            
            sns_client.publish(
                TopicArn=os.environ['SNS_TOPIC_ARN'],
                Message=message,
                Subject='Cost Alert: Daily Budget Exceeded'
            )
    
    return {
        'statusCode': 200,
        'body': json.dumps('Cost check completed')
    }
```

### 2. Automated Cost Optimization

```bash
#!/bin/bash
# automated-cost-optimization.sh

echo "=== Automated Cost Optimization ==="

# 1. Stop ECS service during off-hours (if applicable)
CURRENT_HOUR=$(date +%H)
if [ $CURRENT_HOUR -ge 22 ] || [ $CURRENT_HOUR -le 6 ]; then
  echo "Off-hours detected. Scaling down ECS service..."
  aws ecs update-service \
    --cluster children-drawing-prod-cluster \
    --service children-drawing-prod-service \
    --desired-count 0 \
    --region eu-west-1
fi

# 2. Clean up old S3 objects
echo "Cleaning up old temporary files..."
aws s3 rm s3://children-drawing-production-drawings-$(aws sts get-caller-identity --query Account --output text)/temp/ --recursive

# 3. Delete old CloudWatch logs
echo "Cleaning up old CloudWatch logs..."
aws logs delete-log-group --log-group-name /ecs/children-drawing-prod-old --region eu-west-1 2>/dev/null || true

# 4. Optimize EBS volumes (if any)
echo "Checking for EBS optimization opportunities..."
aws ec2 describe-volumes \
  --filters Name=state,Values=in-use \
  --query 'Volumes[?VolumeType==`gp2`].[VolumeId,Size]' \
  --output table \
  --region eu-west-1

echo "=== Optimization Complete ==="
```

## Cost Governance and Policies

### 1. IAM Cost Control Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyExpensiveInstances",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances"
      ],
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "ForAnyValue:StringNotEquals": {
          "ec2:InstanceType": [
            "t3.micro",
            "t3.small",
            "t3.medium"
          ]
        }
      }
    },
    {
      "Sid": "RequireCostTags",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances",
        "s3:CreateBucket",
        "ecs:CreateService"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestedRegion": "false",
          "aws:RequestTag/Project": "true"
        }
      }
    }
  ]
}
```

### 2. Service Control Policies (SCPs)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyExpensiveServices",
      "Effect": "Deny",
      "Action": [
        "redshift:*",
        "sagemaker:CreateNotebookInstance",
        "ec2:*Spot*"
      ],
      "Resource": "*"
    },
    {
      "Sid": "RestrictRegions",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "eu-west-1",
            "us-east-1"
          ]
        }
      }
    }
  ]
}
```

## Cost Reporting and Dashboards

### 1. CloudWatch Dashboard Configuration

```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/Billing", "EstimatedCharges", "Currency", "USD", {"stat": "Maximum"}]
        ],
        "view": "timeSeries",
        "stacked": false,
        "region": "us-east-1",
        "title": "Monthly Estimated Charges",
        "period": 86400
      }
    },
    {
      "type": "log",
      "properties": {
        "query": "SOURCE '/aws/lambda/cost-monitor'\n| fields @timestamp, @message\n| filter @message like /COST ALERT/\n| sort @timestamp desc\n| limit 20",
        "region": "eu-west-1",
        "title": "Recent Cost Alerts"
      }
    }
  ]
}
```

### 2. Cost and Usage Report (CUR)

```bash
# Enable Cost and Usage Report
aws cur put-report-definition \
  --report-definition '{
    "ReportName": "children-drawing-cur",
    "TimeUnit": "DAILY",
    "Format": "textORcsv",
    "Compression": "GZIP",
    "AdditionalSchemaElements": ["RESOURCES"],
    "S3Bucket": "children-drawing-production-billing-reports",
    "S3Prefix": "cur/",
    "S3Region": "eu-west-1",
    "AdditionalArtifacts": ["REDSHIFT", "ATHENA"],
    "RefreshClosedReports": true,
    "ReportVersioning": "OVERWRITE_REPORT"
  }' \
  --region us-east-1
```

## Best Practices and Recommendations

### 1. Cost Optimization Checklist

- [ ] **Right-size resources**: Use smallest viable instance types
- [ ] **Enable auto-scaling**: Scale down during low usage periods
- [ ] **Use Spot instances**: For fault-tolerant workloads
- [ ] **Implement lifecycle policies**: Automatic S3 storage class transitions
- [ ] **Monitor unused resources**: Regular cleanup of orphaned resources
- [ ] **Use Reserved Instances**: For predictable workloads
- [ ] **Enable cost allocation tags**: Track costs by project/environment
- [ ] **Set up budget alerts**: Proactive cost monitoring
- [ ] **Regular cost reviews**: Monthly optimization assessments
- [ ] **Use AWS Free Tier**: Maximize free tier usage

### 2. Monthly Cost Review Process

1. **Week 1**: Review previous month's costs and trends
2. **Week 2**: Analyze cost anomalies and optimization opportunities
3. **Week 3**: Implement cost optimization measures
4. **Week 4**: Monitor impact and adjust budgets/alerts

### 3. Emergency Cost Control Procedures

```bash
#!/bin/bash
# emergency-cost-control.sh

echo "🚨 EMERGENCY COST CONTROL ACTIVATED 🚨"

# 1. Scale down ECS service to minimum
aws ecs update-service \
  --cluster children-drawing-prod-cluster \
  --service children-drawing-prod-service \
  --desired-count 0 \
  --region eu-west-1

# 2. Delete NAT Gateway (if exists)
NAT_GATEWAY_ID=$(aws ec2 describe-nat-gateways \
  --filter Name=tag:Name,Values=children-drawing-prod-nat-1 \
  --query 'NatGateways[0].NatGatewayId' \
  --output text \
  --region eu-west-1)

if [ "$NAT_GATEWAY_ID" != "None" ]; then
  aws ec2 delete-nat-gateway --nat-gateway-id $NAT_GATEWAY_ID --region eu-west-1
fi

# 3. Stop any running EC2 instances
aws ec2 stop-instances \
  --instance-ids $(aws ec2 describe-instances \
    --filters Name=tag:Project,Values=ChildrenDrawingAnomalyDetection Name=instance-state-name,Values=running \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text \
    --region eu-west-1) \
  --region eu-west-1

# 4. Send notification
aws sns publish \
  --topic-arn arn:aws:sns:eu-west-1:$(aws sts get-caller-identity --query Account --output text):children-drawing-prod-cost-alerts \
  --message "Emergency cost control measures activated. All services have been scaled down." \
  --subject "EMERGENCY: Cost Control Activated" \
  --region eu-west-1

echo "Emergency cost control measures completed."
```

This comprehensive cost monitoring and optimization guide provides the tools and procedures needed to maintain the Children's Drawing Anomaly Detection System within budget while ensuring optimal performance and functionality.