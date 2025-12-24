# GitHub Actions CI/CD Workflows

This directory contains the GitHub Actions workflows for the Children's Drawing Anomaly Detection System.

## Workflows

### 1. Production Deployment (`deploy-production.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

**Jobs:**
1. **Test** - Runs unit tests, property-based tests, linting, and type checking
2. **Security Scan** - Vulnerability scanning with Trivy and dependency checks
3. **Build and Push** - Docker image build and push to ECR (main branch only)
4. **Deploy Infrastructure** - CloudFormation deployment (optional, triggered by commit message or manual input)
5. **Deploy Application** - ECS service deployment (main branch only)
6. **Health Check** - Application health verification with automatic rollback on failure
7. **Monitor Deployment** - Post-deployment monitoring and metrics collection
8. **Notify** - Slack notifications, email alerts, and GitHub issue creation on failure

**Key Features:**
- **Zero-downtime deployments** with health check-based rollback
- **Comprehensive security scanning** at multiple stages
- **Automated rollback** on health check failures
- **Multi-stage health checks** including performance validation
- **Deployment monitoring** with CloudWatch integration
- **Rich notifications** with Slack, email, and GitHub integrations

### 2. Frontend CI (`frontend-ci.yml`)

**Triggers:**
- Push to `main` branch (frontend changes only)
- Pull requests to `main` branch (frontend changes only)

**Jobs:**
1. **Frontend Tests** - Node.js testing, linting, type checking, and building

## Required Secrets

### AWS Configuration
- `AWS_ACCESS_KEY_ID` - AWS access key for deployment
- `AWS_SECRET_ACCESS_KEY` - AWS secret key for deployment

### Optional Secrets
- `S3_BUCKET_PREFIX` - Custom S3 bucket prefix (default: 'children-drawing')
- `ECR_REPOSITORY_URI` - Custom ECR repository URI
- `DOMAIN_NAME` - Custom domain name for CloudFront
- `CERTIFICATE_ARN` - SSL certificate ARN for HTTPS
- `ENABLE_ROUTE53` - Enable Route 53 DNS management (default: 'false')
- `COST_ALERT_EMAIL` - Email for cost alerts
- `SLACK_WEBHOOK_URL` - Slack webhook for notifications
- `NOTIFICATION_EMAIL` - Email for deployment notifications

## Environment Variables

The workflows use the following environment variables:

```yaml
env:
  AWS_REGION: eu-west-1
  ECR_REPOSITORY: children-drawing-anomaly-detection
  ECS_CLUSTER: children-drawing-prod-cluster
  ECS_SERVICE: children-drawing-prod-service
  CONTAINER_NAME: app
```

## Deployment Process

### Automatic Deployment (Push to main)
1. Code is pushed to `main` branch
2. Tests and security scans run automatically
3. Docker image is built and pushed to ECR
4. Application is deployed to ECS
5. Health checks verify deployment success
6. Notifications are sent on completion

### Manual Infrastructure Deployment
1. Trigger workflow manually with `deploy_infrastructure: true`
2. Or include `[deploy-infra]` in commit message
3. CloudFormation templates are validated and deployed

### Rollback Process
1. Health checks fail after deployment
2. Previous task definition is automatically retrieved
3. ECS service is updated to use previous version
4. Rollback health checks verify recovery
5. Failure notifications are sent

## Health Check Strategy

The deployment includes multiple levels of health checks:

1. **Basic Health Check** - `/health` endpoint with retries
2. **API Health Check** - `/api/v1/health` endpoint validation
3. **Static Assets Check** - Static file accessibility
4. **Performance Check** - Response time validation (< 10 seconds)
5. **Post-Rollback Verification** - Health check after rollback

## Security Features

- **Vulnerability Scanning** - Trivy scans for filesystem and container vulnerabilities
- **Dependency Checking** - Safety checks for Python dependencies
- **SARIF Upload** - Security results uploaded to GitHub Security tab
- **Least Privilege** - IAM roles with minimal required permissions
- **Secrets Management** - All sensitive data stored in GitHub Secrets

## Monitoring and Alerting

- **ECS Service Metrics** - Running/desired instance counts
- **CloudWatch Integration** - Error rate monitoring
- **Deployment Summaries** - GitHub step summaries with key metrics
- **Slack Notifications** - Rich notifications with workflow links
- **GitHub Issues** - Automatic issue creation on deployment failures
- **Email Alerts** - Optional email notifications

## Troubleshooting

### Common Issues

1. **Health Check Failures**
   - Check application logs in CloudWatch
   - Verify ECS service is running
   - Check security group configurations

2. **Build Failures**
   - Review test output in workflow logs
   - Check for dependency conflicts
   - Verify Docker build context

3. **Deployment Failures**
   - Check ECS service events
   - Verify IAM permissions
   - Review CloudFormation stack events

### Manual Rollback

If automatic rollback fails:

```bash
# Get previous task definition
aws ecs list-task-definitions --family-prefix children-drawing-prod-cluster-task --status ACTIVE --sort DESC

# Update service
aws ecs update-service --cluster children-drawing-prod-cluster --service children-drawing-prod-service --task-definition <previous-task-def-arn>
```

## Best Practices

1. **Test Locally** - Run tests and linting before pushing
2. **Small Changes** - Keep deployments small and focused
3. **Monitor Deployments** - Watch health checks and metrics
4. **Review Logs** - Check CloudWatch logs for issues
5. **Use Feature Flags** - Enable gradual rollouts when possible

## Property-Based Testing

The workflows include property-based tests that validate:

- **Deployment Trigger Consistency** - Ensures consistent deployment behavior
- **Zero-Downtime Guarantee** - Validates service availability during deployments
- **Infrastructure Reproducibility** - Verifies consistent infrastructure deployment

These tests run automatically and help ensure deployment reliability.