#!/bin/bash

# AWS Infrastructure Deployment Script for Children's Drawing Anomaly Detection System
# This script deploys the CloudFormation templates for the production environment

set -e  # Exit on any error

# Configuration
STACK_NAME="children-drawing-prod"
NETWORK_STACK_NAME="children-drawing-network"
REGION="eu-west-1"
ENVIRONMENT="production"
AWS_PROFILE="${AWS_PROFILE:-d-9067931f77-921400262514-admin+Q}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if AWS CLI is configured
check_aws_cli() {
    print_status "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! AWS_PROFILE=$AWS_PROFILE aws sts get-caller-identity &> /dev/null; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    local account_id=$(AWS_PROFILE=$AWS_PROFILE aws sts get-caller-identity --query Account --output text)
    local current_region=$(AWS_PROFILE=$AWS_PROFILE aws configure get region)
    
    print_success "AWS CLI configured for account: $account_id in region: $current_region"
    
    if [ "$current_region" != "$REGION" ]; then
        print_warning "Current region ($current_region) differs from target region ($REGION)"
        read -p "Continue with region $REGION? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to validate CloudFormation templates
validate_templates() {
    print_status "Validating CloudFormation templates..."
    
    local templates=("main-infrastructure.yaml" "networking.yaml")
    
    for template in "${templates[@]}"; do
        if [ -f "$template" ]; then
            print_status "Validating $template..."
            if AWS_PROFILE=$AWS_PROFILE aws cloudformation validate-template --template-body file://$template --region $REGION > /dev/null; then
                print_success "$template is valid"
            else
                print_error "$template validation failed"
                exit 1
            fi
        else
            print_error "Template $template not found"
            exit 1
        fi
    done
}

# Function to get user input for parameters
get_deployment_parameters() {
    print_status "Gathering deployment parameters..."
    
    # Domain name (optional)
    read -p "Enter domain name (optional, press Enter to skip): " DOMAIN_NAME
    
    # S3 bucket prefix
    read -p "Enter S3 bucket prefix (default: children-drawing): " S3_BUCKET_PREFIX
    S3_BUCKET_PREFIX=${S3_BUCKET_PREFIX:-children-drawing}
    
    # ECR repository URI (optional)
    read -p "Enter ECR repository URI (optional, press Enter to skip): " ECR_REPOSITORY_URI
    
    # Certificate ARN (optional)
    if [ ! -z "$DOMAIN_NAME" ]; then
        read -p "Enter ACM certificate ARN for $DOMAIN_NAME (optional): " CERTIFICATE_ARN
        
        # Route 53 option
        read -p "Create Route 53 hosted zone for $DOMAIN_NAME? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ENABLE_ROUTE53="true"
        else
            ENABLE_ROUTE53="false"
        fi
    else
        ENABLE_ROUTE53="false"
    fi
    
    # Cost alert email
    read -p "Enter email for cost alerts (optional): " COST_ALERT_EMAIL
    
    print_status "Deployment parameters:"
    echo "  Stack Name: $STACK_NAME"
    echo "  Region: $REGION"
    echo "  Environment: $ENVIRONMENT"
    echo "  Domain Name: ${DOMAIN_NAME:-'Not specified'}"
    echo "  S3 Bucket Prefix: $S3_BUCKET_PREFIX"
    echo "  ECR Repository: ${ECR_REPOSITORY_URI:-'Not specified'}"
    echo "  Certificate ARN: ${CERTIFICATE_ARN:-'Not specified'}"
    echo "  Enable Route 53: $ENABLE_ROUTE53"
    echo "  Cost Alert Email: ${COST_ALERT_EMAIL:-'Not specified'}"
    
    read -p "Proceed with deployment? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Deployment cancelled"
        exit 0
    fi
}

# Function to deploy networking stack (optional separate deployment)
deploy_networking() {
    print_status "Deploying networking stack..."
    
    local params=""
    params+="ParameterKey=Environment,ParameterValue=$ENVIRONMENT "
    
    aws cloudformation deploy \
        --template-file networking.yaml \
        --stack-name $NETWORK_STACK_NAME \
        --parameter-overrides $params \
        --capabilities CAPABILITY_IAM \
        --region $REGION \
        --tags Environment=$ENVIRONMENT Project=ChildrenDrawingAnomalyDetection \
        --profile $AWS_PROFILE
    
    if [ $? -eq 0 ]; then
        print_success "Networking stack deployed successfully"
    else
        print_error "Networking stack deployment failed"
        exit 1
    fi
}

# Function to deploy main infrastructure
deploy_main_infrastructure() {
    print_status "Deploying main infrastructure stack..."
    
    # Build parameter overrides
    local params=""
    params+="ParameterKey=Environment,ParameterValue=$ENVIRONMENT "
    params+="ParameterKey=S3BucketPrefix,ParameterValue=$S3_BUCKET_PREFIX "
    
    if [ ! -z "$DOMAIN_NAME" ]; then
        params+="ParameterKey=DomainName,ParameterValue=$DOMAIN_NAME "
    fi
    
    if [ ! -z "$ECR_REPOSITORY_URI" ]; then
        params+="ParameterKey=ECRRepositoryURI,ParameterValue=$ECR_REPOSITORY_URI "
    fi
    
    if [ ! -z "$CERTIFICATE_ARN" ]; then
        params+="ParameterKey=CertificateArn,ParameterValue=$CERTIFICATE_ARN "
    fi
    
    if [ ! -z "$ENABLE_ROUTE53" ]; then
        params+="ParameterKey=EnableRoute53,ParameterValue=$ENABLE_ROUTE53 "
    fi
    
    if [ ! -z "$COST_ALERT_EMAIL" ]; then
        params+="ParameterKey=CostAlertEmail,ParameterValue=$COST_ALERT_EMAIL "
    fi
    
    print_status "Deploying with parameters: $params"
    
    aws cloudformation deploy \
        --template-file main-infrastructure.yaml \
        --stack-name $STACK_NAME \
        --parameter-overrides $params \
        --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
        --region $REGION \
        --tags Environment=$ENVIRONMENT Project=ChildrenDrawingAnomalyDetection \
        --profile $AWS_PROFILE
    
    if [ $? -eq 0 ]; then
        print_success "Main infrastructure stack deployed successfully"
    else
        print_error "Main infrastructure stack deployment failed"
        exit 1
    fi
}

# Function to get stack outputs
get_stack_outputs() {
    print_status "Retrieving stack outputs..."
    
    local outputs=$(AWS_PROFILE=$AWS_PROFILE aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --region $REGION \
        --query 'Stacks[0].Outputs' \
        --output table)
    
    if [ $? -eq 0 ]; then
        print_success "Stack outputs:"
        echo "$outputs"
        
        # Get specific important outputs
        local app_url=$(AWS_PROFILE=$AWS_PROFILE aws cloudformation describe-stacks \
            --stack-name $STACK_NAME \
            --region $REGION \
            --query 'Stacks[0].Outputs[?OutputKey==`ApplicationURL`].OutputValue' \
            --output text)
        
        local alb_dns=$(AWS_PROFILE=$AWS_PROFILE aws cloudformation describe-stacks \
            --stack-name $STACK_NAME \
            --region $REGION \
            --query 'Stacks[0].Outputs[?OutputKey==`ApplicationLoadBalancerDNS`].OutputValue' \
            --output text)
        
        local cloudfront_dns=$(AWS_PROFILE=$AWS_PROFILE aws cloudformation describe-stacks \
            --stack-name $STACK_NAME \
            --region $REGION \
            --query 'Stacks[0].Outputs[?OutputKey==`CloudFrontDistributionDNS`].OutputValue' \
            --output text)
        
        print_success "Deployment completed successfully!"
        echo ""
        echo "Important URLs:"
        echo "  Application URL: $app_url"
        echo "  Load Balancer DNS (HTTP): http://$alb_dns"
        echo "  CloudFront DNS (HTTPS): https://$cloudfront_dns"
        echo ""
        echo "Working Endpoints:"
        echo "  Demo Interface: https://$cloudfront_dns/demo/"
        echo "  API Documentation: https://$cloudfront_dns/docs"
        echo "  Health Check: https://$cloudfront_dns/health"
        echo ""
        echo "Next steps:"
        echo "1. Build and push your Docker image to ECR:"
        echo "   docker build -f Dockerfile.prod -t children-drawing-app:latest ."
        echo "   docker tag children-drawing-app:latest [ECR_URI]:latest"
        echo "   docker push [ECR_URI]:latest"
        echo "2. Update the ECS task definition with your image URI"
        echo "3. Deploy new task definition to ECS service"
        echo "4. Configure your domain DNS (if using custom domain)"
        echo "5. Set up GitHub Actions for automated deployments"
    else
        print_error "Failed to retrieve stack outputs"
    fi
}

# Function to check deployment status
check_deployment_status() {
    print_status "Checking current deployment status..."
    
    # Check if ECS service exists and is running
    local service_status=$(AWS_PROFILE=$AWS_PROFILE aws ecs describe-services \
        --cluster children-drawing-prod-cluster \
        --services children-drawing-prod-service \
        --region $REGION \
        --query 'services[0].{Status:status,RunningCount:runningCount,DesiredCount:desiredCount}' \
        --output text 2>/dev/null || echo "NOT_FOUND")
    
    if [ "$service_status" != "NOT_FOUND" ]; then
        print_success "ECS Service Status: $service_status"
        
        # Test application health
        local alb_dns=$(AWS_PROFILE=$AWS_PROFILE aws elbv2 describe-load-balancers \
            --region $REGION \
            --query 'LoadBalancers[?contains(LoadBalancerName, `children-drawing`)].DNSName' \
            --output text 2>/dev/null)
        
        if [ ! -z "$alb_dns" ]; then
            print_status "Testing application health..."
            if curl -f -s "http://$alb_dns/health" > /dev/null 2>&1; then
                print_success "Application is healthy and responding"
                echo "  Demo Interface: http://$alb_dns/demo/"
                echo "  API Documentation: http://$alb_dns/docs"
            else
                print_warning "Application may not be fully ready yet"
            fi
        fi
    else
        print_warning "ECS service not found - infrastructure may not be deployed"
    fi
}
# Function to estimate costs
    print_status "Cost estimation for the deployed infrastructure:"
    echo ""
    echo "Monthly cost estimates (USD):"
    echo "  ECS Fargate (0.25 vCPU, 0.5 GB): ~\$10-15"
    echo "  Application Load Balancer: ~\$16"
    echo "  NAT Gateway: ~\$32"
    echo "  S3 Storage (demo usage): ~\$2-5"
    echo "  CloudFront: Free tier covers demo usage"
    echo "  Route 53 (if enabled): ~\$1"
    echo "  CloudWatch: Free tier covers basic monitoring"
    echo "  Data Transfer: ~\$1-5"
    echo ""
    echo "  Estimated Total: ~\$62-74/month"
    echo ""
    print_warning "To reduce costs for demo usage:"
    echo "  - Consider disabling NAT Gateway if not needed (-\$32/month)"
    echo "  - Use smaller ECS task sizes"
    echo "  - Monitor S3 storage usage"
    echo "  - Set up cost alerts (included in template)"
}

# Function to show cleanup instructions
show_cleanup_instructions() {
    echo ""
    print_status "To delete the infrastructure:"
    echo "  aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION"
    if [ "$DEPLOY_NETWORKING_SEPARATELY" = "true" ]; then
        echo "  aws cloudformation delete-stack --stack-name $NETWORK_STACK_NAME --region $REGION"
    fi
    echo ""
    print_warning "Note: S3 buckets with content cannot be deleted automatically."
    print_warning "You may need to empty them manually before stack deletion."
}

# Main deployment function
main() {
    echo "=========================================="
    echo "AWS Infrastructure Deployment"
    echo "Children's Drawing Anomaly Detection System"
    echo "=========================================="
    echo ""
    
    # Check prerequisites
    check_aws_cli
    
    # Validate templates
    validate_templates
    
    # Get deployment parameters
    get_deployment_parameters
    
    # Ask about networking deployment
    read -p "Deploy networking as a separate stack? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DEPLOY_NETWORKING_SEPARATELY="true"
        deploy_networking
    else
        DEPLOY_NETWORKING_SEPARATELY="false"
    fi
    
    # Deploy main infrastructure
    deploy_main_infrastructure
    
    # Get outputs
    get_stack_outputs
    
    # Show cost estimation
    estimate_costs
    
    # Show cleanup instructions
    show_cleanup_instructions
    
    print_success "Deployment process completed!"
}

# Handle script arguments
case "${1:-}" in
    "validate")
        validate_templates
        ;;
    "networking")
        check_aws_cli
        validate_templates
        get_deployment_parameters
        deploy_networking
        ;;
    "main")
        check_aws_cli
        validate_templates
        get_deployment_parameters
        deploy_main_infrastructure
        get_stack_outputs
        ;;
    "outputs")
        get_stack_outputs
        ;;
    "costs")
        estimate_costs
        ;;
    "cleanup")
        show_cleanup_instructions
        ;;
    *)
        main
        ;;
esac