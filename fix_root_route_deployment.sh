#!/bin/bash

# Fix Root Route Deployment Script
# This script rebuilds and deploys the container with the root route fix

set -e

echo "🔧 Fixing root route issue by rebuilding and deploying container..."

# Set variables
AWS_REGION="eu-west-1"
ECR_REPOSITORY="921400262514.dkr.ecr.eu-west-1.amazonaws.com/children-drawing-app"
ECS_CLUSTER="children-drawing-prod-cluster"
ECS_SERVICE="children-drawing-prod-service"
TASK_FAMILY="children-drawing-prod-task"

echo "📦 Step 1: Building Docker image with root route fix..."

# Build the Docker image
docker build -f Dockerfile.prod -t children-drawing-app:latest .

echo "🏷️ Step 2: Tagging image for ECR..."

# Tag for ECR
docker tag children-drawing-app:latest ${ECR_REPOSITORY}:latest

echo "🔐 Step 3: Logging into ECR..."

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

echo "⬆️ Step 4: Pushing image to ECR..."

# Push to ECR
docker push ${ECR_REPOSITORY}:latest

echo "🔄 Step 5: Forcing ECS service update..."

# Force ECS service to pull new image
aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION}

echo "⏳ Step 6: Waiting for deployment to complete..."

# Wait for deployment to complete
aws ecs wait services-stable \
    --cluster ${ECS_CLUSTER} \
    --services ${ECS_SERVICE} \
    --region ${AWS_REGION}

echo "✅ Deployment completed successfully!"

echo "🌐 Testing the fix..."

# Test the root endpoint
echo "Testing root endpoint..."
curl -s -o /dev/null -w "%{http_code}" https://d2e6rjfv7d2rgs.cloudfront.net/ || echo "Request failed (might be rate limited)"

echo ""
echo "🎉 Root route fix deployment completed!"
echo ""
echo "📋 Next steps:"
echo "1. Wait 2-3 minutes for CloudFront cache to clear"
echo "2. Test the application at: https://d2e6rjfv7d2rgs.cloudfront.net/"
echo "3. If you still see the API response, wait a bit longer for cache invalidation"
echo ""
echo "🔍 If issues persist, check:"
echo "- ECS service logs: aws logs get-log-events --log-group-name '/ecs/children-drawing-prod' --region eu-west-1"
echo "- ECS service status: aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE} --region ${AWS_REGION}"