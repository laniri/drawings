#!/bin/bash

# CloudFront Cache Invalidation Script
# This script invalidates the CloudFront cache to immediately reflect changes

set -e

echo "🔄 Invalidating CloudFront cache..."

# Set variables
CLOUDFRONT_DISTRIBUTION_ID="E34MC6W2KLQE7H"
AWS_REGION="eu-west-1"

echo "📤 Creating invalidation for distribution: ${CLOUDFRONT_DISTRIBUTION_ID}"

# Create invalidation
INVALIDATION_ID=$(aws cloudfront create-invalidation \
    --distribution-id ${CLOUDFRONT_DISTRIBUTION_ID} \
    --paths "/*" \
    --query 'Invalidation.Id' \
    --output text)

echo "⏳ Invalidation created with ID: ${INVALIDATION_ID}"
echo "⏳ Waiting for invalidation to complete..."

# Wait for invalidation to complete
aws cloudfront wait invalidation-completed \
    --distribution-id ${CLOUDFRONT_DISTRIBUTION_ID} \
    --id ${INVALIDATION_ID}

echo "✅ CloudFront cache invalidation completed!"
echo ""
echo "🌐 You can now test the application at:"
echo "https://d2e6rjfv7d2rgs.cloudfront.net/"