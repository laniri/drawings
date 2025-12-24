#!/bin/bash

# Setup script for GitHub Actions OIDC with AWS
# Run this in your AWS environment where you have admin access

set -e

# Configuration - UPDATE THESE VALUES
GITHUB_USERNAME="laniri"  # Replace with your GitHub username
REPO_NAME="children-drawing-anomaly-detection"  # Replace if different
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="eu-west-1"  # Match your deployment region

echo "Setting up GitHub Actions OIDC for:"
echo "  GitHub: ${GITHUB_USERNAME}/${REPO_NAME}"
echo "  AWS Account: ${AWS_ACCOUNT_ID}"
echo "  Region: ${AWS_REGION}"
echo ""

# Step 1: Create OIDC Identity Provider (if it doesn't exist)
echo "1. Creating OIDC Identity Provider..."
aws iam create-open-id-connect-provider \
    --url https://token.actions.githubusercontent.com \
    --client-id-list sts.amazonaws.com \
    --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 \
    2>/dev/null || echo "OIDC provider already exists"

# Step 2: Create trust policy for GitHub Actions
echo "2. Creating trust policy..."
cat > github-actions-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": [
            "repo:${GITHUB_USERNAME}/${REPO_NAME}:ref:refs/heads/main",
            "repo:${GITHUB_USERNAME}/${REPO_NAME}:pull_request"
          ]
        }
      }
    }
  ]
}
EOF

# Step 3: Create permissions policy for deployment
echo "3. Creating permissions policy..."
cat > github-actions-permissions-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:ListTaskDefinitions"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudformation:DescribeStacks",
        "cloudformation:DescribeStackEvents",
        "cloudformation:DescribeStackResources",
        "cloudformation:GetTemplate",
        "cloudformation:ListStacks",
        "cloudformation:ValidateTemplate",
        "cloudformation:CreateStack",
        "cloudformation:UpdateStack",
        "cloudformation:DeleteStack"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "cloudfront:CreateInvalidation",
        "cloudfront:GetDistribution",
        "cloudfront:ListDistributions"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::children-drawing-*",
        "arn:aws:s3:::children-drawing-*/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": [
        "arn:aws:iam::${AWS_ACCOUNT_ID}:role/children-drawing-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": [
        "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:children-drawing-*"
      ]
    }
  ]
}
EOF

# Step 4: Create the IAM role
echo "4. Creating IAM role..."
ROLE_NAME="GitHubActionsRole-ChildrenDrawing"

aws iam create-role \
    --role-name ${ROLE_NAME} \
    --assume-role-policy-document file://github-actions-trust-policy.json \
    --description "Role for GitHub Actions to deploy Children's Drawing Anomaly Detection System"

# Step 5: Create and attach the permissions policy
echo "5. Creating and attaching permissions policy..."
POLICY_NAME="GitHubActionsPolicy-ChildrenDrawing"

POLICY_ARN=$(aws iam create-policy \
    --policy-name ${POLICY_NAME} \
    --policy-document file://github-actions-permissions-policy.json \
    --description "Permissions for GitHub Actions deployment" \
    --query 'Policy.Arn' --output text)

aws iam attach-role-policy \
    --role-name ${ROLE_NAME} \
    --policy-arn ${POLICY_ARN}

# Step 6: Get the role ARN
ROLE_ARN=$(aws iam get-role --role-name ${ROLE_NAME} --query 'Role.Arn' --output text)

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Add this to your GitHub repository secrets:"
echo "   AWS_ROLE_ARN=${ROLE_ARN}"
echo "   AWS_REGION=${AWS_REGION}"
echo ""
echo "🔧 Update your GitHub Actions workflow with the role ARN above"
echo ""

# Cleanup temporary files
rm -f github-actions-trust-policy.json github-actions-permissions-policy.json

echo "🎉 GitHub Actions OIDC setup complete!"