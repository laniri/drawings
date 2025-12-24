# Deployment Documentation

This directory contains comprehensive deployment guides and documentation for the Children's Drawing Anomaly Detection System.

## AWS Production Deployment

### Core Documentation
- [AWS Production Deployment Guide](aws-production-deployment.md) - Complete AWS deployment instructions
- [Deployment Runbook](deployment-runbook.md) - Step-by-step operational procedures
- [Troubleshooting Guide](troubleshooting-guide.md) - Common issues and solutions

### Specialized Guides
- [Cost Monitoring Guide](cost-monitoring-guide.md) - Cost optimization and monitoring
- [Security & Compliance Guide](security-compliance-guide.md) - Security configurations and compliance

## Local Development

### Development Setup
- [Docker Deployment](docker.md) - Container-based deployment using Docker Compose
- [Environment Setup](environment-setup.md) - Development and production environment configuration

## Quick Start

### Local Development
```bash
# Start all services locally
docker-compose up -d
```

### AWS Production Deployment
```bash
# Deploy to AWS (requires AWS CLI configuration)
cd infrastructure
./deploy.sh
```

## Documentation Structure

```
docs/deployment/
├── README.md                           # This file
├── aws-production-deployment.md        # Main AWS deployment guide
├── deployment-runbook.md              # Operational procedures
├── troubleshooting-guide.md           # Issue resolution
├── cost-monitoring-guide.md           # Cost management
├── security-compliance-guide.md       # Security best practices
├── docker.md                         # Local Docker setup
└── environment-setup.md              # Environment configuration
```

## Getting Started

1. **For Local Development**: Start with [Environment Setup](environment-setup.md)
2. **For AWS Production**: Begin with [AWS Production Deployment Guide](aws-production-deployment.md)
3. **For Operations**: Use the [Deployment Runbook](deployment-runbook.md)
4. **For Issues**: Consult the [Troubleshooting Guide](troubleshooting-guide.md)

## Requirements Coverage

This documentation addresses all deployment requirements:

- ✅ **Infrastructure Setup**: Complete CloudFormation templates and deployment procedures
- ✅ **Troubleshooting**: Comprehensive guide for common deployment issues
- ✅ **Cost Monitoring**: Detailed cost estimation, monitoring, and optimization
- ✅ **Security Compliance**: Security configurations and compliance requirements
- ✅ **Operational Procedures**: Daily operations, maintenance, and incident response