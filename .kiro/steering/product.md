# Product Overview

## Children's Drawing Anomaly Detection System

A machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected norms. The system uses Vision Transformer (ViT) embeddings and autoencoder models trained on age-specific drawing patterns to detect anomalies through reconstruction loss analysis.

**Current Status**: Fully functional and trained with 37,778+ drawings and 8 trained autoencoder models for age groups (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years). Features real-time dashboard updates and optimized threshold management.

## Core Features

- **Drawing Upload & Analysis**: Support for PNG, JPEG, and BMP formats with metadata
- **Age-Based Modeling**: Separate models trained for different age groups
- **Anomaly Detection**: Reconstruction loss-based scoring with configurable thresholds
- **Interactive Interpretability**: Advanced saliency analysis with hoverable regions, zoom/pan, and detailed explanations
- **Web Interface**: Modern React frontend with Material-UI components and real-time configuration management
- **REST API**: FastAPI backend with automatic OpenAPI documentation
- **Enhanced Training**: Offline training system with verbose progress indicators and comprehensive model management
- **Real-time Dashboard**: Dynamic anomaly count updates when threshold percentiles change
- **Optimized Performance**: Fast threshold recalculation using existing analysis results

## Target Users

- Researchers studying child development patterns
- Educational professionals monitoring developmental milestones
- Healthcare providers screening for developmental concerns

## Key Technical Concepts

- **Vision Transformer (ViT)**: Used for feature extraction from drawings
- **Autoencoder Architecture**: Reconstruction-based anomaly detection
- **Age-Group Stratification**: Models trained separately for different age ranges
- **Threshold Management**: Configurable percentile-based anomaly thresholds with real-time updates
- **Dynamic Recalculation**: Optimized threshold updates using existing analysis results
- **Interpretability Engine**: Generates visual explanations using saliency maps