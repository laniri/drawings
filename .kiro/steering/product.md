# Product Overview

## Children's Drawing Anomaly Detection System

A machine learning-powered application that analyzes children's drawings to identify patterns that deviate significantly from age-expected norms. The system uses hybrid embeddings combining Vision Transformer (ViT) visual features with subject category encoding, and subject-aware autoencoder models trained on age-specific drawing patterns to detect anomalies through reconstruction loss analysis.

**Current Status**: Fully functional and trained (v2.0.0 - Subject-Aware) with 37,778+ drawings and 8 trained subject-aware autoencoder models for age groups (2-3, 3-4, 4-5, 5-6, 6-7, 7-8, 8-9, 9-12 years). Features hybrid embeddings (832-dimensional), 64 subject categories, real-time dashboard updates and optimized threshold management.

## Core Features

- **Drawing Upload & Analysis**: Support for PNG, JPEG, and BMP formats with metadata and subject categorization
- **Subject-Aware Modeling**: 64 predefined subject categories (objects, living beings, nature, abstract concepts)
- **Hybrid Embeddings**: 832-dimensional vectors combining visual features (768-dim ViT) and subject encoding (64-dim)
- **Age-Based Modeling**: Separate subject-aware models trained for different age groups
- **Anomaly Detection**: Reconstruction loss-based scoring with subject-contextualized thresholds
- **Interactive Interpretability**: Subject-aware saliency analysis with hoverable regions, zoom/pan, and subject-specific explanations guaranteed for all drawings
- **Web Interface**: Modern React frontend with Material-UI components, subject category selection, and real-time configuration management
- **REST API**: FastAPI backend with automatic OpenAPI documentation and subject-aware endpoints
- **Enhanced Training**: Offline training system with verbose progress indicators and comprehensive subject-aware model management
- **Real-time Dashboard**: Dynamic anomaly count updates when threshold percentiles change, subject distribution analytics
- **Optimized Performance**: Fast threshold recalculation using existing analysis results with subject context

## Target Users

- Researchers studying child development patterns
- Educational professionals monitoring developmental milestones
- Healthcare providers screening for developmental concerns

## Key Technical Concepts

- **Vision Transformer (ViT)**: Used for visual feature extraction from drawings (768-dimensional)
- **Subject Encoding**: One-hot encoding of 64 predefined subject categories (64-dimensional)
- **Hybrid Embeddings**: Concatenation of visual and subject features (832-dimensional total)
- **Subject-Aware Autoencoder Architecture**: Reconstruction-based anomaly detection with subject context
- **Age-Group Stratification**: Models trained separately for different age ranges with subject awareness
- **Subject-Contextualized Thresholds**: Configurable percentile-based anomaly thresholds with subject-specific adjustments
- **Dynamic Recalculation**: Optimized threshold updates using existing analysis results with subject context
- **Subject-Aware Interpretability Engine**: Generates visual explanations using simplified gradient-based saliency maps with subject-specific comparisons and guaranteed availability for all drawings