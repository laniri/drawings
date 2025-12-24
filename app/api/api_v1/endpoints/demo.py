"""
Demo endpoints for displaying sample content and project information.

Provides public access to pre-analyzed sample drawings with complete results,
project descriptions, and technical documentation links.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse

from app.schemas.common import SuccessResponse
from app.services.demo_service import DemoService, get_demo_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def get_demo_page(
    demo_service: DemoService = Depends(get_demo_service),
) -> HTMLResponse:
    """
    Get the complete demo page with all content.

    Returns:
        HTML response with complete demo page content
    """
    try:
        # Get all demo content
        samples = demo_service.get_demo_samples()
        project_description = demo_service.get_project_description()
        medical_disclaimer = demo_service.get_medical_disclaimer()
        technical_links = demo_service.get_technical_links()
        statistics = demo_service.get_demo_statistics()

        # Generate HTML content
        html_content = _generate_demo_html(
            samples=samples,
            project_description=project_description,
            medical_disclaimer=medical_disclaimer,
            technical_links=technical_links,
            statistics=statistics,
        )

        logger.info("Generated demo page successfully")
        return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"Error generating demo page: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate demo page")


@router.get("/samples", response_model=SuccessResponse)
async def get_demo_samples(
    demo_service: DemoService = Depends(get_demo_service),
) -> SuccessResponse:
    """
    Get all demo samples with analysis results.

    Returns:
        List of demo samples with complete analysis data
    """
    try:
        samples = demo_service.get_demo_samples()

        logger.info(f"Retrieved {len(samples)} demo samples")
        return SuccessResponse(
            message=f"Retrieved {len(samples)} demo samples", data={"samples": samples}
        )

    except Exception as e:
        logger.error(f"Error retrieving demo samples: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve demo samples")


@router.get("/samples/{sample_id}", response_model=SuccessResponse)
async def get_demo_sample(
    sample_id: int, demo_service: DemoService = Depends(get_demo_service)
) -> SuccessResponse:
    """
    Get a specific demo sample by ID.

    Args:
        sample_id: ID of the demo sample

    Returns:
        Demo sample with complete analysis data
    """
    try:
        sample = demo_service.get_demo_sample(sample_id)

        if not sample:
            raise HTTPException(
                status_code=404, detail=f"Demo sample {sample_id} not found"
            )

        logger.info(f"Retrieved demo sample {sample_id}")
        return SuccessResponse(
            message=f"Retrieved demo sample {sample_id}", data=sample
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving demo sample {sample_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve demo sample")


@router.get("/project-info", response_model=SuccessResponse)
async def get_project_info(
    demo_service: DemoService = Depends(get_demo_service),
) -> SuccessResponse:
    """
    Get comprehensive project information for demo page.

    Returns:
        Project description with technical details and features
    """
    try:
        project_info = demo_service.get_project_description()

        logger.info("Retrieved project information")
        return SuccessResponse(
            message="Retrieved project information", data=project_info
        )

    except Exception as e:
        logger.error(f"Error retrieving project info: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve project information"
        )


@router.get("/disclaimer", response_model=SuccessResponse)
async def get_medical_disclaimer(
    demo_service: DemoService = Depends(get_demo_service),
) -> SuccessResponse:
    """
    Get medical disclaimer and warnings for demo content.

    Returns:
        Medical disclaimer with all required warnings
    """
    try:
        disclaimer = demo_service.get_medical_disclaimer()

        logger.info("Retrieved medical disclaimer")
        return SuccessResponse(message="Retrieved medical disclaimer", data=disclaimer)

    except Exception as e:
        logger.error(f"Error retrieving medical disclaimer: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve medical disclaimer"
        )


@router.get("/technical-links", response_model=SuccessResponse)
async def get_technical_links(
    demo_service: DemoService = Depends(get_demo_service),
) -> SuccessResponse:
    """
    Get technical links and documentation references.

    Returns:
        Technical links including GitHub repository and documentation
    """
    try:
        links = demo_service.get_technical_links()

        logger.info("Retrieved technical links")
        return SuccessResponse(message="Retrieved technical links", data=links)

    except Exception as e:
        logger.error(f"Error retrieving technical links: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve technical links"
        )


@router.get("/statistics", response_model=SuccessResponse)
async def get_demo_statistics(
    demo_service: DemoService = Depends(get_demo_service),
) -> SuccessResponse:
    """
    Get demo-specific statistics and metrics.

    Returns:
        Demo statistics including sample counts and distributions
    """
    try:
        stats = demo_service.get_demo_statistics()

        logger.info("Retrieved demo statistics")
        return SuccessResponse(message="Retrieved demo statistics", data=stats)

    except Exception as e:
        logger.error(f"Error retrieving demo statistics: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve demo statistics"
        )


def _generate_demo_html(
    samples: List[Dict[str, Any]],
    project_description: Dict[str, Any],
    medical_disclaimer: Dict[str, Any],
    technical_links: Dict[str, Any],
    statistics: Dict[str, Any],
) -> str:
    """
    Generate complete HTML content for demo page.

    Args:
        samples: Demo samples with analysis results
        project_description: Project information
        medical_disclaimer: Medical disclaimer content
        technical_links: Technical documentation links
        statistics: Demo statistics

    Returns:
        Complete HTML content for demo page
    """

    # Generate sample cards HTML
    sample_cards = ""
    for sample in samples:
        anomaly_class = (
            "anomaly"
            if sample.get("analysis_result", {}).get("is_anomaly", False)
            else "normal"
        )
        anomaly_badge = (
            "🔍 Anomaly Detected"
            if sample.get("analysis_result", {}).get("is_anomaly", False)
            else "✅ Normal Pattern"
        )

        sample_cards += f"""
        <div class="sample-card {anomaly_class}">
            <div class="sample-header">
                <h3>{sample.get('title', 'Sample Drawing')}</h3>
                <span class="anomaly-badge {anomaly_class}">{anomaly_badge}</span>
            </div>
            
            <div class="sample-images">
                <div class="image-container">
                    <img src="{sample.get('original_image', '')}" alt="Original Drawing" class="sample-image">
                    <label>Original Drawing</label>
                </div>
                <div class="image-container">
                    <img src="{sample.get('saliency_map', '')}" alt="Saliency Map" class="sample-image">
                    <label>Interpretability Map</label>
                </div>
            </div>
            
            <div class="sample-details">
                <p class="description">{sample.get('description', '')}</p>
                
                <div class="analysis-results">
                    <div class="metric">
                        <span class="label">Age Group:</span>
                        <span class="value">{sample.get('age_group', 'Unknown')}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Anomaly Score:</span>
                        <span class="value">{sample.get('analysis_result', {}).get('anomaly_score', 0):.3f}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Confidence:</span>
                        <span class="value">{sample.get('analysis_result', {}).get('confidence', 0):.3f}</span>
                    </div>
                </div>
                
                <div class="interpretability-section">
                    <h4>🔍 AI Interpretation</h4>
                    <p class="interpretation">{sample.get('interpretability', {}).get('explanation', '')}</p>
                    
                    <div class="key-regions">
                        <h5>Key Regions Analyzed:</h5>
                        <ul>
        """

        # Add key regions
        for region in sample.get("interpretability", {}).get("key_regions", []):
            sample_cards += f"""
                            <li>
                                <strong>{region.get('region', '')}:</strong> 
                                {region.get('description', '')} 
                                <span class="importance">(Importance: {region.get('importance', 0):.2f})</span>
                            </li>
            """

        sample_cards += """
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        """

    # Generate statistics HTML
    stats_html = f"""
    <div class="demo-statistics">
        <div class="stat-item">
            <span class="stat-number">{statistics.get('total_samples', 0)}</span>
            <span class="stat-label">Demo Samples</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">{statistics.get('normal_samples', 0)}</span>
            <span class="stat-label">Normal Patterns</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">{statistics.get('anomaly_samples', 0)}</span>
            <span class="stat-label">Anomalies Detected</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">{statistics.get('interpretability_coverage', '100%')}</span>
            <span class="stat-label">Interpretability Coverage</span>
        </div>
    </div>
    """

    # Complete HTML template
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{project_description.get('title', 'Demo')} - Interactive Demo</title>
        <style>
            {_get_demo_css()}
        </style>
    </head>
    <body>
        <div class="demo-container">
            <!-- Medical Disclaimer -->
            <div class="medical-disclaimer">
                <h2>{medical_disclaimer.get('title', 'Medical Disclaimer')}</h2>
                <p class="primary-warning">{medical_disclaimer.get('primary_warning', '')}</p>
                <div class="disclaimer-details">
    """

    for detail in medical_disclaimer.get("detailed_disclaimer", []):
        html_content += f"<p>• {detail}</p>"

    html_content += f"""
                </div>
                <div class="recommendations">
                    <h4>Recommendations:</h4>
    """

    for rec in medical_disclaimer.get("recommendations", []):
        html_content += f"<p>• {rec}</p>"

    html_content += f"""
                </div>
            </div>
            
            <!-- Project Description -->
            <div class="project-description">
                <h1>{project_description.get('title', 'Demo System')}</h1>
                <h2>{project_description.get('subtitle', '')}</h2>
                <p class="overview">{project_description.get('overview', '')}</p>
                
                <div class="features-section">
                    <h3>🚀 Key Features</h3>
                    <ul class="features-list">
    """

    for feature in project_description.get("key_features", []):
        html_content += f"<li>{feature}</li>"

    html_content += f"""
                    </ul>
                </div>
                
                <div class="technical-approach">
                    <h3>🔬 Technical Approach</h3>
                    <div class="tech-grid">
                        <div class="tech-item">
                            <h4>Feature Extraction</h4>
                            <p>{project_description.get('technical_approach', {}).get('feature_extraction', '')}</p>
                        </div>
                        <div class="tech-item">
                            <h4>Anomaly Detection</h4>
                            <p>{project_description.get('technical_approach', {}).get('anomaly_detection', '')}</p>
                        </div>
                        <div class="tech-item">
                            <h4>Interpretability</h4>
                            <p>{project_description.get('technical_approach', {}).get('interpretability', '')}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Demo Statistics -->
            <div class="statistics-section">
                <h3>📊 Demo Overview</h3>
                {stats_html}
            </div>
            
            <!-- Demo Samples -->
            <div class="samples-section">
                <h3>🎨 Interactive Demo Samples</h3>
                <p class="samples-intro">
                    Explore these pre-analyzed sample drawings to understand how our AI system works. 
                    Each sample includes the original drawing, interpretability visualization, and detailed analysis results.
                </p>
                
                <div class="samples-grid">
                    {sample_cards}
                </div>
            </div>
            
            <!-- Technical Links -->
            <div class="technical-links">
                <h3>🔗 Technical Resources</h3>
                <div class="links-grid">
                    <a href="{technical_links.get('github', {}).get('url', '#')}" target="_blank" class="tech-link github">
                        <span class="link-title">{technical_links.get('github', {}).get('title', 'GitHub')}</span>
                        <span class="link-desc">{technical_links.get('github', {}).get('description', '')}</span>
                    </a>
                    <a href="{technical_links.get('documentation', {}).get('url', '#')}" class="tech-link docs">
                        <span class="link-title">{technical_links.get('documentation', {}).get('title', 'Documentation')}</span>
                        <span class="link-desc">{technical_links.get('documentation', {}).get('description', '')}</span>
                    </a>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="demo-footer">
                <p>This is a demonstration system for educational and research purposes only.</p>
                <p>Current Status: {project_description.get('current_status', {}).get('training_data', 'Active Development')}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html_content


def _get_demo_css() -> str:
    """Get CSS styles for demo page."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .demo-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Medical Disclaimer */
        .medical-disclaimer {
            background: #ffebee;
            border: 3px solid #f44336;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 4px 12px rgba(244, 67, 54, 0.2);
        }
        
        .medical-disclaimer h2 {
            color: #c62828;
            font-size: 1.5em;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .primary-warning {
            font-weight: bold;
            font-size: 1.1em;
            color: #c62828;
            margin-bottom: 16px;
            padding: 12px;
            background: rgba(244, 67, 54, 0.1);
            border-radius: 4px;
        }
        
        .disclaimer-details p, .recommendations p {
            margin-bottom: 8px;
            color: #666;
        }
        
        /* Project Description */
        .project-description {
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .project-description h1 {
            color: #1976d2;
            font-size: 2.5em;
            margin-bottom: 8px;
        }
        
        .project-description h2 {
            color: #666;
            font-size: 1.2em;
            font-weight: normal;
            margin-bottom: 24px;
        }
        
        .overview {
            font-size: 1.1em;
            line-height: 1.8;
            margin-bottom: 32px;
            color: #555;
        }
        
        .features-list {
            list-style: none;
            padding-left: 0;
        }
        
        .features-list li {
            padding: 8px 0;
            padding-left: 24px;
            position: relative;
        }
        
        .features-list li:before {
            content: "✓";
            position: absolute;
            left: 0;
            color: #4caf50;
            font-weight: bold;
        }
        
        .tech-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-top: 16px;
        }
        
        .tech-item {
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #1976d2;
        }
        
        .tech-item h4 {
            color: #1976d2;
            margin-bottom: 8px;
        }
        
        /* Statistics */
        .statistics-section {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .demo-statistics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 24px;
            margin-top: 16px;
        }
        
        .stat-item {
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stat-number {
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #1976d2;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Samples */
        .samples-section {
            background: white;
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .samples-intro {
            color: #666;
            margin-bottom: 24px;
            font-size: 1.1em;
        }
        
        .samples-grid {
            display: grid;
            gap: 32px;
        }
        
        .sample-card {
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 24px;
            background: #fafafa;
        }
        
        .sample-card.anomaly {
            border-color: #ff9800;
            background: #fff8e1;
        }
        
        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        
        .anomaly-badge {
            padding: 4px 12px;
            border-radius: 16px;
            font-size: 0.85em;
            font-weight: bold;
        }
        
        .anomaly-badge.normal {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .anomaly-badge.anomaly {
            background: #fff3e0;
            color: #f57c00;
        }
        
        .sample-images {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        
        .image-container {
            text-align: center;
        }
        
        .sample-image {
            width: 100%;
            max-width: 300px;
            height: 200px;
            object-fit: contain;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
        }
        
        .image-container label {
            display: block;
            margin-top: 8px;
            font-weight: bold;
            color: #666;
        }
        
        .description {
            font-style: italic;
            color: #666;
            margin-bottom: 16px;
        }
        
        .analysis-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }
        
        .metric .label {
            font-weight: bold;
            color: #666;
        }
        
        .metric .value {
            color: #1976d2;
            font-weight: bold;
        }
        
        .interpretability-section {
            border-top: 1px solid #e0e0e0;
            padding-top: 16px;
        }
        
        .interpretability-section h4 {
            color: #1976d2;
            margin-bottom: 12px;
        }
        
        .interpretation {
            background: white;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #1976d2;
            margin-bottom: 16px;
        }
        
        .key-regions ul {
            list-style: none;
            padding-left: 0;
        }
        
        .key-regions li {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .importance {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Technical Links */
        .technical-links {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .links-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }
        
        .tech-link {
            display: block;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            text-decoration: none;
            color: inherit;
            border: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tech-link:hover {
            border-color: #1976d2;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .link-title {
            display: block;
            font-weight: bold;
            color: #1976d2;
            margin-bottom: 8px;
        }
        
        .link-desc {
            color: #666;
            font-size: 0.9em;
        }
        
        /* Footer */
        .demo-footer {
            text-align: center;
            padding: 24px;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .demo-container {
                padding: 16px;
            }
            
            .sample-images {
                grid-template-columns: 1fr;
            }
            
            .analysis-results {
                grid-template-columns: 1fr;
            }
            
            .project-description h1 {
                font-size: 2em;
            }
        }
    """
