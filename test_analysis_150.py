#!/usr/bin/env python3
"""
Test script to regenerate analysis for drawing 37682 (analysis ID 150)
to see if current logic produces better results.
"""

import requests
import json

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
DRAWING_ID = 37682

def regenerate_analysis():
    """Regenerate analysis for drawing 37682."""
    try:
        # Submit analysis request with force reanalysis
        payload = {
            "drawing_id": DRAWING_ID,
            "force_reanalysis": True
        }
        
        print(f"Regenerating analysis for drawing {DRAWING_ID}...")
        response = requests.post(f"{API_BASE_URL}/analysis", json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("New analysis result:")
        print(json.dumps(result, indent=2))
        
        # Extract key metrics
        analysis = result.get("analysis", {})
        print(f"\nKey metrics:")
        print(f"  Anomaly score: {analysis.get('anomaly_score')}")
        print(f"  Normalized score: {analysis.get('normalized_score')}")
        print(f"  Is anomaly: {analysis.get('is_anomaly')}")
        print(f"  Confidence: {analysis.get('confidence')}")
        print(f"  Attribution: {analysis.get('anomaly_attribution')}")
        
        # Check explanation text
        interpretability = result.get("interpretability", {})
        explanation = interpretability.get("explanation_text", "")
        print(f"\nExplanation text:")
        print(f"  {explanation}")
        
        return result
        
    except requests.RequestException as e:
        print(f"Error regenerating analysis: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None

if __name__ == "__main__":
    regenerate_analysis()