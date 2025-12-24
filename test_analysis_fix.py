#!/usr/bin/env python3
"""
Test script to re-analyze a specific drawing and verify the fixes.
"""

import requests
import json
import sys

API_BASE_URL = "http://localhost:8000/api/v1"

def test_analysis_fix(drawing_id: int = 1):
    """Test the analysis fix for a specific drawing."""
    
    print(f"Testing analysis fix for drawing ID {drawing_id}")
    
    try:
        # Force re-analysis of the drawing
        print("1. Triggering re-analysis...")
        # The endpoint expects an AnalysisRequest object with both drawing_id and force_reanalysis
        response = requests.post(f"{API_BASE_URL}/analysis/analyze/{drawing_id}", 
                               json={"drawing_id": drawing_id, "force_reanalysis": True})
        
        if response.status_code != 200:
            print(f"  Response status: {response.status_code}")
            print(f"  Response text: {response.text}")
            # Try without the request body (using default force_reanalysis=False)
            print("  Trying without request body...")
            response = requests.post(f"{API_BASE_URL}/analysis/analyze/{drawing_id}")
            if response.status_code != 200:
                print(f"  Still failed: {response.status_code}")
                print(f"  Response text: {response.text}")
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Re-analysis completed")
        print(f"  New analysis ID: {result['analysis']['id']}")
        
        # Get the analysis results
        analysis_id = result['analysis']['id']
        analysis = result['analysis']
        
        print(f"\n2. Analysis Results:")
        print(f"  Drawing ID: {analysis['drawing_id']}")
        print(f"  Is Anomaly: {analysis['is_anomaly']}")
        print(f"  Anomaly Score: {analysis['anomaly_score']:.6f}")
        print(f"  Normalized Score: {analysis['normalized_score']:.2f}")
        print(f"  Confidence: {analysis['confidence']:.4f} ({analysis['confidence']*100:.1f}%)")
        print(f"  Attribution: {analysis.get('anomaly_attribution', 'None')}")
        print(f"  Analysis Type: {analysis.get('analysis_type', 'Unknown')}")
        
        # Debug: Check if this is actually a new analysis
        print(f"  Analysis Timestamp: {analysis['analysis_timestamp']}")
        
        # Also check the raw API response to see if there are any issues
        print(f"\n  Raw API Response Keys: {list(result.keys())}")
        print(f"  Analysis Keys: {list(analysis.keys())}")
        
        # Check for contradictions
        print(f"\n3. Contradiction Check:")
        contradictions = []
        
        # Check attribution logic
        if not analysis['is_anomaly'] and analysis.get('anomaly_attribution') is not None:
            contradictions.append(f"Normal drawing has attribution: {analysis['anomaly_attribution']}")
        
        if analysis['is_anomaly'] and analysis.get('anomaly_attribution') is None:
            contradictions.append("Anomalous drawing has no attribution")
        
        # Check explanation text
        if 'interpretability' in result and result['interpretability']:
            explanation = result['interpretability']['explanation_text']
            normalized_score = analysis['normalized_score']
            
            if not analysis['is_anomaly']:
                if normalized_score < 40 and "low" not in explanation.lower():
                    contradictions.append(f"Low score ({normalized_score:.1f}) but explanation doesn't mention 'low'")
                elif 40 <= normalized_score < 60 and "moderate" not in explanation.lower():
                    contradictions.append(f"Moderate score ({normalized_score:.1f}) but explanation doesn't mention 'moderate'")
                elif normalized_score >= 60 and "elevated" not in explanation.lower():
                    contradictions.append(f"Elevated score ({normalized_score:.1f}) but explanation doesn't mention 'elevated'")
        
        if contradictions:
            print("  ✗ Contradictions found:")
            for contradiction in contradictions:
                print(f"    - {contradiction}")
        else:
            print("  ✓ No contradictions found!")
        
        # Test confidence endpoint
        print(f"\n4. Testing confidence endpoint...")
        try:
            response = requests.get(f"{API_BASE_URL}/interpretability/{analysis_id}/confidence")
            response.raise_for_status()
            confidence_data = response.json()
            
            print(f"  ✓ Confidence endpoint working")
            print(f"    Overall Confidence: {confidence_data['overall_confidence']:.3f}")
            print(f"    Model Certainty: {confidence_data['model_certainty']:.3f}")
            print(f"    Explanation Reliability: {confidence_data['explanation_reliability']:.3f}")
            print(f"    Data Sufficiency: {confidence_data['data_sufficiency']}")
            
            if confidence_data['warnings']:
                print(f"    Warnings: {len(confidence_data['warnings'])}")
                for warning in confidence_data['warnings']:
                    print(f"      - {warning}")
            
        except requests.RequestException as e:
            print(f"  ✗ Confidence endpoint failed: {e}")
        
        print(f"\n5. Summary:")
        if not contradictions:
            print("  ✓ Analysis results are consistent!")
            print("  ✓ Fixes appear to be working correctly")
        else:
            print("  ✗ Some issues remain - check the contradictions above")
        
        return len(contradictions) == 0
        
    except requests.RequestException as e:
        print(f"✗ API request failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test analysis fixes")
    parser.add_argument(
        "--drawing-id", 
        type=int, 
        default=1, 
        help="Drawing ID to test (default: 1)"
    )
    
    args = parser.parse_args()
    
    success = test_analysis_fix(args.drawing_id)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()