#!/usr/bin/env python3
"""
Script to identify and fix analyses with suspicious normalized scores (exactly 50.0)
that may have been affected by boundary case issues.
"""

import sqlite3
import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

def get_problematic_analyses():
    """Get analyses with exactly 50.0 normalized scores."""
    conn = sqlite3.connect('drawings.db')
    cursor = conn.cursor()
    
    # Find analyses with exactly 50.0 normalized score
    cursor.execute("""
        SELECT a.id, a.drawing_id, a.anomaly_score, a.normalized_score, 
               a.is_anomaly, a.confidence, a.anomaly_attribution,
               d.age_years, d.subject
        FROM anomaly_analyses a
        JOIN drawings d ON a.drawing_id = d.id
        WHERE a.normalized_score = 50.0
        ORDER BY a.id
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    return results

def analyze_drawing_batch(drawing_ids, force_reanalysis=True):
    """Submit batch analysis for specific drawings."""
    payload = {
        "drawing_ids": drawing_ids,
        "force_reanalysis": force_reanalysis
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/analysis/batch", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["batch_id"]
    except requests.RequestException as e:
        print(f"Error submitting batch analysis: {e}")
        return None

def check_batch_progress(batch_id):
    """Check batch analysis progress."""
    try:
        response = requests.get(f"{API_BASE_URL}/analysis/batch/{batch_id}/progress")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error checking batch progress: {e}")
        return {}

def wait_for_batch_completion(batch_id, check_interval=5):
    """Wait for batch to complete."""
    print(f"Waiting for batch {batch_id} to complete...")
    
    while True:
        progress = check_batch_progress(batch_id)
        if not progress:
            print("Failed to get batch progress")
            return {}
        
        status = progress.get("status", "unknown")
        completed = progress.get("completed", 0)
        failed = progress.get("failed", 0)
        total = progress.get("total_drawings", 0)
        
        print(f"  Status: {status} | Completed: {completed}/{total} | Failed: {failed}")
        
        if status == "completed":
            print("Batch analysis completed!")
            return progress
        elif status.startswith("failed"):
            print(f"Batch analysis failed: {status}")
            return progress
        
        time.sleep(check_interval)

def main():
    """Main function to fix problematic analyses."""
    print("Identifying analyses with suspicious normalized scores...")
    
    problematic = get_problematic_analyses()
    
    if not problematic:
        print("No problematic analyses found.")
        return
    
    print(f"Found {len(problematic)} analyses with exactly 50.0 normalized score:")
    print("\nCurrent state:")
    for analysis in problematic:
        analysis_id, drawing_id, anomaly_score, norm_score, is_anomaly, confidence, attribution, age, subject = analysis
        print(f"  Analysis {analysis_id} (Drawing {drawing_id}): age={age}, subject={subject}")
        print(f"    Score: {anomaly_score:.6f}, Normalized: {norm_score}, Anomaly: {bool(is_anomaly)}")
        print(f"    Confidence: {confidence:.3f}, Attribution: {attribution}")
        print()
    
    # Extract drawing IDs for re-analysis
    drawing_ids = [analysis[1] for analysis in problematic]  # drawing_id is at index 1
    
    print(f"Re-analyzing {len(drawing_ids)} drawings with corrected logic...")
    
    # Submit batch analysis
    batch_id = analyze_drawing_batch(drawing_ids, force_reanalysis=True)
    if not batch_id:
        print("Failed to submit batch analysis")
        return
    
    print(f"Submitted batch analysis: {batch_id}")
    
    # Wait for completion
    result = wait_for_batch_completion(batch_id)
    
    if result.get("status") == "completed":
        print(f"\n✅ Successfully re-analyzed {result.get('completed', 0)} drawings")
        print("The analyses should now have corrected normalized scores, confidence levels, and explanations.")
        print("\nYou can now check the updated results:")
        for analysis in problematic[:3]:  # Show first 3 as examples
            analysis_id = analysis[0]
            print(f"  http://localhost:5173/analysis/{analysis_id}")
    else:
        print(f"\n❌ Batch analysis failed or incomplete")
        print(f"Status: {result.get('status', 'unknown')}")

if __name__ == "__main__":
    main()