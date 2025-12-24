#!/usr/bin/env python3
"""
Test API connection and check drawing count.
"""

import requests
import json

API_BASE_URL = "http://localhost:8000/api/v1"

def test_api_connection():
    """Test if the API is accessible."""
    try:
        # Test root endpoint
        response = requests.get("http://localhost:8000/")
        response.raise_for_status()
        print("✓ Backend server is running")
        print(f"  Response: {response.json()}")
        
        # Test drawings endpoint
        response = requests.get(f"{API_BASE_URL}/drawings/?page_size=5")
        response.raise_for_status()
        data = response.json()
        print(f"✓ Drawings API is accessible")
        print(f"  Total drawings: {data.get('total_count', 'unknown')}")
        print(f"  Sample response structure: {list(data.keys())}")
        
        if data.get("drawings"):
            sample_drawing = data["drawings"][0]
            print(f"  Sample drawing keys: {list(sample_drawing.keys())}")
        
        return True
        
    except requests.ConnectionError:
        print("✗ Cannot connect to backend server")
        print("  Make sure the server is running:")
        print("  source venv/bin/activate")
        print("  uvicorn app.main:app --reload")
        return False
        
    except requests.RequestException as e:
        print(f"✗ API request failed: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()