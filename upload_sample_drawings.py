#!/usr/bin/env python3
"""
Upload sample drawings to the anomaly detection system.
"""

import os
import requests
import re
from pathlib import Path

def upload_drawing(filepath, age_years, subject, expert_label="normal"):
    """Upload a single drawing to the system."""
    
    url = "http://localhost:8000/api/v1/drawings/upload"
    
    # Prepare the file
    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f, 'image/png')}
        data = {
            'age_years': age_years,
            'subject': subject,
            'expert_label': expert_label
        }
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error uploading {filepath}: {e}")
            return None

def main():
    """Upload all sample drawings."""
    
    sample_dir = Path('/Users/itay/Desktop/stringent_cleaned_dataset')
    print(sample_dir)
    
    if not sample_dir.exists():
        print("Sample drawings directory not found. Run create_sample_drawings.py first.")
        return
    
    uploaded_count = 0
    failed_count = 0
    
    # Get all PNG files
    png_files = list(sample_dir.glob('*.png'))
    
    print(f"Found {len(png_files)} drawings to upload...")
    
    for filepath in png_files:
        # Parse filename to extract metadata
        # Format: age_X.X_subject_NN.png
        filename = filepath.stem
        
        # Extract age from filename
        age_match = re.search(r'_age(\d+)', filename)
        if not age_match:
            print(f"Could not parse age from {filename}")
            failed_count += 1
            continue
        
        age_years = float(age_match.group(1))
        
        # Extract subject from filename
        subject_match = re.search(r'^([^_]+)', filename)
        if not subject_match:
            print(f"Could not parse subject from {filename}")
            failed_count += 1
            continue
        
        subject = subject_match.group(1)
        
        print(f"Uploading {filename} (age: {age_years}, subject: {subject})...")
        
        result = upload_drawing(filepath, age_years, subject)
        
        if result:
            print(f"  ✓ Uploaded successfully (ID: {result.get('id')})")
            uploaded_count += 1
        else:
            print(f"  ✗ Upload failed")
            failed_count += 1
    
    print(f"\nUpload complete:")
    print(f"  Successfully uploaded: {uploaded_count}")
    print(f"  Failed: {failed_count}")
    
    if uploaded_count > 0:
        print(f"\nNext steps:")
        print(f"1. Check system status: curl http://localhost:8000/api/v1/models/status")
        print(f"2. Auto-create age group models: curl -X POST http://localhost:8000/api/v1/models/auto-create")
        print(f"3. Check model training progress in the web interface")

if __name__ == "__main__":
    main()