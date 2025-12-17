#!/usr/bin/env python3
"""
Create sample drawings for testing the anomaly detection system.
This script generates simple synthetic drawings to demonstrate the training process.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_sample_drawing(age_years, drawing_type="house", size=(224, 224)):
    """Create a simple synthetic drawing based on age and type."""
    
    # Create white background
    img = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(img)
    
    # Age-based complexity
    if age_years < 5:
        # Simple shapes for younger children
        if drawing_type == "house":
            # Simple square house
            draw.rectangle([50, 100, 150, 180], outline='black', width=2)
            draw.polygon([(40, 100), (100, 50), (160, 100)], outline='black', width=2)  # roof
            draw.rectangle([80, 140, 120, 180], outline='black', width=2)  # door
        elif drawing_type == "person":
            # Stick figure
            draw.ellipse([90, 50, 130, 90], outline='black', width=2)  # head
            draw.line([110, 90, 110, 150], fill='black', width=2)  # body
            draw.line([110, 110, 80, 130], fill='black', width=2)  # left arm
            draw.line([110, 110, 140, 130], fill='black', width=2)  # right arm
            draw.line([110, 150, 90, 180], fill='black', width=2)  # left leg
            draw.line([110, 150, 130, 180], fill='black', width=2)  # right leg
            
    elif age_years < 8:
        # More detailed for middle age
        if drawing_type == "house":
            # House with windows
            draw.rectangle([40, 90, 180, 190], outline='black', width=2)
            draw.polygon([(30, 90), (110, 40), (190, 90)], outline='black', width=2)  # roof
            draw.rectangle([60, 150, 90, 190], outline='black', width=2)  # door
            draw.rectangle([110, 120, 140, 150], outline='black', width=2)  # window
            draw.rectangle([150, 120, 170, 150], outline='black', width=2)  # window
        elif drawing_type == "person":
            # Person with more details
            draw.ellipse([85, 40, 135, 90], outline='black', width=2)  # head
            draw.rectangle([100, 90, 120, 140], outline='black', width=2)  # body
            draw.line([100, 100, 70, 120], fill='black', width=2)  # left arm
            draw.line([120, 100, 150, 120], fill='black', width=2)  # right arm
            draw.line([105, 140, 85, 180], fill='black', width=2)  # left leg
            draw.line([115, 140, 135, 180], fill='black', width=2)  # right leg
            # Face features
            draw.ellipse([95, 55, 105, 65], fill='black')  # left eye
            draw.ellipse([115, 55, 125, 65], fill='black')  # right eye
            draw.arc([100, 70, 120, 80], 0, 180, fill='black', width=2)  # smile
            
    else:
        # Complex drawings for older children
        if drawing_type == "house":
            # Detailed house
            draw.rectangle([30, 80, 190, 200], outline='black', width=2)
            draw.polygon([(20, 80), (110, 20), (200, 80)], outline='black', width=2)  # roof
            draw.rectangle([50, 150, 80, 200], outline='black', width=2)  # door
            draw.ellipse([62, 170, 68, 176], fill='black')  # door knob
            # Multiple windows
            draw.rectangle([100, 110, 130, 140], outline='black', width=2)
            draw.line([115, 110, 115, 140], fill='black', width=1)  # window cross
            draw.line([100, 125, 130, 125], fill='black', width=1)
            draw.rectangle([150, 110, 180, 140], outline='black', width=2)
            draw.line([165, 110, 165, 140], fill='black', width=1)
            draw.line([150, 125, 180, 125], fill='black', width=1)
            # Chimney
            draw.rectangle([140, 40, 160, 80], outline='black', width=2)
        elif drawing_type == "person":
            # Detailed person
            draw.ellipse([80, 30, 140, 90], outline='black', width=2)  # head
            draw.rectangle([95, 90, 125, 150], outline='black', width=2)  # body
            draw.line([95, 105, 60, 130], fill='black', width=3)  # left arm
            draw.line([125, 105, 160, 130], fill='black', width=3)  # right arm
            draw.line([100, 150, 80, 200], fill='black', width=3)  # left leg
            draw.line([120, 150, 140, 200], fill='black', width=3)  # right leg
            # Detailed face
            draw.ellipse([90, 50, 100, 60], fill='black')  # left eye
            draw.ellipse([120, 50, 130, 60], fill='black')  # right eye
            draw.ellipse([107, 65, 113, 71], fill='black')  # nose
            draw.arc([95, 75, 125, 85], 0, 180, fill='black', width=2)  # smile
            # Hair
            draw.arc([80, 30, 140, 70], 180, 360, fill='black', width=2)
    
    # Add some random variation
    if random.random() > 0.7:
        # Add some random lines for variation
        for _ in range(random.randint(1, 3)):
            x1, y1 = random.randint(0, size[0]), random.randint(0, size[1])
            x2, y2 = random.randint(0, size[0]), random.randint(0, size[1])
            draw.line([x1, y1, x2, y2], fill='gray', width=1)
    
    return img

def main():
    """Generate sample drawings for different age groups."""
    
    os.makedirs('sample_drawings', exist_ok=True)
    
    # Age groups and their sample counts
    age_groups = [
        (3.5, 4.5, 15),   # 3-4 years: 15 samples
        (5.0, 6.0, 20),   # 5-6 years: 20 samples  
        (7.0, 8.0, 25),   # 7-8 years: 25 samples
        (9.0, 10.0, 20),  # 9-10 years: 20 samples
        (11.0, 12.0, 15), # 11-12 years: 15 samples
    ]
    
    drawing_types = ["house", "person"]
    
    total_created = 0
    
    for age_min, age_max, count in age_groups:
        print(f"Creating {count} drawings for age group {age_min}-{age_max}...")
        
        for i in range(count):
            # Random age within the group
            age = random.uniform(age_min, age_max)
            
            # Random drawing type
            drawing_type = random.choice(drawing_types)
            
            # Create the drawing
            img = create_sample_drawing(age, drawing_type)
            
            # Save with descriptive filename
            filename = f"age_{age:.1f}_{drawing_type}_{i+1:02d}.png"
            filepath = os.path.join('sample_drawings', filename)
            img.save(filepath)
            
            total_created += 1
    
    print(f"\nCreated {total_created} sample drawings in 'sample_drawings/' directory")
    print("\nTo upload these drawings:")
    print("1. Use the web interface at http://localhost:5173")
    print("2. Or use the upload script that will be created next")

if __name__ == "__main__":
    main()