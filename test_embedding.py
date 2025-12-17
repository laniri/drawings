#!/usr/bin/env python3
"""
Test script to debug and fix the embedding generation issue.
"""

import sys
import os
sys.path.append('.')

from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel

def test_vit_processing():
    """Test ViT processing with different approaches."""
    
    # Load a sample image
    image_path = "sample_drawings/age_7.7_house_04.png"
    if not os.path.exists(image_path):
        print(f"Sample image not found: {image_path}")
        return
    
    image = Image.open(image_path)
    print(f"Original image: {image.size}, mode: {image.mode}")
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to expected size
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    print(f"Processed image: {image.size}, mode: {image.mode}")
    
    try:
        # Load ViT processor and model
        print("Loading ViT processor and model...")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        print("Testing different processing approaches...")
        
        # Approach 1: Single image
        try:
            inputs = processor(image, return_tensors="pt")
            print("✓ Approach 1 (single image) works")
            print(f"  Input shape: {inputs['pixel_values'].shape}")
        except Exception as e:
            print(f"✗ Approach 1 failed: {e}")
        
        # Approach 2: List of images
        try:
            inputs = processor([image], return_tensors="pt")
            print("✓ Approach 2 (list of images) works")
            print(f"  Input shape: {inputs['pixel_values'].shape}")
        except Exception as e:
            print(f"✗ Approach 2 failed: {e}")
        
        # Approach 3: With explicit parameters
        try:
            inputs = processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                size={"height": 224, "width": 224}
            )
            print("✓ Approach 3 (explicit parameters) works")
            print(f"  Input shape: {inputs['pixel_values'].shape}")
            
            # Test model inference
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
                print(f"  Embeddings shape: {embeddings.shape}")
                
                # Get CLS token embedding (first token)
                cls_embedding = embeddings[:, 0, :]
                print(f"  CLS embedding shape: {cls_embedding.shape}")
                
        except Exception as e:
            print(f"✗ Approach 3 failed: {e}")
            
    except Exception as e:
        print(f"Failed to load model: {e}")

if __name__ == "__main__":
    test_vit_processing()