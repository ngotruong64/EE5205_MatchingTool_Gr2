import cv2
import os
import sys
import numpy as np
from services.template_matching_service_v2 import process_template_matching

# Setup Paths
DATA_DIR = r'C:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\Data'
TEMPLATE_DIR = os.path.join(DATA_DIR, 'template')
IMG_NAME = 'img004.jpg' 
TEMPLATE_NAME = 'template004.jpg'

img_path = os.path.join(DATA_DIR, IMG_NAME)
template_path = os.path.join(TEMPLATE_DIR, TEMPLATE_NAME)

if not os.path.exists(img_path) or not os.path.exists(template_path):
    print(f"Error: {img_path} or {template_path} not found.")
    sys.exit(1)

# Load Images
with open(img_path, 'rb') as f:
    img_bytes = f.read()
with open(template_path, 'rb') as f:
    template_bytes = f.read()

print(f"Running matching on {IMG_NAME} with {TEMPLATE_NAME}...")

try:
    # Run matching
    results, _ = process_template_matching(
        img_bytes, 
        template_bytes, 
        threshold=0.5, 
        edge_base=True,
        check_overlap=False
    )
    
    print(f"Found {results['count']} matches.")
    for i, match in enumerate(results['matches']):
        print(f"Match {i+1}: {match}")

except Exception as e:
    print(f"Error: {e}")
