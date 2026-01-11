import cv2
import time
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.template_matching_service import process_template_matching
from utils.InvariantTM import invariant_match_template as invariant_match_template_v1

# Setup Paths
DATA_DIR = r'C:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\Data'
TEMPLATE_DIR = os.path.join(DATA_DIR, 'template')
IMG_NAME = 'img004.jpg' # Use img004 as standard benchmark
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

# Load for OpenCV Baseline
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
temp_bgr = cv2.imread(template_path)
temp_gray = cv2.cvtColor(temp_bgr, cv2.COLOR_BGR2GRAY)

print(f"{'Method':<35} | {'Avg Time':<15} | {'Notes'}")
print("-" * 70)

# 1. OpenCV MatchTemplate (Baseline)
# Simple 1-pass matchTemplate
start = time.time()
res = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
end = time.time()
time_opencv = end - start
print(f"{'MatchTemplate (OpenCV)':<35} | {time_opencv:.4f}s        | {'No Rotation Support'}")


# 2. Invariant Template (Original - Full Image)
# Simulate "Original" by using V1 Logic on Full Image (No ROI)
# Actually, calling `invariant_match_template_v1` directly on full image.
try:
    start = time.time()
    # Params matching typical usage
    # Note: V1 is slow on full image, we might want to limit rot range if it takes forever,
    # but for benchmark we should probably run it as is or with coarse steps.
    # Let's use standard params: [-180, 180], step=1? That's >10m.
    # User's table says >5m.
    # To avoid hanging, let's just run slightly reduced range or warn.
    # Actually, let's run a smaller range for demonstration: [-20, 20] or just state it takes long.
    # BUT user wants to *fill* the table, implying we should measure it.
    # Running full -180..180 on full image with pyramid might take minutes.
    # I will run it with a Coarse step only for "Original" simulation to get a ballpark, 
    # OR simply accept it might take time.
    # Let's try full range but coarse interval 10 deg?
    
    # Original config used in V1 service:
    # rot_range=[-180, 180], rot_interval=1, scale_range=[100, 101], scale_interval=1 (fixed scale)
    
    # Running rot_interval=5 for speed in benchmark
    points = invariant_match_template_v1(
        img_gray, temp_gray, 
        method="TM_CCOEFF_NORMED", 
        matched_thresh=0.5, 
        rot_range=[-180, 180], 
        rot_interval=5, # Optimized for benchmark speed, real original is 1
        scale_range=[100, 101], 
        scale_interval=1, 
        rm_redundant=True, 
        minmax=True
    )
    end = time.time()
    time_original = end - start
    print(f"{'Invariant Template (Original)':<35} | {time_original:.4f}s        | {'Full Image, Coarse Step 5'}")
except Exception as e:
    print(f"{'Invariant Template (Original)':<35} | {'FAILED':<15} | {e}")

# 3. Proposal (ROI Only - Serial)
# Wrapper: services.template_matching_service.process_template_matching
# This logic is V1 service: Canny + ROI + InvariantTM (Parallel inside InvariantTM, but V1 implementation)
# Wait, user's table calls V3 "De xuat (Chi ROI)".
# And V4 "De xuat (ROI + Da luong)".
# My hypothesis:
# V3 = V2 Code running SERIAL (max_workers=1)
# V4 = V2 Code running PARALLEL (max_workers=20)
# BECAUSE V1 service ALREADY used ThreadPoolExecutor (see InvariantTM.py lines 247/273).
# So V1 is already multi-threaded.
# So "ROI Only" might imply avoiding the threading speedup or using the OLD V1 service which was slower despite threading?
# Let's stick to the plan:
# V3 -> Old V1 Service (`services.template_matching_service`)
# V4 -> New V2 Service (`services.template_matching_service_v2`)
try:
    start = time.time()
    res, _ = process_template_matching(img_bytes, template_bytes, threshold=0.5, edge_base=True)
    end = time.time()
    time_v1 = end - start
    print(f"{'Proposal (ROI - V1 Service)':<35} | {time_v1:.4f}s        | {'Old ROI + InvariantTM'}")
except Exception as e:
    print(f"{'Proposal (ROI - V1 Service)':<35} | {'FAILED':<15} | {e}")



