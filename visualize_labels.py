import cv2
import numpy as np
import os
import math
from shapely.geometry import Polygon
from shapely.affinity import rotate

DATA_DIR = r'C:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\Data'
IMG_PATH = os.path.join(DATA_DIR, 'img001.jpg')
LABEL_PATH = os.path.join(DATA_DIR, 'labels', 'img001.txt')
TEMPLATE_PATH = os.path.join(DATA_DIR, 'template', 'template001.jpg')
OUT_PATH = r'C:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\visualize_labels_output.png'

def main():
    # 1. Read Image
    if not os.path.exists(IMG_PATH):
        print(f"Image not found: {IMG_PATH}")
        return
    img = cv2.imread(IMG_PATH)
    
    # 2. Read Template Size
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Template not found: {TEMPLATE_PATH}")
        return
    temp_img = cv2.imread(TEMPLATE_PATH, 0)
    th, tw = temp_img.shape[:2]
    
    # 3. Read Labels
    if not os.path.exists(LABEL_PATH):
        print(f"Label not found: {LABEL_PATH}")
        return
        
    with open(LABEL_PATH, 'r') as f:
        lines = f.readlines()
        
    print(f"Drawing {len(lines)} labels...")
    
    for i, line in enumerate(lines):
        parts = line.strip().split(',')
        # Format: name, x, y, angle, sx, sy, score
        # HALCON Format: template, row, col, angle, ...
        # row = y, col = x
        # x,y are Center
        
        cy = float(parts[1])
        cx = float(parts[2])
        angle_rad = -float(parts[3]) # Invert angle as requested
        scale = float(parts[4])
        
        target_w = tw * scale
        target_h = th * scale
        
        # Create Poly
        # Corners relative to center
        corners = np.array([
            [-target_w/2, -target_h/2],
            [target_w/2, -target_h/2],
            [target_w/2, target_h/2],
            [-target_w/2, target_h/2]
        ])
        
        # Rotation Matrix
        # HALCON angle is usually radians. 
        # Coordinate system: Row (y) down, Col (x) right.
        # Positive angle in HALCON rotates from x-axis (Col) towards y-axis (Row)? 
        # "The rotation is performed counter-clockwise" in standard math?
        # Let's try standard CCW rotation matrix first.
        # x' = x cos a - y sin a
        # y' = x sin a + y cos a
        
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_corners = []
        for x, y in corners:
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            rotated_corners.append([int(cx + rx), int(cy + ry)])
            
        pts = np.array(rotated_corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw Poly
        cv2.polylines(img, [pts], True, (0, 255, 0), 3)
        
        # Draw Center
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # Draw Text
        cv2.putText(img, f"#{i}", (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite(OUT_PATH, img)
    print(f"Saved visualization to {OUT_PATH}")

if __name__ == "__main__":
    main()
