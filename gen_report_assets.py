import cv2
import numpy as np
import os
import shutil
from services.template_matching_service import process_template_matching, CONFIG, auto_canny, find_rois_threshold, process_roi
from shapely.geometry import Polygon
from shapely.affinity import rotate

# Setup Output Directory
OUTPUT_DIR = "report_assets"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
# If it exists, we just overwrite files. No need to rmtree which causes PermissionError if files are open.

# 1. Get List of Images
data_dir = "Data"
if not os.path.exists(data_dir):
    print(f"Data directory '{data_dir}' not found.")
    exit()

image_files = sorted([f for f in os.listdir(data_dir) if f.startswith('img') and f.endswith('.jpg')])

for img_file in image_files:
    # Derive paths
    base_name = os.path.splitext(img_file)[0] # img001
    template_name = base_name.replace("img", "template") + ".jpg" # template001.jpg
    
    img_path_full = os.path.join(data_dir, img_file)
    template_path_full = os.path.join(data_dir, "template", template_name)
    
    if not os.path.exists(template_path_full):
        print(f"Skipping {img_file}, template {template_name} not found.")
        continue
        
    print(f"Processing {img_file}...")
    
    # Create valid sub-directory for this image
    img_out_dir = os.path.join(OUTPUT_DIR, base_name)
    if not os.path.exists(img_out_dir):
        os.makedirs(img_out_dir)

    # 1. Load Images
    img = cv2.imread(img_path_full)
    template = cv2.imread(template_path_full)

    if img is None or template is None:
        print(f"Error loading {img_file} or {template_name}!")
        continue

    # Save Original
    cv2.imwrite(os.path.join(img_out_dir, "1_original.jpg"), img)

    # 2. Preprocessing (Exact V1 Logic)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)

    cv2.imwrite(os.path.join(img_out_dir, "2_grayscale.jpg"), img_gray)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=CONFIG['clahe_clip_limit'],
        tileGridSize=CONFIG['clahe_tile_grid_size']
    )
    clahe_image = clahe.apply(img_gray)
    cv2.imwrite(os.path.join(img_out_dir, "3_clahe_enhanced.jpg"), clahe_image)

    # Canny Edges
    edges = auto_canny(clahe_image, sigma=CONFIG['canny_sigma'])
    cv2.imwrite(os.path.join(img_out_dir, "4_canny_edges_raw.jpg"), edges)

    # Morphological Closing
    kernel = np.ones(CONFIG['morph_kernel_size'], np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(img_out_dir, "5_edges_closed.jpg"), edges_closed)

    # 3. Preparation for Matching (Edge Base)
    v = np.median(template_rgb)
    lower = int(max(0, (1.0 - CONFIG['canny_sigma']) * v))
    upper = int(min(255, (1.0 + CONFIG['canny_sigma']) * v))
    img_edges = cv2.Canny(img_gray, lower, upper)
    temp_edges = cv2.Canny(template_rgb, lower, upper)

    img_for_roi = cv2.morphologyEx(img_edges, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(img_out_dir, "6_edges_for_roi_and_matching.jpg"), img_for_roi)

    # 4. Find ROIs
    rois = find_rois_threshold(img_for_roi, edges_closed, template_gray.shape, CONFIG)

    # Draw ROIs
    vis_rois = img.copy()
    for i, roi in enumerate(rois):
        tl, br = roi
        cv2.rectangle(vis_rois, tl, br, (0, 255, 0), 2)
        cv2.putText(vis_rois, str(i), (tl[0], tl[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(os.path.join(img_out_dir, "7_rois_detected.jpg"), vis_rois)

    # --- Measure Processing Time per ROI ---
    import time
    
    # Prepare template for matching
    temp_for_matching = cv2.morphologyEx(temp_edges, cv2.MORPH_CLOSE, kernel)

    roi_times = []
    for i, roi in enumerate(rois):
        start_time = time.time()
        # Run matching for single ROI
        _ = process_roi(roi, img_for_roi, temp_for_matching, 0.5)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        roi_times.append(elapsed_ms)
        # print(f"  ROI {i+1}: {elapsed_ms:.2f} ms") # Reduce noise

    if roi_times:
        avg_conf = sum(roi_times) / len(roi_times)
        print(f"  Avg Time per ROI: {avg_conf:.2f} ms | Total ROIs: {len(rois)}")
    
    # 5. Full Pipeline Result
    with open(img_path_full, 'rb') as f:
        img_bytes = f.read()
    with open(template_path_full, 'rb') as f:
        template_bytes = f.read()

    # Run actual V1 service
    result, result_img = process_template_matching(
        img_bytes, template_bytes, 
        threshold=0.5,
        edge_base=True, 
        check_overlap=True, 
        include_image=True
    )

    if result_img is not None:
        vis_final = img.copy()
        matches = result['matches']
        h, w = template.shape[:2]
        
        for m in matches:
            cx = m['x'] + (w * m['scale']) / 2
            cy = m['y'] + (h * m['scale']) / 2
            
            # Simple box drawing
            rect = ((cx, cy), (w * m['scale'], h * m['scale']), m['angle'])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(vis_final, [box], 0, (0, 0, 255), 2)
            
        cv2.imwrite(os.path.join(img_out_dir, "8_final_result.jpg"), vis_final)

    # --- Comparison Images ---
    def save_comparison(path1, path2, label1, label2, out_name):
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        if img1 is None or img2 is None: return
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2:
            scale = h1 / h2
            img2 = cv2.resize(img2, (int(w2 * scale), h1))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img1, label1, (20, 50), font, 1.5, (0, 255, 255), 3)
        cv2.putText(img2, label2, (20, 50), font, 1.5, (0, 255, 255), 3)
        combined = cv2.hconcat([img1, img2])
        cv2.imwrite(os.path.join(img_out_dir, out_name), combined)

    save_comparison(
        os.path.join(img_out_dir, "2_grayscale.jpg"), 
        os.path.join(img_out_dir, "3_clahe_enhanced.jpg"),
        "Grayscale", "CLAHE Enhanced", "9_compare_preprocessing.jpg"
    )
    save_comparison(
        os.path.join(img_out_dir, "7_rois_detected.jpg"), 
        os.path.join(img_out_dir, "8_final_result.jpg"),
        "ROI Candidates", "Final Matches", "11_compare_result.jpg"
    )

print(f"Done! Check '{OUTPUT_DIR}' for per-image results.")
