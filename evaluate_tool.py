import os
import cv2
import glob
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, scale as shapely_scale
from services.template_matching_service import process_template_matching

# Configuration
DATA_DIR = r'C:\Users\NQT\Desktop\OneDrive\02_ACADEMICS\01_HK1\ComputerVision\00_Thesis\EE5205_MatchingTool_Gr2\Data'
LABEL_DIR = os.path.join(DATA_DIR, 'labels')
TEMPLATE_DIR = os.path.join(DATA_DIR, 'template')
IOU_THRESHOLD = 0.5

def create_rotated_rect(x, y, w, h, angle):
    """
    Creates a helper shapely Polygon for a rotated rectangle.
    x, y: Top-left corner (before rotation? No, usually x,y is top-left of unrotated rect)
    Wait, the service returns x, y as top-left of the bounding rect *before* rotation around center?
    Let's check service logic:
        w_scaled = width * scale_factor
        h_scaled = height * scale_factor
        center_x = point[0] + w_scaled / 2  # point[0] is x
        center_y = point[1] + h_scaled / 2  # point[1] is y
        rect_points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        rotate(poly, angle, origin=(center_x, center_y))
    """
    center_x = x + w / 2
    center_y = y + h / 2
    
    # Create unrotated rectangle (Counter-clockwise order)
    points = [
        (x, y),
        (x + w, y),
        (x + w, y + h),
        (x, y + h)
    ]
    poly = Polygon(points)
    # Rotate around center. Note: Shapely rotates counter-clockwise by default.
    # Check service: rotate(poly, angle, origin=(center_x, center_y), use_radians=False)
    # Service uses degrees.
    rotated_poly = rotate(poly, angle, origin=(center_x, center_y), use_radians=False)
    return rotated_poly

def calculate_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return intersection / union

def parse_label_line(line):
    # Format: template_name, x, y, angle, scale_x, scale_y, score
    parts = line.strip().split(',')
    template_name = parts[0]
    x = float(parts[1])
    y = float(parts[2])
    angle = float(parts[3])
    # Assuming the label format matches the service output roughly
    # output: "x", "y", "angle", "scale", "score"
    # label: scale_x, scale_y might be redundant if aspect ratio is fixed, or separate.
    # Service returns single scale. Let's use scale_x for now or average.
    scale_x = float(parts[4])
    scale_y = float(parts[5]) 
    score = float(parts[6])
    return template_name, x, y, angle, scale_x, scale_y, score

def main():
    image_paths = glob.glob(os.path.join(DATA_DIR, '*.jpg'))
    if not image_paths:
        print("No images found in Data directory.")
        return

    total_tp = 0
    total_fp = 0
    total_fp = 0
    total_fn = 0
    all_roi_times = []
    
    print(f"{'Image':<10} {'Template':<15} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Full Match details'}")
    print("-" * 80)

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_file = os.path.join(LABEL_DIR, img_name.replace('.jpg', '.txt'))
        
        if not os.path.exists(label_file):
            print(f"Label file not found for {img_name}, skipping.")
            continue
            
        # Group GT by template
        gt_by_template = {}
        with open(label_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                t_name, x, y, angle, sx, sy, score = parse_label_line(line)
                if t_name not in gt_by_template:
                    gt_by_template[t_name] = []
                gt_by_template[t_name].append({'x': x, 'y': y, 'angle': angle, 'scale': sx}) # Use sx as scale

        # Read Image
        with open(img_path, 'rb') as f:
            image_bytes = f.read()

        # Process for each template found in GT
        # Note: We only evaluate on templates present in the GT file for this image? 
        # Or should we iterate all available templates?
        # Usually we want to find all instances of all templates. 
        # But if the GT only annotates specific templates, we should probably stick to those 
        # OR run against all templates and treat missing ones in GT as FP?
        # For this script, let's iterate over unique templates found in the GT file for this image
        # to avoid ambiguity about what *should* be there.
        
        unique_templates = list(gt_by_template.keys())
        
        for t_name in unique_templates:
            template_path = os.path.join(TEMPLATE_DIR, t_name)
            if not os.path.exists(template_path):
                print(f"Template {t_name} not found, skipping.")
                continue
                
            with open(template_path, 'rb') as f:
                template_bytes = f.read()
            
            # Get Template Size for polygon construction
            temp_img = cv2.imdecode(np.frombuffer(template_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            t_h, t_w = temp_img.shape[:2]

            # Run Service
            # Note: Threshold is hardcoded in service helper or passed?
            # User requested threshold 0.2 and edge_base=True
            try:
                result, _ = process_template_matching(image_bytes, template_bytes, threshold=0.2, edge_base=True)
                if 'roi_times' in result:
                    all_roi_times.extend(result['roi_times'])
            except Exception as e:
                print(f"Error processing {img_name} with {t_name}: {e}")
                continue

            predictions = result['matches']
            
            # Ground Truths for this template
            gts = gt_by_template[t_name]
            
            # Match Predictions to GTs
            # We need to compute IoU for all pairs and assign matches
            # Simple greedy assignment
            
            gt_polys = []
            for gt in gts:
                # GT is HALCON format: Row(y), Col(x), Angle(rad)
                # Confirmed: x,y are Center. Angle needs to be negated.
                
                target_w = t_w * gt['scale']
                target_h = t_h * gt['scale']
                
                # HALCON: parts[1]=Row(y), parts[2]=Col(x)
                cy = gt['y'] # In parse_label_line, we mapped parts[1]->x, parts[2]->y. Let's re-read that function or just swap here.
                # Currently: parse_label_line: x=parts[1], y=parts[2].
                # So gt['x'] holds Row(y) and gt['y'] holds Col(x).
                
                center_y = gt['x'] 
                center_x = gt['y']
                angle_rad = -gt['angle'] # Negate angle
                
                # Create unrotated box centered at (center_x, center_y)
                points_unrotated = [
                    (center_x - target_w/2, center_y - target_h/2),
                    (center_x + target_w/2, center_y - target_h/2),
                    (center_x + target_w/2, center_y + target_h/2),
                    (center_x - target_w/2, center_y + target_h/2)
                ]
                poly_gt = Polygon(points_unrotated)
                
                # Rotate around center. Shapely uses degrees by default, so use_radians=True
                poly_gt = rotate(poly_gt, angle_rad, origin=(center_x, center_y), use_radians=True)
                
                gt_polys.append({'poly': poly_gt, 'matched': False})

            tp = 0
            fp = 0
            
            # Iterate pred and try to match with a GT
            for pred in predictions:
                # Pred is Top-Left (Unrotated Frame), Angle in Degrees
                p_w = t_w * (pred['scale'] / 100.0) # Service returns %
                p_h = t_h * (pred['scale'] / 100.0)
                
                px, py = pred['x'], pred['y']
                # Pred Center
                pcx = px + p_w / 2
                pcy = py + p_h / 2
                
                points_pred_unrotated = [
                    (px, py),
                    (px + p_w, py),
                    (px + p_w, py + p_h),
                    (px, py + p_h)
                ]
                poly_pred = Polygon(points_pred_unrotated)
                # Pred angle is Degrees
                poly_pred = rotate(poly_pred, pred['angle'], origin=(pcx, pcy), use_radians=False)
                
                best_iou = 0
                best_gt_idx = -1
                
                for idx, gt_item in enumerate(gt_polys):
                    if gt_item['matched']: continue
                    
                    if not poly_pred.is_valid or not gt_item['poly'].is_valid:
                        continue
                        
                    intersection = poly_pred.intersection(gt_item['poly']).area
                    union = poly_pred.union(gt_item['poly']).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                if best_iou >= IOU_THRESHOLD:
                    tp += 1
                    gt_polys[best_gt_idx]['matched'] = True
                else:
                    fp += 1
            
            fn = len([g for g in gt_polys if not g['matched']])
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            print(f"{img_name:<10} {t_name:<15} {prec:.2f}      {rec:.2f}      {f1:.2f}      TP:{tp} FP:{fp} FN:{fn}")

            # --- VISUALIZATION ---
            vis_dir = "evaluation_results"
            os.makedirs(vis_dir, exist_ok=True)
            vis_img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

            # Draw GTs (Green)
            for gt_item in gt_polys:
                if gt_item['poly'].is_empty: continue
                # Get points from shapely poly
                x, y = gt_item['poly'].exterior.coords.xy
                # Shapely returns float, need int for cv2
                pts = np.array([list(z) for z in zip(x, y)]).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2) # Green
                # Optional: Mark center
                # cx, cy = gt_item['poly'].centroid.x, gt_item['poly'].centroid.y
                # cv2.circle(vis_img, (int(cx), int(cy)), 3, (0, 255, 0), -1)

            # Draw Predictions (Red)
            # Re-calculate Pred Polys for drawing (since we didn't store them all in a list outside loop)
            for pred in predictions:
                p_w = t_w * (pred['scale'] / 100.0)
                p_h = t_h * (pred['scale'] / 100.0)
                px, py = pred['x'], pred['y']
                pcx = px + p_w / 2
                pcy = py + p_h / 2
                
                points_pred = [
                    (px, py),
                    (px + p_w, py),
                    (px + p_w, py + p_h),
                    (px, py + p_h)
                ]
                poly_pred = Polygon(points_pred)
                poly_pred = rotate(poly_pred, pred['angle'], origin=(pcx, pcy), use_radians=False)
                
                x, y = poly_pred.exterior.coords.xy
                pts = np.array([list(z) for z in zip(x, y)]).astype(np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(vis_img, [pts], True, (0, 0, 255), 2) # Red

            procecssed_path = os.path.join(vis_dir, f"vis_{img_name}")
            cv2.imwrite(procecssed_path, vis_img)


    print("-" * 80)
    print("-" * 80)
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_rec / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0
    
    print(f"OVERALL: Precision: {overall_prec:.3f}, Recall: {overall_rec:.3f}, F1-Score: {overall_f1:.3f}")
    print(f"CONFUSION MATRIX: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    
    if all_roi_times:
        avg_roi_time = sum(all_roi_times) / len(all_roi_times)
        print(f"Average Processing Time per ROI: {avg_roi_time:.4f}s (Data from {len(all_roi_times)} ROIs)")
    else:
        print("No ROI timing data available.")

if __name__ == "__main__":
    main()
