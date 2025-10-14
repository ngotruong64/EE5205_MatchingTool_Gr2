import cv2
import numpy as np
from typing import Dict, Optional


def visualize_detection_results(
        image: np.ndarray,  # Ảnh gốc (BGR)
        result: Dict,
        template_shape: tuple,
        save_path: Optional[str] = None,
        show: bool = True,
        min_score: float = 0.0,
        max_matches: int = 100
) -> np.ndarray:
    """
    Vẽ kết quả detection lên ảnh gốc và hiển thị/lưu.

    Args:
        image: Ảnh gốc BGR
        result: Dict từ process_template_matching()
        template_shape: (height, width) của template
        save_path: Đường dẫn lưu ảnh (nếu None thì không lưu)
        show: Hiển thị ảnh bằng cv2.imshow()
        min_score: Chỉ vẽ matches có score >= threshold này
        max_matches: Giới hạn số matches vẽ (tránh quá tải)

    Returns:
        Annotated image (BGR)
    """
    annotated = image.copy()
    height, width = template_shape

    # Lọc matches theo score và giới hạn số lượng
    matches = [
                  match for match in result['matches']
                  if match['score'] >= min_score
              ][:max_matches]

    # Màu sắc và styles
    colors = {
        'good': (0, 255, 0),  # Xanh lá
        'overlap': (0, 0, 255),  # Đỏ
        'border': (255, 0, 0),  # Xanh dương
        'text': (255, 255, 255)  # Trắng cho text
    }

    # Statistics cho legend
    good_count = sum(1 for m in matches if not m['overlapped'])
    overlap_count = len(matches) - good_count

    for i, match in enumerate(matches):
        x, y = match['x'], match['y']
        angle = match['angle']

        # FIX 1: Chuyển đổi 'scale' từ phần trăm (vd: 100) sang tỉ lệ (vd: 1.0)
        scale_ratio = match['scale'] / 100.0

        score = match['score']
        overlapped = match['overlapped']

        # Tính kích thước scaled sử dụng scale_ratio
        w_scaled = width * scale_ratio
        h_scaled = height * scale_ratio
        center_x = x + w_scaled / 2
        center_y = y + h_scaled / 2

        # Tính 4 góc của rotated rectangle
        rect = ((center_x, center_y), (w_scaled, h_scaled), angle)
        corners = cv2.boxPoints(rect)
        corners = np.intp(corners)

        # Chọn màu
        color = colors['overlap'] if overlapped else colors['good']

        # Vẽ rotated rectangle
        cv2.drawContours(annotated, [corners], -1, color, 2)  # Giảm độ dày đường viền cho dễ nhìn

        # Vẽ điểm gốc (top-left) thay vì center point để debug
        cv2.circle(annotated, (int(x), int(y)), 5, colors['border'], -1)

        # Vẽ hướng xoay (optional)
        end_x = int(center_x + w_scaled / 2 * np.cos(np.radians(angle)))
        end_y = int(center_y + w_scaled / 2 * np.sin(np.radians(angle)))
        cv2.line(annotated, (int(center_x), int(center_y)), (end_x, end_y), color, 2)

        # FIX 2: Loại bỏ ký tự độ (°) để tránh lỗi font và cast angle sang int
        label = f"S:{score:.2f} A:{int(angle)}"
        if overlapped:
            label = f"⚠️ {label}"

        # Tính vị trí text (trên điểm gốc)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = int(x)
        text_y = int(y - 10)

        # Đảm bảo text không bị vẽ ra ngoài ảnh
        if text_y < 20:
            text_y = int(y + h_scaled + 20)
        if text_x + text_size[0] > annotated.shape[1]:
            text_x = int(annotated.shape[1] - text_size[0] - 5)

        # Background cho text (để dễ đọc)
        cv2.rectangle(annotated,
                      (text_x - 2, text_y - 15),
                      (text_x + text_size[0] + 2, text_y + 5),
                      color, -1)
        cv2.putText(annotated, label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)

    # Thêm legend
    legend_y = 20
    cv2.rectangle(annotated, (5, legend_y - 15), (150, legend_y + 50), (0, 0, 0), -1)  # Background cho legend
    cv2.rectangle(annotated, (10, legend_y), (10 + 15, legend_y + 15), colors['good'], -1)
    cv2.putText(annotated, f"Good: {good_count}", (35, legend_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)

    cv2.rectangle(annotated, (10, legend_y + 25), (10 + 15, legend_y + 40), colors['overlap'], -1)
    cv2.putText(annotated, f"Overlap: {overlap_count}", (35, legend_y + 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1, cv2.LINE_AA)

    # Thêm thông tin tổng quan
    summary = f"Total: {len(matches)} | Template: {width}x{height}"
    cv2.putText(annotated, summary, (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2, cv2.LINE_AA)

    # Hiển thị hoặc lưu
    if show:
        cv2.imshow('Detection Results', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return annotated