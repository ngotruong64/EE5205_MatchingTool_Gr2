from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response, JSONResponse
import time
from services.template_matching_service import process_template_matching
import logging
import cv2
import numpy as np
from utils.result_visualize import visualize_detection_results

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter(prefix="/api", tags=["Template Matching"])

@router.post("/match-template/")
async def match_template(
    image: UploadFile = File(..., description="The larger image to search in (JPEG/PNG)"),
    template: UploadFile = File(..., description="The template image to match (JPEG/PNG)"),
    threshold: float = 0.4,
    edge_base: bool = True,
    check_overlap: bool = False,
    return_image: bool = False  # Thêm tham số để bật/tắt trả về ảnh
):
    """
    Upload an image and a template image, then perform invariant template matching.
    Optionally returns an annotated image with detected matches.

    Args:
        image: The larger image file to search in.
        template: The template image file to match.
        threshold: Similarity threshold for matching (0.0 to 1.0, default: 0.4).
        edge_base: Whether to use edge-based matching (default: True).
        check_overlap: Check overlap between bounding boxes if True.
        return_image: Return the annotated image as an image response (default: False).

    Returns:
        dict: JSON with match details if return_image=False
        Response: Annotated image if return_image=True

    Raises:
        HTTPException: If file format is invalid or processing fails.
    """
    # Kiểm tra định dạng file
    if not image.content_type.startswith("image/") or not template.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files (JPEG/PNG) are supported.")

    # Đọc bytes từ file upload
    try:
        image_bytes = await image.read()
        template_bytes = await template.read()
    except Exception as e:
        logging.error(f"Error reading uploaded files: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded files.")

    # Kiểm tra ngưỡng threshold
    if not 0 <= threshold <= 1:
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1.")

    # Ghi log thông tin request
    logging.info(f"Received request: threshold={threshold}, edge_base={edge_base}, return_image={return_image}")

    # Đo thời gian xử lý
    start_time = time.time()

    # Gọi service xử lý template matching
    try:
        # Luôn yêu cầu trả về ảnh gốc để có thể visualize nếu cần
        results, original_img = process_template_matching(
            image_bytes=image_bytes,
            template_bytes=template_bytes,
            threshold=threshold,
            edge_base=edge_base,
            check_overlap=check_overlap,
            include_image=True
        )
    except Exception as e:
        logging.error(f"Template matching failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Template matching processing failed: {str(e)}")

    # Tính thời gian xử lý
    processing_time = time.time() - start_time
    logging.info(f"Processing completed in {processing_time:.2f} seconds. Matches found: {results['count']}")

    # Nếu return_image=True, vẽ kết quả và trả về ảnh
    if return_image:
        if original_img is None:
            # Xử lý trường hợp không nhận được ảnh gốc
            raise HTTPException(status_code=500, detail="Failed to retrieve original image for visualization.")

        template_shape = tuple(results["template_shape"])
        annotated_img = visualize_detection_results(
            image=original_img,
            result=results,
            template_shape=template_shape,
            show=False,  # Không hiển thị trực tiếp trong server
            save_path=None
        )

        # Encode ảnh đã annotate sang định dạng PNG
        _, buffer = cv2.imencode('.png', annotated_img)
        annotated_bytes = buffer.tobytes()

        # Trả về response dạng ảnh
        return Response(content=annotated_bytes, media_type="image/png")

    # Nếu không, trả về kết quả JSON như mặc định
    return JSONResponse(content=results)