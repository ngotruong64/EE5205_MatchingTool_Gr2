# Object Detection & Template Matching Tool (Group 2)

Dự án này cung cấp các công cụ để thực hiện **Template Matching** bất biến với phép xoay và tỷ lệ (Invariant Template Matching), phục vụ cho các hệ thống thị giác máy tính trong tự động hóa và robotics.

Hệ thống bao gồm:
1.  **API Server (FastAPI)**: Backend xử lý luồng logic chính.
2.  **GUI Client (Tkinter)**: Giao diện người dùng để test nhanh kết quả matching.
3.  **Benchmark & Evaluation Tools**: Các script đánh giá hiệu năng và độ chính xác của thuật toán.

## Tính năng nổi bật
*   **Template Matching bất biến (Invariant)**: Hỗ trợ tìm kiếm mẫu vật thể bị xoay hoặc thay đổi kích thước.
*   **Phương pháp V2 (Proposed)**: Cải tiến tốc độ và độ chính xác so với phương pháp truyền thống, hỗ trợ đa luồng (Multi-threading).
*   **Hỗ trợ ROI (Region of Interest)**: Cho phép chọn vùng quan tâm để tăng tốc độ xử lý.

---

## Cấu trúc dự án

*   `main.py`: Entry point của API Server (FastAPI).
*   `gui_client.py`: Ứng dụng Desktop Client để tương tác trực quan.
*   `services/`: Chứa các service xử lý logic (Template Matching V1, V2).
*   `utils/`: Các hàm tiện ích (xử lý hình ảnh, toán học).
*   `benchmark_methods.py`: Script so sánh tốc độ giữa các phương pháp.
*   `evaluate_tool.py`: Công cụ đánh giá độ chính xác (Precision, Recall, F1) dựa trên Ground Truth.
*   `Data/`: Chứa dữ liệu hình ảnh và nhãn (Labels) để test.

---

## Cài đặt

Yêu cầu: **Python 3.8+**

1.  **Clone dự án:**
    ```bash
    git clone https://github.com/ngotruong64/EE5205_MatchingTool_Gr2.git
    cd EE5205_MatchingTool_Gr2
    ```

2.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Hướng dẫn sử dụng

### 1. Khởi chạy API Server
Server sẽ chạy tại `http://127.0.0.1:8000`.

```bash
python -m uvicorn main:app --reload
```

### 2. Sử dụng GUI Client
Mở một terminal **mới** (sau khi đã chạy API Server) và chạy:

```bash
python gui_client.py
```
*   **Load Image**: Chọn ảnh gốc (Scene).
*   **Load Template**: Chọn ảnh mẫu (Template).
*   **Detect**: Gửi yêu cầu lên server và hiển thị kết quả.

### 3. Chạy Benchmark (So sánh tốc độ)
Script này sẽ chạy so sánh thời gian xử lý giữa OpenCV gốc, Thuật toán Invariant cũ, và Thuật toán V2 đề xuất.

```bash
python benchmark_methods.py
```

### 4. Chạy Evaluation (Đánh giá độ chính xác)
Script này sẽ tính toán Precision, Recall, F1-Score trên tập dữ liệu test (trong thư mục `Data/`).

```bash
python evaluate_tool.py
```
Kết quả đánh giá trực quan sẽ được lưu trong thư mục `evaluation_results/`.

---

## Tác giả
**EE5205 - Group 2**