# Project: Object Detection System (FastAPI & Tkinter)

Dự án này bao gồm một API server được xây dựng bằng **FastAPI** để phát hiện vị trí và góc xoay của vật thể trên mặt phẳng phục vụ hệ thống pick and place và một client giao diện người dùng **Tkinter** để tương tác với chức năng Template Matching.

Kiến trúc này phù hợp cho việc triển khai các thuật toán thị giác máy tính vào các hệ thống tự động hóa và robotics.

## Yêu cầu Hệ thống

* Python 3.8+
* `pip` (Trình quản lý gói của Python)

## 1. Clone project

Clone toàn bộ dự án và checkout sang nhánh develop:

```bash
git clone https://github.com/ngotruong64/EE5205_MatchingTool_Gr2.git
cd ~/EE5205_MatchingTool_Gr2
git checkout develop
```

## 2. Cài đặt các thư viện

```bash
pip install -r requirements.txt
```

## 3. Khởi chạy API server

```bash
python -m uvicorn main:app --reload
```

## 4. Khởi chạy GUI 
Tạo một terminal mới:
```bash
python gui_client.py  
```
Sau đó load ảnh Template và ảnh Gốc và nhấn Detect để chạy.