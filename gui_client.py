import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import requests
import io
import os
import threading

# --- Cấu hình ---
SERVER_URL = "http://127.0.0.1:8000/api/match-template/"
PREVIEW_SIZE = (150, 150)  # Kích thước ảnh xem trước
RESULT_SIZE = (1000, 1000)  # Kích thước tối đa của ảnh kết quả


class TemplateMatcherApp:
    def __init__(self, root):
        """Hàm khởi tạo giao diện chính."""
        self.root = root
        self.root.title("Template Matching Client")
        self.root.geometry("1000x700")

        # Biến lưu đường dẫn file
        self.image_path = tk.StringVar()
        self.template_path = tk.StringVar()

        # Biến cho các tham số
        self.threshold_var = tk.DoubleVar(value=0.4)
        self.edge_base_var = tk.BooleanVar(value=True)
        self.check_overlap_var = tk.BooleanVar(value=False)

        # Tạo các thành phần giao diện
        self.create_widgets()

    def create_widgets(self):
        """Tạo và sắp xếp tất cả các widget trên cửa sổ."""

        # --- KHUNG ĐIỀU KHIỂN (BÊN TRÁI) ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Bảng điều khiển", font=("Helvetica", 16, "bold")).pack(pady=10)

        # --- Chọn ảnh gốc ---
        image_frame = ttk.LabelFrame(control_frame, text="1. Chọn ảnh gốc", padding="10")
        image_frame.pack(fill=tk.X, pady=5)

        ttk.Button(image_frame, text="Browse Image",
                   command=lambda: self.select_file(self.image_path, self.image_preview_label)).pack(fill=tk.X)
        ttk.Label(image_frame, textvariable=self.image_path, wraplength=250).pack(pady=5)
        self.image_preview_label = ttk.Label(image_frame, text="Image Preview")
        self.image_preview_label.pack(pady=5)

        # --- Chọn ảnh mẫu ---
        template_frame = ttk.LabelFrame(control_frame, text="2. Chọn ảnh mẫu (template)", padding="10")
        template_frame.pack(fill=tk.X, pady=5)

        ttk.Button(template_frame, text="Browse Template",
                   command=lambda: self.select_file(self.template_path, self.template_preview_label)).pack(fill=tk.X)
        ttk.Label(template_frame, textvariable=self.template_path, wraplength=250).pack(pady=5)
        self.template_preview_label = ttk.Label(template_frame, text="Template Preview")
        self.template_preview_label.pack(pady=5)

        # --- Cài đặt tham số ---
        params_frame = ttk.LabelFrame(control_frame, text="3. Tùy chỉnh tham số", padding="10")
        params_frame.pack(fill=tk.X, pady=10)

        # Threshold Slider
        ttk.Label(params_frame, text=f"Threshold:").pack(anchor=tk.W)
        self.threshold_slider = ttk.Scale(params_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                          variable=self.threshold_var, command=self.update_threshold_label)
        self.threshold_slider.pack(fill=tk.X)
        self.threshold_label = ttk.Label(params_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.pack()

        # Checkboxes
        ttk.Checkbutton(params_frame, text="Sử dụng Edge-based Matching", variable=self.edge_base_var).pack(anchor=tk.W)
        ttk.Checkbutton(params_frame, text="Kiểm tra chồng lấn (Overlap)", variable=self.check_overlap_var).pack(
            anchor=tk.W)

        # --- Nút thực thi ---
        ttk.Button(control_frame, text="RUN", command=self.start_matching_thread,
                   style="Accent.TButton").pack(fill=tk.X, ipady=10, pady=20)

        # --- Thanh trạng thái ---
        self.status_var = tk.StringVar(value="Sẵn sàng")
        ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM,
                                                                                                   fill=tk.X)

        # --- KHUNG KẾT QUẢ (BÊN PHẢI) ---
        result_frame = ttk.Frame(self.root, padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(result_frame, text="Kết quả", font=("Helvetica", 16, "bold")).pack(pady=10)

        self.result_image_label = ttk.Label(result_frame, text="Kết quả sẽ hiển thị ở đây", anchor=tk.CENTER)
        self.result_image_label.pack(fill=tk.BOTH, expand=True)

    def select_file(self, path_var, preview_label):
        """Mở hộp thoại chọn file và hiển thị ảnh xem trước."""
        file_path = filedialog.askopenfilename(
            title="Chọn file ảnh",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if file_path:
            path_var.set(os.path.basename(file_path))  # Chỉ hiển thị tên file
            self.display_image_preview(file_path, preview_label, PREVIEW_SIZE)
            # Lưu đường dẫn đầy đủ vào một thuộc tính khác
            if path_var == self.image_path:
                self._full_image_path = file_path
            else:
                self._full_template_path = file_path

    def display_image_preview(self, path, label_widget, size):
        """Hiển thị ảnh thu nhỏ trên giao diện."""
        try:
            img = Image.open(path)
            img.thumbnail(size)
            photo = ImageTk.PhotoImage(img)
            label_widget.config(image=photo)
            label_widget.image = photo  # Giữ tham chiếu để ảnh không bị xóa
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh xem trước: {e}")

    def update_threshold_label(self, value):
        """Cập nhật nhãn hiển thị giá trị threshold."""
        self.threshold_label.config(text=f"{float(value):.2f}")

    def start_matching_thread(self):
        """Bắt đầu quá trình matching trong một thread riêng để không làm treo GUI."""
        # Kiểm tra đầu vào
        if not hasattr(self, '_full_image_path') or not hasattr(self, '_full_template_path'):
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn cả ảnh gốc và ảnh mẫu.")
            return

        # Vô hiệu hóa nút để tránh click nhiều lần
        # (Cần tìm đúng nút, ở đây đơn giản hóa bằng cách cập nhật trạng thái)
        self.status_var.set("Đang xử lý, vui lòng chờ...")
        self.root.update_idletasks()  # Cập nhật giao diện ngay lập tức

        # Tạo và bắt đầu thread
        thread = threading.Thread(target=self.call_api)
        thread.daemon = True  # Thread sẽ tự tắt khi chương trình chính thoát
        thread.start()

    def call_api(self):
        """Hàm thực hiện gọi API (chạy trong thread)."""
        params = {
            'threshold': self.threshold_var.get(),
            'edge_base': self.edge_base_var.get(),
            'check_overlap': self.check_overlap_var.get(),
            'return_image': True  # GUI luôn yêu cầu trả về ảnh
        }

        try:
            with open(self._full_image_path, 'rb') as image_file, open(self._full_template_path, 'rb') as template_file:
                # SỬA LỖI Ở ĐÂY: Thêm 'image/png' để chỉ định Content-Type
                files = {
                    'image': (os.path.basename(self._full_image_path), image_file, 'image/png'),
                    'template': (os.path.basename(self._full_template_path), template_file, 'image/png')
                }

                response = requests.post(SERVER_URL, files=files, params=params, timeout=300)
                response.raise_for_status()  # Báo lỗi nếu status code là 4xx hoặc 5xx

                # Cập nhật ảnh kết quả trên main thread
                self.root.after(0, self.display_result_image, response.content)
                self.status_var.set("Hoàn thành!")

        except requests.exceptions.RequestException as e:
            self.root.after(0, messagebox.showerror, "Lỗi API", f"Không thể kết nối hoặc server báo lỗi:\n{e}")
            self.status_var.set("Lỗi!")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Lỗi", f"Đã xảy ra lỗi không xác định:\n{e}")
            self.status_var.set("Lỗi!")

    def display_result_image(self, image_bytes):
        """Hiển thị ảnh kết quả trả về từ API."""
        try:
            img_data = io.BytesIO(image_bytes)
            img = Image.open(img_data)
            img.thumbnail(RESULT_SIZE)  # Thu nhỏ nếu ảnh quá lớn
            photo = ImageTk.PhotoImage(img)

            self.result_image_label.config(image=photo)
            self.result_image_label.image = photo
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh kết quả: {e}")


if __name__ == "__main__":
    root = tk.Tk()

    # Sử dụng style để nút bấm trông đẹp hơn (tùy chọn)
    style = ttk.Style(root)
    style.configure("Accent.TButton", foreground="white", background="dodgerblue", font=("Helvetica", 12, "bold"))

    app = TemplateMatcherApp(root)
    root.mainloop()