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


class TemplateMatcherApp:
    def __init__(self, root):
        """Hàm khởi tạo giao diện chính."""
        self.root = root
        self.root.title("Object detection with orientation")
        self.root.geometry("1000x700")

        # Biến lưu đường dẫn file
        self.image_path = tk.StringVar()
        self.template_path = tk.StringVar()

        # Biến lưu ảnh gốc (PIL Image) và PhotoImage cho kết quả trên Canvas
        self.current_result_image = None
        self.result_photo = None
        self.canvas_image_id = None
        self.canvas_text = None

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

        ttk.Label(control_frame, text="Control panel", font=("Helvetica", 16, "bold")).pack(pady=10)

        # --- Chọn ảnh gốc ---
        image_frame = ttk.LabelFrame(control_frame, text="1. Image", padding="10")
        image_frame.pack(fill=tk.X, pady=5)

        ttk.Button(image_frame, text="Browse Image",
                   command=lambda: self.select_file(self.image_path, self.image_preview_label)).pack(fill=tk.X)
        ttk.Label(image_frame, textvariable=self.image_path, wraplength=250).pack(pady=5)
        self.image_preview_label = ttk.Label(image_frame, text="Image Preview")
        self.image_preview_label.pack(pady=5)

        # --- Chọn ảnh mẫu ---
        template_frame = ttk.LabelFrame(control_frame, text="2. Template", padding="10")
        template_frame.pack(fill=tk.X, pady=5)

        ttk.Button(template_frame, text="Browse Template",
                   command=lambda: self.select_file(self.template_path, self.template_preview_label)).pack(fill=tk.X)
        ttk.Label(template_frame, textvariable=self.template_path, wraplength=250).pack(pady=5)
        self.template_preview_label = ttk.Label(template_frame, text="Template Preview")
        self.template_preview_label.pack(pady=5)

        # --- Cài đặt tham số ---
        params_frame = ttk.LabelFrame(control_frame, text="3. Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=10)

        # Threshold Slider
        ttk.Label(params_frame, text=f"Threshold:").pack(anchor=tk.W)
        self.threshold_slider = ttk.Scale(params_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL,
                                          variable=self.threshold_var, command=self.update_threshold_label)
        self.threshold_slider.pack(fill=tk.X)
        self.threshold_label = ttk.Label(params_frame, text=f"{self.threshold_var.get():.2f}")
        self.threshold_label.pack()

        # Checkboxes
        ttk.Checkbutton(params_frame, text="Edge-based Matching", variable=self.edge_base_var).pack(anchor=tk.W)
        ttk.Checkbutton(params_frame, text="Check overlap", variable=self.check_overlap_var).pack(
            anchor=tk.W)

        # --- Nút thực thi ---
        ttk.Button(control_frame, text="Detect", command=self.start_matching_thread,
                   style="Accent.TButton").pack(fill=tk.X, ipady=10, pady=20)

        # --- Thanh trạng thái ---
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM,
                                                                                                   fill=tk.X)

        # --- KHUNG KẾT QUẢ (BÊN PHẢI) - Đã đổi sang Canvas ---
        result_frame = ttk.Frame(self.root, padding="10")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(result_frame, text="Result", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Canvas để hiển thị ảnh kết quả và hỗ trợ resize/zoom
        self.result_canvas = tk.Canvas(result_frame, bg="gray20", highlightthickness=0)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # Liên kết sự kiện thay đổi kích thước cửa sổ với hàm resize
        self.result_canvas.bind('<Configure>', self.resize_and_display_image)

        # Chữ ban đầu trên canvas
        # Cần đợi canvas được tạo xong để có kích thước, nhưng ta dùng cách này để hiển thị ngay
        # Tọa độ 500, 500 chỉ là ước lượng ban đầu, sẽ được căn giữa khi resize lần đầu
        self.canvas_text = self.result_canvas.create_text(
            10, 10, text="The result will show here.", fill="white",
            font=("Helvetica", 16), anchor=tk.NW
        )

    def select_file(self, path_var, preview_label):
        """Mở hộp thoại chọn file và hiển thị ảnh xem trước."""
        file_path = filedialog.askopenfilename(
            title="Select image",
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

        self.status_var.set("Processing, please wait...")
        self.root.update_idletasks()

        # Tạo và bắt đầu thread
        thread = threading.Thread(target=self.call_api)
        thread.daemon = True
        thread.start()

    def call_api(self):
        """Hàm thực hiện gọi API (chạy trong thread)."""
        params = {
            'threshold': self.threshold_var.get(),
            'edge_base': self.edge_base_var.get(),
            'check_overlap': self.check_overlap_var.get(),
            'return_image': True
        }

        try:
            with open(self._full_image_path, 'rb') as image_file, open(self._full_template_path, 'rb') as template_file:
                # SỬA LỖI Ở ĐÂY: Thêm 'image/png' để chỉ định Content-Type
                files = {
                    'image': (os.path.basename(self._full_image_path), image_file, 'image/png'),
                    'template': (os.path.basename(self._full_template_path), template_file, 'image/png')
                }

                response = requests.post(SERVER_URL, files=files, params=params, timeout=300)
                response.raise_for_status()

                # Cập nhật ảnh kết quả trên main thread
                self.root.after(0, self.display_result_image, response.content)
                self.status_var.set("Finish!")

        except requests.exceptions.RequestException as e:
            self.root.after(0, messagebox.showerror, "Error API", f"Cannot connect or server error:\n{e}")
            self.status_var.set("Error!")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Unknown:\n{e}")
            self.status_var.set("Error!")

    def display_result_image(self, image_bytes):
        """Lưu ảnh gốc và gọi hàm hiển thị/resize."""
        try:
            img_data = io.BytesIO(image_bytes)
            # Lưu trữ ảnh gốc (PIL Image)
            self.current_result_image = Image.open(img_data)

            # Xóa chữ ban đầu trên canvas và đặt lại vị trí
            if self.canvas_text:
                self.result_canvas.delete(self.canvas_text)
                self.canvas_text = None

            # Hiển thị và scale ảnh theo kích thước canvas hiện tại
            self.resize_and_display_image()

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể hiển thị ảnh kết quả: {e}")
            self.current_result_image = None

    def resize_and_display_image(self, event=None):
        """Scale ảnh kết quả để vừa với kích thước Canvas, giữ nguyên tỉ lệ."""
        if not self.current_result_image:
            # Nếu chưa có ảnh, căn giữa lại text nếu nó đang hiển thị
            if self.canvas_text:
                canvas_width = self.result_canvas.winfo_width()
                canvas_height = self.result_canvas.winfo_height()
                self.result_canvas.coords(self.canvas_text, canvas_width // 2, canvas_height // 2)
                self.result_canvas.itemconfig(self.canvas_text, anchor=tk.CENTER)
            return

        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        original_width, original_height = self.current_result_image.size

        # Tính toán tỉ lệ scale để fit vào canvas
        ratio_w = canvas_width / original_width
        ratio_h = canvas_height / original_height

        # Chọn tỉ lệ nhỏ hơn để đảm bảo ảnh vừa vặn hoàn toàn
        scale_ratio = min(ratio_w, ratio_h)

        # Tính kích thước mới
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # Resize ảnh
        resized_img = self.current_result_image.resize((new_width, new_height))

        # Tạo PhotoImage mới và giữ tham chiếu
        self.result_photo = ImageTk.PhotoImage(resized_img)

        # Xóa ảnh cũ trên canvas (nếu có)
        if self.canvas_image_id:
            self.result_canvas.delete(self.canvas_image_id)

        # Hiển thị ảnh mới ở trung tâm canvas
        x = canvas_width // 2
        y = canvas_height // 2

        self.canvas_image_id = self.result_canvas.create_image(x, y, image=self.result_photo)

        # Đảm bảo ảnh mới được hiển thị phía dưới các layer khác (nếu có)
        self.result_canvas.tag_lower(self.canvas_image_id)


if __name__ == "__main__":
    root = tk.Tk()

    # Sử dụng style để nút bấm trông đẹp hơn (tùy chọn)
    style = ttk.Style(root)
    style.configure("Accent.TButton", foreground="green", background="#2a6e9a", font=("Helvetica", 12, "bold"))
    style.map("Accent.TButton", background=[('active', '#3c90c7')])
    style.configure("TLabel", padding=2)

    app = TemplateMatcherApp(root)
    root.mainloop()
