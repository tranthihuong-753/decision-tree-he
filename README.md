# decision-tree-he

## Lưu ý quan trọng về tham số

Thay đổi hai tham số sau **ảnh hưởng trực tiếp đến kết quả huấn luyện và độ chính xác của mô hình**:

```cpp
size_t number_threshold = 31; // 31 Là số cố định giá trị ngưỡng theo nghiên cứu paper 
size_t max_depth = 2;
```

## 📊 Kết quả thực nghiệm

### 🌸 Tập Iris – `ten.cpp`

**Cấu hình tham số**
- `number_threshold (theta) = 31`
- `max_depth = 2`

**Kết quả**
- **Accuracy**: `86.67%`

**F1-score (Macro Average)**
- Number of Classes (L): `3`
- Number of Samples (N): `30`

| Class | Precision | Recall | F1-Score |
|------:|----------:|-------:|---------:|
| 0 | 0.9000 | 0.9000 | 0.9000 |
| 1 | 0.8333 | 1.0000 | 0.9091 |
| 2 | 0.8750 | 0.7000 | 0.7778 |

---

### 🐶🐱 Tập Dog–Cat – `elevent.cpp`

**Cấu hình tham số**
- `number_threshold (theta) = 31`
- `max_depth = 2`

**Kết quả**
- **Accuracy**: `56.52%`

**F1-score (Macro Average)**
- Number of Classes (L): `2`
- Number of Samples (N): `23`

| Class | Precision | Recall | F1-Score |
|------:|----------:|-------:|---------:|
| 0 | 0.7778 | 0.4667 | 0.5833 |
| 1 | 0.4286 | 0.7500 | 0.5455 |

---

## 🚀 Cách chạy chương trình

### 1️⃣ Cấu hình môi trường
Thực hiện cấu hình và cài đặt theo hướng dẫn trong README của repository sau:  
👉 https://github.com/tranthihuong-753/decision-tree-he.git

---

### 2️⃣ Mô tả các file source chính

#### `elevent.cpp`
- Chương trình **hoàn chỉnh** thực hiện:
  - Huấn luyện (train)
  - Dự đoán (predict)
- Trên **dữ liệu đã được mã hóa (Full Homomorphic Encryption)**
- Áp dụng cho tập **Dog–Cat**:
  - 100 chiều đặc trưng
  - 111 mẫu dữ liệu

#### `ten.cpp`
- Chương trình **hoàn chỉnh** thực hiện:
  - Huấn luyện (train)
  - Dự đoán (predict)
- Trên **dữ liệu đã được mã hóa (Full Homomorphic Encryption)**
- Áp dụng cho tập **Iris**:
  - 4 chiều đặc trưng
  - 150 mẫu dữ liệu

---

### 3️⃣ Cấu hình dữ liệu và mô hình
Trước khi biên dịch và chạy, cần chỉnh sửa:
- `src_data`  
  → Đường dẫn tới **file lưu trữ dữ liệu đầu vào**
- `src_model_tree`  
  → Đường dẫn tới **file lưu trữ mô hình cây quyết định dưới dạng mã hóa**

---

### 4️⃣ Cấu hình biên dịch với CMake
Trong file `CMakeLists.txt`:
- Chỉnh sửa:
  - `add_executable(...)`
  - `target_link_libraries(...)`
- Trỏ đúng tới **file `.cpp`muốn chạy** (`elevent.cpp` hoặc `ten.cpp`)

---

## ⚠️ Nhược điểm và hướng cải tiến

### 1️⃣ Số lượng ngưỡng (threshold) cố định
- Hiện tại, giá trị `number_threshold` được cố định, dẫn đến:
  - Mỗi lần huấn luyện luôn sinh ra **32 ngưỡng chia**
  - Không phản ánh tốt phân bố thực tế của dữ liệu
- **Nhược điểm**:
  - Tăng chi phí tính toán không cần thiết
  - Làm chậm quá trình huấn luyện trên dữ liệu mã hóa
- **Hướng cải tiến**:
  - Xây dựng cơ chế **chọn ngưỡng thích nghi (adaptive threshold selection)**
  - Giảm số lượng ngưỡng nhưng vẫn đảm bảo khả năng phân tách tốt
  - Ưu tiên các ngưỡng “đại diện” thay vì xét toàn bộ ngưỡng cố định

---

### 2️⃣ Chi phí lưu và sử dụng khóa Galois
- Việc **lưu và sử dụng Galois keys** trong FHE hiện tại tiêu tốn nhiều thời gian
- **Nhược điểm**:
  - Làm tăng đáng kể thời gian khởi tạo và huấn luyện
- **Hướng cải tiến**:
  - Nghiên cứu các phương án **giảm hoặc tránh sử dụng Galois keys**
  - Đánh giá kỹ **trade-off giữa mức độ bảo mật và thời gian thực thi**
  - Chỉ sử dụng Galois keys trong những phép toán thực sự cần thiết

---

### 3️⃣ Thời gian huấn luyện trên tập Dog–Cat (100 chiều)
- Tập Dog–Cat có:
  - Số chiều đặc trưng lớn (**100 chiều**)
  - Dữ liệu được xử lý hoàn toàn trong miền mã hóa
- **Nhược điểm**:
  - Thời gian huấn luyện dự kiến **kéo dài hơn 1 tuần** khi chạy trên CPU
- **Hướng cải tiến**:
  - Chuyển từ **CPU sang GPU** để tăng tốc các phép toán nặng
  - Khai thác song song (parallelism) trong các phép toán đồng hình

---

### 4️⃣ Suy giảm độ chính xác và chi phí thời gian khi mã hóa
- So sánh giữa huấn luyện trên dữ liệu thường và dữ liệu mã hóa:

| Tiêu chí | Dữ liệu thường | Dữ liệu mã hóa (FHE) |
|--------|----------------|----------------------|
| Accuracy | 100% | ~86.67% |
| Thời gian huấn luyện | ~1 phút | ~53 phút |

- **Nhận xét**:
  - Mã hóa đồng hình làm tăng mạnh thời gian huấn luyện
  - Đồng thời gây suy giảm độ chính xác do:
    - Xấp xỉ số học
    - Giới hạn độ sâu và độ phức tạp mô hình
- **Hướng cải tiến**:
  - Tối ưu tham số mã hóa
  - Nghiên cứu các kỹ thuật xấp xỉ hiệu quả hơn
  - Cân nhắc mô hình lai (hybrid) giữa plaintext và FHE

---

