# decision-tree-he

## Cach cau hinh 

clone ve xoa di 2 thu muc Project_Build, SEAL_Build, SEAL va cau hinh lai 3 thu muc nay theo https://github.com/tranthihuong-753/Build-Microsoft-SEAL.git

## Cau hinh lai CMakeLists.txt

Xem lai duong dan cua include_directories

## Cach chay 

cd Project_Build

cmake --build . --config Release

Release\seal_ckks_example.exe

## Accuracy 

0.33 

Do độ phức tạp tính toán nên hàm soft-step chỉ mới dùng đến bậc 5, số ngưỡng cũng chọn thấp nhất (2) để test 

### seal_ckks_example_1.cpp

Đã dùng bậc 16, 8 cho train, test 

#### 1 

Số ngưỡng 2 

Độ chính xác 0.3 

#### 2 

Số ngưỡng 10 

Độ chính xác 0.3667 

### seal_ckks_example_2.cpp

Thêm mã hóa one-hot best_feature trong compute_W_phi_best

#### 1 

Số ngưỡng 2 

Độ chính xác 0.3 

#### 2 

Số ngưỡng 10 

Độ chính xác 0.233 

// note max_depth cho cả 4 lần train là 2, có thể tăng lên 