cmake_minimum_required(VERSION 3.12)
project(DecisionTreeSEAL LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Tìm thư viện SEAL đã được cài đặt (Find_package hoạt động nhờ bước INSTALL ở Pha 2)
find_package(SEAL REQUIRED)

# Thêm file thực thi
add_executable(seal_ckks_example seal_ckks_example.cpp)
# add_executable(testt testt.cpp)

# Liên kết file thực thi với thư viện SEAL
target_link_libraries(seal_ckks_example SEAL::seal)
# target_link_libraries(testt SEAL::seal)

# Thêm đường dẫn đến thư mục chứa file examples.h
include_directories("C:/hu/decision-tree-he/SEAL/native/examples") 