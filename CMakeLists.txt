cmake_minimum_required(VERSION 3.12)
project(DecisionTreeSEAL LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Tìm thư viện SEAL đã được cài đặt (Find_package hoạt động nhờ bước INSTALL ở Pha 2)
find_package(SEAL REQUIRED)

# Thêm file thực thi
# add_executable(seal_ckks_example seal_ckks_example.cpp)
# add_executable(te te.cpp)
# add_executable(eight_server eight_server.cpp)
# add_executable(ten ten.cpp)
# add_executable(two_p two_p.cpp)
# add_executable(eight eight.cpp)
add_executable(elevent elevent.cpp)
# add_executable(soft-step-thuong soft-step-thuong.cpp)

# Liên kết file thực thi với thư viện SEAL
# target_link_libraries(seal_ckks_example SEAL::seal)
# target_link_libraries(eight_server SEAL::seal)
# target_link_libraries(ten SEAL::seal)
# target_link_libraries(te SEAL::seal)
# target_link_libraries(two_p SEAL::seal)
# target_link_libraries(eight SEAL::seal)
target_link_libraries(elevent SEAL::seal)
# target_link_libraries(soft-step-thuong SEAL::seal)

# Thêm đường dẫn đến thư mục chứa file examples.h
include_directories("C:/hu/decision-tree-he/SEAL/native/examples") 
# include_directories(
#     $ENV{HOME}/decision-tree-he/SEAL/native/examples
# )

