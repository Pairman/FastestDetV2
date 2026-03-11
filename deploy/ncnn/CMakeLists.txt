cmake_minimum_required(VERSION 3.10)
project(fastestdetv2_ncnn_benchmark CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE ncnn)

add_executable(test test.cpp)
target_link_libraries(test PRIVATE ncnn ${OpenCV_LIBS})
