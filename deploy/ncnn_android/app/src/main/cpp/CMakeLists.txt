cmake_minimum_required(VERSION 3.10)

set(ncnn_DIR /tmp/ncnn/install-android-cpu/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(fastestdetv2 SHARED
    jni_interface.cpp
    FastestDetV2.cpp
)

target_link_libraries(fastestdetv2 ncnn jnigraphics)
