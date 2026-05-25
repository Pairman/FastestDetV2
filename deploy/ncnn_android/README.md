## NCNN Android Deployment

Android demo of FastestDetV2 using
[Tencent's NCNN framework](https://github.com/Tencent/ncnn).

### Preparation

Install and setup [JDK 11](https://openjdk.org/projects/jdk/11/), [Android SDK](https://developer.android.com/studio) / [NDK and CMake](https://developer.android.com/studio/projects/install-ndk#cmake).

Export these environment variables:

```sh
# modify the values according to your installations
export JAVA_HOME=JDK11_PATH
export ANDROID_HOME=SDK_PATH
export ANDROID_SDK_ROOT=SDK_PATH
export ANDROID_NDK_HOME=NDK_PATH
```

Place NCNN model files under ```app/src/main/assets/```:

```text
deploy/ncnn_android/app/src/main/assets/fastestdetv2.param
deploy/ncnn_android/app/src/main/assets/fastestdetv2.bin
deploy/ncnn_android/app/src/main/assets/fastestdetv2-2x.param
deploy/ncnn_android/app/src/main/assets/fastestdetv2-2x.bin
```

`assets/` is not generated automatically. You need to prepare and copy both model variants by yourself before building the APK. The app home page exposes a button to switch between `1x` and `2x` at runtime.

### Building

Build NCNN for Android:

```sh
git clone --recursive https://github.com/Tencent/ncnn --depth 1 build/ncnn
cmake -S build/ncnn -B build/ncnn/build-android-arm64-v8a-cpu \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF \
  -DNCNN_VULKAN=OFF \
  -DCMAKE_INSTALL_PREFIX=build/ncnn/install-android-cpu/arm64-v8a
cmake --build build/ncnn/build-android-arm64-v8a-cpu -j
cmake --install build/ncnn/build-android-arm64-v8a-cpu
```

Build release APK:

```sh
./gradlew assembleRelease
```

Release APK will be at ```app/build/outputs/apk/release/app-release.apk```

### Running

Install the APK on the Android device, and run realtime camera inference or benchmark.
