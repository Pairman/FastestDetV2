## NCNN Deployment

### Model Conversion

Install [NCNN](https://github.com/Tencent/ncnn) and [PNNX](https://github.com/pnnx/pnnx):

```sh
pip install ncnn pnnx
```

Convert ```.pt``` TorchScript to ```*.ncnn.param``` and ```*.ncnn.bin``` using PNNX:

```sh
pnnx TORCHSCRIPT_PATH  ncnnparam=fastestdetv2.param ncnnbin=fastestdetv2.bin
```

### Building

Build OpenCV library (https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html):

```sh
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
unzip opencv.zip
cmake -S opencv-4.x -B opencv-4.x/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_python=OFF \
  -DBUILD_opencv_java=OFF \
  -DCMAKE_INSTALL_PREFIX=$PWD/opencv-4.x/install
cmake --build opencv-4.x/build -j
cmake --install opencv-4.x/build
```

Build NCNN library (https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux):

```sh
git clone --recursive https://github.com/Tencent/ncnn --depth 1
cmake -S ncnn -B ncnn/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_BUILD_TOOLS=OFF \
  -DNCNN_BUILD_TESTS=OFF \
  -DNCNN_BUILD_BENCHMARK=OFF \
  -DNCNN_BUILD_EXAMPLES=OFF
cmake --build ncnn/build -j
cmake --install ncnn/build --prefix ncnn/install
```

Configure:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=ncnn/install
```

#### Build benchmark:

```sh
cmake --build build --target bench -j
```

#### Build test:

```sh
cmake --build build --target test -j
```

### Running

#### Run benchmark:

```sh
./build/bench [ITERS] [THREADS]
```

#### Run test:

```sh
./build/test [IN_IMG] [OUTPUT_IMG]
```
