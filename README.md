# CNN pipeline

A pipeline of an ensemble of heterogeneous CNN models.

## Environments

- Install Nvidia GPU driver etc, make sure `nvidia-smi` is working properly.

### Training environment

- Just install miniconda and pytorch should be enough.

### Inference environment

- The inference daemon is developed in C++ to (hopefully) improve real-time performance

#### `libtorch`

- Ubuntu 22.04 provides package `libtorch-dev` but seems it doesn't work.

- We need to manually download the corresponding zip file from
  [here](https://pytorch.org/get-started/locally/) then unzip it.
  - For the sake of easy management, copy `./include/*` to
    `/usr/local/include/torch/` and copy `./lib/*` to `/usr/local/lib/torch/`

#### `OpenCV` and `FFmpeg`

- `OpenCV` (and `FFmpeg` as its Video IO backend) is used to decode/manipulate
  images before sending to them to `libtorch`,

- Refer to instructions
  [here](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/c-cpp/cpp/06_poc/05_cudacodec-vs-ffmpeg)

#### `Swagger`

- Swagger is used to interact with the inference daemon on the fly.

- Build Oatpp:

```Bash
git clone https://github.com/oatpp/oatpp.git
cd oatpp && mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install
```

- Build oatpp-swagger:

```Bash
git clone https://github.com/oatpp/oatpp-swagger.git
cd oatpp-swagger && mkdir build && cd build
cmake ../
make -j4
sudo make install
```

#### Install other libraries

- `readerwriterqueue` for lock-free SPSC queue: `apt install libreaderwriterqueue-dev`
- `spdlog` for logging: `apt install libspdlog-dev`
- `cxxopts` for arguments parsing: `apt install libcxxopts-dev`
- `nlohmann-json3` for JSON support: `apt install nlohmann-json3-dev`
- `cppzmq` for ZeroMQ support: `apt install libzmq3-dev`
- `Magick++` for GIF support: `apt install libmagickcore-dev libmagick++-dev libmagick++-6-headers`
- `clangd` for code intellisense: `apt install clangd`
  - `libstdc++-12` for clangd to work properly: `apt install libstdc++-12-dev`
