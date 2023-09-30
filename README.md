# VGG16-based pipeline

A deep learning pipeline based on (but much smaller than) the good old
[VGGNET-16 model](https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide).

## Prepare environment

### `libtorch`
* While PyTorch is used in training, `libtorch` is used to productionize the
model in C++ to enhance performance.

* Ubuntu 22.04 provides package `libtorch-dev` but seems it doesn't work.

* We need to manually download the corresponding zip file from
[here](https://pytorch.org/get-started/locally/) then unzip it.
  * For the sake of easy management, copy `./include/*` to
  `/usr/local/include/torch/` and copy `./lib/*` to `/usr/local/lib/torch/`
  * Add the share objects path to `LD_LIBRARY_PATH`:
  `export LD_LIBRARY_PATH=/usr/local/lib/torch/:$LD_LIBRARY_PATH`

### `OpenCV` and `FFmpeg`

* `OpenCV` (and `FFmpeg` as its Video IO backend) is used to decode/manipulate images before sending to them to `libtorch`,

* Refer to instructions [here](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/c-cpp/cpp/06_poc/05_cudacodec-vs-ffmpeg)


### `Swagger`

* Swagger is used to interact with the prediction daemon on the fly.

* Build Oatpp:
```Bash
git clone https://github.com/oatpp/oatpp.git
mkdir build
cd build
make -j4
sudo make install
```

* Build oatpp-swagger:
```Bash
git clone https://github.com/oatpp/oatpp-swagger.git
mkdir build
cd build
make -j4
sudo make install
```

### Install other libraries

* `spdlog` for logging: `apt install libspdlog-dev`
* `nlohmann-json3` for JSON support: `apt install nlohmann-json3-dev`
* `ZeroMQ` for message queue: `apt install libzmq3-dev`
