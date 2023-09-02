# VGG16-based pipeline

A deep learning pipeline based on the good old [VGG16 model](https://www.kaggle.com/code/blurredmachine/vggnet-16-architecture-a-complete-guide).

## Prepare environment

### Install `libtorch`
* While PyTorch is used in training, `libtorch` is used to productionize the
model in C++ to enhance performance.

* Ubuntu 22.04 provides package `libtorch-dev` but seems it doesn't work.

* We need to manually download the corresponding zip file from
[here](https://pytorch.org/get-started/locally/) then unzip it.
  * For the sake of easy management, copy `./include/*` to
  `/usr/local/include/torch/` and copy `./lib/*` to `/usr/local/lib/torch/`
  * Add the share objects path to `LD_LIBRARY_PATH`:
  `export LD_LIBRARY_PATH=/usr/local/lib/torch/:$LD_LIBRARY_PATH`

### Install `OpenCV`

* Refer to instructions [here](https://github.com/alex-lt-kong/the-nitty-gritty/tree/main/c-cpp/cpp/06_poc/05_cudacodec-vs-ffmpeg)

### Install other libraries

```
apt install nlohmann-json3-dev
apt install libspdlog-dev
```

## Useful commands

* `debug.cpp`/`debug.py` can infer images from a directory with images. We may
need to extract frames from a video/GIF file and save all the frames to a
directory:
```
IN_FILE=/tmp/video.mp4
OUT_FILES=/tmp/video_%05d.jpg
# Ignore FPS if input file is a gif animation.
FPS=5
ffmpeg -i "${IN_FILE}" -r "${FPS}" "${OUT_FILES}"
```