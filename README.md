# CNN pipeline

A pipeline of an ensemble of heterogeneous CNN models.

## Environments

- Install Nvidia GPU driver etc, make sure `nvidia-smi` is working properly.

### Training environment

- Install miniconda, then `pytorch` (need to refer to its official site
  for guideline), `scikit-learn`, `pandas` and `matplotlib` should be enough.

### Inference environment

- The inference daemon is developed in C++ to (hopefully) improve real-time performance
  - Basically all the dependencies can be handled by vcpkg, except perhaps: cuda, OpenCV, FFmpeg and libtorch

#### `libtorch`

- Ubuntu 22.04 provides package `libtorch-dev` but seems it doesn't work.
  - vcpkg also provides libtorch repo, but seems it doesnt work either.

- We need to manually download the corresponding zip file from
  [here](https://pytorch.org/get-started/locally/) then unzip it.
  - `libtorch` is just a bunch of .so and .h files, to make libtorch available to all users, including non-root users,
    and prevent system-level pollution, one may consider copying the ./libtorch to a new /apps/libtorch directory, if
    /apps is the directory that multiple users use to run apps
  - To make cmake aware of this location, set "-DCMAKE_PREFIX_PATH="/apps/libtorch/share/cmake/""

#### `OpenCV` and `FFmpeg`

- `OpenCV` (and `FFmpeg` as its Video IO backend) is used to decode/manipulate
  images before sending to them to `libtorch`,

- Refer to instructions
  [here](https://github.com/alex-lt-kong/cuda-motion/blob/main/helper/build-notes.md)

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
