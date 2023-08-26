# VGG16-based pipeline

A framework built to be a component in a pipeline of data processing using PyTorch.

## Install Libtorch

* Ubuntu 22.04 provides package `libtorch-dev` but seems it doesn't work.

* We need to manually download the corresponding zip file from
[here](https://pytorch.org/get-started/locally/) then unzip it.
  * For the sake of easy management, copy `./include/*` to
  `/usr/local/include/torch/` and copy `./lib/*` to `/usr/local/lib/torch/`
  * Add the share objects path to `LD_LIBRARY_PATH`:
  `export LD_LIBRARY_PATH=/usr/local/lib/torch/:$LD_LIBRARY_PATH`
